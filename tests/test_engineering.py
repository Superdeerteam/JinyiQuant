"""Behavioral checks for configuration, isolation, archives and architecture."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jinyiquant import runner
from jinyiquant.checks import batch_checks, numerical_equal, prediction_checks, verify_artifacts
from jinyiquant.config import DataConfig, Period, RunConfig, load_config
from jinyiquant.data import Batch, build_batch
from jinyiquant.io import file_sha256, read_json, write_json
from jinyiquant.models import LinearModel
from jinyiquant.runner import execute_pipeline, failed_config
from jinyiquant.tracking import Recorder, claim_final_access, new_run_dir


def config_values():
    return {
        "experiment_id": "engineering-test",
        "hypothesis": "Verify isolated research infrastructure",
        "data": {
            "bars_path": "bars.parquet", "labels_path": "labels.parquet",
            "source": "synthetic", "version": "test-v1", "feature_columns": ["close"],
            "lookback": 3, "min_history": 2, "sample_times": ["10:00"],
        },
        "splits": {
            "train": {"start": "2020-01-02", "end": "2020-01-03"},
            "validation": {"start": "2020-01-03", "end": "2020-01-04"},
            "final": {"start": "2020-01-04", "end": "2020-01-05"},
        },
    }


@pytest.mark.parametrize("path,value", [
    (("unrecognized",), 1),
    (("splits", "train", "end"), "2020-01-03 00:00:01"),
    (("splits", "train", "start"), "2020-01-03"),
    (("splits", "validation", "start"), "2020-01-03T00:00:00+08:00"),
    (("data", "feature_columns"), ["close", "target"]),
    (("data", "feature_columns"), ["close", "close"]),
    (("data", "min_history"), 4),
    (("data", "sample_times"), ["14:31"]),
    (("data", "allowed_start"), "2020-01-03"),
    (("model",), {"alpha": float("nan")}),
    (("budget",), {"wall_seconds": 0}),
    (("evaluation_split",), "final"),
])
def test_config_rejects_ambiguous_or_unsafe_research_inputs(path, value):
    raw = config_values()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_resolved_config_snapshot_is_immutable_and_checksum_tracks_effective_parameters(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, config_values())
    config = load_config(path)
    assert config.data.bars_path == (tmp_path / "bars.parquet").resolve()
    assert config.output_root == (tmp_path / "runs").resolve()
    assert config.evaluation_split == "validation"
    assert config.checksum == RunConfig.model_validate(config.snapshot()).checksum
    reordered = dict(reversed(list(config.snapshot().items())))
    assert RunConfig.model_validate(reordered).checksum == config.checksum
    changed = config.snapshot()
    changed["model"]["alpha"] = 2.0
    assert RunConfig.model_validate(changed).checksum != config.checksum
    with pytest.raises(ValidationError):
        config.seed = 7
    with pytest.raises(ValidationError):
        config.splits.train.end = pd.Timestamp("2020-02-01")


def synthetic_batch(values):
    values = np.asarray(values, dtype=float).reshape(-1, 1, 1)
    keys = pd.DataFrame({"date": pd.date_range("2020-01-02 10:00", periods=len(values), freq="D"),
                         "instrument": "A"})
    return Batch(keys, values, np.ones_like(values, dtype=bool),
                 keys.date.to_numpy(copy=True)[:, None], {"min_section_coverage": 1.0})


def test_training_preprocessing_is_frozen_during_inference_and_survives_persistence(tmp_path):
    training = synthetic_batch([-1, 0, 1])
    validation = synthetic_batch([2, 100])
    model = LinearModel("ridge", alpha=1.0)
    progress = model.fit(training, [-3, 0, 3])
    assert progress["loss"] == pytest.approx(0.375)
    np.testing.assert_allclose(model.mean, [0, 1])
    np.testing.assert_allclose(model.scale, [np.sqrt(2 / 3), 1])
    before = (model.mean.copy(), model.scale.copy(), model.weights.copy())
    np.testing.assert_allclose(model.predict(validation), [4.5, 225])
    for old, current in zip(before, [model.mean, model.scale, model.weights]):
        np.testing.assert_array_equal(current, old)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = LinearModel.load(path)
    np.testing.assert_array_equal(restored.predict(validation), model.predict(validation))
    assert set(vars(training)) == {"keys", "x", "observed", "history_dates", "coverage", "history_feature_count"}
    assert training.history_feature_count is None


def test_mean_baseline_and_invalid_training_inputs():
    model = LinearModel("mean")
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(synthetic_batch([1]))
    training = synthetic_batch([1, 2, 3])
    model.fit(training, [1, 4, 7])
    np.testing.assert_array_equal(model.predict(synthetic_batch([-1000, 1000])), [4, 4])
    for target in ([1, 2], [1, np.nan, 3]):
        with pytest.raises(ValueError, match="aligned finite"):
            model.fit(training, target)


@pytest.mark.parametrize("leaky_windows", [False, True])
def test_future_check_exercises_single_section_even_for_constant_model(monkeypatch, leaky_windows):
    from jinyiquant import checks

    cutoff = pd.Timestamp("2020-01-02 10:00")
    bars = pd.DataFrame({"date": [cutoff], "instrument": ["A"], "close": [10.0]})
    config = DataConfig.model_validate(config_values()["data"])
    batch = build_batch(bars, bars[["date", "instrument"]], config)
    model = LinearModel("mean")
    model.fit(batch, [0.1])
    original_build = checks.build_batch

    def build_with_future_leak(frame, keys, options):
        result = original_build(frame, keys, options)
        if (frame.date > cutoff).any():
            result.x += 1
        return result

    if leaky_windows:
        monkeypatch.setattr(checks, "build_batch", build_with_future_leak)
    results = prediction_checks(model, batch, bars, config, 1e-10, 1e-8)
    future = next(item for item in results if item["name"] == "future_market_invariance")
    assert future["details"]["altered_future_rows"] > 0
    assert future["details"]["injected_future_rows"] == 1
    assert future["passed"] is (not leaky_windows)


def test_acceptance_detects_nonfinite_features_future_history_and_boundary_crossing():
    batch = synthetic_batch([1, 2, 3])
    targets = batch.keys.assign(target=1.0, label_end=batch.keys.date + pd.Timedelta(minutes=30))
    split = Period(start="2020-01-02", end="2020-01-05")
    checks = {item["name"]: item["passed"] for item in batch_checks(batch, targets, split, 1.0)}
    assert all(checks.values())
    batch.x[0, 0, 0] = np.nan
    batch.history_dates[0, 0] = np.datetime64("2020-01-06")
    targets.loc[0, "label_end"] = split.end
    checks = {item["name"]: item["passed"] for item in batch_checks(batch, targets, split, 1.0)}
    assert checks["finite_features"] is False
    assert checks["causal_history"] is False
    assert checks["label_boundary"] is False


def test_acceptance_rejects_observed_but_insufficient_history():
    raw = config_values()["data"]
    config = DataConfig.model_validate({**raw, "min_history": 3})
    bars = pd.DataFrame({"date": [pd.Timestamp("2020-01-02 10:00")],
                         "instrument": ["A"], "close": [10.0]})
    batch = build_batch(bars, bars[["date", "instrument"]], config)
    assert batch.coverage["min_section_coverage"] == 0
    targets = bars.assign(target=1.0, label_end=bars.date + pd.Timedelta(minutes=30))
    results = batch_checks(batch, targets, Period(start="2020-01-02", end="2020-01-03"), 1.0)
    assert next(item for item in results if item["name"] == "sample_coverage")["passed"] is False


def test_reproducibility_tolerance_does_not_accept_missing_or_nonfinite_metrics():
    assert numerical_equal({"value": [1.0, None]}, {"value": [1.000001, None]}, 1e-5, 0)
    assert not numerical_equal({"value": [1.0]}, {"value": [1.01]}, 1e-5, 0)
    assert not numerical_equal({"value": 1.0}, {}, 1e-5, 0)
    assert not numerical_equal(float("nan"), float("nan"), 1e-5, 0)


def test_event_schema_and_nested_credentials_are_redacted(tmp_path):
    run_dir = new_run_dir(tmp_path)
    recorder = Recorder(run_dir, "test-experiment")
    recorder.event("loading", "started", nested={"api_key": "example-key"},
                   reason="password=example-password https://user:example-pass@host/path")
    event = read_json(run_dir / "events.jsonl")
    assert event["schema_version"] == 1
    assert event["timestamp_utc"].endswith("+00:00")
    assert event["stage"] == "loading" and event["event_type"] == "started"
    assert event["run_id"] == run_dir.name and event["experiment_id"] == "test-experiment"
    serialized = json.dumps(event)
    assert "example-key" not in serialized
    assert "example-password" not in serialized
    assert "example-pass" not in serialized


def test_invalid_configuration_archive_never_contains_rejected_credential_values(tmp_path):
    raw = config_values()
    secret_value = "test-sensitive-value-a72f"
    raw["api_key"] = secret_value
    with pytest.raises(ValidationError) as raised:
        RunConfig.model_validate(raw)
    run_dir, exit_code = failed_config(tmp_path / "bad-config.json", raised.value)
    assert exit_code == 2
    manifest = read_json(run_dir / "manifest.json")
    assert manifest["status"] == "failed" and manifest["exit_code"] == 2
    assert not read_json(run_dir / "checks.json")["passed"]
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["stage"] == "configuration" and event["event_type"] == "failed" for event in events)
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert secret_value.encode() not in path.read_bytes(), path.name


def test_pipeline_failure_redacts_credentials_in_every_archived_output(tmp_path, monkeypatch):
    run_dir = new_run_dir(tmp_path)
    write_json(run_dir / "config.json", config_values())
    secret_value = "test-private-error-a72f"

    def fail_pipeline(config, destination):
        raise RuntimeError("Upstream rejected password=" + secret_value)

    monkeypatch.setattr(runner, "execute_pipeline", fail_pipeline)
    assert runner.worker(run_dir) == 1
    assert not read_json(run_dir / "checks.json")["passed"]
    for path in run_dir.iterdir():
        assert secret_value.encode() not in path.read_bytes(), path.name


def test_identical_predictions_cannot_prove_reproducibility_after_seed_change(tmp_path):
    """Deterministic baselines can hide changed seeds; the recorded recipe must match."""
    reference, current = tmp_path / "reference", tmp_path / "current"
    raw = config_values()
    original_config = RunConfig.model_validate(raw)
    changed_config = RunConfig.model_validate({**raw, "seed": original_config.seed + 1})
    for directory, config in ((reference, original_config), (current, changed_config)):
        directory.mkdir()
        write_json(directory / "config.json", config.snapshot())
        write_json(directory / "data.json", {"source": "test", "version": "v1"})
        write_json(directory / "metrics.json", {"rank_ic_mean": 0.25})
        write_json(directory / "checks.json", {"passed": True, "checks": []})
        pd.DataFrame({"date": [pd.Timestamp("2020-01-03 10:00")], "instrument": ["A"],
                      "score": [0.25]}).to_parquet(directory / "predictions.parquet", index=False)
        inventory = [{"path": path.name, "sha256": file_sha256(path)} for path in directory.iterdir()]
        write_json(directory / "manifest.json", {"status": "completed", "artifacts": inventory,
                    "config_sha256": config.checksum, "code": {"content_sha256": "same-source"}})
    with pytest.raises(ValueError, match="[Cc]onfig|[Ss]eed|[Pp]arameter|recipe"):
        runner._reproducibility(current, reference, changed_config)


def test_final_access_limit_survives_experiment_renaming(tmp_path):
    raw = config_values()
    raw.update(output_root=str(tmp_path), evaluation_split="final",
               final_access={"purpose": "One predeclared final check", "max_views": 1})
    claim_final_access(RunConfig.model_validate(raw), "first-run")
    raw["experiment_id"] = "renamed-experiment"
    with pytest.raises(ValueError, match="view limit"):
        claim_final_access(RunConfig.model_validate(raw), "second-run")


def test_artifact_verification_detects_tampering_missing_files_and_path_escape(tmp_path):
    run_dir = new_run_dir(tmp_path)
    artifact = run_dir / "metrics.json"
    artifact.write_text('{"rank_ic_mean":0.1}', encoding="utf-8")
    entry = {"path": artifact.name, "sha256": file_sha256(artifact)}
    write_json(run_dir / "manifest.json", {"artifacts": [entry]})
    assert verify_artifacts(run_dir)["passed"]
    artifact.write_text('{"rank_ic_mean":0.9}', encoding="utf-8")
    assert not verify_artifacts(run_dir)["passed"]
    write_json(run_dir / "manifest.json", {"artifacts": [{**entry, "path": "missing.json"}]})
    assert not verify_artifacts(run_dir)["passed"]
    external = tmp_path / "external.json"
    external.write_text("external", encoding="utf-8")
    write_json(run_dir / "manifest.json", {"artifacts": [
        {"path": "../external.json", "sha256": file_sha256(external)}]})
    assert not verify_artifacts(run_dir)["passed"]


def test_architecture_keeps_orchestration_out_of_reusable_core_modules():
    root = Path(__file__).resolve().parents[1] / "src" / "jinyiquant"
    forbidden = {"runner", "cli", "suite", "tracking", "comparison"}
    for module in ("io", "config", "data", "models", "evaluation", "checks"):
        tree = ast.parse((root / f"{module}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {node.module.split(".")[-1]} if node.module else {item.name for item in node.names}
                local = node.level > 0 or (node.module or "").startswith("jinyiquant.")
                assert not (local and names & forbidden), f"{module} imports orchestration: {names}"
            elif isinstance(node, ast.Import):
                names = {item.name.split(".")[1] for item in node.names if item.name.startswith("jinyiquant.")}
                assert not names & forbidden, f"{module} imports orchestration: {names}"


def test_model_kind_literal_matches_torch_backend_contract():
    from typing import get_args

    from jinyiquant.config import TORCH_MODEL_KINDS, ModelConfig

    kinds = set(get_args(ModelConfig.model_fields["kind"].annotation))
    assert kinds == {"ridge", "mean", *TORCH_MODEL_KINDS}
    assert ModelConfig(kind="mlplob").torch_backend
    assert not ModelConfig(kind="ridge").torch_backend


@pytest.mark.parametrize("explicit_universe", [False, True])
def test_evaluation_answer_removal_changes_coverage_but_never_predictions(tmp_path, explicit_universe):
    """Label availability/value is future information and cannot choose inference keys."""
    bars, labels = [], []
    for day in ("2020-01-02", "2020-01-03", "2020-01-04"):
        cutoff = pd.Timestamp(day + " 10:00")
        for stock in range(4):
            instrument = f"{stock:06d}.SZ"
            for lag in range(3):
                bars.append({"date": cutoff - pd.Timedelta(minutes=lag), "instrument": instrument,
                             "close": float(10 + stock + lag / 10)})
            labels.append({"date": cutoff, "instrument": instrument, "target": float(stock - 1.5) / 100,
                           "label_end": cutoff + pd.Timedelta(minutes=30)})
    bars_path, labels_path = tmp_path / "bars.parquet", tmp_path / "labels.parquet"
    pd.DataFrame(bars).to_parquet(bars_path, index=False)
    targets = pd.DataFrame(labels)
    targets.to_parquet(labels_path, index=False)
    raw = config_values()
    raw["data"].update(bars_path=str(bars_path), labels_path=str(labels_path))
    if explicit_universe:
        universe_path = tmp_path / "universe.parquet"
        targets[["date", "instrument"]].to_parquet(universe_path, index=False)
        raw["data"]["instruments_path"] = str(universe_path)
    config = RunConfig.model_validate(raw)
    baseline_dir, changed_dir = tmp_path / "complete-answers", tmp_path / "changed-answers"
    for directory in (baseline_dir, changed_dir):
        directory.mkdir()
        write_json(directory / "manifest.json", {})
    execute_pipeline(config, baseline_dir)
    evaluation = targets.date.eq(pd.Timestamp("2020-01-03 10:00"))
    changed = targets.drop(targets.index[evaluation][0]).copy()
    changed.loc[changed.date.eq(pd.Timestamp("2020-01-03 10:00")), "target"] *= -1234
    changed.to_parquet(labels_path, index=False)
    execute_pipeline(config, changed_dir)
    pd.testing.assert_frame_equal(pd.read_parquet(baseline_dir / "predictions.parquet"),
                                  pd.read_parquet(changed_dir / "predictions.parquet"), check_exact=True)
    assert len(pd.read_parquet(changed_dir / "predictions.parquet")) == 4
    assert len(pd.read_parquet(changed_dir / "scored_predictions.parquet")) == 3
    results = read_json(changed_dir / "checks.json")
    assert results["passed"]
    coverage = next(item for item in results["checks"] if item["name"] == "expected_universe_coverage")
    assert coverage["details"]["missing_answers"] == 1
    assert coverage["details"]["minimum"] == 0.75
