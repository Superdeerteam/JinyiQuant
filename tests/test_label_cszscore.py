"""Variance-matched timestamp z-score labels: formula, control invariance, checkpoints and recovery."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, EvaluationConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from jinyiquant.io import file_sha256, write_json  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402
from scripts import run_label_cszscore as csz  # noqa: E402

TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
DEFAULT = np.array([-.0040, -.0030, -.0020, -.0010, .0200, .0210, .0220, .0230], dtype=np.float32)


def setup(*, timestamps=2, per=4, target=None, mode=csz.MODE, seed=7):
    dates = []
    for index in range(timestamps):
        dates.extend([pd.Timestamp("2020-01-02") + pd.Timedelta(minutes=30 * index)] * per)
    keys = pd.DataFrame({"date": pd.to_datetime(dates),
                         "instrument": [f"S{item}" for item in range(per)] * timestamps})
    x = np.random.default_rng(seed).normal(size=(len(keys), 8, 12)).astype(np.float32)
    batch = Batch(keys, x, np.ones_like(x, dtype=bool),
                  np.full((len(x), 8), np.datetime64("NaT", "ns")), {})
    data = DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic",
                      version="1", feature_columns=lob_columns(3), lookback=8, min_history=2,
                      sample_times=TIMES)
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=1, batch_size=4,
                         target_scaling="train_std", target_centering=mode, device="cpu", log_every_steps=100)
    values = DEFAULT if target is None else np.asarray(target, dtype=np.float32)
    target = np.resize(values, len(keys)).astype(np.float32) if len(values) != len(keys) else values
    return TLOBAdapter(config, data), batch, target, data


def expected_transform(keys, target):
    raw = np.asarray(target, dtype=np.float32).astype(np.float64)
    timestamps = pd.to_datetime(keys["date"]).to_numpy()
    centered = np.zeros_like(raw)
    sigmas = np.zeros(len(np.unique(timestamps)), dtype=np.float64)
    for index, timestamp in enumerate(pd.unique(timestamps)):
        selected = timestamps == timestamp
        centered[selected] = raw[selected] - raw[selected].mean()
        sigmas[index] = raw[selected].std(ddof=0)
    std_rows = np.repeat(sigmas, [int((timestamps == timestamp).sum()) for timestamp in pd.unique(timestamps)])
    q = np.divide(centered, std_rows, out=np.zeros_like(centered), where=std_rows > 0)
    c = centered.std(ddof=0) / q.std(ddof=0)
    return raw, centered, q, c, (c * q).astype(np.float32), sigmas


def test_formula_uses_full_timestamp_float64_ddof_zero_and_matches_pool_variance():
    model, batch, target, _ = setup(timestamps=2, per=4)
    mask = np.ones(len(target), dtype=bool)
    selected, stats = model._training_targets(batch.keys, target, mask)
    raw, centered, q, c, transformed, sigmas = expected_transform(batch.keys, target)
    assert stats["method"] == csz.MODE and stats["ddof"] == 0
    assert stats["timestamps"] == 2 and stats["rows"] == len(target)
    assert stats["centered_std"] == pytest.approx(centered.std(ddof=0))
    assert stats["q_std"] == pytest.approx(q.std(ddof=0))
    assert stats["c"] == pytest.approx(c)
    assert stats["transformed_std"] == pytest.approx(centered.std(ddof=0))
    assert stats["float32_transformed_std"] == pytest.approx(transformed.astype(np.float64).std(ddof=0))
    assert stats["transformed_std"] == pytest.approx(stats["centered_std"])
    assert [row["samples"] for row in stats["per_timestamp"]] == [4, 4]
    assert [row["std"] for row in stats["per_timestamp"]] == pytest.approx(sigmas.tolist())
    np.testing.assert_array_equal(selected, transformed)
    # ddof=1 would give a different per-timestamp standard deviation; the contract is ddof=0.
    assert stats["per_timestamp"][0]["std"] != pytest.approx(target[:4].astype(np.float64).std(ddof=1))


def test_equal_timestamp_standard_deviations_reduce_to_the_centered_control():
    # A constant shift changes each timestamp mean but keeps every within-timestamp std equal.
    target = np.array([-.003, -.001, .001, .003, .001, .003, .005, .007], dtype=np.float32)
    model, batch, target, _ = setup(target=target)
    selected, stats = model._training_targets(batch.keys, target, np.ones(len(target), dtype=bool))
    raw, centered, _, c, _, _ = expected_transform(batch.keys, target)
    assert stats["zero_variance_timestamps"] == 0
    assert c == pytest.approx(centered.std(ddof=0))
    np.testing.assert_allclose(selected, centered.astype(np.float32), atol=1e-12, rtol=0)
    assert stats["transformed_std"] == pytest.approx(stats["centered_std"])


def test_zero_variance_timestamp_is_kept_with_zero_q():
    target = np.array([.01, .01, .01, .01, -.003, -.001, .001, .003], dtype=np.float32)
    model, batch, target, _ = setup(target=target)
    selected, stats = model._training_targets(batch.keys, target, np.ones(len(target), dtype=bool))
    _, centered, q, c, transformed, _ = expected_transform(batch.keys, target)
    assert stats["zero_variance_timestamps"] == 1
    assert stats["zero_variance_timestamp_list"] == [str(batch.keys.date.iloc[0])]
    np.testing.assert_array_equal(selected[:4], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(selected, transformed)
    assert stats["q_std"] == pytest.approx(q.std(ddof=0))
    assert stats["c"] == pytest.approx(c)
    assert stats["transformed_std"] == pytest.approx(centered.std(ddof=0))


def test_whole_pool_zero_q_standard_deviation_fails_before_training():
    target = np.full(8, .01, dtype=np.float32)
    model, batch, target, _ = setup(target=target)
    with pytest.raises(ValueError, match="zero or nonfinite pool q standard deviation"):
        model._training_targets(batch.keys, target, np.ones(len(target), dtype=bool))


def test_original_target_scale_samples_and_batch_order_are_unchanged(monkeypatch):
    centering, batch, target, _ = setup(mode="timestamp_mean_train")
    candidate, _, _, _ = setup(mode=csz.MODE)
    original = torch.randperm
    calls = []

    def spy(n, *args, **kwargs):
        result = original(n, *args, **kwargs)
        calls.append(result.clone())
        return result

    monkeypatch.setattr(torch, "randperm", spy)
    left = centering.fit(batch, target)
    right = candidate.fit(batch, target)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0].numpy(), calls[1].numpy())
    assert left["target_scale"] == pytest.approx(right["target_scale"])
    assert left["target_scale"] == pytest.approx(target.astype(np.float64).std(ddof=0))
    assert (left["steps"], left["samples"], left["samples_seen"]) == (
        right["steps"], right["samples"], right["samples_seen"])
    assert candidate.target_centering_stats["fallback_mean"] == pytest.approx(centering.intercept)


@pytest.mark.parametrize("filename", ["weights.pt", "weights.json"])
def test_csz_checkpoint_keeps_metadata_and_does_not_multiply_c_at_inference(tmp_path, filename):
    model, batch, target, _ = setup()
    model.fit(batch, target)
    expected = model.predict(batch)
    path = tmp_path / filename
    (model.export_text if filename.endswith("json") else model.save)(path)
    restored = TLOBAdapter.load(path)
    assert restored.config["target_centering"] == csz.MODE
    assert restored.target_centering_stats == model.target_centering_stats
    np.testing.assert_allclose(restored.predict(batch), expected, atol=1e-12, rtol=1e-9)
    payload = (torch.load(path, weights_only=True) if filename.endswith("pt")
               else json.loads(path.read_text(encoding="utf-8")))
    assert payload["label"] == "official_vwap_return_timestamp_zscore_train_rms_matched"
    assert payload["prediction_units"] == "relative_return_score"
    assert payload["target_centering_stats"]["method"] == csz.MODE
    assert "per_timestamp" in payload["target_centering_stats"]
    assert "timestamp_means" not in payload
    with torch.inference_mode():
        parts = [model.network(torch.from_numpy(batch.x[offset:offset + model.config["batch_size"]])).cpu().numpy()
                 for offset in range(0, len(batch.x), model.config["batch_size"])]
    manual = np.concatenate(parts).astype(np.float64) * model.target_scale
    np.testing.assert_allclose(expected, manual, atol=1e-12, rtol=1e-9)
    c = model.target_centering_stats["c"]
    assert not np.allclose(expected, manual * c, atol=1e-12, rtol=1e-9)


def test_centered_and_raw_checkpoints_remain_readable(tmp_path):
    for mode in ("none", "timestamp_mean_train"):
        model, batch, target, _ = setup(mode=mode)
        model.fit(batch, target)
        path = tmp_path / f"{mode}.pt"
        model.save(path)
        restored = TLOBAdapter.load(path)
        assert restored.config["target_centering"] == mode
        np.testing.assert_allclose(restored.predict(batch), model.predict(batch), atol=1e-12, rtol=1e-9)


def test_missing_history_fallback_matches_the_centered_control(tmp_path):
    model, batch, target, _ = setup()
    model.fit(batch, target)
    observed = batch.observed.copy()
    observed[0] = False
    probe = Batch(batch.keys, batch.x, observed, batch.history_dates, {})
    model.save(tmp_path / "weights.pt")
    restored = TLOBAdapter.load(tmp_path / "weights.pt")
    scores = restored.predict(probe)
    assert scores[0] == pytest.approx(restored.target_centering_stats["fallback_mean"], abs=1e-15)
    assert scores[0] == pytest.approx(0.0, abs=1e-10)


def test_centered_error_centers_in_float64_and_preserves_near_tied_ranks():
    # These two float32 labels collapse under float32 mean subtraction but stay distinct in float64.
    values = np.array([.02941324934363365, .0028422242030501366, .05467129871249199,
                       -.07364540547132492, .10000000149011612, .10000000894069672,
                       .05988462269306183, .003972210921347141], dtype=np.float32)
    timestamps = [pd.Timestamp("2024-03-01 09:30")] * 8
    targets = pd.DataFrame({"date": timestamps, "instrument": [f"S{item}" for item in range(8)],
                            "target": values})
    predictions = targets[["date", "instrument"]].assign(score=values.astype(np.float64))
    collapsed = (values - np.float32(values.astype(np.float64).mean())).astype(np.float32)
    assert len(np.unique(collapsed)) < len(np.unique(values))
    evaluation = EvaluationConfig(min_cross_section=3)
    centered, daily, _ = csz.centered_error(predictions, targets, evaluation)
    official, official_daily, _ = csz.evaluate(predictions, targets, evaluation, return_sections=True)
    assert daily.rank_ic.iloc[0] == pytest.approx(official_daily.rank_ic.iloc[0], abs=1e-12)
    assert daily.long_short_return.iloc[0] == pytest.approx(official_daily.long_short_return.iloc[0], abs=1e-12)
    assert centered["mse"] != pytest.approx(official["mse"])


def test_plan_guard_accepts_only_the_csz_target_change():
    reference = {"experiment_id": "old", "hypothesis": "old", "output_root": "old",
                 "model": {"target_centering": "none", "target_scaling": "train_std", "max_steps": 1500},
                 "budget": {"wall_seconds": 3600.0, "max_rss_mb": 6144.0, "max_train_samples": 100000}}
    candidate = deepcopy(reference)
    candidate.update(experiment_id="new", hypothesis="new", output_root="new",
                     budget={"wall_seconds": 900.0, "max_rss_mb": 9472.0, "max_train_samples": 8})
    candidate["model"]["target_centering"] = csz.MODE
    assert csz.assert_only_target_transform_changes(reference, candidate, "window") is None
    candidate["model"]["target_scaling"] = "none"
    with pytest.raises(ValueError):
        csz.assert_only_target_transform_changes(reference, candidate, "window")


def test_six_fit_completeness_rejects_a_partial_replay():
    attempts = [{"status": "completed"}] * 5 + [{"status": "failed"}]
    with pytest.raises(ValueError, match="Six completed"):
        csz.require_complete_attempts(attempts)
    assert len(csz.require_complete_attempts([{"status": "completed"}] * 6)) == 6


def test_artifact_identity_rejects_a_changed_weight(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    expected = file_sha256(path)
    assert csz.verify_artifact(path, expected, "fit")
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="artifact identity"):
        csz.verify_artifact(path, expected, "fit")


def test_recovery_requires_all_six_fits_and_predictions_without_training(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {"schema_version": 1})
    plan = {"windows": [{"name": "research_a"}, {"name": "research_b"}]}
    source = tmp_path / "source"
    (source / "inference").mkdir(parents=True)
    write_json(source / "manifest.json", {"status": "failed", "plan_sha256": file_sha256(plan_path)})
    attempts = []
    for window in plan["windows"]:
        fitting = source / "fits" / f"{window['name']}_s11"
        fitting.mkdir(parents=True)
        for filename in ("model.pt", "training.json", "training_identity.json"):
            (fitting / filename).write_text("{}", encoding="utf-8")
        inference = source / "inference" / window["name"]
        inference.mkdir(parents=True)
        for seed in (11, 29, 47):
            (inference / f"seed{seed}_predictions.parquet").write_bytes(b"predictions")
            attempts.append({"window": window["name"], "seed": seed, "status": "completed"})
    # The helper requires all six fits for each window; add the remaining two seeds.
    for window in plan["windows"]:
        for seed in (29, 47):
            fitting = source / "fits" / f"{window['name']}_s{seed}"
            fitting.mkdir(parents=True, exist_ok=True)
            for filename in ("model.pt", "training.json", "training_identity.json"):
                (fitting / filename).write_text("{}", encoding="utf-8")
    write_json(source / "attempts.json", attempts)
    assert len(csz.check_recovery_source(plan, source, plan_path)) == 6
    (source / "fits" / "research_a_s29" / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="missing fit artifacts"):
        csz.check_recovery_source(plan, source, plan_path)


def test_one_technical_retry_per_fit_and_twelve_start_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(csz, "RUNS", tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {"schema_version": 1})
    path = tmp_path / "fits" / "research_a_s11"
    assert csz.claim_fit(plan_path, "research_a", 11, path)[0] == 1
    assert csz.claim_fit(plan_path, "research_a", 11, path)[0] == 2
    with pytest.raises(ValueError, match="At most one technical retry"):
        csz.claim_fit(plan_path, "research_a", 11, path)


def test_incomplete_fit_directory_can_be_reused_for_a_retry_but_a_completed_one_cannot(tmp_path):
    output = tmp_path / "fit"
    csz.prepare_job_output(output, "fit")
    (output / "events.jsonl").write_text("attempt-one\n", encoding="utf-8")
    csz.prepare_job_output(output, "fit", allow_existing=True)
    assert (output / "events.jsonl").read_text(encoding="utf-8") == "attempt-one\n"
    with pytest.raises(FileExistsError, match="already exists"):
        csz.prepare_job_output(output, "fit")
    (output / "model.pt").write_bytes(b"weights")
    with pytest.raises(ValueError, match="completed fit"):
        csz.prepare_job_output(output, "fit", allow_existing=True)
    inference = tmp_path / "inference"
    csz.prepare_job_output(inference, "infer")
    with pytest.raises(FileExistsError):
        csz.prepare_job_output(inference, "infer", allow_existing=True)
