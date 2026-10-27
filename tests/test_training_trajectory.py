"""Behavior checks for the continuous training trajectory: budgets, resume and exposure."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from jinyiquant.config import DataConfig, ModelConfig, Period, RunConfig, Splits  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from jinyiquant.trajectory import (  # noqa: E402
    CHECKPOINT_FILENAME,
    CHECKPOINT_METADATA_FILENAME,
    batches_per_epoch,
    derive_budget_steps,
    effective_epochs_for_steps,
    exposure_report,
    file_sha256,
    learning_rate_at,
    load_plan,
    plan_digest,
    reconcile_exposure,
    steps_for_effective_epochs,
    validate_budget_points,
)
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "training_trajectory_20261027_plan.json"


def training_case(*, days: int = 6, instruments: int = 5, times=("10:00", "11:00"),
                  batch_size: int = 4, epochs: int = 2):
    data = DataConfig(bars_path="unused", labels_path="unused", source="synthetic", version="1",
                      feature_columns=lob_columns(3), lookback=8, min_history=2,
                      sample_times=times)
    config = ModelConfig(kind="mlplob", hidden_dim=8, num_layers=2, batch_size=batch_size,
                         epochs=epochs, device="cpu", target_scaling="train_std",
                         log_every_steps=1)
    dates = pd.bdate_range("2023-08-01", periods=days)
    keys = pd.DataFrame([{"date": pd.Timestamp(f"{date.date()} {time}"), "instrument": f"S{index:03d}"}
                         for date in dates for time in times for index in range(instruments)])
    generator = np.random.default_rng(11)
    x = generator.normal(size=(len(keys), 8, 12)).astype(np.float32)
    observed = np.ones_like(x, dtype=bool)
    history = np.full((len(keys), 8), np.datetime64("NaT", "ns"))
    batch = Batch(keys, x, observed, history, {})
    target = generator.normal(scale=0.01, size=len(keys)).astype(np.float32)
    return data, config, batch, target


def recorder(events: list):
    return SimpleNamespace(event=lambda stage, event, **context: events.append((event, context)))


def losses(events: list) -> list[float]:
    return [context["loss"] for event, context in events if event == "progress"]


def adapter(config, data, seed: int = 42) -> TLOBAdapter:
    return TLOBAdapter(config, data, seed=seed)


def test_budget_conversion_is_exposure_based_and_not_a_frozen_step_list():
    # One effective epoch is one pass over the usable pool measured in samples.
    assert batches_per_epoch(10, 4) == 3
    assert steps_for_effective_epochs(2.0, 10, 4) == 5
    assert steps_for_effective_epochs(1.32, 479_432, 32) == 19_777
    assert effective_epochs_for_steps(19_777, 479_432, 32) == pytest.approx(1.32, abs=1e-4)
    assert steps_for_effective_epochs(5.28, 10, 4) == 13
    # Doubling the declared exposure doubles the steps up to rounding.
    assert steps_for_effective_epochs(2.64, 479_432, 32) == 39_553
    assert steps_for_effective_epochs(5.28, 479_432, 32) == 79_106
    for bad in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            steps_for_effective_epochs(bad, 100, 4)


def test_budget_points_must_increase_and_keep_unique_names():
    points = validate_budget_points([{"name": "a", "effective_epochs": 1.32},
                                     {"name": "b", "effective_epochs": 2.64},
                                     {"name": "c", "effective_epochs": 5.28}])
    assert [point["effective_epochs"] for point in points] == [1.32, 2.64, 5.28]
    rows = derive_budget_steps(points, 1024, 32)
    assert [row["steps"] for row in rows] == [42, 84, 169]
    assert [row["segment_steps"] for row in rows] == [42, 42, 85]
    half_step = 0.5 * 32 / 1024
    for row in rows:
        assert abs(row["realised_effective_epochs"] - row["effective_epochs"]) <= half_step
        assert row["realised_effective_epochs"] == pytest.approx(
            row["steps"] * 32 / 1024, abs=1e-12)
    with pytest.raises(ValueError, match="increase"):
        validate_budget_points([{"name": "a", "effective_epochs": 2.0},
                                {"name": "b", "effective_epochs": 2.0}])
    with pytest.raises(ValueError, match="unique"):
        validate_budget_points([{"name": "a", "effective_epochs": 1.0},
                                {"name": "a", "effective_epochs": 2.0}])


def test_learning_rate_rule_is_frozen_across_budget_points():
    for step in (0, 1, 19_800, 79_200):
        assert learning_rate_at("constant", 1e-4, step) == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="schedule"):
        learning_rate_at("cosine_to_zero", 1e-4, 10)
    data, config, batch, target = training_case()
    progress = adapter(config, data).fit(batch, target, step_budget=5, track_exposure=True)
    assert progress["lr_schedule"] == "constant"
    assert progress["learning_rate"] == pytest.approx(config.learning_rate)
    assert progress["steps"] == 5
    with pytest.raises(ValueError):
        ModelConfig(**{**config.model_dump(), "lr_schedule": "cosine"})
    with pytest.raises(ValueError, match="learning-rate schedule"):
        TLOBAdapter({"kind": "mlplob", "hidden_dim": 8, "num_layers": 2, "device": "cpu",
                     "cpu_threads": 1, "lr_schedule": "cosine"}, data)


def test_checkpoint_roundtrip_reproduces_the_uninterrupted_trajectory(tmp_path):
    """Save, resume and continue must be bit-identical to one uninterrupted run."""
    data, config, batch, target = training_case()
    budget, split = 11, 5
    straight = adapter(config, data)
    straight_progress = straight.fit(batch, target, step_budget=budget, track_exposure=True)
    segments = tmp_path / "segments"
    adapter(config, data).fit(batch, target, step_budget=split, track_exposure=True,
                              checkpoint_at_steps=[split], checkpoint_dir=segments)
    payload = torch.load(segments / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    assert payload["position"]["steps"] == split
    assert payload["position"]["usable_samples"] == straight_progress["usable_samples"]
    assert payload["lr_schedule"] == "constant"
    assert payload["learning_rate"] == pytest.approx(config.learning_rate)
    assert set(payload["rng"]) == {"torch", "generator", "numpy", "python", "cuda"}
    assert payload["optimizer_state_dict"]["state"], "optimizer state must be checkpointed"
    metadata = json.loads((segments / CHECKPOINT_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["position"]["steps"] == split and metadata["fingerprint"]

    second = adapter(config, data)
    second_progress = second.fit(batch, target, step_budget=budget, track_exposure=True, resume=payload)
    assert second_progress["resumed"] is True
    assert second_progress["steps"] == straight_progress["steps"] == budget
    assert second_progress["samples_seen"] == straight_progress["samples_seen"]
    assert second_progress["epochs"] == pytest.approx(straight_progress["epochs"])
    for key, value in straight.network.state_dict().items():
        assert torch.equal(value, second.network.state_dict()[key]), key
    np.testing.assert_array_equal(straight.exposure_counts, second.exposure_counts)
    np.testing.assert_array_equal(straight.predict(batch), second.predict(batch))
    # The restored global RNG streams are still the checkpointed ones.
    numpy_state = payload["rng"]["numpy"]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1].numpy())
    assert random.getstate() == payload["rng"]["python"]
    assert torch.equal(torch.get_rng_state(), payload["rng"]["torch"])


def test_per_step_losses_are_identical_after_resume(tmp_path):
    data, config, batch, target = training_case()
    budget, split = 9, 4
    straight_events = []
    TLOBAdapter(config, data, seed=42, recorder=recorder(straight_events)).fit(
        batch, target, step_budget=budget)
    segments = tmp_path / "seg"
    segment_events = []
    TLOBAdapter(config, data, seed=42, recorder=recorder(segment_events)).fit(
        batch, target, step_budget=split, checkpoint_at_steps=[split],
        checkpoint_dir=segments)
    payload = torch.load(segments / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    TLOBAdapter(config, data, seed=42, recorder=recorder(segment_events)).fit(
        batch, target, step_budget=budget, resume=payload)
    straight_losses, segment_losses = losses(straight_events), losses(segment_events)
    assert len(straight_losses) == len(segment_losses) == budget
    assert segment_losses == straight_losses
    assert all(np.isfinite(value) and value > 0 for value in straight_losses)


def test_resume_fails_closed_when_the_pool_or_sampler_position_changes(tmp_path):
    data, config, batch, target = training_case()
    segments = tmp_path / "segments"
    first = adapter(config, data)
    first.fit(batch, target, step_budget=4, track_exposure=True,
              checkpoint_at_steps=[4], checkpoint_dir=segments, observer=None)
    payload = torch.load(segments / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)

    drifted_target = np.asarray(target, dtype=np.float32).copy()
    drifted_target[0] += 1e-3
    with pytest.raises(ValueError, match="identical pool"):
        adapter(config, data).fit(batch, drifted_target, step_budget=6, resume=payload, observer=None)

    tampered = {**payload, "fingerprint": {**payload["fingerprint"], "combined_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="identical pool"):
        adapter(config, data).fit(batch, target, step_budget=6, resume=tampered, observer=None)

    moved = {**payload, "position": {**payload["position"], "batch_index": 99}}
    with pytest.raises(ValueError, match="batch position|batch order"):
        adapter(config, data).fit(batch, target, step_budget=6, resume=moved,
                                   track_exposure=True, observer=None)

    schedule = {**payload, "lr_schedule": "cosine"}
    with pytest.raises(ValueError, match="schedule changed"):
        adapter(config, data).fit(batch, target, step_budget=6, resume=schedule, observer=None)

    with pytest.raises(ValueError, match="exposure counters"):
        adapter(config, data).fit(batch, target, step_budget=6, resume=payload, observer=None)


def test_exposure_accounting_reconciles_with_the_declared_budget():
    data, config, batch, target = training_case()
    model = adapter(config, data)
    progress = model.fit(batch, target, step_budget=7, track_exposure=True, observer=None)
    report = model.exposure_report()
    assert report["steps"] == progress["steps"] == 7
    assert report["samples_seen"] == progress["samples_seen"] == 7 * config.batch_size
    assert report["usable_samples"] == len(target)
    assert report["coverage"] == (report["unique_samples"] / report["usable_samples"])
    assert 0 < report["coverage"] <= 1
    assert report["effective_epochs"] == pytest.approx(7 * config.batch_size / len(target))
    assert len(report["per_day"]) == 6
    assert sum(row["usable_samples"] for row in report["per_day"]) == len(target)
    assert sum(row["samples_seen"] for row in report["per_day"]) == report["samples_seen"]
    assert report["per_epoch"] and report["per_epoch"][0]["steps"] >= 1

    declared = {"steps": 7, "effective_epochs": report["effective_epochs"],
                "effective_epochs_atol": 1e-9, "min_coverage": report["coverage"]}
    assert reconcile_exposure(report, declared)["passed"] is True

    # A configuration that claims a longer trajectory than it ran must fail.
    overstated = {"steps": 7, "effective_epochs": 5.28, "effective_epochs_atol": 1e-3,
                  "min_coverage": 0.0}
    verdict = reconcile_exposure(report, overstated)
    assert verdict["passed"] is False and verdict["issues"]
    under_covered = {"steps": 7, "effective_epochs": report["effective_epochs"],
                     "effective_epochs_atol": 1e-9, "min_coverage": 1.0}
    if report["coverage"] < 1.0:
        assert reconcile_exposure(report, under_covered)["passed"] is False


def test_exposure_counter_survives_a_resume(tmp_path):
    data, config, batch, target = training_case()
    straight = adapter(config, data)
    straight.fit(batch, target, step_budget=9, track_exposure=True, observer=None)
    straight_counts = straight.exposure_counts.copy()
    segments = tmp_path / "segments"
    first = adapter(config, data)
    first.fit(batch, target, step_budget=4, track_exposure=True,
              checkpoint_at_steps=[4], checkpoint_dir=segments, observer=None)
    payload = torch.load(segments / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    second = adapter(config, data)
    second.fit(batch, target, step_budget=9, resume=payload, track_exposure=True, observer=None)
    np.testing.assert_array_equal(straight_counts, second.exposure_counts)
    assert second.exposure_report()["samples_seen"] == 9 * config.batch_size


def test_exposure_report_rejects_a_mismatched_counter_or_mask():
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-08-01 10:00"] * 4),
                         "instrument": ["A", "B", "C", "D"]})
    with pytest.raises(ValueError, match="aligned nonempty"):
        exposure_report(keys, np.zeros(4, dtype=bool), np.zeros(0, dtype=np.int64),
                        batch_size=2, steps=1)
    with pytest.raises(ValueError, match="one entry per usable"):
        exposure_report(keys, np.ones(4, dtype=bool), np.zeros(3, dtype=np.int64),
                        batch_size=2, steps=1)
    with pytest.raises(ValueError, match="steps \\* batch_size"):
        exposure_report(keys, np.ones(4, dtype=bool), np.zeros(4, dtype=np.int64),
                        batch_size=2, steps=3)


def test_pre_registered_plan_declares_the_budget_points_and_a_fixed_schedule():
    plan = load_plan(PLAN)
    assert plan["lr_schedule"] == "constant"
    assert [point["effective_epochs"] for point in plan["budget_points"]] == [1.32, 2.64, 5.28]
    assert plan_digest(plan) == plan_digest(load_plan(PLAN))
    recipe = ROOT / plan["recipe_config"]
    assert recipe.is_file(), recipe
    assert file_sha256(recipe) == plan["recipe_sha256"]
    assert float(plan["exposure"]["min_coverage"]) > 0
    assert float(plan["exposure"]["effective_epochs_atol"]) >= 0


def test_cloud_job_manifest_lists_code_data_commands_and_retrieval():
    from scripts.cloud_trajectory_manifest import build_manifest

    manifest = build_manifest(PLAN, ROOT, digest=False)
    assert manifest["schema_version"] == 1
    assert manifest["experiment_id"] == "training-trajectory-20261027"
    assert len(manifest["code"]["commit"]) == 40
    assert manifest["data"]["files"]
    assert all({"path", "exists", "sha256"} <= set(row) for row in manifest["data"]["files"])
    assert manifest["budget"]["batch_size"] == 32
    assert [row["steps"] for row in manifest["budget"]["points"]] == [19_777, 39_553, 79_106]
    # The recipe points at the cloud paths; absent locally, they are flagged, not invented.
    assert all({"field", "shard", "exists", "bytes", "sha256"} <= set(row)
               for row in manifest["data"]["files"])
    assert all(row["exists"] is False and row["sha256"] is None
               for row in manifest["data"]["files"])
    for name in ("upload", "run", "fetch"):
        assert name in manifest["commands"]
    assert manifest["retrieve"]["checkpoints"] and manifest["retrieve"]["budget_points"]


def test_trajectory_config_only_relaxes_the_budget_control(tmp_path):
    from scripts.run_training_trajectory import trajectory_config

    plan = load_plan(PLAN)
    config, guard = trajectory_config(plan, ROOT, tmp_path / "runs")
    assert config.model.max_steps is None
    # Only the step budget is relaxed; the epoch ceiling already covers the last point.
    assert guard["deviations"] == ["model.max_steps", "output_root"]
    assert config.model.epochs == 12

    recipe = json.loads((ROOT / plan["recipe_config"]).read_text(encoding="utf-8"))
    recipe["model"]["hidden_dim"] = 48
    drifted = tmp_path / plan["recipe_config"]
    drifted.parent.mkdir(parents=True, exist_ok=True)
    drifted.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since pre-registration"):
        trajectory_config(plan, tmp_path, tmp_path / "runs")


def test_driver_runs_all_budget_points_and_resumes_between_them(tmp_path):
    from scripts import run_training_trajectory

    data, config, batch, target = training_case(days=4, instruments=4, times=("10:00",))
    run_config = RunConfig(
        experiment_id="synthetic-trajectory", hypothesis="synthetic resume wiring check",
        output_root=tmp_path / "runs", seed=42, data=data, model=config,
        splits=Splits(train=Period(start="2023-08-01", end="2023-08-04"),
                      validation=Period(start="2023-08-04", end="2023-08-07"),
                      final=Period(start="2023-08-07", end="2023-08-10")))
    plan = {"_path": "synthetic-plan",
            "exposure": {"min_coverage": 0.0, "effective_epochs_atol": 1e-9},
            "budget_points": [{"name": "p1", "effective_epochs": 0.5},
                              {"name": "p2", "effective_epochs": 1.0},
                              {"name": "p3", "effective_epochs": 1.5}]}
    prepared = {"bundle": SimpleNamespace(metadata={"source": "synthetic", "version": "1"}),
                "train": batch, "targets": target,
                "fit_mask": np.ones(len(target), dtype=bool)}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Source capture is a byte-for-byte archive of the checkout; the trajectory wiring is
    # what this test covers, so the snapshot is stubbed with the same shape.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(run_training_trajectory, "capture_source",
                        lambda run_dir: {"commit": "0" * 40, "content_sha256": "0" * 64,
                                        "snapshot": None, "files": {}})
    result = run_training_trajectory.run_trajectory(plan, run_config, prepared, batch, run_dir)
    monkeypatch.undo()
    assert result["status"] == "completed", result["error"]
    assert [row["name"] for row in result["trajectory"]] == ["p1", "p2", "p3"]
    assert [row["resumed"] for row in result["trajectory"]] == [False, True, True]
    steps = [row["steps"] for row in result["trajectory"]]
    assert steps == [2, 4, 6]
    for name in ("manifest.json", "config.json", "data.json", "budget.json", "exposure.json",
                 "trajectory.json", "events.jsonl", "trajectory/training_state.pt",
                 "trajectory/training_state.json"):
        assert (run_dir / name).is_file(), name
    for row in result["trajectory"]:
        assert (run_dir / "budget_points" / f"{row['name']}.json").is_file()
        assert (run_dir / "budget_points" / f"{row['name']}_predictions.parquet").is_file()
        assert (run_dir / row["model"]).is_file()
        weights = run_dir / row["weights"]
        assert weights.is_file() and weights.name == f"modified_tlob_weights-{row['name']}.json"
        reloaded = TLOBAdapter.load(weights, device="cpu")
        np.testing.assert_array_equal(reloaded.predict(batch), TLOBAdapter.load(
            run_dir / row["model"], device="cpu").predict(batch))
        assert 0 < row["coverage"] <= 1
    assert result["trajectory"][-1]["coverage"] > result["trajectory"][0]["coverage"]
    for index, row in enumerate(result["trajectory"], start=1):
        point_state = run_dir / "trajectory" / "points" / f"{index:02d}-{row['name']}"
        assert (point_state / "training_state.pt").is_file()
        metadata = json.loads((point_state / "training_state.json").read_text(encoding="utf-8"))
        assert metadata["budget_point"]["name"] == row["name"]
        assert metadata["position"]["steps"] == row["steps"]
        assert metadata["optimizer"]["state_entries"] > 0 and metadata["optimizer"]["step"] > 0
        assert metadata["rng_streams"] == ["cuda", "generator", "numpy", "python", "torch"]
        assert metadata["sampler"]["epoch_batches_sha256"] or metadata["sampler"]["batch_index"] == 0
        assert metadata["exposure_counters"]["rows"] == len(target)
        assert row["checkpoint"].startswith("trajectory/points/")
    assert (run_dir / "trajectory" / "training_state.pt").is_file()
    document = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert [point["steps"] for point in document["budget"]] == steps
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_data_inventory_lists_each_shard_with_its_digest(tmp_path):
    from scripts.cloud_trajectory_manifest import data_inventory

    shards = tmp_path / "bars"
    shards.mkdir()
    (shards / "202301.0.feather").write_bytes(b"january")
    (shards / "202302.0.feather").write_bytes(b"february")
    single = tmp_path / "instruments.parquet"
    single.write_bytes(b"instrument table")
    data = DataConfig(bars_path=shards, labels_path=single, source="synthetic", version="1",
                      feature_columns=lob_columns(3), lookback=8)
    rows = data_inventory(data)
    assert [row["shard"] for row in rows] == ["202301.0.feather", "202302.0.feather",
                                              "instruments.parquet"]
    assert rows[0]["bytes"] == len(b"january") and rows[0]["exists"] is True
    assert rows[0]["sha256"] != rows[1]["sha256"]
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert [row["sha256"] for row in data_inventory(data, digest=False)] == [None, None, None]
