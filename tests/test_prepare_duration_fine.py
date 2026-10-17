"""The fine-grid freeze must emit schema-valid configs and refuse ambiguous replication."""

import zipfile

import numpy as np
import pandas as pd
import pytest
from test_training_round import reseal

from jinyiquant.config import RunConfig
from jinyiquant.io import file_sha256, read_json, write_json
from scripts.prepare_training_duration_fine import (
    BASELINE,
    EXPERIMENT,
    PERIODS,
    REPLICATION_SEEDS,
    SCREEN_SEED,
    STEPS,
    apply_source_remap,
    parse_remap,
    prepare_replication,
    prepare_screen,
    reference_evidence,
)

TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
FEATURES = [f"{side}_{kind}{level}" for level in (1, 2, 3)
            for side, kind in (("ask", "price"), ("ask", "volume"), ("bid", "price"), ("bid", "volume"))]
RANK_IC = [0.05, 0.06, 0.04, 0.07]
LONG_SHORT = [1.0e-4, 2.0e-4, 1.5e-4, 0.5e-4]


def reference_run(path, period, seed, start):
    path.mkdir(parents=True)
    daily = pd.DataFrame({"date": pd.date_range(start, periods=4).strftime("%Y-%m-%d"),
                          "rank_ic": RANK_IC, "long_short_return": LONG_SHORT, "turnover": [1.4] * 4})
    daily.to_csv(path / "daily.csv", index=False)
    sections = pd.DataFrame({"timestamp": pd.date_range(start, periods=4).strftime("%Y-%m-%d") + " 09:30:00",
                             "rank_ic": RANK_IC, "long_short_return": LONG_SHORT})
    sections.to_csv(path / "by_timestamp.csv", index=False)
    metrics = {"rank_ic_mean": float(np.mean(RANK_IC)), "long_short_return_mean": float(np.mean(LONG_SHORT)),
               "turnover_mean": float(np.mean([1.4] * 4)),
               "min_daily_long_short_return": float(np.min(LONG_SHORT)), "mse": 1e-5}
    write_json(path / "metrics.json", metrics)
    write_json(path / "checks.json", {"passed": True})
    write_json(path / "learning_curve.json", [{"steps": 1500, "epoch": 0.6, "train_mse": 0.01}])
    with zipfile.ZipFile(path / "source.zip", "w") as archive:
        archive.writestr("README", "test source snapshot")
    config = {
        "schema_version": 1, "experiment_id": "reference", "hypothesis": "archived reference",
        "output_root": str(path), "seed": seed, "evaluation_split": "validation",
        "data": {"bars_path": "data/bars", "labels_path": "data/labels", "source": "official", "version": "t",
                 "feature_columns": FEATURES, "lookback": 128, "min_history": 1, "history_days": 10,
                 "sample_times": TIMES, "training_sample_times": TIMES, "price_scale": 100},
        "splits": {"train": {"start": start, "end": "2023-07-18"},
                   "validation": {"start": "2023-07-18", "end": "2023-07-22"},
                   "final": {"start": "2023-08-03", "end": "2023-08-04"}},
        "model": {"kind": "modified_tlob", "max_steps": 1500, "epochs": 12, "batch_size": 32,
                  "learning_rate": 0.0001, "target_scaling": "train_std"},
        "budget": {"wall_seconds": 3600, "max_rss_mb": 6144},
        "logging": {"learning_curves": True, "learning_curve_every_steps": 250},
    }
    write_json(path / "config.json", config)
    write_json(path / "manifest.json", {"status": "completed"})
    reseal(path)
    return path


@pytest.fixture
def frozen_root(tmp_path):
    starts = dict(zip(PERIODS, ("2023-07-10", "2023-07-17")))
    inputs = []
    for period in PERIODS:
        for seed in (11, 29, 47):
            run = reference_run(tmp_path / "refs" / f"{period}_s{seed}", period, seed, starts[period])
            inputs.append({"arm": "candidate", "period": period, "seed": seed, "run": str(run),
                           "manifest_sha256": file_sha256(run / "manifest.json")})
    comparison = tmp_path / "runs/training-progression/step3c-full-analysis"
    comparison.mkdir(parents=True)
    write_json(comparison / "comparison.json", {"checks": {"passed": True}, "final_views": 0,
                                               "inputs": inputs})
    return tmp_path


def test_screen_freeze_validates_every_config_and_registers_the_explicit_baseline(frozen_root):
    references, sources, calibration = reference_evidence(
        frozen_root / "runs/training-progression/step3c-full-analysis/comparison.json")
    result = prepare_screen(frozen_root, references, sources, calibration)
    assert result["prepared"] and result["baseline_budget"] == BASELINE and result["configs"] == len(STEPS) * 2
    assert set(calibration["pooled_seed_std"]) == {"rank_ic", "long_short_return", "turnover",
                                                   "worst_day_long_short_return", "opening_rank_ic",
                                                   "opening_long_short_return"}
    plan = read_json(frozen_root / "configs/training_duration_fine_20261016_plan.json")
    assert [entry["name"] for entry in plan["runs"]] == [f"{period}_s{SCREEN_SEED}_steps{steps}"
                                                         for steps in STEPS for period in PERIODS]
    assert plan["total_wall_seconds"] == 20000 and plan["final_views"] == 0
    for entry in plan["runs"]:
        config = read_json(frozen_root / "configs" / entry["config"])
        snapshot = RunConfig.model_validate(config).snapshot()
        assert snapshot["model"]["max_steps"] == entry["steps"] and snapshot["seed"] == SCREEN_SEED
        assert snapshot["experiment_id"] == EXPERIMENT and snapshot["evaluation_split"] == "validation"
    prereg = read_json(frozen_root / "runs/training-duration-fine-20261016/preregistration.json")
    assert prereg["experiment"]["baseline_budget"] == BASELINE
    assert prereg["experiment"]["budgets"] == list(STEPS)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_screen(frozen_root, references, sources, calibration)


def decision_file(root, *, selected=2000, sha=None, drift=True):
    path = root / "decision.json"
    write_json(path, {"experiment_id": EXPERIMENT,
                      "registration_sha256": file_sha256(root / "runs/training-duration-fine-20261016/"
                                                         "preregistration.json") if sha is None else sha,
                      "historical_reference_drift": {"passed": drift},
                      "recommend_replication_budget": selected})
    return path


def test_replication_plan_freezes_only_the_recommended_candidate(frozen_root):
    references, sources, calibration = reference_evidence(
        frozen_root / "runs/training-progression/step3c-full-analysis/comparison.json")
    prepare_screen(frozen_root, references, sources, calibration)
    result = prepare_replication(frozen_root, sources, decision_file(frozen_root))
    assert result["selected_budget"] == 2000 and result["configs"] == 8
    plan = read_json(frozen_root / "configs/training_duration_fine_20261016_replication_plan.json")
    assert [(entry["seed"], entry["steps"], entry["arm"]) for entry in plan["runs"]] == [
        (seed, steps, "baseline" if steps == BASELINE else "candidate")
        for seed in REPLICATION_SEEDS for period in PERIODS for steps in (BASELINE, 2000)]
    for entry in plan["runs"]:
        config = read_json(frozen_root / "configs" / entry["config"])
        snapshot = RunConfig.model_validate(config).snapshot()
        assert snapshot["model"]["max_steps"] == entry["steps"] and snapshot["seed"] == entry["seed"]
    assert plan["replication_of"]["selected_budget"] == 2000 and plan["final_views"] == 0
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_replication(frozen_root, sources, decision_file(frozen_root))


@pytest.mark.parametrize("problem", ["wrong_experiment", "wrong_sha", "drift_failed",
                                     "baseline_recommended", "unregistered_budget"])
def test_replication_refuses_ambiguous_or_failed_decisions(frozen_root, problem):
    references, sources, calibration = reference_evidence(
        frozen_root / "runs/training-progression/step3c-full-analysis/comparison.json")
    prepare_screen(frozen_root, references, sources, calibration)
    path = decision_file(frozen_root, selected=2000)
    if problem == "wrong_experiment":
        value = read_json(path)
        value["experiment_id"] = "other-experiment"
        write_json(path, value)
    elif problem == "wrong_sha":
        path = decision_file(frozen_root, selected=2000, sha="0" * 64)
    elif problem == "drift_failed":
        path = decision_file(frozen_root, selected=2000, drift=False)
    elif problem == "baseline_recommended":
        path = decision_file(frozen_root, selected=BASELINE)
    else:
        path = decision_file(frozen_root, selected=5000)
    with pytest.raises(ValueError):
        prepare_replication(frozen_root, sources, path)


def test_reference_recipe_drift_is_rejected(frozen_root):
    run = frozen_root / "refs/july_18_21_s11"
    config = read_json(run / "config.json")
    config["model"]["target_scaling"] = "none"
    write_json(run / "config.json", config)
    reseal(run)
    comparison = frozen_root / "runs/training-progression/step3c-full-analysis/comparison.json"
    value = read_json(comparison)
    for item in value["inputs"]:
        if item["run"] == str(run):
            item["manifest_sha256"] = file_sha256(run / "manifest.json")
    write_json(comparison, value)
    with pytest.raises(ValueError, match="recipe differs"):
        reference_evidence(comparison)


def windows_frozen_root(tmp_path):
    """Archives recorded with the original machine's D:\\ absolute paths, files on the server."""
    server = tmp_path / "srv"
    starts = dict(zip(PERIODS, ("2023-07-10", "2023-07-17")))
    inputs = []
    for period in PERIODS:
        for seed in (11, 29, 47):
            run = reference_run(server / "refs" / f"{period}_s{seed}", period, seed, starts[period])
            recorded = f"D:\\JinyiQuant\\srv\\refs\\{period}_s{seed}"
            config = read_json(run / "config.json")
            config["data"].update(bars_path="D:\\JinyiQuant\\srv\\data\\bars",
                                  labels_path="D:\\JinyiQuant\\srv\\cpt_jyc_2026_vwap.rar",
                                  instruments_path="D:\\JinyiQuant\\srv\\instruments.parquet",
                                  instrument_map_path="D:\\JinyiQuant\\srv\\mapping.csv",
                                  cache_dir="D:\\JinyiQuant\\srv\\cache")
            write_json(run / "config.json", config)
            reseal(run)
            inputs.append({"arm": "candidate", "period": period, "seed": seed, "run": recorded,
                           "manifest_sha256": file_sha256(run / "manifest.json")})
    comparison = tmp_path / "runs/training-progression/step3c-full-analysis"
    comparison.mkdir(parents=True)
    write_json(comparison / "comparison.json", {"checks": {"passed": True}, "final_views": 0,
                                               "inputs": inputs})
    return tmp_path, server


def test_parse_remap_normalizes_and_rejects_ambiguous_rules():
    assert parse_remap([r"D:\JinyiQuant=/srv/jyq", "D:\\JinyiQuant\\x.rar=/srv/x"]) == (
        ("D:/JinyiQuant", "/srv/jyq"), ("D:/JinyiQuant/x.rar", "/srv/x"))
    assert parse_remap(None) == parse_remap([]) == ()
    for bad in ("no-equals", "=missing", "old=", "D:/x=relative"):
        with pytest.raises(ValueError):
            parse_remap([bad])
    with pytest.raises(ValueError, match="mapped once"):
        parse_remap(["D:/a=/x", "D:\\a=/y"])


def test_windows_archives_are_remapped_verified_and_recorded(tmp_path):
    root, server = windows_frozen_root(tmp_path)
    (server / "data/bars").mkdir(parents=True)
    (server / "labels-dir").mkdir()
    (server / "instruments.parquet").write_bytes(b"stub")
    (server / "mapping.csv").write_bytes(b"stub")
    rules = parse_remap([rf"D:\JinyiQuant\srv\cpt_jyc_2026_vwap.rar={server / 'labels-dir'}",
                         rf"D:\JinyiQuant\srv={server}"])
    comparison = root / "runs/training-progression/step3c-full-analysis/comparison.json"
    references, sources, calibration = reference_evidence(comparison, rules)
    assert all(r["run"].startswith(str(server / "refs")) for r in references)
    apply_source_remap(sources, rules)
    for snapshot in sources.values():
        assert snapshot["data"]["bars_path"] == str(server / "data/bars")
        assert snapshot["data"]["labels_path"] == str(server / "labels-dir")
        assert snapshot["data"]["instrument_map_path"] == str(server / "mapping.csv")
    prepare_screen(root, references, sources, calibration, remap=rules)
    prereg = read_json(root / "runs/training-duration-fine-20261016/preregistration.json")
    assert prereg["experiment"]["path_remap"]["rules"] == [f"{old}={new}" for old, new in rules]
    for entry in read_json(root / "configs/training_duration_fine_20261016_plan.json")["runs"]:
        config = read_json(root / "configs" / entry["config"])
        assert config["data"]["bars_path"] == str(server / "data/bars")
        assert config["data"]["labels_path"] == str(server / "labels-dir")


def test_remap_to_missing_inputs_fails_before_any_plan_is_written(tmp_path):
    root, server = windows_frozen_root(tmp_path)
    rules = parse_remap([rf"D:\JinyiQuant\srv={server}"])
    references, sources, calibration = reference_evidence(
        root / "runs/training-progression/step3c-full-analysis/comparison.json", rules)
    with pytest.raises(ValueError, match="missing on this machine"):
        apply_source_remap(sources, rules)
    assert not (root / "configs/training_duration_fine_20261016_plan.json").exists()


def test_unresolvable_windows_run_paths_are_rejected(tmp_path):
    root, server = windows_frozen_root(tmp_path)
    comparison = root / "runs/training-progression/step3c-full-analysis/comparison.json"
    with pytest.raises(ValueError, match="Cannot find reference run"):
        reference_evidence(comparison, parse_remap([r"D:\JinyiQuant\srv=/somewhere/else"]))
