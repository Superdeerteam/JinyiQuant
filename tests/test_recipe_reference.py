"""Cumulative recipe comparison permits declared heterogeneity, not changed evaluation."""

import inspect
from pathlib import Path

import pandas as pd
import pytest
from test_training_round import archive, reseal

from jinyiquant.evaluation import evaluate
from jinyiquant.io import file_sha256, read_json, write_json
from scripts.compare_recipe_reference import compare


def make_plan(tmp_path, *, full=False):
    pairs = []
    for period, start in (("first", "2023-07-18"), ("second", "2023-07-25")) if full else (("first", "2023-07-18"),):
        for seed in (11, 29, 47) if full else (11,):
            pair = {"period": period, "seed": seed}
            for arm in ("baseline", "candidate"):
                path = archive(tmp_path / f"{period}_{seed}_{arm}", epochs=12, seed=seed, start=start)
                config = read_json(path / "config.json")
                config["model"].update(batch_size=32, target_scaling="none" if arm == "baseline" else "train_std")
                config["splits"]["train"] = {
                    "start": str(pd.Timestamp(start) - pd.Timedelta(days=4 if arm == "baseline" else 14)), "end": start}
                write_json(path / "config.json", config)
                write_json(path / "data.json", {"files": {"bars": [{"path": "bars", "sha256": "fixed-bars"}]},
                                                "requested_start": config["splits"]["train"]["start"]})
                curve = read_json(path / "learning_curve.json")
                curve[-1]["steps"] = 1500
                write_json(path / "learning_curve.json", curve)
                manifest = read_json(path / "manifest.json")
                manifest["code"]["files"] = {"src/jinyiquant/evaluation.py": file_sha256(Path(inspect.getfile(evaluate))),
                                               "modified_tlob/adapter.py": arm + "-implementation"}
                write_json(path / "manifest.json", manifest)
                reseal(path)
                pair[arm] = str(path)
            pairs.append(pair)
    path = tmp_path / "plan.json"
    write_json(path, {"schema_version": 1, "experiment_id": "recipe-reference-test", "pairs": pairs,
                      "bootstrap_samples": 100, "bootstrap_seed": 4})
    return path


def test_recipe_changes_risk_cost_and_hashes_are_descriptive_and_inputs_unchanged(tmp_path):
    plan = make_plan(tmp_path, full=True)
    source_files = [path for pair in read_json(plan)["pairs"] for arm in ("baseline", "candidate")
                    for path in Path(pair[arm]).iterdir()]
    before = {str(path): file_sha256(path) for path in source_files}
    destination = tmp_path / "comparison"
    result = compare(plan, destination)
    assert result["complete_two_period_three_seed_grid"]
    assert result["conclusion"] == "descriptive_only" and not result["model_promoted"]
    assert not result["single_factor_attribution"] and result["final_views"] == 0
    changes = result["recipe_changes"][0]
    assert {"splits.train.start", "model.target_scaling"}.issubset(
        row["field"] for row in changes["config_differences"])
    assert changes["core_source_differences"][0]["field"] == "modified_tlob/adapter.py"
    assert all("min_daily_long_short_return" in row and "extreme_return" in row and "resources" in row
               for row in result["summary"])
    assert len(result["summary"]) == 12
    assert all(row["mean"] == pytest.approx(0) for row in result["paired_intervals"])
    assert len(pd.read_csv(destination / "paired_daily.csv")) == 24
    manifest = read_json(destination / "manifest.json")
    assert manifest["plan_sha256"] == file_sha256(plan)
    assert all(row["sha256"] == file_sha256(destination / row["path"]) for row in manifest["artifacts"])
    assert before == {str(path): file_sha256(path) for path in source_files}
    with pytest.raises(ValueError, match="new directory"):
        compare(plan, destination)


@pytest.mark.parametrize("problem", ["times", "evaluation", "steps", "batch", "environment", "files",
                                    "evaluation_hash", "full_keys", "seed", "validation"])
def test_fixed_comparison_contract_rejects_changes_before_output(tmp_path, problem):
    plan = make_plan(tmp_path)
    pair = read_json(plan)["pairs"][0]
    path = Path(pair["candidate"])
    config, manifest = read_json(path / "config.json"), read_json(path / "manifest.json")
    if problem == "times":
        config["data"]["sample_times"] = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
    elif problem == "evaluation":
        config["evaluation"]["periods_per_year"] = 365
    elif problem == "batch":
        config["model"]["batch_size"] = 64
    elif problem == "seed":
        config["seed"] = 29
    elif problem == "validation":
        config["splits"]["validation"]["end"] = "2023-07-23"
    elif problem == "environment":
        manifest["environment"]["python"] = "different-version"
    elif problem == "evaluation_hash":
        manifest["code"]["files"]["src/jinyiquant/evaluation.py"] = "different-evaluator"
    elif problem == "files":
        data = read_json(path / "data.json")
        data["files"]["bars"][0]["sha256"] = "changed-bars"
        write_json(path / "data.json", data)
    elif problem == "steps":
        curve = read_json(path / "learning_curve.json")
        curve[-1]["steps"] = 1499
        write_json(path / "learning_curve.json", curve)
    else:
        predictions = pd.read_parquet(path / "predictions.parquet")
        predictions.iloc[1:].to_parquet(path / "predictions.parquet", index=False)
    write_json(path / "config.json", config)
    write_json(path / "manifest.json", manifest)
    reseal(path)
    with pytest.raises(ValueError):
        compare(plan, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_recomputed_but_different_targets_are_rejected(tmp_path):
    plan = make_plan(tmp_path)
    path = Path(read_json(plan)["pairs"][0]["candidate"])
    targets = pd.read_parquet(path / "targets.parquet")
    targets["target"] *= 2
    targets.to_parquet(path / "targets.parquet", index=False)
    metrics, daily = evaluate(pd.read_parquet(path / "scored_predictions.parquet"), targets,
                              read_json(path / "config.json")["evaluation"], undefined_constant_portfolios=True)
    write_json(path / "metrics.json", metrics)
    daily.to_csv(path / "daily.csv", index=False)
    reseal(path)
    with pytest.raises(ValueError, match="keys or targets differ"):
        compare(plan, tmp_path / "rejected")


def test_same_arm_period_cannot_mix_training_windows_across_seeds(tmp_path):
    plan = make_plan(tmp_path, full=True)
    path = Path(read_json(plan)["pairs"][1]["candidate"])
    config = read_json(path / "config.json")
    config["splits"]["train"]["start"] = "2023-07-05"
    write_json(path / "config.json", config)
    reseal(path)
    with pytest.raises(ValueError, match="Training data, splits or keys changed"):
        compare(plan, tmp_path / "rejected")


def test_matching_pair_cannot_change_evaluation_targets_across_seeds(tmp_path):
    plan = make_plan(tmp_path, full=True)
    pair = read_json(plan)["pairs"][1]
    for arm in ("baseline", "candidate"):
        path = Path(pair[arm])
        targets = pd.read_parquet(path / "targets.parquet")
        targets["target"] *= 2
        targets.to_parquet(path / "targets.parquet", index=False)
        metrics, daily = evaluate(pd.read_parquet(path / "scored_predictions.parquet"), targets,
                                  read_json(path / "config.json")["evaluation"], undefined_constant_portfolios=True)
        write_json(path / "metrics.json", metrics)
        daily.to_csv(path / "daily.csv", index=False)
        reseal(path)
    with pytest.raises(ValueError, match="Evaluation population or targets changed across seeds"):
        compare(plan, tmp_path / "rejected")
