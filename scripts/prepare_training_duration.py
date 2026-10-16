"""Freeze the duration screen and calibrate its rules from six archived reference runs.

This reads completed July research archives only. It does not train, inspect final
data, or overwrite an existing plan, configuration directory, or preregistration.
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import verify_artifacts
from jinyiquant.config import RunConfig
from jinyiquant.io import file_sha256, read_json, utc_now, write_json

EXPERIMENT = "training-duration-20261015"
PERIODS = ("july_18_21", "july_25_28")
SEEDS = (11, 29, 47)
STEPS = (1500, 5000, 10000)
TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
METRICS = {
    "rank_ic": "rank_ic_mean",
    "long_short_return": "long_short_return_mean",
    "turnover": "turnover_mean",
    "worst_day_long_short_return": "min_daily_long_short_return",
}
HYPOTHESIS = (
    "At fixed ten-day/all-eight-time raw12/train_std recipe and learning rate 1e-4, "
    "compare 1500, 5000 and 10000 optimizer updates to determine whether longer "
    "training improves later-date ranking and gross long-short results."
)


def reference_evidence(comparison_path):
    comparison = read_json(comparison_path)
    if not comparison.get("checks", {}).get("passed") or comparison.get("final_views") != 0:
        raise ValueError("Reference comparison must pass and contain no final views")
    inputs = [item for item in comparison["inputs"] if item["arm"] == "candidate"]
    expected = {(period, seed) for period in PERIODS for seed in SEEDS}
    if len(inputs) != len(expected) or {(x["period"], x["seed"]) for x in inputs} != expected:
        raise ValueError("Reference requires exactly two periods by three unique seeds")
    rows, references, configs = [], [], {}
    for item in sorted(inputs, key=lambda x: (x["period"], x["seed"])):
        path = Path(item["run"])
        if file_sha256(path / "manifest.json") != item["manifest_sha256"]:
            raise ValueError(f"Reference manifest changed: {path}")
        if not verify_artifacts(path)["passed"]:
            raise ValueError(f"Reference artifacts failed: {path}")
        manifest, config, checks, metrics = [read_json(path / f"{name}.json")
                                            for name in ("manifest", "config", "checks", "metrics")]
        if manifest.get("status") != "completed" or not checks.get("passed"):
            raise ValueError(f"Reference did not complete successfully: {path}")
        effective = RunConfig.model_validate(config).snapshot()
        model = effective["model"]
        expected_model = {"max_steps": 1500, "epochs": 12, "batch_size": 32,
                          "learning_rate": 0.0001, "target_scaling": "train_std",
                          "book_representation": "raw", "time_loss_weighting": "none",
                          "trade_features": "zero"}
        if (any(model[key] != value for key, value in expected_model.items())
                or effective["evaluation_split"] != "validation"
                or effective["seed"] != item["seed"]
                or len(effective["data"]["feature_columns"]) != 12
                or effective["data"]["sample_times"] != list(TIMES)
                or effective["data"]["training_sample_times"] != list(TIMES)):
            raise ValueError(f"Reference recipe differs: {path}")
        curve = read_json(path / "learning_curve.json")
        if curve[-1]["steps"] != 1500:
            raise ValueError(f"Reference did not reach 1500 steps: {path}")
        daily = pd.read_csv(path / "daily.csv")
        sections = pd.read_csv(path / "by_timestamp.csv")
        opening = sections[pd.to_datetime(sections["timestamp"]).dt.strftime("%H:%M") == "09:30"]
        if len(daily) != 4 or len(opening) != 4 or daily["date"].nunique() != 4:
            raise ValueError(f"Reference needs four complete daily/opening observations: {path}")
        values = {key: float(metrics[field]) for key, field in METRICS.items()}
        values.update(opening_rank_ic=float(opening["rank_ic"].mean()),
                      opening_long_short_return=float(opening["long_short_return"].mean()))
        recomputed = {"rank_ic": float(daily["rank_ic"].mean()),
                      "long_short_return": float(daily["long_short_return"].mean()),
                      "turnover": float(daily["turnover"].mean()),
                      "worst_day_long_short_return": float(daily["long_short_return"].min())}
        if (not np.isfinite(list(values.values())).all()
                or not np.isfinite(opening[["rank_ic", "long_short_return"]]).all().all()
                or any(not np.isclose(values[k], v, atol=1e-12, rtol=1e-9) for k, v in recomputed.items())):
            raise ValueError(f"Undefined metrics or metrics/daily mismatch: {path}")
        rows.append({"period": item["period"], "seed": item["seed"], **values})
        names = ("manifest.json", "config.json", "metrics.json", "daily.csv", "by_timestamp.csv",
                 "checks.json", "learning_curve.json", "source.zip")
        references.append({"period": item["period"], "seed": item["seed"], "run": str(path),
                           "sha256": {name: file_sha256(path / name) for name in names},
                           "metrics": values})
        configs[item["period"], item["seed"]] = effective
    frame = pd.DataFrame(rows)
    metrics = list(rows[0].keys())[2:]
    period_stats = {}
    for period, group in frame.groupby("period", sort=True):
        period_stats[period] = {metric: {"mean": float(group[metric].mean()),
                                        "std": float(group[metric].std(ddof=1)),
                                        "n": len(group)} for metric in metrics}
    pooled = {metric: float(np.sqrt(np.mean([period_stats[p][metric]["std"] ** 2 for p in PERIODS])))
              for metric in metrics}
    return references, configs, {
        "method": "sqrt(mean(within-period sample variance across the three seeds)); ddof=1",
        "units": "Returns use original fractional-return units; multiply by 10000 for basis points.",
        "interpretation": "Seed variability conditional on these windows, not independent-date uncertainty.",
        "periods": period_stats, "pooled_seed_std": pooled,
        "overall_means": {metric: float(frame[metric].mean()) for metric in metrics},
        "source_comparison": str(comparison_path), "source_comparison_sha256": file_sha256(comparison_path),
    }


def prepare(root):
    root = Path(root).resolve()
    plan_path = root / "configs/training_duration_20261015_plan.json"
    config_dir = root / "configs/training_duration_20261015"
    prereg_path = root / "runs/training-duration-20261015/preregistration.json"
    for path in (plan_path, config_dir, prereg_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    comparison = root / "runs/training-progression/step3c-full-analysis/comparison.json"
    references, sources, calibration = reference_evidence(comparison)
    output_root = root / "runs/training-duration-20261015/screen"
    configs, entries = {}, []
    for steps in STEPS:
        for period in PERIODS:
            raw = deepcopy(sources[period, 11])
            raw.update(experiment_id=EXPERIMENT, hypothesis=HYPOTHESIS, output_root=str(output_root))
            raw["model"].update(max_steps=steps, book_representation="raw",
                                time_loss_weighting="none", trade_features="zero")
            raw["budget"].update(wall_seconds=3600.0, max_rss_mb=6144.0)
            raw["logging"].update(learning_curves=True, learning_curve_every_steps=1500)
            config = RunConfig.model_validate(raw).snapshot()
            name = f"{period}_s11_steps{steps}"
            configs[name] = config
            entries.append({"name": name, "period": period, "seed": 11, "steps": steps,
                            "arm": "baseline" if steps == 1500 else f"candidate_{steps}",
                            "config": f"{config_dir.name}/{name}.json"})
    for period in PERIODS:
        baseline = deepcopy(configs[f"{period}_s11_steps1500"])
        for steps in STEPS[1:]:
            candidate = deepcopy(configs[f"{period}_s11_steps{steps}"])
            candidate["model"]["max_steps"] = baseline["model"]["max_steps"]
            if candidate != baseline:
                raise ValueError("Generated duration pair changes more than max_steps")
    rules = {
        "delta": "candidate minus same-period same-seed 1500-step baseline",
        "noise_threshold_source": "calibration.pooled_seed_std; multiplier=1.0 fixed before screening",
        "primary_metrics": ["rank_ic", "long_short_return"],
        "risk_directions": {"worst_day_long_short_return": "maximize", "opening_rank_ic": "maximize",
                            "opening_long_short_return": "maximize", "turnover": "minimize"},
        "screen": {
            "seed": 11, "required_periods": list(PERIODS),
            "every_period_primary_delta_min": 0.0,
            "aggregate_improvement": "Equal-period mean delta of at least one primary >= its pooled_seed_std",
            "risk_gate": "In every period, each risk metric regression <= its own pooled_seed_std",
            "failed_action": "Do not expand seeds for a failed budget",
        },
        "selection": {
            "eligible": "Only budgets passing the complete screen including risk gate",
            "priority": "Higher equal-period mean RankIC; exact RankIC ties choose fewer steps",
            "shorter_budget_tie": "If absolute differences between eligible budgets are below both primary "
                                  "pooled_seed_std thresholds, select the shorter budget",
            "none_pass": "Keep the 1500-step reference; stop without seed expansion",
            "replication_limit": "Replicate only the selected budget, not both screened budgets",
        },
        "replication": {
            "additional_seeds": [29, 47], "periods": list(PERIODS), "complete_pairs": 6,
            "baseline": "Rerun 1500 steps with the identical new core source for both additional seeds",
            "every_pair_primary_delta_min": 0.0,
            "aggregate_improvement": "At least one primary mean delta >= its pooled_seed_std and the same "
                                     "metric has paired-day block 95% interval lower bound > 0",
            "risk_gate": "Every period-seed pair: each risk metric regression <= its own pooled_seed_std",
            "uncertainty": {"method": "Average seeds within each day; moving two-day blocks within each period",
                            "block_days": 2, "confidence": 0.95, "bootstrap_samples": 2000,
                            "bootstrap_seed": 20261015},
            "failure_action": "Keep the 1500-step reference, report failed or inconclusive evidence",
        },
        "final_decision": "Passing supports a replacement research candidate only; no formal champion promotion",
        "intermediate_validation": "Diagnostic only; do not select unregistered intermediate steps or alter LR",
        "stopping": "Only 1500/5000/10000 screen; stop at declared caps or failures. No extra budgets, LR search, "
                    "or final-split access. Preserve all attempts and reference weights.",
    }
    plan = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "hypothesis": HYPOTHESIS,
        "output_root": "../runs/training-duration-20261015/screen", "total_wall_seconds": 23000,
        "final_views": 0, "model_promoted": False, "runs": entries,
        "preregistration": "../runs/training-duration-20261015/preregistration.json",
        "run_order": "Budget first: 1500, 5000, 10000; within budget July18-21 then July25-28; seed11 only",
        "change": {"factor": "model.max_steps", "baseline": 1500, "candidates": [5000, 10000],
                   "hypothesis": HYPOTHESIS},
        "rules": rules,
    }
    preregistration = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
        "hypothesis": HYPOTHESIS, "references": references, "calibration": calibration,
        "experiment": {
            "budgets": list(STEPS), "periods": list(PERIODS), "screen_seed": 11, "replication_seeds": [29, 47],
            "checkpoints": {"every_steps": 1500, "include_initial_and_final": True, "selection": "final only"},
            "immutable_recipe": {"training_days": 10, "training_and_evaluation_times": list(TIMES),
                                 "input": "raw12", "target_scaling": "train_std", "learning_rate": 0.0001,
                                 "batch_size": 32, "epochs_ceiling": 12, "time_loss_weighting": "none",
                                 "trade_features": "zero", "smoothing": False},
            "evaluation_scope": "Same July research windows, full eight-time inference universe; final_views=0",
            "reference_reuse": "Historical1500 archives calibrate noise and check reproduction only. All screen "
                               "pairs use new identical core code; archived weights are not overwritten.",
        },
        "rules": rules, "resource_budget": {"screen_runs": 6, "per_run_wall_seconds": 3600,
                                            "screen_total_wall_seconds": 23000, "max_rss_mb": 6144},
        "limitations": ["Only eight distinct research validation days; previous screening creates selection bias.",
                        "Local approximate gross metrics exclude fees and slippage; no official score claim.",
                        "Longer budgets may overfit; lower training MSE alone is not an improvement.",
                        "No formal promotion and no final-split data access."],
        "final_views": 0, "model_promoted": False,
        "generator": {"script": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
        "checks": {"reference_artifacts_verified": True, "reference_metrics_recomputed": True,
                   "six_configs_schema_valid": True, "single_factor_pairs": True,
                   "reference_seed_period_grid_complete": True},
    }
    config_dir.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        write_json(config_dir / f"{name}.json", config)
    write_json(plan_path, plan)
    preregistration["outputs"] = {
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "configs": [{"path": str(config_dir / f"{name}.json"),
                     "sha256": file_sha256(config_dir / f"{name}.json")} for name in configs],
    }
    write_json(prereg_path, preregistration)
    return {"prepared": True, "training_started": False, "plan": str(plan_path),
            "preregistration": str(prereg_path), "configs": len(configs),
            "pooled_seed_std": calibration["pooled_seed_std"]}


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(prepare(parser.parse_args().root), indent=2, ensure_ascii=False))
