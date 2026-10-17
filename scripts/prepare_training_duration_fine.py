"""Freeze the fine-grained duration screen around the retained 1500-step reference.

The 2026-10-15 duration round kept 1500 updates; 5000 and 10000 failed their
preregistered screen. This round adds a finer grid (500-3000) around the
retained budget, including budgets below it, and registers an explicit
baseline_budget so grids extending below the reference stay comparable.

Default mode freezes the seed-11 screen plan from archived July references.
"--replication <decision.json>" generates the preregistered seed-29/47
replication plan for the screen's single recommended budget. Both modes only
read completed research archives; they never train, open the final split, or
overwrite an existing plan, configuration directory, or preregistration.

Archived configs record the original machine's absolute Windows data paths.
"--remap OLD=NEW" (repeatable; exact matches win over prefixes) rewrites them
for this server, verifies every remapped input exists, and records the rules
in the preregistration. Example:

    --remap 'D:\\JinyiQuant\\cpt_jyc_2026_vwap.rar=/srv/labels-july' \\
    --remap 'D:\\JinyiQuant=/srv/JinyiQuant'
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from jinyiquant.checks import verify_artifacts
from jinyiquant.config import RunConfig
from jinyiquant.io import file_sha256, read_json, utc_now, write_json

EXPERIMENT = "training-duration-fine-20261016"
PERIODS = ("july_18_21", "july_25_28")
SEEDS = (11, 29, 47)
SCREEN_SEED = 11
REPLICATION_SEEDS = (29, 47)
STEPS = (500, 750, 1000, 1500, 2000, 2500, 3000)
BASELINE = 1500
CURVE_EVERY_STEPS = 250
SCREEN_WALL_SECONDS = 20000
REPLICATION_WALL_SECONDS = 12000
PER_RUN_WALL_SECONDS = 3600.0
MAX_RSS_MB = 6144.0
TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
METRICS = {
    "rank_ic": "rank_ic_mean",
    "long_short_return": "long_short_return_mean",
    "turnover": "turnover_mean",
    "worst_day_long_short_return": "min_daily_long_short_return",
}
HYPOTHESIS = (
    "At the fixed ten-day/all-eight-time raw12/train_std recipe and learning rate 1e-4, "
    "compare 500, 750, 1000, 1500, 2000, 2500 and 3000 optimizer updates on a fine grid "
    "around the retained 1500-step reference to locate the best training duration; the "
    "previous 5000 and 10000 update budgets failed their preregistered screen."
)


PATH_FIELDS = ("bars_path", "labels_path", "instruments_path", "instrument_map_path", "cache_dir")
VERIFIED_FIELDS = ("bars_path", "labels_path", "instruments_path", "instrument_map_path")


def parse_remap(pairs):
    """Parse OLD=NEW rules; backslashes normalize to '/', exact matches beat prefixes."""
    rules = []
    for item in pairs or []:
        old, separator, new = item.partition("=")
        if not separator or not old.strip() or not new.strip():
            raise ValueError(f"--remap must be OLD=NEW: {item}")
        old = old.strip().replace("\\", "/")
        new = new.strip().replace("\\", "/")
        if not PurePosixPath(new).is_absolute():
            raise ValueError(f"--remap NEW must be an absolute path: {item}")
        rules.append((old, new))
    if len({old for old, _ in rules}) != len(rules):
        raise ValueError("Each --remap OLD prefix may only be mapped once")
    return tuple(rules)


def remap_path(value, rules):
    """Return (remapped-or-original value, whether a rule matched)."""
    if not isinstance(value, str) or not value:
        return value, False
    normalized = value.replace("\\", "/")
    for old, new in rules:
        if normalized == old:
            return new, True
    for old, new in rules:
        if normalized.startswith(old + "/"):
            return new + normalized[len(old):], True
    return value, False


def resolve_run_path(recorded, rules, *, kind="reference run"):
    path = Path(recorded)
    if path.exists():
        return path
    candidate, mapped = remap_path(recorded, rules)
    if mapped and Path(candidate).exists():
        return Path(candidate)
    raise ValueError(f"Cannot find {kind} on this machine (recorded {recorded})"
                     + f"; remap resolved to {candidate}" if mapped else "; no remap rule matched")


def apply_source_remap(sources, rules):
    """Rewrite archived data paths for this server and verify migrated inputs exist."""
    if not rules:
        return sources
    missing = []
    for snapshot in sources.values():
        for field in PATH_FIELDS:
            value = snapshot["data"].get(field)
            if value is None:
                continue
            candidate, mapped = remap_path(value, rules)
            if not mapped:
                continue
            snapshot["data"][field] = candidate
            if field in VERIFIED_FIELDS and not Path(candidate).exists():
                missing.append(f"{field}: {candidate}")
    if missing:
        raise ValueError("Remapped input paths are missing on this machine: " + "; ".join(sorted(missing)))
    return sources


def reference_evidence(comparison_path, remap=()):
    """Verify the six archived 1500-step references and calibrate seed noise."""
    comparison = read_json(comparison_path)
    if not comparison.get("checks", {}).get("passed") or comparison.get("final_views") != 0:
        raise ValueError("Reference comparison must pass and contain no final views")
    inputs = [item for item in comparison["inputs"] if item["arm"] == "candidate"]
    expected = {(period, seed) for period in PERIODS for seed in SEEDS}
    if len(inputs) != len(expected) or {(x["period"], x["seed"]) for x in inputs} != expected:
        raise ValueError("Reference requires exactly two periods by three unique seeds")
    rows, references, configs = [], [], {}
    for item in sorted(inputs, key=lambda x: (x["period"], x["seed"])):
        path = resolve_run_path(item["run"], remap)
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


def rules():
    return {
        "delta": "candidate minus same-period same-seed 1500-step baseline",
        "final_decision": "Passing supports a replacement research candidate only; no formal champion promotion",
        "intermediate_validation": "Diagnostic only; do not select unregistered intermediate steps or alter LR",
        "noise_threshold_source": "calibration.pooled_seed_std; multiplier=1.0 fixed before screening",
        "primary_metrics": ["rank_ic", "long_short_return"],
        "replication": {
            "additional_seeds": list(REPLICATION_SEEDS),
            "periods": list(PERIODS),
            "aggregate_improvement": "At least one primary mean delta >= its pooled_seed_std and the same metric "
                                     "has paired-day block 95% interval lower bound > 0",
            "baseline": "Rerun 1500 steps with the identical new core source for both additional seeds",
            "budgets": "Replicate only the single selected screen budget with its 1500-step baseline pairs",
            "complete_pairs": 4,
            "every_pair_primary_delta_min": 0.0,
            "failure_action": "Keep the 1500-step reference, report failed or inconclusive evidence",
            "risk_gate": "Every period-seed pair: each risk metric regression <= its own pooled_seed_std",
            "uncertainty": {
                "block_days": 2,
                "bootstrap_samples": 2000,
                "bootstrap_seed": 20261016,
                "confidence": 0.95,
                "method": "Average seeds within each day; moving two-day blocks within each period",
            },
        },
        "risk_directions": {
            "opening_long_short_return": "maximize",
            "opening_rank_ic": "maximize",
            "turnover": "minimize",
            "worst_day_long_short_return": "maximize",
        },
        "screen": {
            "aggregate_improvement": "Equal-period mean delta of at least one primary >= its pooled_seed_std",
            "every_period_primary_delta_min": 0.0,
            "failed_action": "Do not expand seeds for a failed budget",
            "required_periods": list(PERIODS),
            "risk_gate": "In every period, each risk metric regression <= its own pooled_seed_std",
            "seed": SCREEN_SEED,
        },
        "selection": {
            "eligible": "Only budgets passing the complete screen including risk gate",
            "none_pass": "Keep the 1500-step reference; stop without seed expansion",
            "priority": "Higher equal-period mean RankIC; exact RankIC ties choose fewer steps",
            "shorter_budget_tie": "If absolute differences between eligible budgets are below both primary "
                                  "pooled_seed_std thresholds, select the shorter budget",
        },
        "stopping": "Only the seven preregistered budgets 500-3000 screen; stop at declared caps or failures. "
                    "No extra budgets, LR search, or final-split access. Preserve all attempts and reference "
                    "weights.",
    }


def _config(sources, period, seed, steps, output_root):
    raw = deepcopy(sources[period, seed])
    raw.update(experiment_id=EXPERIMENT, hypothesis=HYPOTHESIS, output_root=str(output_root))
    raw["seed"] = seed
    raw["model"].update(max_steps=steps, book_representation="raw",
                        time_loss_weighting="none", trade_features="zero")
    raw["budget"].update(wall_seconds=PER_RUN_WALL_SECONDS, max_rss_mb=MAX_RSS_MB)
    raw["logging"].update(learning_curves=True, learning_curve_every_steps=CURVE_EVERY_STEPS)
    return RunConfig.model_validate(raw).snapshot()


def prepare_screen(root, references, sources, calibration, remap=()):
    plan_path = root / "configs/training_duration_fine_20261016_plan.json"
    config_dir = root / "configs/training_duration_fine_20261016"
    prereg_path = root / "runs/training-duration-fine-20261016/preregistration.json"
    for path in (plan_path, config_dir, prereg_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    output_root = root / "runs/training-duration-fine-20261016/screen"
    configs, entries = {}, []
    for steps in STEPS:
        for period in PERIODS:
            config = _config(sources, period, SCREEN_SEED, steps, output_root)
            name = f"{period}_s{SCREEN_SEED}_steps{steps}"
            configs[name] = config
            entries.append({"name": name, "period": period, "seed": SCREEN_SEED, "steps": steps,
                            "arm": "baseline" if steps == BASELINE else f"candidate_{steps}",
                            "config": f"{config_dir.name}/{name}.json"})
    for period in PERIODS:
        baseline = deepcopy(configs[f"{period}_s{SCREEN_SEED}_steps{BASELINE}"])
        for steps in STEPS:
            if steps == BASELINE:
                continue
            candidate = deepcopy(configs[f"{period}_s{SCREEN_SEED}_steps{steps}"])
            candidate["model"]["max_steps"] = baseline["model"]["max_steps"]
            if candidate != baseline:
                raise ValueError("Generated duration pair changes more than max_steps")
    experiment_rules = rules()
    plan = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "hypothesis": HYPOTHESIS,
        "output_root": "../runs/training-duration-fine-20261016/screen",
        "total_wall_seconds": SCREEN_WALL_SECONDS,
        "final_views": 0, "model_promoted": False, "runs": entries,
        "preregistration": "../runs/training-duration-fine-20261016/preregistration.json",
        "run_order": "Ascending budget: 500, 750, 1000, 1500, 2000, 2500, 3000; within budget "
                     "July18-21 then July25-28; seed11 only",
        "change": {"factor": "model.max_steps", "baseline": BASELINE,
                   "candidates": [steps for steps in STEPS if steps != BASELINE],
                   "hypothesis": HYPOTHESIS},
        "rules": experiment_rules,
    }
    preregistration = {
        "schema_version": 1, "experiment_id": EXPERIMENT, "created_at_utc": utc_now(),
        "hypothesis": HYPOTHESIS, "references": references, "calibration": calibration,
        "experiment": {
            "budgets": list(STEPS), "baseline_budget": BASELINE, "periods": list(PERIODS),
            "screen_seed": SCREEN_SEED, "replication_seeds": list(REPLICATION_SEEDS),
            "path_remap": {"rules": [f"{old}={new}" for old, new in remap],
                           "note": "Original machine's absolute paths rewritten for this server; "
                                   "remapped inputs verified to exist"} if remap else None,
            "checkpoints": {"every_steps": CURVE_EVERY_STEPS, "include_initial_and_final": True,
                            "selection": "final only"},
            "immutable_recipe": {"training_days": 10, "training_and_evaluation_times": list(TIMES),
                                 "input": "raw12", "target_scaling": "train_std", "learning_rate": 0.0001,
                                 "batch_size": 32, "epochs_ceiling": 12, "time_loss_weighting": "none",
                                 "trade_features": "zero", "smoothing": False},
            "evaluation_scope": "Same July research windows, full eight-time inference universe; final_views=0",
            "reference_reuse": "Historical1500 archives calibrate noise and check reproduction only. All screen "
                               "pairs use new identical core code; archived weights are not overwritten.",
        },
        "rules": experiment_rules, "resource_budget": {"screen_runs": len(entries),
                                                       "per_run_wall_seconds": int(PER_RUN_WALL_SECONDS),
                                                       "screen_total_wall_seconds": SCREEN_WALL_SECONDS,
                                                       "max_rss_mb": int(MAX_RSS_MB)},
        "limitations": ["Only eight distinct research validation days; repeated screening creates selection bias.",
                        "Local approximate gross metrics exclude fees and slippage; no official score claim.",
                        "Adjacent fine-grid budgets may differ by less than seed noise; sub-noise ties select "
                        "the shorter budget.",
                        "Longer budgets may overfit; lower training MSE alone is not an improvement.",
                        "No formal promotion and no final-split data access."],
        "final_views": 0, "model_promoted": False,
        "generator": {"script": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
        "checks": {"reference_artifacts_verified": True, "reference_metrics_recomputed": True,
                   "config_schema_valid": True, "single_factor_pairs": True,
                   "reference_seed_period_grid_complete": True,
                   "explicit_baseline_budget_registered": True},
    }
    config_dir.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        write_json(config_dir / f"{name}.json", config)
    write_json(plan_path, plan)
    preregistration["outputs"] = {
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "configs": [{"path": str(config_dir / f"{name}.json"),
                     "sha256": file_sha256(config_dir / f"{name}.json")} for name in sorted(configs)],
    }
    write_json(prereg_path, preregistration)
    return {"prepared": True, "training_started": False, "plan": str(plan_path),
            "preregistration": str(prereg_path), "configs": len(configs),
            "baseline_budget": BASELINE, "pooled_seed_std": calibration["pooled_seed_std"]}


def prepare_replication(root, sources, decision_path):
    decision = read_json(decision_path)
    prereg_path = root / "runs/training-duration-fine-20261016/preregistration.json"
    if not prereg_path.is_file():
        raise FileNotFoundError(f"Missing frozen preregistration: {prereg_path}")
    if decision.get("experiment_id") != EXPERIMENT:
        raise ValueError("Decision belongs to a different experiment")
    if decision.get("registration_sha256") != file_sha256(prereg_path):
        raise ValueError("Decision was not produced from the current frozen preregistration")
    if not decision.get("historical_reference_drift", {}).get("passed"):
        raise ValueError("Historical reference drift failed; replication is blocked")
    selected = decision.get("recommend_replication_budget")
    if selected not in STEPS or selected == BASELINE:
        raise ValueError("Decision does not recommend a preregistered candidate budget")
    plan_path = root / "configs/training_duration_fine_20261016_replication_plan.json"
    config_dir = root / "configs/training_duration_fine_20261016_replication"
    for path in (plan_path, config_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    output_root = root / "runs/training-duration-fine-20261016/replication"
    configs, entries = {}, []
    for seed in REPLICATION_SEEDS:
        for period in PERIODS:
            for arm, steps in (("baseline", BASELINE), ("candidate", selected)):
                config = _config(sources, period, seed, steps, output_root)
                name = f"{period}_s{seed}_steps{steps}"
                configs[name] = config
                entries.append({"name": name, "period": period, "seed": seed, "steps": steps,
                                "arm": arm, "config": f"{config_dir.name}/{name}.json"})
    for period in PERIODS:
        for seed in REPLICATION_SEEDS:
            baseline = deepcopy(configs[f"{period}_s{seed}_steps{BASELINE}"])
            candidate = deepcopy(configs[f"{period}_s{seed}_steps{selected}"])
            candidate["model"]["max_steps"] = baseline["model"]["max_steps"]
            if candidate != baseline:
                raise ValueError("Generated replication pair changes more than max_steps")
    plan = {
        "schema_version": 1, "experiment_id": f"{EXPERIMENT}-replication", "hypothesis": HYPOTHESIS,
        "output_root": "../runs/training-duration-fine-20261016/replication",
        "total_wall_seconds": REPLICATION_WALL_SECONDS,
        "final_views": 0, "model_promoted": False, "runs": entries,
        "preregistration": "../runs/training-duration-fine-20261016/preregistration.json",
        "replication_of": {"decision": str(decision_path),
                           "decision_sha256": file_sha256(decision_path),
                           "selected_budget": selected, "baseline_budget": BASELINE},
        "run_order": f"Seed29 stage then seed47; within seed July18-21 then July25-28; "
                     f"1500-step baseline before {selected}-step candidate in each period",
        "change": {"factor": "model.max_steps", "baseline": BASELINE, "candidate": selected,
                   "hypothesis": HYPOTHESIS},
        "rules": rules()["replication"],
    }
    config_dir.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        write_json(config_dir / f"{name}.json", config)
    write_json(plan_path, plan)
    return {"prepared": True, "training_started": False, "plan": str(plan_path), "configs": len(configs),
            "selected_budget": selected, "baseline_budget": BASELINE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replication", type=Path, default=None,
                        help="Generate the replication plan from a completed screen decision.json")
    parser.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                        help="Rewrite archived absolute paths for this server (repeatable)")
    args = parser.parse_args()
    root = args.root.resolve()
    rules = parse_remap(args.remap)
    comparison = root / "runs/training-progression/step3c-full-analysis/comparison.json"
    references, sources, calibration = reference_evidence(comparison, rules)
    apply_source_remap(sources, rules)
    if args.replication is None:
        result = prepare_screen(root, references, sources, calibration, remap=rules)
    else:
        result = prepare_replication(root, sources, Path(args.replication).resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
