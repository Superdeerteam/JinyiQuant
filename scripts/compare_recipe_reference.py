"""Describe cumulative recipe quality against a reference; no causal attribution or promotion.

Plan: {schema_version: 1, experiment_id, pairs: [{period, seed, baseline, candidate}],
bootstrap_samples: 2000, bootstrap_seed: 20261015}. Paths are relative to the plan.
"""

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.evaluation import evaluate
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.protocol import recipe, without_locations
from scripts.compare_training_round import (
    KEYS,
    METRICS,
    SUMMARY,
    _curve_finished,
    _keys,
    _load,
    paired_intervals,
)


def differences(left, right, prefix=""):
    """Report every changed, added or removed field without treating missing as null."""
    rows = []
    for key in sorted(left.keys() | right.keys()):
        name = f"{prefix}.{key}" if prefix else key
        if key in left and key in right and isinstance(left[key], dict) and isinstance(right[key], dict):
            rows.extend(differences(left[key], right[key], name))
        elif key not in left or key not in right or left[key] != right[key]:
            rows.append({"field": name, "baseline_present": key in left, "candidate_present": key in right,
                         "baseline": left.get(key), "candidate": right.get(key)})
    return rows


def compare(plan_path, destination):
    plan_path, destination = Path(plan_path).resolve(), Path(destination).resolve()
    plan = read_json(plan_path)
    required = {"schema_version", "experiment_id", "pairs"}
    if (not required.issubset(plan) or set(plan) - required - {"bootstrap_samples", "bootstrap_seed"}
            or plan["schema_version"] != 1 or not isinstance(plan["pairs"], list) or not plan["pairs"]):
        raise ValueError("Invalid explicit reference comparison plan")
    if destination.exists():
        raise ValueError("Comparison output must be a new directory")
    evaluation_hash = file_sha256(Path(inspect.getfile(evaluate)))
    summary, daily_rows, sections_rows, paired_rows, inputs, changes = [], [], [], [], [], []
    seen, paths, windows, arm_recipes, arm_sources, period_identity = set(), set(), {}, {}, {}, {}
    evaluation_populations = {}
    fixed_contract = None
    for pair in plan["pairs"]:
        if (set(pair) != {"period", "seed", "baseline", "candidate"}
                or not isinstance(pair["period"], str) or not pair["period"].strip() or pair["period"] == "all"
                or type(pair["seed"]) is not int or pair["seed"] < 0):
            raise ValueError("Each pair needs a named period, integer seed and two archive paths")
        key = pair["period"], pair["seed"]
        if key in seen:
            raise ValueError("Duplicate period/seed pair")
        seen.add(key)
        loaded = {}
        for arm in ("baseline", "candidate"):
            path = (plan_path.parent / pair[arm]).resolve()
            if path in paths or destination == path or destination.is_relative_to(path):
                raise ValueError("Archives must be distinct and output cannot modify an input archive")
            paths.add(path)
            run, daily, sections = _load(path)
            curve = read_json(path / "learning_curve.json")
            _curve_finished(curve, run.config)
            steps, batch_size = curve[-1]["steps"], run.config["model"].get("batch_size")
            if type(steps) is not int or steps <= 0 or type(batch_size) is not int or batch_size <= 0:
                raise ValueError("Positive actual steps and batch_size are required")
            if run.config["seed"] != pair["seed"]:
                raise ValueError("Declared seed differs from archived run")
            source = {name: checksum for name, checksum in run.manifest.get("code", {}).get("files", {}).items()
                      if name.endswith(".py") and name.startswith(("src/jinyiquant/", "modified_tlob/"))}
            if source.get("src/jinyiquant/evaluation.py") != evaluation_hash:
                raise ValueError("Archived evaluation source must match the current evaluator hash")
            environment = run.manifest.get("environment")
            files = without_locations(run.data.get("files"))
            if not environment or not files:
                raise ValueError("Software environment and data source files are required")
            contract = {"evaluation": run.config["evaluation"], "sample_times": run.config["data"]["sample_times"],
                        "environment": environment, "actual_steps": steps, "batch_size": batch_size,
                        "files": files, "evaluation_source_sha256": evaluation_hash}
            if fixed_contract is not None and contract != fixed_contract:
                raise ValueError("Evaluation times/config, environment, actual steps/batch or source files differ")
            fixed_contract = contract
            if pair["period"] in windows and windows[pair["period"]] != run.period:
                raise ValueError("A period must use the same validation split")
            windows[pair["period"]] = run.period
            fixed = recipe(run.config, across_grid=True)
            fixed["splits"].pop("train")
            if arm in arm_recipes and (fixed != arm_recipes[arm] or source != arm_sources[arm]):
                raise ValueError("Each arm must keep one recipe and core implementation across seeds/periods")
            arm_recipes[arm], arm_sources[arm] = fixed, source
            identity = (run.data, run.config["splits"], _keys(path, "train_history.parquet"))
            previous = period_identity.get((pair["period"], arm))
            if previous is not None and (previous[:2] != identity[:2] or not previous[2].equals(identity[2])):
                raise ValueError("Training data, splits or keys changed across seeds within one arm/period")
            period_identity[pair["period"], arm] = identity
            resources = run.manifest.get("resources", {})
            if any(not isinstance(resources.get(field), (int, float)) or not np.isfinite(resources[field])
                   or resources[field] < 0 for field in ("wall_seconds", "peak_rss_mb")):
                raise ValueError("Recorded wall time and peak RSS are required")
            metadata = {"period": pair["period"], "seed": pair["seed"], "arm": arm}
            summary.append({**metadata, "run": str(path), **{field: run.metrics[field] for field in SUMMARY},
                            "training_samples": len(identity[2]), "actual_steps": steps, "batch_size": batch_size,
                            "actual_epochs": curve[-1]["epoch"], "resources": resources,
                            "correlation_coverage": run.metrics["correlation_coverage"]})
            daily_rows.append(daily.assign(**metadata))
            sections_rows.append(sections.assign(**metadata))
            inputs.append({**metadata, "run": str(path), "manifest_sha256": file_sha256(path / "manifest.json")})
            loaded[arm] = run, daily, source
        left, right = loaded["baseline"][0], loaded["candidate"][0]
        for name, columns in (("predictions.parquet", KEYS), ("scored_predictions.parquet", KEYS),
                              ("targets.parquet", KEYS + ["target"])):
            population = _keys(left.path, name, columns)
            if not population.equals(_keys(right.path, name, columns)):
                raise ValueError("Paired full/scored prediction keys or targets differ")
            prior = evaluation_populations.get((pair["period"], name))
            if prior is not None and not population.equals(prior):
                raise ValueError("Evaluation population or targets changed across seeds within one period")
            evaluation_populations[pair["period"], name] = population
        changes.append({"period": pair["period"], "seed": pair["seed"],
                        "config_differences": differences(left.config, right.config),
                        "core_source_differences": differences(loaded["baseline"][2], loaded["candidate"][2])})
        joined = loaded["baseline"][1][["date", *METRICS]].merge(
            loaded["candidate"][1][["date", *METRICS]], on="date", validate="one_to_one",
            suffixes=("_baseline", "_candidate"))
        paired_rows.append(joined.assign(period=pair["period"], seed=pair["seed"]))
    ordered = sorted(windows.values())
    if any(pd.Timestamp(left[1]) > pd.Timestamp(right[0]) for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Validation periods must not overlap")
    paired = pd.concat(paired_rows, ignore_index=True)
    intervals = paired_intervals(paired, bootstrap_samples=plan.get("bootstrap_samples", 2000),
                                 seed=plan.get("bootstrap_seed", 20261015))
    seeds = {seed for _, seed in seen}
    complete = len(windows) >= 2 and len(seeds) >= 3 and seen == {(p, s) for p in windows for s in seeds}
    result = {"schema_version": 1, "experiment_id": plan["experiment_id"], "created_at": utc_now(),
              "comparison_kind": "cumulative_recipe_reference", "plan": plan, "inputs": inputs,
              "fixed_contract": fixed_contract, "recipe_changes": changes, "summary": summary,
              "paired_intervals": intervals, "complete_two_period_three_seed_grid": complete,
              "conclusion": "descriptive_only", "model_promoted": False, "single_factor_attribution": False,
              "final_views": 0, "confidence": .95, "block_days": 2,
              "limits": "Selected cumulative recipes, not a single-factor experiment. Local approximate gross metrics; "
                        "seed-averaged paired moving two-day blocks within periods. Few research dates and selection "
                        "bias remain. Inspect worst-day/tail risk and resources; positive main-metric CIs do not promote a model."}
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "comparison.json", result)
    write_json(destination / "paired_intervals.json", intervals)
    paired.to_csv(destination / "paired_daily.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(destination / "daily.csv", index=False)
    pd.concat(sections_rows, ignore_index=True).to_csv(destination / "by_timestamp.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in row.items() if k not in {"resources", "correlation_coverage"}},
                   **row["resources"]} for row in summary]).to_csv(destination / "summary.csv", index=False)
    shutil.copyfile(__file__, destination / Path(__file__).name)
    write_json(destination / "manifest.json", {
        "schema_version": 1, "status": "completed", "created_at": utc_now(), "inputs": inputs,
        "plan_sha256": file_sha256(plan_path), "script_sha256": file_sha256(Path(__file__)),
        "dependency_sha256": {name: file_sha256(Path(inspect.getfile(function))) for name, function in
                              (("comparison", _load), ("evaluation", evaluate), ("protocol", recipe))},
        "artifacts": [{"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
                      for path in sorted(destination.iterdir()) if path.is_file()]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.plan, args.output)
        print(json.dumps({"output": str(args.output.resolve()), "conclusion": result["conclusion"]}))
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"exit_code": 2, "error": str(exc)}))
        raise SystemExit(2) from exc
