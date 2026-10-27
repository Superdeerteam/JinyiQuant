"""Offline alpha=0.5 causal smoothing on explicit completed validation archives.

Plan: {schema_version: 1, experiment_id, runs: [{period, seed, run}],
bootstrap_samples: 2000, bootstrap_seed: 20261015}. Paths are relative to the plan.
EWMA uses pandas' adjust=False recurrence; no alpha search or model promotion.
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
"""

import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.comparison import archived_cost_scenarios
from jinyiquant.evaluation import EVALUATION_VERSION, _validated, evaluate
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.protocol import recipe
from scripts.compare_training_round import (
    KEYS,
    METRICS,
    SUMMARY,
    _curve_finished,
    _keys,
    _load,
    paired_intervals,
)

ALPHA = .5


def smooth_scores(predictions, *, alpha=ALPHA):
    """Use the complete current cross section, then each stock's observed daily history."""
    if not 0 < alpha <= 1 or set(predictions) != {*KEYS, "score"}:
        raise ValueError("Smoothing requires 0 < alpha <= 1 and date/instrument/score only")
    frame = _validated(predictions, "score", "predictions").sort_values(KEYS, ignore_index=True)
    if alpha == 1:
        return frame
    sections = frame.groupby("date", sort=False).score
    mean, std = sections.transform("mean"), sections.transform("std", ddof=0)
    z = (frame.score - mean) / std.where(std != 0, 1.0)
    history = z.groupby([frame.date.dt.normalize(), frame.instrument], sort=False)
    smoothed = history.transform(lambda values: values.ewm(alpha=alpha, adjust=False).mean())
    frame["score"] = mean + std * smoothed
    if not np.isfinite(frame.score).all():
        raise ValueError("Smoothing produced nonfinite scores")
    return frame


def compare(plan_path, destination):
    plan_path, destination = Path(plan_path).resolve(), Path(destination).resolve()
    plan = read_json(plan_path)
    required = {"schema_version", "experiment_id", "runs"}
    if (not required.issubset(plan) or set(plan) - required - {"bootstrap_samples", "bootstrap_seed"}
            or plan["schema_version"] != 1 or not isinstance(plan["runs"], list) or not plan["runs"]):
        raise ValueError("Invalid explicit smoothing plan; alpha is fixed at 0.5")
    if destination.exists():
        raise ValueError("Smoothing output must be a new directory")
    inputs, outputs, metrics_rows, daily_rows, section_rows, pairs = [], [], [], [], [], []
    seen, paths, windows, period_data, fixed_recipe = set(), set(), {}, {}, None
    complete_coverage = True
    for item in plan["runs"]:
        if (set(item) != {"period", "seed", "run"} or not isinstance(item["period"], str)
                or not item["period"].strip() or item["period"] == "all"
                or type(item["seed"]) is not int or item["seed"] < 0):
            raise ValueError("Each run needs a named period, integer seed and archive path")
        key = item["period"], item["seed"]
        path = (plan_path.parent / item["run"]).resolve()
        if key in seen or path in paths or destination == path or destination.is_relative_to(path):
            raise ValueError("Duplicate run or output inside an input archive")
        seen.add(key)
        paths.add(path)
        run, _, _ = _load(path)
        _curve_finished(read_json(path / "learning_curve.json"), run.config)
        if run.config["seed"] != item["seed"]:
            raise ValueError("Declared seed differs from archived run")
        if item["period"] in windows and windows[item["period"]] != run.period:
            raise ValueError("A period must use one validation interval")
        windows[item["period"]] = run.period
        current_recipe = recipe(run.config, across_grid=True)
        current_recipe["splits"].pop("train")
        if fixed_recipe is not None and current_recipe != fixed_recipe:
            raise ValueError("Smoothing runs must share one fixed model and evaluation recipe")
        fixed_recipe = current_recipe
        identity = (run.data, run.config["splits"], _keys(path, "train_history.parquet"))
        previous = period_data.get(item["period"])
        if previous is not None and (previous[:2] != identity[:2] or not previous[2].equals(identity[2])):
            raise ValueError("Training data, splits or keys changed across seeds within one period")
        period_data[item["period"]] = identity
        full = smooth_scores(pd.read_parquet(path / "predictions.parquet"), alpha=1)
        start, end = map(pd.Timestamp, run.period)
        if not ((full.date >= start) & (full.date < end)
                & full.date.dt.strftime("%H:%M").isin(run.config["data"]["sample_times"])).all():
            raise ValueError("Full predictions must stay inside declared validation dates/times")
        scored = smooth_scores(pd.read_parquet(path / "scored_predictions.parquet"), alpha=1)
        original_subset = scored[KEYS].merge(full, on=KEYS, how="left", validate="one_to_one")
        if not np.array_equal(original_subset.score.to_numpy(), scored.score.to_numpy()):
            raise ValueError("Archived scored predictions are not an unchanged subset of full predictions")
        targets = pd.read_parquet(path / "targets.parquet")
        metadata = {"period": item["period"], "seed": item["seed"]}
        arm_daily, arm_outputs = {}, {}
        # Smooth the FULL inference universe before applying label-available scoring keys.
        for arm, predictions in (("baseline", full), ("candidate", smooth_scores(full))):
            selected = scored[KEYS].merge(predictions, on=KEYS, how="left", validate="one_to_one")
            metrics, daily, sections = evaluate(selected, targets, run.config["evaluation"], return_sections=True,
                undefined_constant_portfolios=run.config["evaluation"].get("constant_portfolios") == "undefined")
            for metric in ("ic", "rank_ic"):
                coverage = metrics["correlation_coverage"][metric]
                complete_coverage &= (coverage["eligible_section_count"] > 0 and
                                      coverage["eligible_section_count"] == coverage["valid_section_count"])
            metrics_rows.append({**metadata, "arm": arm, **metrics})
            daily_rows.append(daily.assign(**metadata, arm=arm))
            section_rows.append(sections.assign(**metadata, arm=arm))
            arm_daily[arm], arm_outputs[arm] = daily, (predictions, selected)
        paired = arm_daily["baseline"][["date", *METRICS]].merge(
            arm_daily["candidate"][["date", *METRICS]], on="date", validate="one_to_one",
            suffixes=("_baseline", "_candidate"))
        pairs.append(paired.assign(**metadata))
        output_name = f"run_{len(outputs):03d}"
        outputs.append((output_name, arm_outputs))
        inputs.append({**metadata, "run": str(path), "output": output_name,
                       "manifest_sha256": file_sha256(path / "manifest.json"), "config": run.config,
                       "files_sha256": {name: file_sha256(path / name) for name in
                            ("predictions.parquet", "scored_predictions.parquet", "targets.parquet")}})
    ordered = sorted(windows.values())
    if any(pd.Timestamp(left[1]) > pd.Timestamp(right[0]) for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Research validation periods must not overlap")
    paired = pd.concat(pairs, ignore_index=True)
    intervals = paired_intervals(paired, bootstrap_samples=plan.get("bootstrap_samples", 2000),
                                 seed=plan.get("bootstrap_seed", 20261015))
    seeds = {seed for _, seed in seen}
    complete = len(windows) >= 2 and len(seeds) >= 3 and seen == {(p, s) for p in windows for s in seeds}
    guards = {}
    for metric in ("rank_ic", "long_short_return"):
        rows = [row for row in intervals if row["metric"] == metric and row["period"] != "all"]
        guards[metric] = {
            "nonnegative_point_change_each_period": all(row["mean"] is not None and row["mean"] >= 0 for row in rows),
            "nonnegative_95ci_lower_each_period": all(row["interval"] is not None and row["interval"][0] >= 0
                                                       for row in rows)}
    result = {"schema_version": 1, "experiment_id": plan["experiment_id"], "created_at": utc_now(),
              "evaluation_version": EVALUATION_VERSION,
              "plan": plan, "plan_sha256": file_sha256(plan_path), "inputs": inputs,
              "alpha": ALPHA, "script_sha256": file_sha256(Path(__file__)),
              "dependency_sha256": {name: file_sha256(Path(inspect.getfile(function)))
                                     for name, function in (("comparison", _load), ("evaluation", evaluate))},
              "checks": {"passed": True, "full_universe_before_label_filter": True,
                         "complete_two_period_three_seed_grid": complete,
                         "complete_target_eligible_correlation_coverage": bool(complete_coverage)},
              "summary": [{**{key: row[key] for key in ("period", "seed", "arm")},
                           **{key: row[key] for key in SUMMARY}} for row in metrics_rows],
              "paired_intervals": intervals, "quality_guardrails": guards,
              "conclusion": "review_required" if complete else "insufficient_evidence",
              "model_promoted": False, "final_views": 0, "confidence": .95, "block_days": 2,
              "method": "Current full-cross-section population z-score; per-stock per-day adjust=False EWMA; "
                        "first observation unchanged in z units; reconstruct with current mean/std. "
                        "Candidate minus baseline; seed-averaged paired moving two-day blocks within periods.",
              "limits": "Local approximate gross returns and turnover, including overnight transitions. "
                        "Lower turnover alone is insufficient: RankIC and long-short returns must both be preserved "
                        "with uncertainty reviewed. No calibrated noninferiority margin or automatic promotion; "
                        "few research dates and model-selection bias remain."}
    result["evaluation_contract"] = archived_cost_scenarios(
        (Path(plan_path).resolve().parent / plan["runs"][0]["run"]).resolve())
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "comparison.json", result)
    write_json(destination / "metrics.json", metrics_rows)
    write_json(destination / "checks.json", result["checks"])
    write_json(destination / "paired_intervals.json", intervals)
    paired.to_csv(destination / "paired_daily.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(destination / "daily.csv", index=False)
    pd.concat(section_rows, ignore_index=True).to_csv(destination / "by_timestamp.csv", index=False)
    for name, arms in outputs:
        folder = destination / name
        folder.mkdir()
        for arm, (predictions, selected) in arms.items():
            predictions.to_parquet(folder / f"{arm}_predictions.parquet", index=False)
            selected.to_parquet(folder / f"{arm}_scored_predictions.parquet", index=False)
    shutil.copyfile(__file__, destination / Path(__file__).name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.plan, args.output)
        print(json.dumps({"output": str(args.output.resolve()), "conclusion": result["conclusion"]}))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"exit_code": 2, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
