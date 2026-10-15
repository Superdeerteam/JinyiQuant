"""Read-only comparison of explicit archived run pairs; never train or promote a model.

Plan: {schema_version: 1, experiment_id, change: {factor, baseline, candidate,
hypothesis}, pairs: [{period, seed, baseline, candidate}], bootstrap_samples: 2000,
bootstrap_seed: 20261015}. Run paths are relative to the plan. Partial grids are
reported as insufficient evidence; complete research grids need 2 periods/3 seeds.
Training-only expansions support data.training_sample_times and splits.train.start;
the latter accepts a date or a period-to-date mapping per arm. Evaluation stays fixed.
Optional screen_periods declares the two expected screening windows; otherwise the
pair declarations supply them. screen_gate(result) also audits an archived comparison.
"""

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from jinyiquant.checks import numerical_equal, verify_artifacts
from jinyiquant.comparison import _block_indices, _fair_pair, _read_run
from jinyiquant.evaluation import evaluate
from jinyiquant.io import file_sha256, read_json, utc_now, write_json
from jinyiquant.protocol import recipe, validate_change

KEYS = ["date", "instrument"]
METRICS = ("mse", "ic", "rank_ic", "long_return", "short_return", "long_short_return", "turnover")
SUMMARY = ("mse", "mae", "ic_mean", "rank_ic_mean", "rank_ic_ir", "long_return_mean", "short_return_mean",
           "long_short_return_mean", "long_short_sharpe", "extreme_return", "min_daily_long_short_return",
           "turnover_mean", "score_change_mae", "rank_persistence_mean")
EXPANSIONS = {"data.training_sample_times", "splits.train.start"}


def _period_change(change, period):
    result = dict(change)
    if change["factor"] == "splits.train.start":
        for arm in ("baseline", "candidate"):
            result[arm] = change[arm][period] if isinstance(change[arm], dict) else change[arm]
    return result


def _validate_change(change, periods):
    if not isinstance(change, dict):
        raise ValueError("Change must be an explicit object")
    if change.get("factor") not in EXPANSIONS:
        return validate_change(change)
    if (set(change) != {"factor", "baseline", "candidate", "hypothesis"}
            or not isinstance(change["hypothesis"], str) or not change["hypothesis"].strip()):
        raise ValueError("Data expansion requires factor, baseline, candidate and hypothesis")
    if change["factor"] == "data.training_sample_times":
        for arm in ("baseline", "candidate"):
            value = change[arm]
            if (not isinstance(value, list) or not value or not all(isinstance(t, str) for t in value)
                    or len(set(value)) != len(value)):
                raise ValueError("Training times must be explicit nonempty unique lists")
        if not set(change["baseline"]) < set(change["candidate"]):
            raise ValueError("Candidate training times must strictly include baseline times")
    else:
        for arm in ("baseline", "candidate"):
            if isinstance(change[arm], dict) and set(change[arm]) != periods:
                raise ValueError("Training-start mapping must cover exactly the declared periods")
        for period in periods:
            local = _period_change(change, period)
            dates = [pd.Timestamp(local[arm]) for arm in ("baseline", "candidate")]
            if any(pd.isna(date) or date.tzinfo for date in dates) or dates[1] >= dates[0]:
                raise ValueError("Candidate training start must precede baseline start")


def _curve_finished(curve, config):
    if not curve:
        raise ValueError("Learning curve is missing its final checkpoint")
    last, model = curve[-1], config["model"]
    cap = model.get("max_steps")
    if cap is None:
        complete = last["epoch"] == model["epochs"]
    else:
        complete = (0 < last["epoch"] <= model["epochs"] and 0 < last["steps"] <= cap
                    and (last["steps"] == cap or last["epoch"] == model["epochs"]))
    if not complete:
        raise ValueError("Learning curve has not reached its declared update cap or epoch limit")


def _fair_expansion(left, right, change):
    """Normalize only declared data expansion, then reuse strict pair checks."""
    configs, data = [copy.deepcopy(run.config) for run in (left, right)], [copy.deepcopy(run.data) for run in (left, right)]
    for index, arm in enumerate(("baseline", "candidate")):
        config = configs[index]
        if change["factor"] == "data.training_sample_times":
            if config["data"].get("training_sample_times") != change[arm]:
                raise ValueError("Training times differ from the declared change")
            if not set(change[arm]).issubset(config["data"]["sample_times"]):
                raise ValueError("Training times must stay within the fixed evaluation times")
            config["data"]["training_sample_times"] = change["baseline"]
        else:
            if pd.Timestamp(config["splits"]["train"]["start"]) != pd.Timestamp(change[arm]):
                raise ValueError("Training start differs from the declared change")
            if pd.Timestamp(data[index]["requested_start"]) != pd.Timestamp(change[arm]):
                raise ValueError("Data request differs from the declared training start")
            warm = pd.Timestamp(change[arm]) - pd.Timedelta(days=config["data"].get("history_days", 10))
            if config["data"].get("allowed_start") is not None:
                warm = max(warm, pd.Timestamp(config["data"]["allowed_start"]))
            if pd.Timestamp(data[index]["bars_start"]) != warm:
                raise ValueError("Historical preload start is not derived from the declared training start")
            config["splits"]["train"]["start"] = left.config["splits"]["train"]["start"]
            for field in ("requested_start", "bars_start", "bar_rows", "label_rows", "missing_targets"):
                data[index].pop(field, None)
            for field in ("labels_input_rows", "outside_universe_label_rows", "outside_universe_label_instruments",
                          "labels_retained_rows"):
                data[index].get("alignment", {}).pop(field, None)
    identity = {"factor": "control.identity", "baseline": None, "candidate": None, "hypothesis": change["hypothesis"]}
    _fair_pair(replace(left, config=configs[0], data=data[0]), replace(right, config=configs[1], data=data[1]), identity)
    keys = [_keys(run.path, "train_history.parquet") for run in (left, right)]
    for run, frame in zip((left, right), keys):
        start, end = (pd.Timestamp(run.config["splits"]["train"][field]) for field in ("start", "end"))
        times = run.config["data"].get("training_sample_times") or run.config["data"]["sample_times"]
        if not ((frame.date >= start) & (frame.date < end) & frame.date.dt.strftime("%H:%M").isin(times)).all():
            raise ValueError("Training keys lie outside their declared dates/times")
    joined = keys[0].merge(keys[1], on=KEYS, how="outer", validate="one_to_one", indicator=True)
    extra = joined.loc[joined._merge == "right_only"]
    if (joined._merge == "left_only").any() or extra.empty:
        raise ValueError("Expanded training keys must strictly include all baseline keys")
    if change["factor"] == "data.training_sample_times":
        allowed = extra.date.dt.strftime("%H:%M").isin(set(change["candidate"]) - set(change["baseline"]))
    else:
        allowed = (extra.date >= pd.Timestamp(change["candidate"])) & (extra.date < pd.Timestamp(change["baseline"]))
    if not allowed.all():
        raise ValueError("Additional training keys are outside the declared expansion")


def paired_intervals(paired, *, bootstrap_samples=2000, seed=20261015):
    """Average seeds within day, then resample shared moving two-day blocks.

Overall draws are stratified by period, so blocks cannot bridge research gaps.
Undefined daily observations are retained and prevent a confidence interval.
"""
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if paired.empty or paired.duplicated(["period", "seed", "date"]).any():
        raise ValueError("Paired results require unique nonempty period/seed/date keys")
    results, all_days, all_draws = [], {}, {}
    for period_index, (period, group) in enumerate(paired.groupby("period", sort=True)):
        expected = pd.MultiIndex.from_product([sorted(group.seed.unique()), sorted(group.date.unique())],
                                             names=["seed", "date"])
        if set(group.set_index(["seed", "date"]).index) != set(expected):
            raise ValueError("Incomplete paired seed/day grid")
        for metric in METRICS:
            columns = [metric + "_baseline", metric + "_candidate"]
            values = group[columns].to_numpy(dtype=float)
            row = {"period": period, "metric": metric, "difference": "candidate minus baseline",
                   "days": int(group.date.nunique()), "seeds": int(group.seed.nunique()),
                   "undefined_observations": int((~np.isfinite(values)).sum()),
                   "mean": None, "interval": None, "status": "undefined"}
            if np.isfinite(values).all():
                delta = group.assign(delta=values[:, 1] - values[:, 0])
                days = delta.groupby("date", sort=True).delta.mean().to_numpy()
                row.update(mean=float(days.mean()), status="insufficient_days")
                if len(days) >= 2:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, period_index]))
                    draws = np.array([days[_block_indices(len(days), 2, rng)].mean()
                                      for _ in range(bootstrap_samples)])
                    row.update(status="estimated", interval=np.quantile(draws, [.025, .975]).tolist(),
                               seed_mean_std=float(delta.groupby("seed").delta.mean().std())
                               if group.seed.nunique() > 1 else None)
                    all_days.setdefault(metric, []).append(days)
                    all_draws.setdefault(metric, []).append(draws * len(days))
            results.append(row)
    period_count = paired.period.nunique()
    for metric in METRICS:
        days = all_days.get(metric, [])
        row = {"period": "all", "metric": metric, "difference": "candidate minus baseline",
               "status": "undefined", "mean": None, "interval": None}
        if len(days) == period_count:
            length = sum(map(len, days))
            draws = np.sum(all_draws[metric], axis=0) / length
            row.update(status="estimated", days=length, mean=float(np.concatenate(days).mean()),
                       interval=np.quantile(draws, [.025, .975]).tolist())
        results.append(row)
    return results


def _keys(path, filename, columns=KEYS):
    frame = pd.read_parquet(path / filename)[columns]
    if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"Invalid archived keys: {path / filename}")
    return frame.sort_values(KEYS).reset_index(drop=True)


def screen_gate(comparison):
    """Pure screening evidence from a validated comparison; never authorize a run or promotion."""
    plan, checks = comparison["plan"], comparison["checks"]
    declared = [(row["period"], row["seed"]) for row in plan["pairs"]]
    explicit = "screen_periods" in plan
    expected = plan["screen_periods"] if explicit else sorted({period for period, _ in declared})
    valid_periods = (isinstance(expected, list) and len(expected) == 2
                     and all(isinstance(p, str) and p and p != "all" for p in expected)
                     and len(set(expected)) == 2)
    periods = sorted(expected) if valid_periods else []
    seeds = sorted({seed for _, seed in declared})
    grid = {(period, seed) for period in periods for seed in seeds}
    summaries = {}
    for row in comparison["summary"]:
        summaries.setdefault((row["period"], row["seed"], row["arm"]), []).append(row)
    prerequisites = {
        "comparison_checks_passed": checks.get("passed") is True,
        "paired_checks_passed": checks.get("paired_keys_labels_data_and_single_factor") is True,
        "complete_correlation_coverage": checks.get("complete_target_eligible_correlation_coverage") is True,
        "two_expected_periods": valid_periods,
        "complete_declared_seed_period_grid": bool(seeds) and len(declared) == len(set(declared))
            and set(declared) == grid,
        "complete_unique_arm_summaries": set(summaries) == {(p, s, a) for p, s in grid
            for a in ("baseline", "candidate")} and all(len(rows) == 1 for rows in summaries.values()),
    }

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

    rows = []
    for period, seed in sorted(grid):
        for metric, field in (("rank_ic", "rank_ic_mean"), ("long_short_return", "long_short_return_mean")):
            values = []
            for arm in ("baseline", "candidate"):
                found = summaries.get((period, seed, arm), [])
                value = found[0].get(field) if len(found) == 1 else None
                values.append(float(value) if finite(value) else None)
            delta = values[1] - values[0] if None not in values else None
            delta = delta if finite(delta) else None
            passed = delta is not None and delta >= 0
            rows.append({"period": period, "seed": seed, "metric": metric,
                         "baseline": values[0], "candidate": values[1], "delta": delta, "passed": passed,
                         "reason": None if passed else "undefined_or_nonfinite" if delta is None else "negative_change"})
    return {"schema_version": 1, "passed": all(prerequisites.values()) and bool(rows)
            and all(row["passed"] for row in rows),
            "rule": "Every seed in both expected periods: finite candidate-minus-baseline RankIC and long-short >= 0",
            "expected_periods": expected, "expected_periods_source": "plan.screen_periods" if explicit else "plan.pairs",
            "seeds": seeds, "prerequisites": prerequisites,
            "failed_prerequisites": [key for key, passed in prerequisites.items() if not passed], "rows": rows,
            "risk_and_resource_review_required": True, "automatic_action": False, "model_promoted": False}


def _load(path):
    run = _read_run(path)
    if run.manifest.get("status") != "completed" or run.checks.get("passed") is not True:
        raise ValueError(f"Only completed, passing runs may be compared: {path}")
    required = {"config.json", "data.json", "metrics.json", "checks.json", "daily.csv", "predictions.parquet",
                "scored_predictions.parquet", "targets.parquet", "train_history.parquet", "learning_curve.json"}
    inventory = {item["path"] for item in run.manifest.get("artifacts", [])}
    if not required.issubset(inventory) or not verify_artifacts(path)["passed"]:
        raise ValueError(f"Missing or damaged comparison artifacts: {path}")
    metrics, daily, sections = evaluate(pd.read_parquet(path / "scored_predictions.parquet"),
        pd.read_parquet(path / "targets.parquet"), run.config["evaluation"], return_sections=True,
        undefined_constant_portfolios=run.config["evaluation"].get("constant_portfolios") == "undefined")
    for key in (*SUMMARY, "sample_keys_sha256", "targets_sha256", "correlation_coverage"):
        if not numerical_equal(metrics[key], run.metrics.get(key), 1e-12, 1e-10):
            raise ValueError(f"Saved predictions do not reproduce {key}: {path}")
    for metric in METRICS:
        if not np.allclose(daily[metric], run.daily[metric], atol=1e-12, rtol=1e-10, equal_nan=True):
            raise ValueError(f"Saved predictions do not reproduce daily {metric}: {path}")
    return run, daily, sections


def compare(plan_path, destination, *, plots=False):
    plan_path, destination = Path(plan_path).resolve(), Path(destination).resolve()
    plan = read_json(plan_path)
    required = {"schema_version", "experiment_id", "change", "pairs"}
    if (not required.issubset(plan) or set(plan) - required - {"bootstrap_samples", "bootstrap_seed", "screen_periods"}
            or plan["schema_version"] != 1 or not plan["pairs"]):
        raise ValueError("Invalid explicit comparison plan")
    _validate_change(plan["change"], {pair["period"] for pair in plan["pairs"]})
    summary, daily_rows, section_rows, curve_rows, pairs, inputs = [], [], [], [], [], []
    windows, seen, paths, recipes, period_data = {}, set(), set(), {}, {}
    implementation, runtime, correlation_complete = None, None, True
    for pair in plan["pairs"]:
        if set(pair) != {"period", "seed", "baseline", "candidate"} or pair["period"] == "all":
            raise ValueError("Each pair needs period, seed, baseline and candidate; 'all' is reserved")
        pair_key = pair["period"], pair["seed"]
        if pair_key in seen:
            raise ValueError("Duplicate period/seed pair")
        seen.add(pair_key)
        loaded = {}
        for arm in ("baseline", "candidate"):
            path = (plan_path.parent / pair[arm]).resolve()
            if path in paths or destination == path or destination.is_relative_to(path):
                raise ValueError("Run paths must be distinct and output must not modify an input archive")
            paths.add(path)
            run, daily, sections = _load(path)
            source = {name: checksum for name, checksum in run.manifest.get("code", {}).get("files", {}).items()
                      if name.endswith(".py") and name.startswith(("src/jinyiquant/", "modified_tlob/"))}
            environment = run.manifest.get("environment")
            if not source or (implementation is not None and source != implementation):
                raise ValueError("Comparison requires the same core source implementation")
            if not environment or (runtime is not None and environment != runtime):
                raise ValueError("Comparison requires the same software/hardware environment")
            implementation, runtime = source, environment
            for metric in ("ic", "rank_ic"):
                coverage = run.metrics["correlation_coverage"][metric]
                correlation_complete &= (coverage["eligible_section_count"] > 0 and
                                         coverage["eligible_section_count"] == coverage["valid_section_count"])
            if run.config["seed"] != pair["seed"]:
                raise ValueError("Declared seed differs from archived run")
            window = run.period
            if pair["period"] in windows and windows[pair["period"]] != window:
                raise ValueError("A named period must use the same validation interval")
            windows[pair["period"]] = window
            fixed = recipe(run.config, across_grid=True)
            fixed["splits"].pop("train")
            if arm in recipes and recipes[arm] != fixed:
                raise ValueError("An arm must use one fixed recipe across periods/seeds")
            recipes[arm] = fixed
            identity = (run.data, run.config["splits"], _keys(path, "train_history.parquet"))
            previous = period_data.get((pair["period"], arm))
            if previous is not None and (previous[:2] != identity[:2] or not previous[2].equals(identity[2])):
                raise ValueError("Data, splits or training keys changed across seeds")
            period_data[pair["period"], arm] = identity
            metadata = {"period": pair["period"], "seed": pair["seed"], "arm": arm}
            curve = read_json(path / "learning_curve.json")
            _curve_finished(curve, run.config)
            summary.append({**metadata, "run": str(path), **{key: run.metrics[key] for key in SUMMARY},
                            "training_samples": len(identity[2]), "steps": curve[-1]["steps"],
                            "epochs": curve[-1]["epoch"],
                            **{key: value for key, value in curve[-1].items()
                               if key.startswith(("train_", "validation_"))},
                            "resources": run.manifest.get("resources", {})})
            curve_rows.extend({**row, **metadata} for row in curve)
            daily_rows.append(daily.assign(**metadata))
            section_rows.append(sections.assign(**metadata))
            inputs.append({**metadata, "run": str(path), "manifest_sha256": file_sha256(path / "manifest.json")})
            loaded[arm] = run, daily
        left, right = loaded["baseline"][0], loaded["candidate"][0]
        if plan["change"]["factor"] in EXPANSIONS:
            _fair_expansion(left, right, _period_change(plan["change"], pair["period"]))
        else:
            _fair_pair(left, right, plan["change"])
            if not _keys(left.path, "train_history.parquet").equals(_keys(right.path, "train_history.parquet")):
                raise ValueError("Paired keys/labels differ: train_history.parquet")
        for filename, columns in (("predictions.parquet", KEYS), ("targets.parquet", KEYS + ["target"])):
            if not _keys(left.path, filename, columns).equals(_keys(right.path, filename, columns)):
                raise ValueError(f"Paired keys/labels differ: {filename}")
        paired = loaded["baseline"][1][["date", *METRICS]].merge(
            loaded["candidate"][1][["date", *METRICS]], on="date", validate="one_to_one",
            suffixes=("_baseline", "_candidate"))
        pairs.append(paired.assign(period=pair["period"], seed=pair["seed"]))
    ordered = sorted(windows.values())
    if any(pd.Timestamp(a[1]) > pd.Timestamp(b[0]) for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Research validation periods must not overlap")
    paired = pd.concat(pairs, ignore_index=True)
    intervals = paired_intervals(paired, bootstrap_samples=plan.get("bootstrap_samples", 2000),
                                seed=plan.get("bootstrap_seed", 20261015))
    seeds = sorted({seed for _, seed in seen})
    complete = len(windows) >= 2 and len(seeds) >= 3 and seen == {(p, s) for p in windows for s in seeds}
    rank = [row for row in intervals if row["metric"] == "rank_ic" and row["period"] != "all"]
    positive = bool(rank) and all(row["status"] == "estimated" and row["mean"] > 0 for row in rank)
    result = {"schema_version": 1, "experiment_id": plan["experiment_id"], "created_at": utc_now(),
              "plan": plan, "inputs": inputs, "script_sha256": file_sha256(Path(__file__)),
              "checks": {"passed": True, "paired_keys_labels_data_and_single_factor": True,
                         "complete_two_period_three_seed_grid": complete,
                         "complete_target_eligible_correlation_coverage": bool(correlation_complete)},
              "summary": summary, "paired_intervals": intervals,
              "conclusion": "改善线索" if complete and positive and correlation_complete else "证据不足",
              "model_promoted": False, "final_views": 0, "confidence": .95, "block_days": 2,
              "method": "Candidate minus baseline; average seeds within day; moving day blocks within each period",
              "limits": "Local approximate metrics, few research dates, exploratory comparisons and selection bias; "
                        "no practical threshold or automatic promotion. Review all metrics, costs and independent reruns."}
    result["screen_gate"] = screen_gate(result)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "comparison.json", result)
    write_json(destination / "checks.json", result["checks"])
    paired.to_csv(destination / "paired_daily.csv", index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(destination / "daily.csv", index=False)
    pd.concat(section_rows, ignore_index=True).to_csv(destination / "by_timestamp.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(destination / "learning_curves.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in row.items() if k != "resources"}, **row["resources"]}
                  for row in summary]).to_csv(destination / "summary.csv", index=False)
    if plots:
        from jinyiquant.research_plots import plot_learning_curve
        for index, pair in enumerate(plan["pairs"]):
            for arm in ("baseline", "candidate"):
                rows = [r for r in curve_rows if r["period"] == pair["period"]
                        and r["seed"] == pair["seed"] and r["arm"] == arm]
                plot_learning_curve(rows, destination / f"curve_{index}_{arm}.png", arm)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    try:
        result = compare(args.plan, args.output, plots=args.plots)
        print(json.dumps({"output": str(args.output.resolve()), "conclusion": result["conclusion"]}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"exit_code": 2, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
