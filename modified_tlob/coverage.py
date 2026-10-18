"""Fixed-compute training-date-coverage study: exposure, baseline noise and decision gates.

The only treatment is how many training dates a fixed optimizer budget sees. Every
gate below is derived from the archived baseline arm before any candidate outcome
exists; nothing here selects seeds, scores or step counts after the fact.
"""

from __future__ import annotations

import numpy as np

from .ensemble import SEEDS

NOISE_METRICS = ("rank_ic_mean", "long_short_return_mean", "turnover_mean", "min_daily_long_short_return")
DELTA_METRICS = ("rank_ic", "long_short_return", "turnover")


def declared_exposure(samples_by_arm, *, steps, batch_size):
    """Sample exposures a fixed-step, fixed-batch fit actually consumes.

    A larger pool is not more compute: the same steps visit a smaller share of it.
    """
    if type(steps) is not int or type(batch_size) is not int or steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive integers")
    declared = {str(key): value for key, value in samples_by_arm.items()}
    if sorted(declared) != ["10", "40"]:
        raise ValueError("Declare exactly the 10-day and 40-day arms")
    arms = {}
    for key in ("10", "40"):
        samples = declared[key]
        if type(samples) is not int or samples < 1:
            raise ValueError("Arm sample counts must be positive integers")
        arms[key] = {"samples": samples, "sample_exposures": steps * batch_size,
                     "epochs": steps * batch_size / samples}
    return {"steps": steps, "batch_size": batch_size, "sample_exposures": steps * batch_size, "arms": arms,
            "interpretation": "Equal optimizer compute; the 40-day pool is traversed fewer times, so this "
                              "round tests generalization at equal compute, not the value of more history."}


def baseline_seed_noise(window_metrics, *, metrics=NOISE_METRICS):
    """Noise unit per metric: sqrt(mean over windows of the cross-seed sample variance).

    ``window_metrics`` maps window name -> metric name -> seed -> that seed's window mean.
    Windows are the sampling strata, seeds the replicates; days and stocks are never
    treated as independent evidence here.
    """
    if not isinstance(window_metrics, dict) or len(window_metrics) < 2:
        raise ValueError("Calibration requires at least two baseline windows")
    result = {"method": "sqrt(mean_over_windows(sample_variance_across_seeds)); baseline arm only",
              "seeds": list(SEEDS), "metric_scope": "local_approximation", "noise": {}, "per_window": {}}
    for metric in metrics:
        variances = []
        for period, values in sorted(window_metrics.items()):
            row = values.get(metric)
            if not isinstance(row, dict) or set(row) != {str(s) for s in SEEDS}:
                raise ValueError(f"{metric}: every window requires exactly seeds {list(SEEDS)}")
            sample = np.asarray([row[str(seed)] for seed in SEEDS], dtype=float)
            if not np.isfinite(sample).all():
                raise ValueError(f"{metric}: nonfinite baseline window mean")
            variances.append(float(sample.var(ddof=1)))
            result["per_window"].setdefault(period, {})[metric] = {
                "seed_means": {str(seed): float(value) for seed, value in zip(SEEDS, sample)},
                "seed_std": float(np.sqrt(sample.var(ddof=1))),
                "spread": float(sample.max() - sample.min())}
        noise = float(np.sqrt(np.mean(variances)))
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError(f"{metric}: baseline seed noise must be positive")
        result["noise"][metric] = noise
    return result


def minimum_mean_gain(noise, *, practical_improvement, noise_multiplier):
    """Predeclared materiality floor: the larger of practical and measured noise units."""
    for value, name in ((practical_improvement, "practical_improvement"), (noise_multiplier, "noise_multiplier")):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    basis = noise["noise"]
    if not isinstance(basis, dict) or "rank_ic_mean" not in basis:
        raise ValueError("Noise calibration is missing the primary metric")
    measured = float(noise_multiplier) * float(basis["rank_ic_mean"])
    return {"primary_metric": "rank_ic_mean", "practical_improvement": float(practical_improvement),
            "noise_multiplier": float(noise_multiplier), "noise_scaled": measured,
            "minimum_mean_gain": max(float(practical_improvement), measured)}


def coverage_decision(paired, summaries, intervals, policy, *, cost_measured, exposure):
    """Predeclared accept/reject/inconclusive gates for the date-coverage comparison.

    ``paired`` holds one row per evaluation day with candidate-minus-baseline deltas,
    already averaged across the registered seeds; the seeds are not independent days.
    """
    periods = set(paired.period)
    if (len(periods) != 2 or set(r["period"] for r in summaries) != periods or len(summaries) != 2
            or paired.duplicated(["period", "date"]).any() or not paired.groupby("period").size().eq(10).all()
            or [r["block_days"] for r in intervals] != [5, 2]):
        raise ValueError("Decision requires the complete two-window ten-day grid and both fixed block lengths")
    noise = policy["noise"]
    primary = next(r for r in intervals[0]["rows"] if r["period"] == "all" and r["metric"] == "rank_ic")
    window_means = paired.groupby("period").rank_ic.mean()
    checks = {"minimum_mean_gain": primary["mean"] >= policy["minimum_mean_gain"],
              "both_windows_improve": bool((window_means > 0).all()),
              "gross_mean_nondecreasing": bool(paired.long_short_return.mean() >= 0),
              "turnover_mean_nonincreasing": bool(paired.turnover.mean() <= 0),
              "worst_day_guardrail": all(r["worst_day_delta"] >= -noise["min_daily_long_short_return"]
                                         for r in summaries),
              "inference_cost_measured": bool(cost_measured)}
    interval_checks = []
    for result in intervals:
        rows = {r["metric"]: r for r in result["rows"] if r["period"] == "all"}
        interval_checks.append({"block_days": result["block_days"],
            "positive_rank_ic_lower": rows["rank_ic"]["interval"][0] > 0,
            "gross_lower_above_negative_noise": rows["long_short_return"]["interval"][0]
                >= -noise["long_short_return_mean"],
            "turnover_upper_below_noise": rows["turnover"]["interval"][1] <= noise["turnover_mean"]})
    point_pass = all(value for key, value in checks.items() if key != "inference_cost_measured")
    interval_pass = all(all(value for key, value in row.items() if key != "block_days") for row in interval_checks)
    result = "reject" if not point_pass else "accept" if interval_pass and cost_measured else "inconclusive"
    return {"schema_version": 1, "result": result, "scope": "local_research_only",
            "only_factor": "training dates 10 -> 40 at fixed steps, batch, learning rate and recipe",
            "action": "retain_40_day_candidate_for_next_stage" if result == "accept"
                      else "retain_10_day_reference",
            "checks": checks, "interval_checks": interval_checks,
            "mean_rank_ic_delta": primary["mean"], "minimum_mean_gain": policy["minimum_mean_gain"],
            "period_mean_deltas": {str(k): float(v) for k, v in window_means.items()},
            "exposure": exposure, "cost_measured": bool(cost_measured),
            "platform_verified": False, "model_promoted": False, "final_views": 0,
            "interpretation": "A non-accept result rejects an improvement claim at this fixed exposure, not the "
                              "value of more history: the 40-day pool is only traversed for a fraction of an epoch, "
                              "and step counts are frozen for this round.",
            "limitations": "Two research windows and three seeds are selection data; local gross metrics exclude "
                           "fees and slippage; platform time and memory limits remain unverified."}
