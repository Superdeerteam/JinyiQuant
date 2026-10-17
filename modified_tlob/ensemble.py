"""Fixed three-seed averaging in original return units, without fitting or selection."""

import numpy as np

SEEDS = (11, 29, 47)
KEYS = ["date", "instrument"]


def average_predictions(frames):
    if set(frames) != set(SEEDS):
        raise ValueError("Exactly seeds 11, 29, 47 are required")
    ordered = []
    for seed in SEEDS:
        frame = frames[seed]
        if (set(frame) != {*KEYS, "score"} or frame.empty or frame.isna().any().any()
                or frame.duplicated(KEYS).any() or not np.isfinite(frame.score).all()):
            raise ValueError("Require unique, finite original-scale predictions")
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        if ordered and not frame[KEYS].equals(ordered[0][KEYS]):
            raise ValueError("Prediction keys differ; dropping or positional joining is forbidden")
        ordered.append(frame)
    return ordered[0][KEYS].assign(score=np.mean([f.score.to_numpy(dtype=np.float64) for f in ordered], axis=0))


class FixedSeedEnsemble:
    def __init__(self, models):
        if set(models) != set(SEEDS) or any(models[s].seed != s for s in SEEDS):
            raise ValueError("Exactly the fixed seed models are required")
        reference = models[11]
        for model in models.values():
            if model.config != reference.config or model.preprocessing != reference.preprocessing:
                raise ValueError("All members must use identical model and preprocessing recipes")
        self.models = models

    @classmethod
    def load(cls, paths, *, device="cuda", recorder=None):
        from .adapter import TLOBAdapter

        return cls({int(seed): TLOBAdapter.load(path, device=device, recorder=recorder)
                    for seed, path in paths.items()})

    def predict(self, batch):
        # Adapter.predict restores each fitted target_scale, including missing-history fallback.
        values = [self.models[seed].predict(batch).astype(np.float64) for seed in SEEDS]
        return np.mean(values, axis=0)


def paired_block_intervals(daily, *, block_days, samples=20000, seed=20261017):
    """Moving blocks within each window; never bridge the gap or treat stocks as days."""
    required = {"period", "date", "rank_ic", "long_short_return", "turnover"}
    if (not required.issubset(daily) or daily.duplicated(["period", "date"]).any()
            or daily.empty or type(block_days) is not int or block_days < 1 or samples < 100):
        raise ValueError("Invalid paired daily block input")
    metrics = ["rank_ic", "long_short_return", "turnover"]
    rng = np.random.default_rng(seed)
    draws, rows = [], []
    for period, group in daily.groupby("period", sort=True):
        group = group.sort_values("date")
        values = group[metrics].to_numpy(dtype=float)
        n = len(values)
        if n < block_days or not np.isfinite(values).all():
            raise ValueError("Insufficient days or undefined paired daily metrics")
        starts = rng.integers(0, n - block_days + 1, size=(samples, int(np.ceil(n / block_days))))
        indices = (starts[..., None] + np.arange(block_days)).reshape(samples, -1)[:, :n]
        sampled = values[indices].mean(axis=1)
        draws.append((n, sampled))
        for i, metric in enumerate(metrics):
            rows.append({"period": period, "metric": metric, "days": n, "mean": float(values[:, i].mean()),
                         "interval": np.quantile(sampled[:, i], [.025, .975]).tolist()})
    total = sum(n for n, _ in draws)
    pooled = sum(n * values for n, values in draws) / total
    for i, metric in enumerate(metrics):
        rows.append({"period": "all", "metric": metric, "days": total,
                     "mean": float(daily[metric].mean()),
                     "interval": np.quantile(pooled[:, i], [.025, .975]).tolist()})
    return {"block_days": block_days, "samples": samples, "seed": seed, "confidence": .95, "rows": rows}


def research_decision(paired, summaries, intervals, policy, *, cost_measured):
    """A local research decision; platform acceptance remains a separate measured requirement."""
    periods = set(paired.period)
    if (len(periods) != 2 or set(r["period"] for r in summaries) != periods or len(summaries) != 2
            or paired.duplicated(["period", "date"]).any() or not paired.groupby("period").size().eq(10).all()
            or [r["block_days"] for r in intervals] != [5, 2]):
        raise ValueError("Decision requires the complete two-window ten-day grid and both fixed block lengths")
    primary = next(r for r in intervals[0]["rows"] if r["period"] == "all" and r["metric"] == "rank_ic")
    means = paired.groupby("period").rank_ic.mean()
    noise = policy["noise"]
    checks = {"minimum_mean_gain": primary["mean"] >= policy["minimum_mean_gain"],
              "both_windows_improve": len(means) == 2 and bool((means > 0).all()),
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
    interval_pass = all(all(value for key, value in row.items() if key != "block_days")
                        for row in interval_checks)
    result = "reject" if not point_pass else "accept" if interval_pass and cost_measured else "inconclusive"
    return {"schema_version": 1, "result": result, "scope": "local_research_only",
            "action": "retain_ensemble_research_candidate" if result == "accept" else "retain_seed11_reference",
            "checks": checks, "interval_checks": interval_checks, "mean_rank_ic_delta": primary["mean"],
            "minimum_mean_gain": policy["minimum_mean_gain"], "period_mean_deltas": means.to_dict(),
            "platform_verified": False, "model_promoted": False, "final_views": 0,
            "limitations": "Old eight-day calibration is uncertain; research selection is not independent confirmation. "
                          "Gross local metrics exclude fees/slippage. Full platform time/memory remains unverified."}
