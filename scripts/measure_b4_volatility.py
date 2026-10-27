"""Measure baseline volatility before pre-registering promotion thresholds (B4).

No training, no model search, no sealed-interval access. The only metric entry
point is ``jinyiquant.evaluation.evaluate`` (eval-v2). Scores are reused from the
frozen multiscale replay artifacts that already back ``reeval-a-20261026``.

Predeclared before any metric is computed (``plan.json`` is written first):

* windows: the five already-viewed development months covered by the frozen
  replay (2023-11, 2024-01, 2024-03, 2024-04, 2024-06), outside the sealed
  interval ``[2024-10-08, 2025-01-01)``;
* objects: 9 frozen members (3 trunks x seeds 11/29/47), the three 3-seed
  equal-weight ensembles and the nine-checkpoint equal-weight blend that is the
  current submission baseline;
* statistics M1-M6 (window dispersion, seed dispersion, day-block sampling
  standard error, placebo paired differences between equal-quality arms,
  difference autocorrelation, per-metric scales);
* the threshold estimator that turns those measurements into one primary
  improvement amount, secondary regression limits and the resource budget.

The measured scale is the day-block resampling of paired daily differences.
Seed counts and same-day stocks are never treated as independent date evidence:
seeds appear only as a descriptive dispersion, stocks are never resampled, and
the interval unit is one trading day inside a declared disjoint month.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from jinyiquant.comparison import _block_indices
from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import METRIC_SPEC, aggregate_metric, evaluate

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b"
VERIFY = ROOT / "runs/multiscale-replay-verify-20261026/20261026T140057.561539Z_1a57d4b38acc"
WINDOWS = (
    ("nov2023", SCREEN, "2023-11-01", "2023-12-01"),
    ("jan2024", SCREEN, "2024-01-01", "2024-02-01"),
    ("mar2024", SCREEN, "2024-03-01", "2024-04-01"),
    ("apr2024", VERIFY, "2024-04-01", "2024-05-01"),
    ("jun2024", VERIFY, "2024-06-01", "2024-07-01"),
)
TRUNKS = {
    "tlob": ("tlob_s11", "tlob_s29", "tlob_s47"),
    "mlp": ("mlp_s11", "mlp_s29", "mlp_s47"),
    "patch": ("patch_s11", "patch_s29", "patch_s47"),
}
MEMBERS = tuple(member for trunk in TRUNKS.values() for member in trunk)
ENSEMBLES = ("tlob_3seed", "mlp_3seed", "patch_3seed")
REFERENCE = "nine_blend"
OBJECT_IDS = MEMBERS + ENSEMBLES + (REFERENCE,)
# Equal-quality arms: three 3-seed ensembles plus the nine-checkpoint blend that
# contains them. These six unordered pairs estimate the difference scale that a
# nominally equivalent alternative shows on the same trading days.
PLACEBO_PAIRS = (
    ("tlob_3seed", "mlp_3seed"),
    ("tlob_3seed", "patch_3seed"),
    ("tlob_3seed", REFERENCE),
    ("mlp_3seed", "patch_3seed"),
    ("mlp_3seed", REFERENCE),
    ("patch_3seed", REFERENCE),
)
PRIMARY = "rank_ic_mean"
# metric -> rounding grid used to round the measured scale up to a stable digit
SECONDARY = {
    "ic_mean": 1e-4,
    "ic_ir": 1e-3,
    "long_short_sharpe": 1e-2,
    "long_short_return_mean": 1e-7,
    "min_daily_long_short_return": 1e-7,
    "turnover_mean": 1e-4,
}
NET5 = "net_return_mean_5bp"
NET5_GRID = 1e-7
PRIMARY_GRID = 1e-4
RELATIVE_FLOOR = 0.02  # 2% of the baseline level: minimal practical relevance
BOOTSTRAP = {"block_days": 5, "draws": 4000, "seed": 20261027, "confidence": 0.95}
DAILY_COLUMNS = ("rank_ic", "ic", "long_short_return", "turnover")
SEALED = ("2024-10-08", "2025-01-01")
EXPERIMENT_ID = "b4-volatility-20261027"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / 1e6)


def evaluation_config(rate: float) -> EvaluationConfig:
    return EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=rate)


def plan_document() -> dict:
    """The measurement plan; identical bytes every run so tampering is detectable."""
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": "B4",
        "purpose": "measure baseline volatility on already-viewed development months before "
                   "pre-registering one promotion threshold, secondary regression lines and a "
                   "resource budget",
        "no_training": True,
        "sealed_interval": list(SEALED),
        "window_rule": "The five months are exactly the frozen multiscale replay coverage that "
                       "already backs reeval-a-20261026; they are viewed research data, so they "
                       "measure volatility but cannot serve as independent confirmation. The "
                       "sealed interval is never read and the window set is fixed before any "
                       "metric is computed.",
        "windows": [{"name": name, "start": start, "end": end, "replay_run": rel(replay)}
                    for name, replay, start, end in WINDOWS],
        "objects": [
            *[{"id": member, "role": "member", "definition": f"frozen {member} checkpoint"}
              for member in MEMBERS],
            *[{"id": ensemble, "role": "ensemble",
               "definition": f"equal-weight {trunk} seeds 11/29/47", "members": list(members)}
              for (trunk, members), ensemble in zip(TRUNKS.items(), ENSEMBLES)],
            {"id": REFERENCE, "role": "baseline",
             "definition": "equal-weight blend of the nine frozen members (current submission build)",
             "members": list(MEMBERS)},
        ],
        "cost_scenarios": {"0bp": 0.0, "5bp": 0.0005},
        "bootstrap": BOOTSTRAP | {"unit": "consecutive trading-day blocks inside one month; "
                                          "blocks never cross the month gaps"},
        "primary_metric": PRIMARY,
        "secondary_metrics": list(SECONDARY) + [NET5],
        "statistics": {
            "M1_window_dispersion": "per-window metric level for every object; mean, sample sd "
                                    "(ddof=1), min and max across the five months",
            "M2_seed_dispersion": "within month and trunk, sample sd across the three seeds; reported "
                                  "as a descriptive dispersion only - seeds are not independent dates",
            "M3_day_block_sampling_se": "standard deviation of the block-resampled mean level of the "
                                        "baseline (and every object); the volatility of the measured "
                                        "level under trading-day resampling",
            "M4_placebo_paired_differences": "signed mean paired daily difference and 5-day block 95% "
                                             "interval for the six equal-quality pairs; the largest "
                                             "absolute pair difference is the spurious-difference scale",
            "M5_difference_autocorrelation": "lag 1-5 autocorrelation of the placebo daily difference "
                                             "series, used to justify consecutive-day blocks",
            "M6_secondary_metric_scales": "the same placebo maximum computed for each secondary "
                                          "metric and for net return at the 5bp stress rate",
        },
        "threshold_estimator": {
            "rule": "min_improvement = round_up(PRIMARY_GRID, max(placebo_max_abs_primary_difference, "
                    "RELATIVE_FLOOR * baseline_primary_level)); secondary_limit[m] = "
                    "round_up(grid[m], placebo_max_abs_difference[m]); the threshold is therefore "
                    "the larger of one measured spurious-difference scale and one relative floor",
            "primary_grid": PRIMARY_GRID,
            "relative_floor": RELATIVE_FLOOR,
            "secondary_grids": SECONDARY,
            "net5_grid": NET5_GRID,
            "placebo_pairs": [list(pair) for pair in PLACEBO_PAIRS],
            "note": "the estimator is fixed before the measurements are computed; the resulting "
                    "numbers are copied into the pre-registration and re-derived by the gate",
        },
        "evidence_rules": {
            "date_evidence_unit": "one trading day; resampling unit is a block of consecutive "
                                  "trading days inside one declared disjoint period",
            "seeds_are_not_date_evidence": True,
            "stocks_within_a_day_are_not_resampled": True,
            "paired": "baseline and candidate are differenced on the identical trading-day index",
            "minimum_periods": 2,
            "minimum_days": 40,
        },
        "supersedes": {
            "retired_thresholds": [
                {"name": "min_rank_ic_gain", "value": 0.0046,
                 "source": "configs/epoch_share_selection_20261025_plan.json",
                 "reason": "two times the cross-seed sd of a single-seed RankIC level; seed "
                           "dispersion is not trading-day evidence and the number was never "
                           "measured against paired daily differences"},
                {"name": "min_rank_ic_gain", "value": 0.0023,
                 "source": "configs/grouped_ranking_20261025_plan.json",
                 "reason": "one times the same cross-seed sd; same defect, kept only as history"},
            ],
            "usable_for_promotion": False,
        },
    }


def load_window(name: str, replay: Path, start: str, end: str) -> dict:
    base = replay / "windows" / name
    targets = pd.read_parquet(base / "targets.parquet")
    canonical = targets[["date", "instrument"]].reset_index(drop=True)
    if canonical.duplicated().any():
        raise ValueError(f"{name}: duplicate target keys")
    ordered = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    frames = {}
    shas = {}
    for member in MEMBERS:
        path = base / member / "predictions.parquet"
        frame = pd.read_parquet(path)
        merged = frame.merge(canonical, on=["date", "instrument"], how="inner", validate="one_to_one")
        if len(merged) != len(canonical):
            raise ValueError(f"{name}/{member}: predictions do not cover every target key")
        merged = merged.sort_values(["date", "instrument"]).reset_index(drop=True)
        if not merged[["date", "instrument"]].equals(ordered[["date", "instrument"]]):
            raise ValueError(f"{name}/{member}: key order mismatch")
        frames[member] = merged["score"].to_numpy(dtype=float)
        shas[member] = sha256_file(path)
    dates = pd.to_datetime(ordered["date"])
    if ((dates >= pd.Timestamp(SEALED[0])) & (dates < pd.Timestamp(SEALED[1]))).any():
        raise ValueError(f"{name}: window enters the sealed interval")
    if (dates < pd.Timestamp(start)).any() or (dates >= pd.Timestamp(end)).any():
        raise ValueError(f"{name}: window bounds do not match the declared period")
    return {"name": name, "start": start, "end": end, "replay_rel": rel(replay),
            "targets": ordered, "scores": frames, "prediction_sha256": shas,
            "targets_sha256": sha256_file(base / "targets.parquet")}


def object_scores(window: dict) -> dict[str, np.ndarray]:
    scores = dict(window["scores"])
    for (trunk, members), ensemble in zip(TRUNKS.items(), ENSEMBLES):
        scores[ensemble] = sum(scores[member] for member in members) / len(members)
    scores[REFERENCE] = sum(scores[member] for member in MEMBERS) / len(MEMBERS)
    return {object_id: scores[object_id] for object_id in OBJECT_IDS}


def evaluate_window(window: dict) -> dict:
    """Worker task: evaluate every object of one window at 0bp and 5bp."""
    scores = object_scores(window)
    keys = window["targets"][["date", "instrument"]]
    targets = window["targets"]
    results = {}
    for object_id, values in scores.items():
        predictions = keys.assign(score=values)
        metrics_0, daily_0, _ = evaluate(predictions, targets, evaluation_config(0.0),
                                         return_sections=True, undefined_constant_portfolios=True)
        metrics_5, daily_5, _ = evaluate(predictions, targets, evaluation_config(0.0005),
                                         return_sections=True, undefined_constant_portfolios=True)
        results[object_id] = {"metrics_0bp": metrics_0, "daily_0bp": daily_0,
                             "metrics_5bp": metrics_5, "daily_5bp": daily_5}
    return {"window": window["name"], "objects": results}


def block_resample_values(values: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator,
                          block_days: int) -> float:
    total = 0.0
    count = 0
    for positions in groups:
        sampled = positions[_block_indices(len(positions), min(block_days, len(positions)), rng)]
        total += float(values[sampled].sum())
        count += len(sampled)
    return total / count


def month_groups(dates: pd.Series) -> list[np.ndarray]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates).dt.normalize()})
    frame["month"] = frame["date"].dt.strftime("%Y-%m")
    return [np.asarray(indices) for _, indices in frame.groupby("month").indices.items()]


def block_statistics(values: np.ndarray, dates: pd.Series, *, block_days: int, draws: int,
                     seed: int, confidence: float) -> dict:
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("block statistics require finite daily values")
    groups = month_groups(dates)
    rng = np.random.default_rng(seed)
    sampled = np.array([block_resample_values(values, groups, rng, block_days) for _ in range(draws)])
    alpha = (1 - confidence) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    return {
        "mean": float(values.mean()),
        "sampling_se": float(sampled.std(ddof=1)),
        "low": float(low),
        "high": float(high),
        "confidence": confidence,
        "group_days": [int(len(group)) for group in groups],
        "draws": draws,
        "block_days": block_days,
        "seed": seed,
    }


def lag_autocorrelation(values: np.ndarray, lag: int) -> float:
    """Serial correlation of a daily difference series (time-series diagnostic, not a metric)."""
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) <= lag + 2:
        return float("nan")
    return float(series.autocorr(lag=lag))


def round_up(grid: float, value: float) -> float:
    return float(np.ceil(value / grid - 1e-12) * grid)


def aggregate(frame: pd.DataFrame, metric: str) -> float | None:
    column, statistic, _ = METRIC_SPEC[metric]
    return aggregate_metric(frame[column].to_numpy(dtype=float), statistic, 252.0)


def measure(windows: dict[str, dict], evaluated: dict[str, dict]) -> dict:
    """M1-M6 plus the derived thresholds, all from the per-window eval-v2 artifacts."""
    objects = OBJECT_IDS
    names = [name for name, *_ in WINDOWS]
    d0 = {object_id: {name: evaluated[name]["objects"][object_id]["daily_0bp"] for name in names}
          for object_id in objects}
    d5 = {object_id: {name: evaluated[name]["objects"][object_id]["daily_5bp"] for name in names}
          for object_id in objects}

    grouped = {name: {object_id: pd.concat(
        [d0[object_id][window] for window in names], ignore_index=True).sort_values("date").reset_index(drop=True)
        for object_id in objects} for name in ("daily_0bp",)}
    pooled0 = grouped["daily_0bp"]
    pooled5 = {object_id: pd.concat([d5[object_id][window] for window in names], ignore_index=True)
               .sort_values("date").reset_index(drop=True) for object_id in objects}
    for object_id in objects:
        if not pooled0[object_id]["date"].equals(pooled5[object_id]["date"]):
            raise ValueError(f"{object_id}: 0bp and 5bp daily indexes differ")

    def aggregated(object_id: str, metric: str) -> float | None:
        return aggregate(pooled0[object_id], metric)

    levels = {object_id: {metric: aggregated(object_id, metric) for metric in
                          (PRIMARY, *SECONDARY)} for object_id in objects}
    for object_id in objects:
        levels[object_id][NET5] = float(pooled5[object_id]["net_return"].mean())

    # M1: window dispersion of every object on the primary metric.
    window_dispersion = {}
    for object_id in objects:
        values = [aggregate(d0[object_id][name], PRIMARY) for name in names]
        finite = [value for value in values if value is not None and np.isfinite(value)]
        window_dispersion[object_id] = {
            "by_window": dict(zip(names, values)),
            "mean": float(np.mean(finite)), "std": float(np.std(finite, ddof=1)),
            "min": float(np.min(finite)), "max": float(np.max(finite)),
            "range": float(np.max(finite) - np.min(finite)),
        }

    # M2: seed dispersion inside month and trunk (descriptive only).
    seed_dispersion = {}
    for trunk, members in TRUNKS.items():
        per_window = {}
        for name in names:
            values = [aggregate(d0[member][name], PRIMARY) for member in members]
            per_window[name] = float(np.std(values, ddof=1))
        seed_dispersion[trunk] = {"by_window_sd": per_window,
                                  "median_sd": float(np.median(list(per_window.values()))),
                                  "max_sd": float(np.max(list(per_window.values())))}
    seed_dispersion["note"] = ("descriptive dispersion only; three seeds are not three dates and "
                               "seed count never widens or narrows a date-block interval")

    # M3: day-block sampling standard error of the measured level.
    sampling = {}
    for object_id in objects:
        frame = pooled0[object_id]
        pooled = block_statistics(frame["rank_ic"].to_numpy(dtype=float), frame["date"],
                                  block_days=BOOTSTRAP["block_days"], draws=BOOTSTRAP["draws"],
                                  seed=BOOTSTRAP["seed"], confidence=BOOTSTRAP["confidence"])
        by_window = {}
        for name in names:
            window_frame = d0[object_id][name]
            by_window[name] = block_statistics(window_frame["rank_ic"].to_numpy(dtype=float),
                                               window_frame["date"], block_days=BOOTSTRAP["block_days"],
                                               draws=BOOTSTRAP["draws"], seed=BOOTSTRAP["seed"],
                                               confidence=BOOTSTRAP["confidence"])["sampling_se"]
        sampling[object_id] = {"pooled": pooled, "by_window_sampling_se": by_window}

    # M4/M6: placebo paired differences between equal-quality arms.
    #
    # A ratio/quantile metric (IC_IR, Sharpe, worst day) has no additive daily
    # decomposition, so its difference is taken on the pooled metric level. Only
    # mean-type metrics also get a moving-block interval of the daily difference;
    # those intervals describe sampling noise and are never used to widen or
    # narrow a ratio-metric difference.
    placebo = {}
    placebo_scale = {metric: 0.0 for metric in (PRIMARY, *SECONDARY, NET5)}
    for left, right in PLACEBO_PAIRS:
        name = f"{left}__vs__{right}"
        frame = pooled0[left]
        if not frame["date"].equals(pooled0[right]["date"]):
            raise ValueError("placebo pair daily indexes differ")
        record = {"left": left, "right": right, "paired_days": int(len(frame)), "metrics": {}}
        for metric in (PRIMARY, *SECONDARY):
            column, statistic, _ = METRIC_SPEC[metric]
            difference = float(levels[right][metric]) - float(levels[left][metric])
            entry = {"metric_difference": difference, "abs_metric_difference": abs(difference)}
            if statistic == "mean" and column in frame:
                block = block_statistics((pooled0[right][column].to_numpy(dtype=float)
                                          - pooled0[left][column].to_numpy(dtype=float)),
                                         frame["date"], block_days=BOOTSTRAP["block_days"],
                                         draws=BOOTSTRAP["draws"], seed=BOOTSTRAP["seed"],
                                         confidence=BOOTSTRAP["confidence"])
                entry["daily_difference_block_interval"] = block
                if abs(block["mean"] - difference) > 1e-9:
                    raise ValueError(f"{metric}: daily-mean difference disagrees with metric difference")
            record["metrics"][metric] = entry
            placebo_scale[metric] = max(placebo_scale[metric], abs(difference))
        net_difference = float(levels[right][NET5]) - float(levels[left][NET5])
        net_block = block_statistics((pooled5[right]["net_return"].to_numpy(dtype=float)
                                      - pooled5[left]["net_return"].to_numpy(dtype=float)),
                                     frame["date"], block_days=BOOTSTRAP["block_days"],
                                     draws=BOOTSTRAP["draws"], seed=BOOTSTRAP["seed"],
                                     confidence=BOOTSTRAP["confidence"])
        record["metrics"][NET5] = {"metric_difference": net_difference,
                                   "abs_metric_difference": abs(net_difference),
                                   "daily_difference_block_interval": net_block}
        placebo_scale[NET5] = max(placebo_scale[NET5], abs(net_difference))
        placebo[name] = record

    # M5: autocorrelation of the placebo daily differences (primary metric).
    autocorrelation = {}
    for name, record in placebo.items():
        left, right = record["left"], record["right"]
        differences = (pooled0[right]["rank_ic"].to_numpy(dtype=float)
                       - pooled0[left]["rank_ic"].to_numpy(dtype=float))
        autocorrelation[name] = {str(lag): lag_autocorrelation(differences, lag) for lag in range(1, 6)}
    mean_autocorrelation = {str(lag): float(np.nanmean([value[str(lag)] for value in autocorrelation.values()]))
                            for lag in range(1, 6)}

    # Derived thresholds, exactly as the pre-declared estimator states.
    reference_level = levels[REFERENCE][PRIMARY]
    relative_floor_value = RELATIVE_FLOOR * reference_level
    raw_primary = max(placebo_scale[PRIMARY], relative_floor_value)
    derived = {
        "primary_metric": PRIMARY,
        "baseline_primary_level": float(reference_level),
        "relative_floor": RELATIVE_FLOOR,
        "relative_floor_value": float(relative_floor_value),
        "placebo_max_abs_primary_difference": float(placebo_scale[PRIMARY]),
        "min_improvement_raw": float(raw_primary),
        "min_improvement": round_up(PRIMARY_GRID, raw_primary),
        "secondary_limits": {metric: round_up(grid, placebo_scale[metric])
                             for metric, grid in SECONDARY.items()},
        "secondary_placebo_scales": {metric: float(placebo_scale[metric]) for metric in SECONDARY},
        "net5_limit": round_up(NET5_GRID, placebo_scale[NET5]),
        "net5_placebo_scale": float(placebo_scale[NET5]),
        "placebo_scale_definition": "max over the six equal-quality pairs of |mean paired daily difference|",
        "resource_budget": {
            "platform_ram_limit_mib": 24576,
            "headroom_fraction": 0.15,
            "promotion_peak_rss_mib": 20889,
            "note": "24 GiB platform caliber on host RSS across all subprocesses; 15% headroom "
                    "leaves 3687 MiB for platform variance; GPU memory and wall time stay "
                    "report-only until the platform declares their limits",
        },
    }
    return {
        "estimator": {
            "rule": "min_improvement = round_up(primary_grid, max(placebo_max_abs_primary_difference, "
                    "relative_floor * baseline_primary_level)); secondary_limit[m] = "
                    "round_up(grid[m], placebo_max_abs_difference[m])",
            "primary_grid": PRIMARY_GRID,
            "relative_floor": RELATIVE_FLOOR,
            "secondary_grids": dict(SECONDARY),
            "net5_grid": NET5_GRID,
            "bootstrap": BOOTSTRAP,
            "placebo_pairs": [list(pair) for pair in PLACEBO_PAIRS],
        },
        "levels": levels,
        "window_dispersion": window_dispersion,
        "seed_dispersion": seed_dispersion,
        "sampling_standard_error": sampling,
        "placebo_pairs": placebo,
        "placebo_scale": {metric: float(value) for metric, value in placebo_scale.items()},
        "difference_autocorrelation": {"by_pair": autocorrelation, "mean_by_lag": mean_autocorrelation},
        "derived_thresholds": derived,
        "days": int(len(pooled0[REFERENCE])),
        "periods": names,
    }


def write_json(document, path: Path) -> str:
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.write_bytes(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/b4-volatility-20261027")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = utc_now()
    timer = time.time()
    rss_samples = [peak_rss_mb()]

    plan = plan_document()
    plan_sha = digest(plan)
    output = args.output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(plan, output / "plan.json")
    print(f"predeclared measurement plan {rel(output / 'plan.json')} sha256={plan_sha}")

    windows = {name: load_window(name, replay, start, end) for name, replay, start, end in WINDOWS}
    rss_samples.append(peak_rss_mb())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        evaluated = {result["window"]: result for result in executor.map(evaluate_window,
                                                                        [windows[name] for name in windows])}
    rss_samples.append(peak_rss_mb())

    for name in windows:
        for object_id in OBJECT_IDS:
            entry = evaluated[name]["objects"][object_id]
            object_dir = output / "objects" / object_id
            object_dir.mkdir(parents=True, exist_ok=True)
            write_csv(entry["daily_0bp"], object_dir / f"daily_0bp_{name}.csv")
            write_csv(entry["daily_5bp"], object_dir / f"daily_5bp_{name}.csv")
            write_json({"schema_version": 1, "object_id": object_id, "window": name,
                        "evaluation_version": entry["metrics_0bp"]["evaluation_version"],
                        "metrics_0bp": entry["metrics_0bp"], "metrics_5bp": entry["metrics_5bp"]},
                       object_dir / f"metrics_{name}.json")

    measurements = measure(windows, evaluated)
    volatility = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "phase": "B4",
        "status": "completed",
        "created_at": utc_now(),
        "evaluation_version": evaluated["nov2023"]["objects"][REFERENCE]["metrics_0bp"]["evaluation_version"],
        "plan_sha256": plan_sha,
        "windows": [{"name": name, "start": windows[name]["start"], "end": windows[name]["end"],
                     "replay_run": windows[name]["replay_rel"],
                     "targets_sha256": windows[name]["targets_sha256"]} for name in windows],
        "prediction_sha256": {name: windows[name]["prediction_sha256"] for name in windows},
        "baseline_object": REFERENCE,
        "measurements": measurements,
        "limitations": [
            "already-viewed development months measure volatility; they cannot confirm a candidate",
            "local approximate metrics; no official combined score and no stress stability",
            "seed dispersion is descriptive only and is never used as date evidence",
            "windows are evaluated separately, so each month starts a new rebalance; turnover and "
            "cost at the five month boundaries are undefined and counted, which makes the pooled "
            "turnover level differ trivially from the single pooled reeval-a-20261026 archive",
        ],
    }
    volatility_sha = write_json(volatility, output / "volatility.json")
    checks = {
        "schema_version": 1,
        "passed": True,
        "items": {
            "sealed_interval_not_read": True,
            "contract_version": volatility["evaluation_version"],
            "windows": list(windows),
            "days": measurements["days"],
            "periods": len(measurements["periods"]),
            "objects": len(OBJECT_IDS),
            "placebo_pairs": len(PLACEBO_PAIRS),
            "daily_index_identical_across_objects": True,
            "bootstrap_block_days": BOOTSTRAP["block_days"],
        },
    }
    write_json(checks, output / "checks.json")
    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "kind": "measurement",
        "status": "completed",
        "hypothesis": plan["purpose"],
        "started_at": started,
        "ended_at": utc_now(),
        "evaluation_version": volatility["evaluation_version"],
        "plan_sha256": plan_sha,
        "volatility_sha256": volatility_sha,
        "artifacts": [{"path": path.relative_to(output).as_posix(),
                       "sha256": sha256_file(path), "bytes": path.stat().st_size}
                      for path in sorted(output.rglob("*")) if path.is_file()],
        "resources": {"peak_rss_mb": max(rss_samples), "wall_seconds": round(time.time() - timer, 3),
                      "workers": args.workers},
        "sealed_interval": list(SEALED),
        "code": {"commit": git_commit(), "script_sha256": sha256_file(Path(__file__).resolve()),
                 "evaluation_source_sha256": sha256_file(
                     ROOT / "src/jinyiquant/evaluation.py"),
                 "working_tree_dirty": bool(git_status())},
    }
    write_json(manifest, output / "manifest.json")
    derived = measurements["derived_thresholds"]
    print(f"wrote {rel(output)}")
    print(f"min_improvement={derived['min_improvement']} "
          f"(placebo={derived['placebo_max_abs_primary_difference']:.6f}, "
          f"relative={derived['relative_floor_value']:.6f})")
    print(f"secondary_limits={derived['secondary_limits']} net5={derived['net5_limit']}")
    return 0


def git_status() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
