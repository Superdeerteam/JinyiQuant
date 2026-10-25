"""Freeze the V2-CSZ seed-11 baseline volatility and the V3-open admission gate.

The V3-open round adds a 09:30 opening residual head on top of the frozen V2
backbone.  Before any head training starts, this entry point

1. reconstructs the three development-month target sets from the official
   label files exactly as the frozen replay did,
2. re-scores the archived V2-CSZ seed-11 predictions (read-only, no training),
3. measures the baseline dispersion of the 09:30 view: per-day RankIC,
   long-short return, full-day turnover, ICIR, Sharpe, contiguous 5-day block
   intervals, monthly means and volatility/volume stress buckets,
4. re-derives the frozen local style/industry residualized (``processed``) view
   on the previous-day exposure snapshot, and
5. writes the pre-registered admission gate and its evidence files:

   ``configs/v3_open_20261024_gate.json``
   ``runs/v3-open-20261024/baseline_metrics.json``
   ``runs/v3-open-20261024/baseline_recompute_check.json``

All metrics stay the local ``jinyiquant.evaluation`` approximation: gross
forward label returns, no fees, no execution, no official score.  Nothing in
this script trains a model or reads the locked final-check ranges.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import METRIC_SPEC  # noqa: E402
from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402
from modified_tlob.risk import (  # noqa: E402
    INDUSTRY_DUMMIES,
    STYLE_FACTORS,
    previous_exposure_date,
    trailing_median,
)

DATA_ROOT = Path("D:/进益资本杯/数据")
REPLAY_RUN = Path(
    "D:/进益资本杯/runs/label-contrast-eval-20261024/20261024T171905.318633Z_ad9cc8a7400e"
)
PLAN_PATH = ROOT / "configs" / "label_contrast_20261024_plan.json"
ARCHIVE_PATH = Path("D:/进益资本杯/modified_tlob/label_contrast_20261024.json")
OUT_ROOT = ROOT / "runs" / "v3-open-20261024"
GATE_PATH = ROOT / "configs" / "v3_open_20261024_gate.json"

BASELINE = "v2_60d_csz"
SEED = 11
BASELINE_WEIGHTS = Path(
    "D:/进益资本杯/runs/model-versions/v2-cloud-20261023/60d/modified_tlob_weights.json"
)
BASELINE_WEIGHTS_SHA256 = "3e44451670320c282f7421b77d773bc2af8190cbdf02a62c2a1f7e6b9284c0aa"
WINDOWS = ("nov2023", "jan2024", "mar2024")
VIEWS = ("raw_full", "raw_common", "processed")
CLOCK = "09:30"
SECONDARY_CLOCK = "10:00"
BLOCK_DAYS = 5
SECONDARY_BLOCK_DAYS = 2
SAMPLES = 20000
BOOTSTRAP_SEED = 20261024
CONFIDENCE = 0.95
EXPECTED_0930_RANK_IC = 0.02040
EXPECTED_1000_RANK_IC = 0.10849
RECOMPUTE_TOLERANCE = 1e-12
# Gate effect floor: an improvement must be at least one baseline block-5
# standard error of the 09:30 daily RankIC (no arbitrary "plus five percent").
MIN_GAIN_SIGMAS = 1.0
# Degradation tolerances are expressed in baseline volatility units.
DEGRADATION_SIGMAS = 1.0
# Worst-day tolerance: one baseline daily long-short standard deviation.
WORST_DAY_SIGMAS = 1.0


# --------------------------------------------------------------------- inputs


def load_evaluation_module():
    """Import the frozen label-contrast evaluation entry point for its conventions."""
    path = ROOT / "scripts" / "label_contrast_evaluation.py"
    spec = importlib.util.spec_from_file_location("label_contrast_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_contract() -> dict:
    """Plan contract with the worktree-safe absolute data paths."""
    plan = read_json(PLAN_PATH)
    plan["data"].update({
        "bars_path": str(DATA_ROOT / "cpt_jyc_2026_e2e_bar1m"),
        "labels_path": str(DATA_ROOT / "cpt_jyc_2026_vwap"),
        "instruments_path": str(DATA_ROOT / "cpt_jyc_2026_instruments(1).parquet"),
        "instrument_map_path": str(DATA_ROOT / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"),
        "cache_dir": None,
    })
    return plan


def evaluation_config(plan: dict) -> EvaluationConfig:
    block = plan["evaluation"]
    return EvaluationConfig(quantile=float(block["quantile"]),
                            periods_per_year=int(block["periods_per_year"]),
                            min_cross_section=int(block["min_cross_section"]),
                            constant_portfolios="undefined")


def load_exposure() -> tuple[np.ndarray, dict, pd.Series]:
    """Exposure snapshots used by every development month (previous-day convention)."""
    paths = [DATA_ROOT / "cpt_jyc_2026_exposure" / "cpt_jyc_2026_exposure_2023_part_0001.parquet",
             DATA_ROOT / "cpt_jyc_2026_exposure" / "cpt_jyc_2026_exposure_2024_part_0001.parquet"]
    columns = ["date", "instrument", *STYLE_FACTORS, *INDUSTRY_DUMMIES, "ret"]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame.duplicated(["date", "instrument"]).any():
        raise ValueError("exposure table contains duplicate date/instrument keys")
    values = frame[["ret", *STYLE_FACTORS, *INDUSTRY_DUMMIES]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("exposure table contains undefined values")
    if not frame[list(INDUSTRY_DUMMIES)].isin([0, 1]).all().all():
        raise ValueError("industry dummies must be binary")
    if not (frame[list(INDUSTRY_DUMMIES)].sum(axis=1) == 1).all():
        raise ValueError("industry dummies must sum to one per stock")
    dates = np.array(sorted(frame["date"].unique()), dtype="datetime64[ns]")
    by_date = {date: group for date, group in frame.groupby("date", sort=True)}
    volatility = frame.groupby("date")["ret"].apply(lambda x: float(np.median(np.abs(x))))
    return dates, by_date, volatility


def load_volume(months: list[str]) -> pd.Series:
    """Daily sampled volume total from the official label months (regime baseline)."""
    frames = []
    for month in months:
        path = DATA_ROOT / "cpt_jyc_2026_vwap" / f"cpt_jyc_2026_vwap_{month}.parquet"
        frame = pd.read_parquet(path, columns=["date", "volume"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    return frame.groupby("date")["volume"].sum().astype(float)


# ----------------------------------------------------------------- statistics


def bootstrap_draws(values_by_period: list[np.ndarray], *, block_days: int = BLOCK_DAYS,
                    samples: int = SAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Moving-block resamples inside each period; blocks never cross months."""
    rng = np.random.default_rng(seed)
    draws = []
    for values in values_by_period:
        n = len(values)
        if n < block_days:
            raise ValueError("period has fewer days than the block length")
        starts = rng.integers(0, n - block_days + 1, size=(samples, int(np.ceil(n / block_days))))
        indices = (starts[..., None] + np.arange(block_days)).reshape(samples, -1)[:, :n]
        draws.append(values[indices])
    return np.concatenate(draws, axis=1)


def interval(draws: np.ndarray) -> dict:
    low, high = np.quantile(draws, [0.025, 0.975]).tolist()
    return {"ci_low": float(low), "ci_high": float(high),
            "se": float((high - low) / (2 * 1.96))}


def json_float(value) -> float | None:
    """JSON-safe float: undefined statistics stay null, never zero."""
    return None if value is None or not np.isfinite(float(value)) else float(value)


def daily_series(panel: pd.DataFrame, clock: str = CLOCK) -> pd.DataFrame:
    """One row per trading day for the given clock."""
    frame = panel.copy()
    frame["clock"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%H:%M")
    part = frame.loc[frame["clock"] == clock, ["date", "rank_ic", "long_short_return", "turnover"]]
    part = part.sort_values("date").reset_index(drop=True)
    if part["date"].duplicated().any() or len(part) == 0:
        raise ValueError(f"invalid {clock} daily series")
    return part


def series_stats(values: np.ndarray, periods: list[np.ndarray], *, annualize: bool,
                 floor_quantile: float = 0.05) -> dict:
    """Mean dispersion, block intervals, ICIR/Sharpe bootstrap floors, worst days."""
    values = np.asarray(values, dtype=float)
    draws = bootstrap_draws(periods)
    mean_ci = interval(draws.mean(axis=1))
    block2 = interval(bootstrap_draws(periods, block_days=SECONDARY_BLOCK_DAYS).mean(axis=1))
    std = float(values.std(ddof=1))
    if std <= 0:
        raise ValueError("zero variance series")
    if annualize:
        ratio = draws.mean(axis=1) / draws.std(axis=1, ddof=1) * np.sqrt(252.0)
        observed_ratio = float(values.mean() / std * np.sqrt(252.0))
        kind = "long_short_sharpe"
    else:
        ratio = draws.mean(axis=1) / draws.std(axis=1, ddof=1)
        observed_ratio = float(values.mean() / std)
        kind = "rank_ic_ir"
    ordered = np.sort(values)
    worst_five = float(ordered[:5].mean()) if len(ordered) >= 5 else float(ordered.mean())
    return {
        kind: float(observed_ratio),
        f"{kind}_floor_p05": float(np.quantile(ratio, floor_quantile)),
        "mean": float(values.mean()),
        "daily_std": std,
        "block5": mean_ci,
        "block2": block2,
        f"{kind}_se": float((np.quantile(ratio, 0.975) - np.quantile(ratio, 0.025)) / (2 * 1.96)),
        "worst_five_day_mean": worst_five,
        "p05_daily": float(np.quantile(values, 0.05)),
    }


def histogram_se(period_values: list[np.ndarray]) -> float:
    """Independent-month standard error of a pooled mean (stress buckets).

    Months with a single bucket day contribute no within-month variance term;
    the independent-month sum is then a conservative (wide) approximation.
    """
    period_values = [values for values in period_values if len(values)]
    total = sum(len(values) for values in period_values)
    if total < 2:
        raise ValueError("stress bucket needs at least two days")
    variance = 0.0
    for values in period_values:
        n = len(values)
        if n < 2:
            continue
        variance += (n / total) ** 2 * float(values.var(ddof=1)) / n
    return float(np.sqrt(variance))


def regimes_for_days(days: list[pd.Timestamp], exposure_dates: np.ndarray,
                     volatility: pd.Series, volume: pd.Series, *, window: int = 20) -> pd.DataFrame:
    rows = []
    for day in days:
        previous = previous_exposure_date(exposure_dates, day)
        vol_state = float(volatility.loc[previous])
        vol_base = trailing_median(volatility, exposure_dates, before=previous, window=window)
        volume_state = float(volume.loc[previous])
        volume_base = trailing_median(volume, exposure_dates, before=previous, window=window)
        rows.append({"date": day, "snapshot_date": previous,
                     "volatility_state": vol_state, "volatility_baseline": vol_base,
                     "volume_state": volume_state, "volume_baseline": volume_base,
                     "vol_high": bool(vol_state > vol_base),
                     "volume_high": bool(volume_state > volume_base)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ recompute


def main() -> int:
    lce = load_evaluation_module()
    plan = load_contract()
    evaluation = evaluation_config(plan)
    freeze_scan = scan_runs()
    if freeze_scan["candidate_training_runs"]:
        raise RuntimeError("refusing to freeze a gate after candidate training exists: "
                           f"{freeze_scan['candidate_training_runs']}")
    exposure_dates, exposure_by_date, volatility = load_exposure()
    volume = load_volume(["2023-09", "2023-10", "2023-11", "2023-12",
                          "2024-01", "2024-02", "2024-03"])

    window_plan = [window for window in plan["windows"] if window["name"] in WINDOWS]
    if [window["name"] for window in window_plan] != list(WINDOWS):
        raise ValueError("development windows are missing from the plan")

    windows_out, recompute = [], {}
    daily_by = {view: [] for view in VIEWS}
    all_days = []
    archived_checks = {}
    for window in window_plan:
        name = window["name"]
        window_dir = REPLAY_RUN / "windows" / name
        archived_metrics = read_json(window_dir / BASELINE / "metrics.json")
        predictions = pd.read_parquet(window_dir / BASELINE / "predictions.parquet")
        if set(predictions.columns) != {"date", "instrument", "score"}:
            raise ValueError(f"{name}: predictions are not a score table")
        targets, counts = lce.window_targets(plan, window)
        metrics_full, daily_full, sections_full = lce.score_full(predictions, targets, evaluation)
        frame, diagnostics = lce.neutralize(predictions, exposure_by_date, exposure_dates,
                                            quantile=evaluation.quantile)
        if len(frame) < 0.99 * len(predictions):
            raise ValueError(f"{name}: more than 1% of keys lack a prior-day exposure")
        if diagnostics["timestamp"].nunique() != len(pd.unique(predictions["date"])):
            raise ValueError(f"{name}: residualized section grid changed")
        scored = {}
        for view, column in (("raw_common", "raw_score"), ("processed", "processed_score")):
            metrics, daily, sections, keys_sha = lce.score_common(frame, column, targets, evaluation)
            scored[view] = {"metrics": metrics, "daily": daily, "sections": sections,
                            "keys_sha": keys_sha}
        if scored["raw_common"]["keys_sha"] != scored["processed"]["keys_sha"]:
            raise ValueError(f"{name}: raw and processed views use different samples")
        scored["raw_full"] = {"metrics": metrics_full, "daily": daily_full,
                              "sections": sections_full,
                              "keys_sha": metrics_full["sample_keys_sha256"]}

        # --- reproduction against the archived run (raw view, byte-level identity).
        diffs = {metric: abs(scored["raw_full"]["metrics"][metric] - archived_metrics[metric])
                 for metric in METRIC_SPEC
                 if scored["raw_full"]["metrics"][metric] is not None
                 and archived_metrics[metric] is not None}
        identity = {
            "window": name,
            "sample_keys_sha256_match":
                scored["raw_full"]["metrics"]["sample_keys_sha256"]
                == archived_metrics["sample_keys_sha256"],
            "targets_sha256_match":
                scored["raw_full"]["metrics"]["targets_sha256"] == archived_metrics["targets_sha256"],
            "sample_count_match":
                scored["raw_full"]["metrics"]["sample_count"] == archived_metrics["sample_count"],
            "targets": counts,
            "max_metric_abs_diff": max(diffs.values()),
        }
        if max(diffs.values()) > RECOMPUTE_TOLERANCE:
            raise ValueError(f"{name}: raw metrics do not reproduce the archive "
                             f"({max(diffs.values())})")
        if not all(identity[key] for key in ("sample_keys_sha256_match", "targets_sha256_match",
                                             "sample_count_match")):
            raise ValueError(f"{name}: target/sample identity differs from the archive")
        archived_checks[name] = identity

        # --- by-time reproduction against the archived by_time.csv.
        archived_by_time = pd.read_csv(window_dir / BASELINE / "by_time.csv").rename(
            columns={"clock": "time"})
        recomputed_by_time = pd.DataFrame(lce.by_time(scored["raw_full"]["sections"]))
        merged = archived_by_time.merge(recomputed_by_time, on="time", suffixes=("_arch", "_recomp"))
        clock_diff = float(np.max(np.abs(merged["rank_ic_mean_arch"] - merged["rank_ic_mean_recomp"])))
        if clock_diff > RECOMPUTE_TOLERANCE:
            raise ValueError(f"{name}: by-time RankIC does not reproduce ({clock_diff})")
        archived_checks[name]["by_time_rank_ic_max_abs_diff"] = clock_diff

        # --- archived processed/raw_common values are the style-view cross-check.
        recorded = archived_archive_rows(name)
        for view in ("raw_common", "processed"):
            recorded_metrics = recorded[view]
            for metric in METRIC_SPEC:
                left, right = scored[view]["metrics"][metric], recorded_metrics.get(metric)
                if (left is None) != (right is None):
                    raise ValueError(f"{name}/{view}: undefined metric mismatch for {metric}")
                if left is not None and abs(left - right) > RECOMPUTE_TOLERANCE:
                    raise ValueError(f"{name}/{view}: archived {metric} differs "
                                     f"({abs(left - right)})")
        archived_checks[name]["processed_view_matches_archive"] = True
        processed_by_time = archived_clock_rank_ic(name, "processed")
        for clock in (CLOCK, SECONDARY_CLOCK):
            recomputed = float(daily_series(scored["processed"]["sections"], clock)
                               ["rank_ic"].mean())
            if abs(recomputed - processed_by_time[clock]) > RECOMPUTE_TOLERANCE:
                raise ValueError(f"{name}: processed {clock} RankIC does not reproduce the "
                                 f"archive ({abs(recomputed - processed_by_time[clock])})")
        archived_checks[name]["processed_by_time_match"] = True

        windows_out.append({
            "name": name, "label": window["label"],
            "start": str(pd.Timestamp(window["start"]))[:10],
            "end": str(pd.Timestamp(window["end"]))[:10],
            "trading_days": int(predictions["date"].dt.normalize().nunique()),
            "sections": int(len(sections_full)),
            "prediction_keys": int(len(predictions)),
            "target_keys": int(len(targets)),
            "targets": counts,
            "targets_sha256": scored["raw_full"]["metrics"]["targets_sha256"],
            "sample_keys_sha256": scored["raw_full"]["metrics"]["sample_keys_sha256"],
            "exposure_common_ratio": float(len(frame) / len(predictions)),
            "mean_style_explained": float(diagnostics["style_explained"].mean()),
            "max_style_explained": float(diagnostics["style_explained"].max()),
        })
        for view in VIEWS:
            daily = scored[view]["daily"].copy()
            daily["window"] = name
            daily_by[view].append(daily)
        sections_by_view = {view: scored[view]["sections"].assign(window=name) for view in VIEWS}
        recompute.setdefault(name, {})["sections"] = sections_by_view
        recompute[name]["daily"] = {view: scored[view]["daily"] for view in VIEWS}
        all_days.extend(sorted(pd.unique(sections_full["date"].dt.normalize())))

    # ---------------------------------------------------------- baseline tables
    clock_parts = {}
    for view in VIEWS:
        frames = []
        for name in WINDOWS:
            sections = recompute[name]["sections"][view]
            part = daily_series(sections, CLOCK)
            part["window"] = name
            frames.append(part)
            clock_parts.setdefault((view, SECONDARY_CLOCK), []).append(
                daily_series(sections, SECONDARY_CLOCK).assign(window=name))
        clock_parts[(view, CLOCK)] = frames

    ten_oclock = {}
    for view in VIEWS:
        ten_oclock[view] = float(np.mean(
            [part["rank_ic"].mean() for part in clock_parts[(view, SECONDARY_CLOCK)]]))
    nine_oclock_raw = float(np.mean([part["rank_ic"].mean()
                                     for part in clock_parts[("raw_full", CLOCK)]]))
    nine_oclock_processed = float(np.mean(
        [part["rank_ic"].mean() for part in clock_parts[("processed", CLOCK)]]))
    if abs(nine_oclock_raw - EXPECTED_0930_RANK_IC) > 5e-5:
        raise ValueError(f"09:30 RankIC {nine_oclock_raw} does not match the expected baseline")
    if abs(ten_oclock["raw_full"] - EXPECTED_1000_RANK_IC) > 5e-5:
        raise ValueError(f"10:00 RankIC {ten_oclock['raw_full']} does not match the expected baseline")

    # pooled daily view per view: all eight timestamps equally weighted per day.
    pooled_daily = {}
    for view in VIEWS:
        frame = pd.concat(daily_by[view], ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"])
        pooled_daily[view] = frame.sort_values(["window", "date"]).reset_index(drop=True)

    pooled_stats = {}
    for view in VIEWS:
        frame = pooled_daily[view]
        rank_ic_parts = [frame.loc[frame["window"] == name, "rank_ic"].to_numpy(dtype=float)
                         for name in WINDOWS]
        long_short_parts = [frame.loc[frame["window"] == name, "long_short_return"]
                            .to_numpy(dtype=float) for name in WINDOWS]
        turnover_parts = []
        for name in WINDOWS:
            values = frame.loc[frame["window"] == name, "turnover"].to_numpy(dtype=float)
            turnover_parts.append(values[np.isfinite(values)])
        turnover_draws = bootstrap_draws(turnover_parts)
        worst = frame.loc[frame["long_short_return"].idxmin()]
        pooled_stats[view] = {
            "rank_ic_full_day": series_stats(np.concatenate(rank_ic_parts), rank_ic_parts,
                                             annualize=False),
            "long_short_full_day": series_stats(np.concatenate(long_short_parts), long_short_parts,
                                                annualize=True),
            "turnover_full_day": {
                "mean": float(np.concatenate(turnover_parts).mean()),
                "block5": interval(turnover_draws.mean(axis=1)),
            },
            "worst_days_full_day": [
                {"window": row.window, "date": str(row.date)[:10],
                 "long_short_return": float(row.long_short_return),
                 "rank_ic": float(row.rank_ic)}
                for row in frame.sort_values("long_short_return").head(5).itertuples()],
            "worst_day_full_day": {"window": worst.window, "date": str(worst.date)[:10],
                                   "long_short_return": float(worst.long_short_return)},
        }

    per_clock = {}
    for view in VIEWS:
        rows = []
        for clock in plan["analysis"]["timestamps"]:
            per_window = []
            for name in WINDOWS:
                part = daily_series(recompute[name]["sections"][view], clock)
                per_window.append(part)
            averaged = float(np.mean([part["rank_ic"].mean() for part in per_window]))
            values = np.concatenate([part["rank_ic"].to_numpy(dtype=float) for part in per_window])
            rows.append({
                "time": clock,
                "sections": int(sum(len(part) for part in per_window)),
                "rank_ic_window_mean": averaged,
                "rank_ic_pooled_mean": float(values.mean()),
                "long_short_return_window_mean": float(np.mean(
                    [part["long_short_return"].mean() for part in per_window])),
                "turnover_window_mean": float(np.mean([part["turnover"].mean() for part in per_window])),
            })
        per_clock[view] = rows

    # ------------------------------------------------- 09:30 baseline dispersion
    baseline = {}
    for view in VIEWS:
        parts_frames = clock_parts[(view, CLOCK)]
        frame = pd.concat(parts_frames, ignore_index=True)
        parts = [part["rank_ic"].to_numpy(dtype=float) for part in parts_frames]
        rank_ic = series_stats(np.concatenate(parts), parts, annualize=False)
        long_short = series_stats(
            np.concatenate([frame.loc[frame["window"] == name, "long_short_return"]
                            .to_numpy(dtype=float) for name in WINDOWS]),
            [frame.loc[frame["window"] == name, "long_short_return"].to_numpy(dtype=float)
             for name in WINDOWS], annualize=True)
        turnover_parts = []
        for name in WINDOWS:
            values = frame.loc[frame["window"] == name, "turnover"].to_numpy(dtype=float)
            turnover_parts.append(values[np.isfinite(values)])
        turnover_draws = bootstrap_draws(turnover_parts)
        turnover_ci = interval(turnover_draws.mean(axis=1))
        turnover = {
            "mean": float(np.concatenate(turnover_parts).mean()),
            "daily_std": float(np.concatenate(turnover_parts).std(ddof=1)),
            "block5": turnover_ci,
        }
        monthly = {}
        for name in WINDOWS:
            part = frame.loc[frame["window"] == name]
            month_draws = bootstrap_draws([part["rank_ic"].to_numpy(dtype=float)])
            monthly[name] = {
                "days": int(len(part)),
                "rank_ic_mean": float(part["rank_ic"].mean()),
                "rank_ic_block5": interval(month_draws.mean(axis=1)),
                "long_short_return_mean": float(part["long_short_return"].mean()),
                "turnover_mean": float(part["turnover"].mean()),
            }
        worst = frame.sort_values("long_short_return").head(5)
        baseline[view] = {
            "days": int(len(frame)),
            "rank_ic": rank_ic,
            "long_short_return": long_short,
            "turnover": turnover,
            "monthly": monthly,
            "worst_days": [
                {"window": row.window, "date": str(row.date)[:10],
                 "long_short_return": float(row.long_short_return),
                 "rank_ic": float(row.rank_ic)} for row in worst.itertuples()],
            "daily_std_long_short": float(np.concatenate(
                [frame.loc[frame["window"] == name, "long_short_return"]
                 .to_numpy(dtype=float) for name in WINDOWS]).std(ddof=1)),
        }

    # stress buckets on the 09:30 view (prior-day volatility/volume regimes).
    stress = {}
    regime_frame = regimes_for_days(all_days, exposure_dates, volatility, volume)
    regime_frame["date"] = pd.to_datetime(regime_frame["date"])
    for view in VIEWS:
        joined = pd.concat(clock_parts[(view, CLOCK)], ignore_index=True).merge(
            regime_frame[["date", "vol_high", "volume_high"]], on="date", how="left",
            validate="one_to_one")
        if joined[["vol_high", "volume_high"]].isna().any().any():
            raise ValueError("regime states are missing for some days")
        buckets = {}
        conditions = {
            "high_volatility": joined["vol_high"].to_numpy(),
            "low_volatility": ~joined["vol_high"].to_numpy(),
            "high_volume": joined["volume_high"].to_numpy(),
            "low_volume": ~joined["volume_high"].to_numpy(),
            "high_volatility_high_volume": (joined["vol_high"] & joined["volume_high"]).to_numpy(),
            "high_volatility_low_volume": (joined["vol_high"] & ~joined["volume_high"]).to_numpy(),
        }
        for state, mask in conditions.items():
            part = joined.loc[mask].sort_values("date")
            month_parts_ic = [part.loc[(part["window"] == name), "rank_ic"].to_numpy(dtype=float)
                              for name in WINDOWS if (part["window"] == name).any()]
            month_parts_ls = [part.loc[(part["window"] == name), "long_short_return"]
                              .to_numpy(dtype=float)
                              for name in WINDOWS if (part["window"] == name).any()]
            buckets[state] = {
                "days": int(len(part)),
                "rank_ic_mean": float(part["rank_ic"].mean()),
                "rank_ic_se": histogram_se(month_parts_ic),
                "long_short_return_mean": float(part["long_short_return"].mean()),
                "long_short_se": histogram_se(month_parts_ls),
                "months": {name: int((part["window"] == name).sum())
                           for name in WINDOWS},
            }
        stress[view] = buckets

    # ----------------------------------------------------------- gate numbers
    primary = baseline["processed"]
    full_day = pooled_stats["processed"]
    s5_primary = primary["rank_ic"]["block5"]["se"]
    s5_raw = baseline["raw_full"]["rank_ic"]["block5"]["se"]
    s5_full_day_turnover = full_day["turnover_full_day"]["block5"]["se"]
    s5_full_day_long_short = full_day["long_short_full_day"]["block5"]["se"]
    s5_ten_oclock = interval(bootstrap_draws(
        [part["rank_ic"].to_numpy(dtype=float)
         for part in clock_parts[("raw_full", SECONDARY_CLOCK)]]).mean(axis=1))["se"]
    month_floor = {name: -float(primary["monthly"][name]["rank_ic_block5"]["se"])
                   for name in WINDOWS}
    worst_day_floor = (full_day["worst_day_full_day"]["long_short_return"]
                       - WORST_DAY_SIGMAS * full_day["long_short_full_day"]["daily_std"])
    stress_floors = {}
    for state in ("high_volatility", "low_volatility", "high_volume", "low_volume",
                  "high_volatility_high_volume", "high_volatility_low_volume"):
        bucket = stress["processed"][state]
        stress_floors[state] = {
            "rank_ic_floor": bucket["rank_ic_mean"] - DEGRADATION_SIGMAS * bucket["rank_ic_se"],
            "long_short_floor": bucket["long_short_return_mean"]
            - DEGRADATION_SIGMAS * bucket["long_short_se"],
            "days": bucket["days"],
            "months": bucket["months"],
        }

    gate = {
        "schema_version": 1,
        "gate_id": "v3-open-20261024-gate-v1",
        "experiment_id": "v3-open-20261024",
        "created_at_utc": utc_now(),
        "frozen_before_head_training": True,
        "freeze_evidence": {
            "candidate_training_runs_present_at_freeze":
                freeze_scan["candidate_training_runs"],
            "smoke_or_implementation_runs_present_at_freeze":
                freeze_scan["smoke_or_implementation_runs"],
            "no_candidate_training_at_freeze":
                not freeze_scan["candidate_training_runs"],
            "baseline_replay_started_at_utc": read_json(
                REPLAY_RUN / "manifest.json").get("started_at"),
            "baseline_replay_status": read_json(REPLAY_RUN / "manifest.json").get("status"),
            "note": "Written before the first 09:30 residual-head candidate training run; smoke "
                    "runs are implementation checks and are not candidate evidence. The gate is "
                    "never edited after a candidate result is known.",
        },
        "scope": {
            "primary_view": "processed",
            "primary_view_definition":
                "pooled over the three development months of per-day 09:30 RankIC after the "
                "frozen local style/industry residualization (previous-day exposure snapshot, "
                "median +/- 3 * 1.4826 * MAD winsorization, section z-score, equal-weight OLS on "
                "the 10 style factors + 32 industry dummies)",
            "report_view": "raw_full",
            "report_view_definition": "same per-day 09:30 RankIC on the raw official vwap labels "
                                      "without residualization",
            "development_months": list(WINDOWS),
            "development_days": int(baseline["raw_full"]["days"]),
            "locked_ranges_read": [],
            "label_horizon_minutes": int(plan["data"]["label_horizon_minutes"]),
            "timestamps": plan["analysis"]["timestamps"],
            "local_approximation": True,
            "not_used": "july2023 (context only), final/locked check ranges, any refit on "
                        "jan2024/mar2024",
        },
        "evaluation_protocol": {
            "quantile": float(plan["evaluation"]["quantile"]),
            "periods_per_year": int(plan["evaluation"]["periods_per_year"]),
            "min_cross_section": int(plan["evaluation"]["min_cross_section"]),
            "daily_aggregation": "equal timestamp mean; undefined correlations stay missing",
            "paired_daily_delta": "candidate minus V2-CSZ baseline on the same date and clock",
            "block_bootstrap": {
                "block_days": [BLOCK_DAYS, SECONDARY_BLOCK_DAYS],
                "samples": SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence": CONFIDENCE,
                "rule": "contiguous moving blocks inside each month; blocks never bridge months; "
                        "pooled draws weighted by days",
            },
            "comparison": "each check is evaluated on the same development days for candidate and "
                          "baseline; missing or incomplete candidate evidence fails the check",
            "reference_implementation": [
                "scripts/label_contrast_evaluation.py: neutralize, score_common, paired_block",
                "scripts/freeze_v3_open_gate.py: baseline dispersion, block intervals, floors",
            ],
        },
        "baseline": {
            "version": BASELINE,
            "seed": SEED,
            "weights_sha256": BASELINE_WEIGHTS_SHA256,
            "replay_run_dir": str(REPLAY_RUN),
            "replay_run_id": REPLAY_RUN.name,
            "headline": {
                "rank_ic_0930_raw_full_window_mean": nine_oclock_raw,
                "rank_ic_0930_processed_window_mean": nine_oclock_processed,
                "rank_ic_1000_raw_full_window_mean": ten_oclock["raw_full"],
                "rank_ic_0930_raw_full_pooled_daily_mean":
                    baseline["raw_full"]["rank_ic"]["mean"],
                "rank_ic_0930_processed_pooled_daily_mean":
                    primary["rank_ic"]["mean"],
            },
            "noise_units": {
                "block5_se_rank_ic_0930_processed": s5_primary,
                "block5_se_rank_ic_0930_raw_full": s5_raw,
                "block5_se_rank_ic_1000_raw_full": s5_ten_oclock,
                "block5_se_turnover_full_day_processed": s5_full_day_turnover,
                "block5_se_long_short_full_day_processed": s5_full_day_long_short,
                "daily_std_long_short_full_day_processed":
                    full_day["long_short_full_day"]["daily_std"],
                "block5_se_rank_ic_0930_processed_monthly":
                    {name: primary["monthly"][name]["rank_ic_block5"]["se"] for name in WINDOWS},
            },
            "dispersion": {
                "rank_ic_0930_processed": primary["rank_ic"],
                "long_short_0930_processed": primary["long_short_return"],
                "turnover_0930_processed": primary["turnover"],
                "rank_ic_full_day_processed": full_day["rank_ic_full_day"],
                "long_short_full_day_processed": full_day["long_short_full_day"],
                "turnover_full_day_processed": full_day["turnover_full_day"],
                "worst_days_0930_processed": primary["worst_days"],
                "worst_days_full_day_processed": full_day["worst_days_full_day"],
                "stress_0930_processed": stress["processed"],
            },
        },
        "thresholds": {
            "primary_0930_rank_ic": {
                "view": "processed",
                "dataset": "pooled development months",
                "checks": [
                    {"id": "mean_delta_at_least_minimum_gain",
                     "rule": "paired daily 09:30 RankIC delta mean > 0 and >= minimum_gain "
                             "(one baseline block-5 SE)",
                     "threshold": float(MIN_GAIN_SIGMAS * s5_primary)},
                    {"id": "block5_lower_positive",
                     "rule": "block-5 95% interval lower bound > 0",
                     "threshold": 0.0},
                    {"id": "block2_lower_positive",
                     "rule": "block-2 95% interval lower bound > 0",
                     "threshold": 0.0},
                    {"id": "months_positive",
                     "rule": "per-month mean delta > 0 in at least 2 of 3 months",
                     "threshold": 2},
                ],
                "minimum_gain": float(MIN_GAIN_SIGMAS * s5_primary),
                "derivation": "minimum_gain = 1.0 x baseline block-5 standard error of the 09:30 "
                              "processed daily RankIC; no fixed percentage is used",
                "raw_view_consistency": {
                    "rule": "raw_full 09:30 RankIC delta mean >= -1.0 x baseline block-5 SE",
                    "threshold": float(-DEGRADATION_SIGMAS * s5_raw),
                },
            },
            "monthly_non_degradation": {
                "rule": "per-month processed 09:30 RankIC delta >= -1.0 x that month's baseline "
                        "block-5 SE",
                "floors": {name: float(month_floor[name]) for name in WINDOWS},
            },
            "seed_consistency": {
                "stage_seed11": "apply primary_0930_rank_ic and every guardrail to seed 11",
                "stage_final": [
                    {"id": "same_direction",
                     "rule": "seed 11 and seed 29 pooled processed 09:30 RankIC delta means are "
                             "both > 0 (same direction)"},
                    {"id": "seed_average_block5_lower_positive",
                     "rule": "the two-seed average paired daily delta (mean of the two seeds on "
                             "each date) has a block-5 95% interval lower bound > 0"},
                    {"id": "each_seed_two_of_three_months",
                     "rule": "each seed improves in at least 2 of the 3 development months"},
                    {"id": "each_seed_guardrails",
                     "rule": "every degradation guardrail and stress bucket passes for each seed"},
                ],
                "seeds_required": [11, 29],
            },
            "degradation_guardrails": {
                "view": "processed",
                "dataset": "pooled development months; full-day aggregates average the eight "
                           "sampled timestamps per day",
                "turnover": {
                    "metric": "full-day mean turnover",
                    "rule": "candidate <= baseline + 1.0 x baseline block-5 SE",
                    "threshold": float(full_day["turnover_full_day"]["mean"]
                                       + DEGRADATION_SIGMAS * s5_full_day_turnover),
                    "baseline": float(full_day["turnover_full_day"]["mean"]),
                },
                "rank_ic_ir": {
                    "metric": "full-day RankIC IR (daily mean / sample std, not annualized)",
                    "rule": "candidate >= 5th percentile of the block-bootstrapped baseline "
                            "distribution",
                    "threshold": float(full_day["rank_ic_full_day"]["rank_ic_ir_floor_p05"]),
                    "baseline": float(full_day["rank_ic_full_day"]["rank_ic_ir"]),
                },
                "long_short_mean": {
                    "metric": "full-day mean long-short return",
                    "rule": "candidate >= baseline - 1.0 x baseline block-5 SE",
                    "threshold": float(full_day["long_short_full_day"]["mean"]
                                       - DEGRADATION_SIGMAS * s5_full_day_long_short),
                    "baseline": float(full_day["long_short_full_day"]["mean"]),
                },
                "long_short_sharpe": {
                    "metric": "full-day annualized long-short Sharpe (sqrt(252))",
                    "rule": "candidate >= 5th percentile of the block-bootstrapped baseline "
                            "distribution",
                    "threshold": float(
                        full_day["long_short_full_day"]["long_short_sharpe_floor_p05"]),
                    "baseline": float(full_day["long_short_full_day"]["long_short_sharpe"]),
                },
                "worst_day": {
                    "metric": "worst single-day full-day long-short return",
                    "rule": "candidate >= baseline worst day - 1.0 x baseline daily long-short "
                            "standard deviation",
                    "threshold": float(worst_day_floor),
                    "baseline": float(full_day["worst_day_full_day"]["long_short_return"]),
                    "baseline_worst_date": full_day["worst_day_full_day"]["date"],
                    "baseline_daily_std":
                        float(full_day["long_short_full_day"]["daily_std"]),
                },
                "rank_ic_1000_raw_full": {
                    "metric": "10:00 raw-full RankIC (main-clock protection)",
                    "rule": "candidate 10:00 RankIC delta mean >= -1.0 x baseline block-5 SE",
                    "threshold": float(-DEGRADATION_SIGMAS * s5_ten_oclock),
                    "baseline": float(ten_oclock["raw_full"]),
                },
            },
            "stress_scenarios": {
                "definition": "prior-day volatility state and sampled-volume state versus their "
                              "trailing 20-day medians (strictly before the state); 09:30 clock",
                "rule": "candidate bucket 09:30 RankIC mean >= floor; bucket 09:30 long-short mean "
                        ">= floor; floors are the baseline bucket mean minus 1.0 x the bucket "
                        "standard error",
                "buckets": {state: {
                    "days": value["days"],
                    "rank_ic_floor": float(stress_floors[state]["rank_ic_floor"]),
                    "long_short_floor": float(stress_floors[state]["long_short_floor"]),
                } for state, value in stress_floors.items()},
            },
            "packaging": {
                "upload_files": sorted([
                    "modified_tlob_weights.json", "provenance.json", "requirements.txt",
                    "submission_config.json", "modified_tlob_flat.py", "jinyiquant_flat.py",
                    "inference.ipynb"]),
                "checks": [
                    {"id": "exact_file_set", "rule": "the upload archive contains exactly the "
                                                     "seven frozen file names"},
                    {"id": "payload_hashes", "rule": "every payload file sha256 in provenance.json "
                                                     "matches the packaged file bytes"},
                    {"id": "flat_equals_source", "rule": "offline flat-package predictions equal the "
                                                         "source-package predictions bit for bit "
                                                         "(max absolute difference == 0.0)"},
                    {"id": "no_original_imports", "rule": "the flat package imports neither the "
                                                          "jinyiquant nor the modified_tlob folder"},
                    {"id": "offline_only", "rule": "packaging and replay attempts no network call"},
                    {"id": "weights_unchanged", "rule": "provenance reports weights_changed == false "
                                                        "for modified_tlob_weights.json"},
                ],
            },
            "weight_identity": {
                "checks": [
                    {"id": "weights_sha256_chain",
                     "rule": "the sha256 of the evaluated checkpoint equals the sha256 in the run "
                             "manifest, the package provenance and the packaged weights file"},
                    {"id": "baseline_weights_frozen",
                     "rule": "the V2 backbone block keeps sha256 "
                             f"{BASELINE_WEIGHTS_SHA256}"},
                    {"id": "version_label", "rule": "weights metadata carries the version tag "
                                                    "'v3-open-20261024' and the training seed (11 or 29)"},
                    {"id": "config_sha256", "rule": "the model config sha256 recorded in the run "
                                                    "manifest equals the committed config used for "
                                                    "the run"},
                    {"id": "sample_keys_identity", "rule": "evaluated predictions cover exactly the "
                                                           "replay sample keys and target keys"},
                    {"id": "no_weight_edit_after_evaluation", "rule": "the packaged weights sha256 "
                                                                      "equals the evaluated weights sha256"},
                ],
            },
        },
        "decision_rules": {
            "seed11_screen": "promote to seed 29 only if primary_0930_rank_ic passes on seed 11, "
                             "monthly_non_degradation passes, and every degradation guardrail and "
                             "stress bucket passes",
            "final_promotion": "accept V3-open only if seed_consistency.stage_final passes for both "
                               "seeds (guardrails and stress buckets are evaluated per seed), and "
                               "packaging plus weight_identity checks pass",
            "fail": "any failed check keeps the frozen V2-CSZ baseline as the submission model",
            "no_threshold_edit_after_results": True,
            "no_official_score_claim": True,
        },
        "files": {
            "gate": "configs/v3_open_20261024_gate.json",
            "baseline_metrics": "runs/v3-open-20261024/baseline_metrics.json",
            "recompute_check": "runs/v3-open-20261024/baseline_recompute_check.json",
            "source_script": "scripts/freeze_v3_open_gate.py",
            "source_script_sha256": file_sha256(ROOT / "scripts" / "freeze_v3_open_gate.py"),
        },
    }

    # ------------------------------------------------------- gate self-test
    guardrails = gate["thresholds"]["degradation_guardrails"]
    stress_buckets = gate["thresholds"]["stress_scenarios"]["buckets"]
    gate_self_test = {
        "minimum_gain_positive": gate["thresholds"]["primary_0930_rank_ic"]["minimum_gain"] > 0,
        "monthly_floors_negative": all(value < 0 for value in month_floor.values()),
        "raw_consistency_floor_negative":
            gate["thresholds"]["primary_0930_rank_ic"]["raw_view_consistency"]["threshold"] < 0,
        "turnover_threshold_above_baseline":
            guardrails["turnover"]["threshold"] > guardrails["turnover"]["baseline"],
        "rank_ic_ir_floor_below_baseline":
            guardrails["rank_ic_ir"]["threshold"] < guardrails["rank_ic_ir"]["baseline"],
        "long_short_floor_below_baseline":
            guardrails["long_short_mean"]["threshold"] < guardrails["long_short_mean"]["baseline"],
        "sharpe_floor_below_baseline":
            guardrails["long_short_sharpe"]["threshold"]
            < guardrails["long_short_sharpe"]["baseline"],
        "worst_day_floor_below_baseline":
            guardrails["worst_day"]["threshold"] < guardrails["worst_day"]["baseline"],
        "ten_oclock_delta_floor_negative":
            guardrails["rank_ic_1000_raw_full"]["threshold"] < 0,
        "stress_floors_below_bucket_means": all(
            stress_buckets[state]["rank_ic_floor"] < stress["processed"][state]["rank_ic_mean"]
            and stress_buckets[state]["long_short_floor"]
            < stress["processed"][state]["long_short_return_mean"]
            for state in stress_buckets),
    }
    if not all(gate_self_test.values()):
        raise ValueError(f"gate self-test failed: {gate_self_test}")

    baseline_metrics = {
        "schema_version": 1,
        "created_at_utc": gate["created_at_utc"],
        "experiment_id": "v3-open-20261024",
        "purpose": "V2-CSZ seed-11 baseline dispersion for the V3-open 09:30 residual-head gate",
        "source": {
            "version": BASELINE,
            "seed": SEED,
            "weights": str(BASELINE_WEIGHTS),
            "weights_sha256": file_sha256(BASELINE_WEIGHTS),
            "replay_run_dir": str(REPLAY_RUN),
            "replay_run_id": REPLAY_RUN.name,
            "plan_path": str(PLAN_PATH),
            "plan_sha256": file_sha256(PLAN_PATH),
            "repo_commit": "e27e2e7",
            "read_only_inputs": True,
        },
        "conventions": {
            "metric_scope": "local_approximation",
            "rank_ic": "Pearson correlation of average ranks inside one timestamp",
            "daily": "equal mean over the eight sampled timestamps",
            "rank_ic_ir": "daily mean / sample std (ddof=1), not annualized",
            "long_short_sharpe": "daily mean / sample std (ddof=1) * sqrt(252)",
            "turnover": "0.5 * sum(abs(net weight_t - net weight_t-1)) with the previous sampled "
                        "timestamp, including the overnight step",
            "portfolio": "equal tail weights, floor(n * quantile), minimum one",
            "style_residualization": "previous-day exposure snapshot; median +/- 3 * 1.4826 * MAD "
                                     "winsorization; section z-score (ddof=0); equal-weight OLS "
                                     "without intercept on 10 style factors + 32 industry dummies",
            "turnover_undefined": "the first sampled timestamp of each window has no previous "
                                  "timestamp, so its turnover stays null",
            "views": {
                "raw_full": "all prediction keys scored on the raw official labels",
                "raw_common": "raw scores restricted to the exposure-common sample",
                "processed": "style/industry residualized scores on the same exposure-common "
                             "sample as raw_common",
            },
        },
        "windows": windows_out,
        "headline": {
            "rank_ic_0930_window_mean": {
                "raw_full": nine_oclock_raw,
                "processed": nine_oclock_processed,
                "expected_reference_raw_full": EXPECTED_0930_RANK_IC,
            },
            "rank_ic_1000_window_mean": {
                "raw_full": ten_oclock["raw_full"],
                "processed": ten_oclock["processed"],
                "expected_reference_raw_full": EXPECTED_1000_RANK_IC,
            },
            "rank_ic_0930_pooled_daily_mean": {
                view: baseline[view]["rank_ic"]["mean"] for view in VIEWS},
        },
        "per_clock": per_clock,
        "nine_thirty_baseline": {view: baseline[view] for view in VIEWS},
        "pooled_full_day": {view: pooled_stats[view] for view in VIEWS},
        "stress_regimes": {
            "definition": "prior-day volatility (median abs ret) and sampled volume versus their "
                          "trailing 20-day medians strictly before the state",
            "buckets": {view: stress[view] for view in VIEWS},
        },
        "daily_0930": {
            view: [
                {"window": name,
                 "date": str(row.date)[:10],
                 "rank_ic": json_float(row.rank_ic),
                 "long_short_return": json_float(row.long_short_return),
                 "turnover": json_float(row.turnover)}
                for name in WINDOWS
                for row in clock_parts[(view, CLOCK)][WINDOWS.index(name)].itertuples()
            ] for view in VIEWS
        },
        "gate_noise_units": gate["baseline"]["noise_units"],
    }

    recompute_check = {
        "schema_version": 1,
        "created_at_utc": gate["created_at_utc"],
        "experiment_id": "v3-open-20261024",
        "purpose": "reproduce the archived V2-CSZ seed-11 metrics and self-test the gate before "
                   "freezing it",
        "replay_run_dir": str(REPLAY_RUN),
        "tolerance": RECOMPUTE_TOLERANCE,
        "headline": {
            "recomputed_rank_ic_0930_raw_full_window_mean": nine_oclock_raw,
            "expected_rank_ic_0930": EXPECTED_0930_RANK_IC,
            "rank_ic_0930_abs_diff": abs(nine_oclock_raw - EXPECTED_0930_RANK_IC),
            "recomputed_rank_ic_1000_raw_full_window_mean": ten_oclock["raw_full"],
            "expected_rank_ic_1000": EXPECTED_1000_RANK_IC,
            "rank_ic_1000_abs_diff": abs(ten_oclock["raw_full"] - EXPECTED_1000_RANK_IC),
        },
        "per_window": archived_checks,
        "gate_self_test": {"all_passed": True, "checks": gate_self_test},
        "all_passed": True,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUT_ROOT / "baseline_metrics.json", baseline_metrics)
    write_json(OUT_ROOT / "baseline_recompute_check.json", recompute_check)
    write_json(GATE_PATH, gate)
    print(json.dumps({
        "gate": str(GATE_PATH),
        "baseline_metrics": str(OUT_ROOT / "baseline_metrics.json"),
        "recompute_check": str(OUT_ROOT / "baseline_recompute_check.json"),
        "rank_ic_0930_raw": nine_oclock_raw,
        "rank_ic_0930_processed": nine_oclock_processed,
        "rank_ic_1000_raw": ten_oclock["raw_full"],
        "s5_processed": s5_primary,
        "s5_raw": s5_raw,
        "s5_1000": s5_ten_oclock,
        "full_day_turnover": full_day["turnover_full_day"],
        "full_day_icir": full_day["rank_ic_full_day"]["rank_ic_ir"],
        "full_day_icir_floor": full_day["rank_ic_full_day"]["rank_ic_ir_floor_p05"],
        "full_day_sharpe": full_day["long_short_full_day"]["long_short_sharpe"],
        "full_day_sharpe_floor":
            full_day["long_short_full_day"]["long_short_sharpe_floor_p05"],
        "full_day_long_short": full_day["long_short_full_day"]["mean"],
        "worst_day_floor": worst_day_floor,
        "worst_day_baseline": full_day["worst_day_full_day"],
        "stress": stress_floors,
        "monthly": primary["monthly"],
    }, indent=1, default=str))
    return 0


def read_json_tolerant(path: Path) -> dict:
    """Read an archive whose Windows paths were written with single backslashes."""
    raw = path.read_text(encoding="utf-8")
    hexdigits = set("0123456789abcdefABCDEF")
    out, index = [], 0
    while index < len(raw):
        char = raw[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        nxt = raw[index + 1] if index + 1 < len(raw) else ""
        if nxt in '"\\/bfnrt':
            out.append(char + nxt)
            index += 2
        elif nxt == "u" and all(c in hexdigits for c in raw[index + 2:index + 6]):
            out.append(raw[index:index + 6])
            index += 6
        else:
            out.append("\\\\")
            index += 1
    return json.loads("".join(out))


_ARCHIVE: dict = {}


def archive_payload() -> dict:
    if "data" not in _ARCHIVE:
        _ARCHIVE["data"] = read_json_tolerant(ARCHIVE_PATH)
    return _ARCHIVE["data"]


def archived_archive_rows(name: str) -> dict:
    """Archived processed/raw_common metric blocks from the consolidation archive."""
    rows = {}
    for block in archive_payload()["results"]:
        if block["window"] == name and block["version"] == BASELINE:
            rows[block["variant"]] = block
    if set(rows) != set(VIEWS):
        raise ValueError(f"{name}: archived variants missing")
    return rows


def archived_clock_rank_ic(name: str, variant: str) -> dict:
    """Archived per-clock RankIC means for one window and view."""
    for block in archive_payload()["eight_time_points"]:
        if (block["window"] == name and block["version"] == BASELINE
                and block["variant"] == variant):
            return {row["time"]: float(row["rank_ic_mean"]) for row in block["rows"]}
    raise ValueError(f"{name}/{variant}: archived by-time rows missing")


def scan_runs() -> dict:
    """Classify runs recorded before the gate freeze; candidates must not exist yet."""
    candidates, smoke = [], []
    for manifest_path in sorted(OUT_ROOT.glob("*/*/manifest.json")):
        manifest = read_json_tolerant(manifest_path)
        relative = str(manifest_path.parent.relative_to(ROOT)).replace("\\", "/")
        entry = {"path": relative,
                 "run_id": manifest.get("run_id"),
                 "experiment_id": manifest.get("experiment_id"),
                 "status": manifest.get("status"),
                 "steps": (manifest.get("training") or {}).get("steps")}
        implementation = ("smoke" in relative.lower()
                          or "smoke" in str(manifest.get("experiment_id", "")).lower())
        if implementation:
            smoke.append(entry)
        else:
            candidates.append(entry)
    return {"candidate_training_runs": candidates,
            "smoke_or_implementation_runs": smoke}


if __name__ == "__main__":
    raise SystemExit(main())
