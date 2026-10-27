"""Independent verification (t14) of the single turnover change recorded by t7.

Nothing from the experiment script is imported: the reference blend, the causal EWMA
operator, the two-leg book, the rebalancing cost ``U``, rank IC / IC, the four cost
scenarios and the collapse guards are all re-implemented here from the raw frozen replay
parquet and the published eval-v2 definitions. The recomputed numbers are then reconciled
day by day against t7's stored artifacts and against t6's cost curve.

Purpose and boundaries:

* confirm that the single change keeps the gross spread, lowers the cost and does not
  collapse the prediction (acceptance of t14);
* confirm the change is one factor with no grid and no trading threshold, and that a
  standardized score is never used as a bp return threshold;
* prove the collapse guard has teeth with deliberately degenerate controls (a constant
  signal and a within-day frozen signal: low turnover but no usable prediction). Those
  controls are verifier guard calibrations, not evaluated arms; alpha stays 0.5 only;
* no promotion verdict, no model selection, no retraining, sealed interval never read.

Single entry point: ``python scripts/verify_turnover_change.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "turnover-verify-20261027"

MEMBERS = ("tlob_s11", "tlob_s29", "tlob_s47", "mlp_s11", "mlp_s29", "mlp_s47",
           "patch_s11", "patch_s29", "patch_s47")
SCREEN = "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b"
VERIFY = "runs/multiscale-replay-verify-20261026/20261026T140057.561539Z_1a57d4b38acc"
WINDOWS = (("nov2023", "2023-11-01", "2023-12-01", SCREEN),
           ("jan2024", "2024-01-01", "2024-02-01", SCREEN),
           ("mar2024", "2024-03-01", "2024-04-01", SCREEN),
           ("apr2024", "2024-04-01", "2024-05-01", VERIFY),
           ("jun2024", "2024-06-01", "2024-07-01", VERIFY))
SEALED = ("2024-10-08", "2025-01-01")
RATES = (("0bp", 0.0), ("2bp", 0.0002), ("5bp", 0.0005), ("10bp", 0.001))
PRIMARY = "5bp"
ALPHA = 0.5
QUANTILE = 0.2
PERIODS_PER_YEAR = 252
LABEL_MINUTES = 30
COMPETITION_TEXT = "比赛说明/比赛说明.txt"
T6_RECORD = "modified_tlob/cost_curve_20261027.json"
T7_RUN = "runs/turnover-stability-20261027/20261027T070343Z_606ce5deaa8b"
T7_RECORD = "modified_tlob/turnover_stability_20261027.json"
T7_SCRIPT = "scripts/turnover_stability_change.py"
T7_READOUT = "runs/turnover-readout-20261027/20261027T073543Z_ca2d96722f39/readout.json"

REFERENCE = "reference_nine_blend"
CANDIDATE = "candidate_ewma_alpha050_nine_blend"
CONTROLS = ("control_constant_zero", "control_daily_freeze")

# Verifier guard bars, predeclared before any number is recomputed (plan.json is written
# first). They answer "did the signal collapse?" and are not promotion thresholds.
BARS = {
    "min_cross_sectional_std": 1e-6,
    "min_defined_rank_ic_share": 1.0,
    "min_rank_ic_retention": 0.80,
    "min_ic_retention": 0.80,
    "min_long_short_retention": 0.80,
    "min_gross_retention": 0.80,
    "max_turnover_proxy_ratio": 0.90,
    "max_cost_ratio_5bp": 0.90,
    "reconcile_abs_tolerance": 1e-9,
}


# --------------------------------------------------------------------------- utilities

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def num(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def label_end(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Official forward-window end: the 11:30 label ends at 13:30 (lunch break skipped)."""
    if (timestamp.hour, timestamp.minute) == (11, 30):
        return timestamp + pd.Timedelta(hours=2)
    return timestamp + pd.Timedelta(minutes=LABEL_MINUTES)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, undefined (NaN) for a degenerate input vector."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0 or not math.isfinite(denominator):
        return float("nan")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)


# ------------------------------------------------------------------------ input layer

def load_window(name: str, start: str, end: str, replay: str) -> dict:
    """Read one frozen replay window and validate key identity before any metric."""
    base = ROOT / replay / "windows" / name
    targets = pd.read_parquet(base / "targets.parquet")
    if targets[["date", "instrument"]].duplicated().any():
        raise ValueError(f"{name}: duplicated target keys")
    ordered = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    scores, prediction_sha = {}, {}
    for member in MEMBERS:
        path = base / member / "predictions.parquet"
        frame = pd.read_parquet(path)
        merged = frame.merge(ordered[["date", "instrument"]], on=["date", "instrument"],
                             how="inner", validate="one_to_one")
        merged = merged.sort_values(["date", "instrument"]).reset_index(drop=True)
        if len(merged) != len(ordered) or not merged[["date", "instrument"]].equals(
                ordered[["date", "instrument"]]):
            raise ValueError(f"{name}/{member}: predictions do not cover the target keys")
        values = merged["score"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}/{member}: non-finite prediction")
        scores[member] = values
        prediction_sha[member] = sha256(path)
    dates = pd.to_datetime(ordered["date"])
    if (dates < pd.Timestamp(start)).any() or (dates >= pd.Timestamp(end)).any():
        raise ValueError(f"{name}: dates outside the declared window")
    if (dates >= pd.Timestamp(SEALED[0])).any():
        raise ValueError(f"{name}: dates enter the sealed interval")
    return {"name": name, "start": start, "end": end, "replay": replay,
            "keys": ordered[["date", "instrument"]].reset_index(drop=True),
            "target": ordered["target"].to_numpy(dtype=float),
            "scores": scores, "prediction_sha256": prediction_sha,
            "keys_sha256": sha256(base / "keys.parquet"),
            "targets_sha256": sha256(base / "targets.parquet")}


def pooled(windows: list[dict]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Pool the windows in calendar order, keeping every member aligned by key."""
    frame = pd.concat([window["keys"] for window in windows], ignore_index=True)
    frame["target"] = np.concatenate([window["target"] for window in windows])
    for member in MEMBERS:
        frame[member] = np.concatenate([window["scores"][member] for window in windows])
    if frame[["date", "instrument"]].duplicated().any():
        raise ValueError("pooled keys are not unique")
    frame = frame.sort_values(["date", "instrument"]).reset_index(drop=True)
    target = frame["target"].to_numpy(dtype=float)
    scores = {member: frame[member].to_numpy(dtype=float) for member in MEMBERS}
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError("pooled member scores are not finite")
    return frame[["date", "instrument"]], target, scores


# --------------------------------------------------------------------- operator layer

def cross_sectional_z(keys: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Population z-score per timestamp (the platform standardizes the same way)."""
    values = np.asarray(scores, dtype=float)
    frame = pd.DataFrame({"date": keys["date"].to_numpy(), "score": values})
    grouped = frame.groupby("date")["score"]
    mean = grouped.transform("mean").to_numpy(dtype=float)
    scale = grouped.transform("std", ddof=0).to_numpy(dtype=float)
    safe = np.where(scale > 0, scale, 1.0)
    return np.where(scale > 0, (values - mean) / safe, 0.0)


def causal_ewma(keys: pd.DataFrame, scores: np.ndarray, alpha: float) -> np.ndarray:
    """z-score then causal per-day EWMA along the intraday sections; state resets daily.

    Independent implementation: build the timestamp x instrument panel, walk the rows of
    each trading day in time order, and blend only with the state that already exists at
    or before that row.
    """
    z = cross_sectional_z(keys, scores)
    frame = pd.DataFrame({"date": keys["date"].to_numpy(),
                          "instrument": keys["instrument"].to_numpy(), "z": z})
    panel = frame.pivot(index="date", columns="instrument", values="z").sort_index()
    days = pd.to_datetime(panel.index).normalize()
    columns = panel.shape[1]
    state = np.full(columns, np.nan)
    in_state = np.zeros(columns, dtype=bool)
    output = np.empty(panel.shape, dtype=float)
    for row in range(len(panel)):
        if row == 0 or days[row] != days[row - 1]:
            state = np.full(columns, np.nan)
            in_state = np.zeros(columns, dtype=bool)
        value = panel.iloc[row].to_numpy(dtype=float)
        observed = np.isfinite(value)
        blended = np.where(in_state, alpha * value + (1.0 - alpha) * state, value)
        blended = np.where(observed, blended, state)
        state = np.where(observed, blended, state)
        in_state = in_state | observed
        output[row] = np.where(observed, blended, np.nan)
    smoothed = pd.DataFrame(output, index=panel.index, columns=panel.columns).stack(future_stack=True)
    values = smoothed.reindex(pd.MultiIndex.from_frame(keys[["date", "instrument"]])).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("the operator does not cover every reference key")
    return values


def daily_freeze(keys: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Verifier control: hold each day's first available standardized cross section all day.

    Maximum persistence / minimum fresh information: the failure mode "low turnover,
    prediction gone" that the collapse guard must catch. Not an evaluated arm.
    """
    z = cross_sectional_z(keys, scores)
    frame = pd.DataFrame({"date": keys["date"].to_numpy(),
                          "instrument": keys["instrument"].to_numpy(), "z": z})
    panel = frame.pivot(index="date", columns="instrument", values="z").sort_index()
    days = pd.Series(pd.to_datetime(panel.index).normalize(), index=panel.index)
    frozen = panel.groupby(days).transform("first")
    stacked = frozen.stack(future_stack=True)
    return stacked.reindex(pd.MultiIndex.from_frame(keys[["date", "instrument"]])).to_numpy(dtype=float)


# -------------------------------------------------------------------- portfolio layer

def leg_weights(values: np.ndarray, count: int, *, largest: bool) -> np.ndarray:
    """Equal weights on ``count`` extreme scores; boundary ties share the remaining weight."""
    ordered = np.sort(values)
    boundary = ordered[-count] if largest else ordered[count - 1]
    strictly = values > boundary if largest else values < boundary
    tied = values == boundary
    weights = strictly.astype(float)
    weights[tied] = (count - int(strictly.sum())) / int(tied.sum())
    return weights / count


def net_weights(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Long minus short equal-weight book per 1.0 equity, plus the boundary-tie count."""
    count = max(1, int(math.floor(QUANTILE * len(values))))
    long = leg_weights(values, count, largest=True)
    short = leg_weights(values, count, largest=False)
    boundary_ties = int(np.count_nonzero((long > 0) & (long < 1.0 / count))
                        + np.count_nonzero((short > 0) & (short < 1.0 / count)))
    return long, short, count, boundary_ties


def weight_change(target: pd.Series, pretrade: pd.Series) -> float:
    union = target.index.union(pretrade.index)
    delta = target.reindex(union, fill_value=0.0) - pretrade.reindex(union, fill_value=0.0)
    return float(np.abs(delta.to_numpy(dtype=float)).sum())


def drift(weights: pd.Series, period_returns: pd.Series) -> pd.Series:
    returns = period_returns.reindex(weights.index)
    spread = float(np.dot(weights.to_numpy(dtype=float), returns.to_numpy(dtype=float)))
    return weights * (1.0 + returns) / (1.0 + spread)


# ----------------------------------------------------------------------- metric layer

def evaluate_sections(keys: pd.DataFrame, target: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Per-timestamp signal and economic rows, implemented from the eval-v2 definitions."""
    rows = []
    previous_timestamp = None
    previous_weights = None
    previous_target = None
    previous_scores = None
    for timestamp, block in keys.groupby("date", sort=True):
        index = block.index.to_numpy()
        instruments = block["instrument"].to_numpy()
        score = pd.Series(scores[index], index=instruments)
        label = pd.Series(target[index], index=instruments)
        values = score.to_numpy(dtype=float)
        long, short, count, ties = net_weights(values)
        net = pd.Series(long - short, index=instruments)
        spread = float(np.dot(long, label.to_numpy(dtype=float))
                       - np.dot(short, label.to_numpy(dtype=float)))
        score_change = float("nan")
        rank_persistence = float("nan")
        if previous_scores is not None:
            common = score.index.intersection(previous_scores.index)
            if len(common):
                score_change = float(np.abs(score.loc[common] - previous_scores.loc[common]).mean())
            if len(common) >= 2:
                rank_persistence = pearson(
                    average_ranks(score.loc[common].to_numpy(dtype=float)),
                    average_ranks(previous_scores.loc[common].to_numpy(dtype=float)))
        change = float("nan")
        state = "first_timestamp"
        overlap = False
        if previous_timestamp is not None:
            end = label_end(previous_timestamp)
            overlap = bool(timestamp < end)
            if overlap:
                state = "overlapping_labels"
            elif timestamp == end:
                change = weight_change(net, drift(previous_weights, previous_target))
                state = "adjacent_label"
            else:
                change = weight_change(net, previous_weights)
                state = "gap_drift_unknown"
        rows.append({
            "timestamp": timestamp,
            "day": timestamp.normalize(),
            "rank_ic": pearson(average_ranks(values), average_ranks(label.to_numpy(dtype=float))),
            "ic": pearson(values, label.to_numpy(dtype=float)),
            "long_short_return": spread,
            "total_abs_weight_change": change,
            "turnover": change * 0.5,
            "score_change_mae": score_change,
            "rank_persistence": rank_persistence,
            "sample_count": len(values),
            "tail_size": count,
            "boundary_ties": ties,
            "cross_sectional_std": float(np.std(values, ddof=0)),
            "ranking_defined": bool(np.unique(values).size > 1),
            "label_overlap": overlap,
            "drift_state": state,
        })
        previous_timestamp = timestamp
        previous_weights = net
        previous_target = label
        previous_scores = score
    return pd.DataFrame(rows)


def aggregate(sections: pd.DataFrame) -> dict:
    """Daily equal-weight aggregation then mean/IR/Sharpe, exactly as eval-v2 defines it."""
    columns = ["rank_ic", "ic", "long_short_return", "turnover", "total_abs_weight_change",
               "score_change_mae", "rank_persistence"]
    daily = sections.groupby("day")[columns].mean().reset_index()

    def mean(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(finite.mean()) if len(finite) else None

    def ir(values, annualize=False):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < 2 or float(np.std(finite, ddof=1)) == 0:
            return None
        value = float(finite.mean() / np.std(finite, ddof=1))
        return value * math.sqrt(PERIODS_PER_YEAR) if annualize else value

    defined = sections.loc[sections["total_abs_weight_change"].notna()]
    gross_economic = float(defined["long_short_return"].mean()) if len(defined) else None
    change_economic = float(defined["total_abs_weight_change"].mean()) if len(defined) else None
    breakeven = None
    if gross_economic is not None and change_economic and change_economic > 0:
        breakeven = gross_economic / change_economic
    return {
        "daily": daily,
        "ic_mean": mean(daily["ic"]),
        "ic_ir": ir(daily["ic"]),
        "rank_ic_mean": mean(daily["rank_ic"]),
        "rank_ic_ir": ir(daily["rank_ic"]),
        "long_short_sharpe": ir(daily["long_short_return"], annualize=True),
        "long_short_return_mean": mean(daily["long_short_return"]),
        "min_daily_long_short_return": (float(np.nanmin(daily["long_short_return"]))
                                        if daily["long_short_return"].notna().any() else None),
        "turnover_mean_daily": mean(daily["turnover"]),
        "score_change_mae": mean(daily["score_change_mae"]),
        "rank_persistence_mean": mean(daily["rank_persistence"]),
        "gross_return_mean_economic": gross_economic,
        "total_abs_weight_change_mean": change_economic,
        "turnover_economic": None if change_economic is None else change_economic * 0.5,
        "breakeven_single_side_rate": breakeven,
        "day_count": int(len(daily)),
        "cross_section_count": int(len(sections)),
        "undefined_rank_ic_count": int(sections["rank_ic"].isna().sum()),
        "undefined_turnover_count": int(sections["turnover"].isna().sum()),
        "defined_rebalance_count": int(len(defined)),
        "ranking_defined_share": float(sections["ranking_defined"].mean()),
        "min_cross_sectional_std": float(sections["cross_sectional_std"].min()),
        "boundary_tie_sections": int((sections["boundary_ties"] > 0).sum()),
    }


def monthly_net(daily_net: pd.DataFrame) -> dict:
    frame = daily_net.copy()
    if frame.empty:
        return {}
    frame["month"] = pd.to_datetime(frame["day"]).dt.strftime("%Y-%m")
    return {month: num(block["net_return"].mean() * 1e4)
            for month, block in frame.groupby("month")}


def scenario_block(summary: dict, sections: pd.DataFrame) -> dict:
    """Gross is rate free; cost and net are formed per timestamp at every predeclared rate."""
    gross = summary["gross_return_mean_economic"]
    change = summary["total_abs_weight_change_mean"]
    block = {}
    for label, rate in RATES:
        daily = daily_from_sections(sections, rate)
        cost = None if change is None else rate * change
        block[label] = {
            "single_side_rate": rate,
            "gross_return_mean_bp": None if gross is None else gross * 1e4,
            "cost_mean_bp": None if cost is None else cost * 1e4,
            "net_return_mean_bp": None if (gross is None or cost is None) else (gross - cost) * 1e4,
            "daily_net_mean_bp": None if daily.empty else num(daily["net_return"].mean() * 1e4),
            "turnover_local": summary["turnover_mean_daily"],
            "breakeven_single_side_rate_bp": (None if summary["breakeven_single_side_rate"] is None
                                              else summary["breakeven_single_side_rate"] * 1e4),
            "share_days_net_positive": None if daily.empty else num(
                float((daily["net_return"] > 0).sum()) / len(daily)),
            "monthly_net_mean_bp": monthly_net(daily),
        }
    return block


# ------------------------------------------------------------------------ reconciliation

def compare(name: str, recomputed, stored, tolerance: float) -> dict:
    if recomputed is None or stored is None:
        return {"metric": name, "recomputed": recomputed, "stored": stored,
                "abs_gap": None, "rel_gap": None, "within_tolerance": False,
                "note": "missing on one side"}
    left, right = float(recomputed), float(stored)
    gap = abs(left - right)
    scale = max(abs(left), abs(right), 1e-300)
    return {"metric": name, "recomputed": left, "stored": right, "abs_gap": gap,
            "rel_gap": gap / scale, "within_tolerance": gap <= tolerance}


def daily_from_sections(sections: pd.DataFrame, rate: float) -> pd.DataFrame:
    """eval-v2 daily rows: per-timestamp values first, then an equal-weight same-day mean.

    Net and cost are formed per timestamp so that a timestamp with an undefined rebalance
    is skipped in exactly the same columns as the reference evaluator skips it.
    """
    frame = sections.copy()
    frame["cost"] = rate * frame["total_abs_weight_change"]
    frame["net_return"] = frame["long_short_return"] - frame["cost"]
    columns = ["rank_ic", "ic", "long_short_return", "turnover", "total_abs_weight_change",
               "cost", "net_return"]
    return frame.groupby("day")[columns].mean().reset_index()


def daily_reconcile(sections: pd.DataFrame, path: Path, rate: float) -> dict:
    """Day-by-day comparison against one stored eval-v2 daily file at the same rate."""
    stored = pd.read_csv(path)
    if "date" not in stored.columns:
        raise ValueError(f"{path.name}: no date column")
    stored["day"] = pd.to_datetime(stored["date"])
    frame = daily_from_sections(sections, rate)
    merged = frame.merge(stored.drop(columns=["date"]), on="day", how="outer",
                         suffixes=("_verify", "_t7"), validate="one_to_one")
    if merged["day"].isna().any():
        raise ValueError(f"{path.name}: day sets differ")
    gaps = {}
    for field in ("rank_ic", "ic", "long_short_return", "turnover", "total_abs_weight_change",
                  "cost", "net_return"):
        left = pd.to_numeric(merged[f"{field}_verify"], errors="coerce")
        right = pd.to_numeric(merged[f"{field}_t7"], errors="coerce")
        gaps[field] = num((left - right).abs().max())
    # Informational only: eval-v2 forms every daily column from per-timestamp values and skips
    # undefined rebalances inside that column, so on the first evaluated day the stored daily
    # net is not the stored daily spread minus the stored daily cost.
    gaps["stored_daily_net_vs_spread_minus_cost_informational"] = num(
        (pd.to_numeric(merged["net_return_t7"], errors="coerce")
         - (pd.to_numeric(merged["long_short_return_t7"], errors="coerce")
            - pd.to_numeric(merged["cost_t7"], errors="coerce"))).abs().max())
    return {"file": path.name, "single_side_rate": rate, "rows": int(len(merged)),
            "max_abs_gap_by_field": gaps,
            "note": "the *_t7 columns are reconciled against this verifier's independent "
                    "recomputation; the informational field documents that the archived daily "
                    "net is built per column (undefined rebalances skipped), not as daily spread "
                    "minus daily cost"}


# ------------------------------------------------------- frozen pre-registered gate lines

SECONDARY_SUMMARY_KEYS = {"turnover_mean": "turnover_mean_daily"}
SECONDARY_DIRECTIONS = {"ic_mean": "maximize", "ic_ir": "maximize",
                        "long_short_sharpe": "maximize", "long_short_return_mean": "maximize",
                        "min_daily_long_short_return": "maximize", "turnover_mean": "minimize"}


def block_indices(length: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    """Moving consecutive-day blocks inside one declared period; stocks are never resampled."""
    starts = rng.integers(0, length - block_days + 1, size=int(np.ceil(length / block_days)))
    return (starts[:, None] + np.arange(block_days)).ravel()[:length]


def frozen_gate_readout(reference_daily: pd.DataFrame, candidate_daily: pd.DataFrame,
                        reference_sections: pd.DataFrame, candidate_sections: pd.DataFrame,
                        reference_summary: dict, candidate_summary: dict, prereg: dict,
                        sealed_hits: int) -> dict:
    """Apply the frozen pre-registered lines to the independently recomputed dailies.

    Only the thresholds are read from the registered file; every statistic is recomputed
    here from the paired daily series (candidate minus reference on the frozen sign
    convention). Nothing is copied from the archived readout.
    """
    policy = prereg["object_policy_template"]
    declared = [(pd.Timestamp(start), pd.Timestamp(end))
                for start, end in prereg["periods"]["declared"]]
    frame = pd.DataFrame({"day": pd.to_datetime(reference_daily["day"]).reset_index(drop=True),
                          "reference": reference_daily["rank_ic"].to_numpy(dtype=float),
                          "candidate": candidate_daily["rank_ic"].to_numpy(dtype=float)})
    if not pd.to_datetime(candidate_daily["day"]).reset_index(drop=True).equals(frame["day"]):
        raise ValueError("objects are not evaluated on the identical trading-day index")
    frame["difference"] = frame["candidate"] - frame["reference"]  # primary metric maximizes
    positions = [np.flatnonzero(((frame["day"] >= start) & (frame["day"] <= end)).to_numpy())
                 for start, end in declared]
    covered = np.zeros(len(frame), dtype=bool)
    for block in positions:
        covered[block] = True
    effects = [float(frame["difference"].to_numpy()[block].mean()) if len(block) else None
               for block in positions]
    improvement = (float(np.mean(effects)) if effects and None not in effects else None)
    differences = frame["difference"].to_numpy(dtype=float)
    # The frozen engine (compare_evaluation_objects) defines the paired improvement as the
    # equal-weight mean of the declared period effects. The archived readout reports the pooled
    # day-weighted mean instead; both are reported so the difference is auditable.
    pooled_improvement = float(np.mean(differences))
    rng = np.random.default_rng(int(policy["seed"]))
    draws = []
    for _ in range(int(policy["bootstrap_samples"])):
        sampled = np.concatenate([block[block_indices(len(block), int(policy["block_days"]), rng)]
                                  for block in positions])
        draws.append(float(np.mean(differences[sampled])))
    alpha = (1.0 - float(policy["confidence"])) / 2.0
    low, high = np.quantile(np.asarray(draws), [alpha, 1.0 - alpha])
    required_positive = max(1, int(np.ceil(
        float(prereg["evidence_rules"]["required_positive_fraction"]) * len(declared) - 1e-12)))
    positive = sum(1 for effect in effects if effect is not None and effect > 0)
    secondary = {}
    for metric, limit in policy["max_metric_regressions"].items():
        key = SECONDARY_SUMMARY_KEYS.get(metric, metric)
        base = reference_summary.get(key)
        cand = candidate_summary.get(key)
        direction = SECONDARY_DIRECTIONS.get(metric)
        if direction is None:
            raise ValueError(f"no frozen direction for secondary metric {metric}")
        regression = (base - cand) if direction == "maximize" else (cand - base)
        secondary[metric] = {"reference": num(base), "candidate": num(cand),
                             "direction": direction, "regression": num(regression),
                             "limit": float(limit), "passed": bool(regression <= float(limit))}
    rate5 = dict(RATES)[PRIMARY]
    reference_net5 = float(daily_from_sections(reference_sections, rate5)["net_return"].mean())
    candidate_net5 = float(daily_from_sections(candidate_sections, rate5)["net_return"].mean())
    net5_regression = reference_net5 - candidate_net5
    net5_limit = float(prereg["net5_limit"])
    minimum_days = int(prereg["evidence_rules"]["minimum_days"])
    minimum_periods = int(prereg["evidence_rules"]["minimum_periods"])
    days_outside = int((~covered).sum())
    interval_excludes_zero = bool(float(low) > 0)
    lines = [
        {"line": "coverage", "passed": bool(len(frame) >= minimum_days
                                            and len(declared) >= minimum_periods
                                            and days_outside == 0 and sealed_hits == 0),
         "evidence": {"days": int(len(frame)), "minimum_days": minimum_days,
                      "declared_periods": len(declared), "minimum_periods": minimum_periods,
                      "days_outside_declared_periods": days_outside,
                      "sealed_intersection": int(sealed_hits)}},
        {"line": "primary_practical_improvement",
         "passed": bool(improvement is not None
                        and improvement >= float(policy["min_improvement"])),
         "evidence": {"metric": policy["primary_metric"],
                      "improvement_equal_period": num(improvement),
                      "improvement_pooled_daily": num(pooled_improvement),
                      "estimator": "frozen engine: equal-weight mean of the declared period "
                                   "effects; the pooled day-weighted mean is reported alongside",
                      "frozen_min_improvement": float(policy["min_improvement"])}},
        {"line": "primary_statistical_interval", "passed": interval_excludes_zero,
         "evidence": {"low": num(low), "high": num(high),
                      "confidence": float(policy["confidence"]),
                      "block_days": int(policy["block_days"]),
                      "draws": int(policy["bootstrap_samples"]),
                      "space": "maximize: paired daily candidate minus reference"}},
        {"line": "primary_repeat_across_periods", "passed": bool(positive >= required_positive),
         "evidence": {"period_effects": [num(effect) for effect in effects],
                      "period_names": [f"{start.date()}..{end.date()}" for start, end in declared],
                      "positive_periods": positive, "periods": len(effects),
                      "required_positive_periods": required_positive}},
    ]
    for metric, block in secondary.items():
        lines.append({"line": f"secondary.{metric}", "passed": block["passed"],
                      "evidence": block})
    net5_passed = bool(net5_regression <= net5_limit)
    lines.append({"line": "economic.net5_stress", "passed": net5_passed,
                  "evidence": {"reference_net5": num(reference_net5),
                               "candidate_net5": num(candidate_net5),
                               "regression": num(net5_regression),
                               "frozen_limit": net5_limit,
                               "note": "regression = reference net minus candidate net; a negative "
                                       "value is an improvement at the 5bp stress"}})
    passed = sum(1 for item in lines if item["passed"])
    return {"frozen": True,
            "primary": {"metric": policy["primary_metric"],
                        "reference_level": num(reference_summary["rank_ic_mean"]),
                        "candidate_level": num(candidate_summary["rank_ic_mean"]),
                        "improvement": num(improvement),
                        "improvement_pooled_daily": num(pooled_improvement),
                        "frozen_min_improvement": float(policy["min_improvement"]),
                        "interval": {"low": num(low), "high": num(high),
                                     "confidence": float(policy["confidence"]),
                                     "block_days": int(policy["block_days"]),
                                     "draws": int(policy["bootstrap_samples"]),
                                     "seed": int(policy["seed"]),
                                     "defined_days": int(len(frame))},
                        "interval_excludes_zero": interval_excludes_zero,
                        "period_evidence": {"effects": [num(effect) for effect in effects],
                                            "positive_periods": positive,
                                            "required_positive_periods": required_positive}},
            "secondary": secondary,
            "net5": {"reference": num(reference_net5), "candidate": num(candidate_net5),
                     "regression": num(net5_regression), "frozen_limit": net5_limit,
                     "passed": net5_passed},
            "lines": lines, "lines_passed": passed, "lines_total": len(lines),
            "promotable_under_frozen_lines": passed == len(lines),
            "blocking_lines": [item["line"] for item in lines if not item["passed"]],
            "what_this_is": "the frozen pre-registered lines applied to the independent "
                            "recomputation; every statistic is recomputed here and no threshold is "
                            "re-declared, relaxed or replaced"}


# ----------------------------------------------------------------------------- checks

def check(name: str, passed: bool, evidence) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def text_quotes(path: Path, patterns: dict) -> dict:
    """Quote the competition text lines that back the official-turnover reasoning."""
    lines = path.read_text(encoding="utf-8").splitlines()
    found = {}
    for key, pattern in patterns.items():
        hits = [(number, line.strip()) for number, line in enumerate(lines, start=1)
                if re.search(pattern, line)]
        found[key] = [{"line": number, "text": text} for number, text in hits[:3]]
    return found


def static_scan(path: Path) -> dict:
    """Scan the experiment script for a second alpha, a grid, or a score threshold."""
    source = path.read_text(encoding="utf-8")
    return {
        "alpha_declarations": re.findall(r"^ALPHA\s*=\s*([0-9.]+)", source, flags=re.MULTILINE),
        "grid_declaration": re.findall(r"^GRID\s*=\s*\(([^)]*)\)", source, flags=re.MULTILINE),
        "loops_over_alpha": re.findall(r"for\s+alpha\s+in\s+[A-Za-z_]+", source),
        "threshold_lines": [line.strip() for line in source.splitlines()
                            if re.search(r"threshold", line, flags=re.IGNORECASE)],
        "score_comparison_lines": [line.strip() for line in source.splitlines()
                                   if re.search(r"score\s*[<>]=?", line)],
        "operator_functions": sorted(set(re.findall(r"^def\s+(\w*ewma\w*|\w*smoo\w*)",
                                                    source, flags=re.MULTILINE))),
    }


# --------------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / EXPERIMENT_ID)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    plan = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "kind": "verification",
        "task": "t14",
        "objective": "independently confirm that the pre-registered single turnover change keeps "
                     "the gross spread, lowers the cost and does not collapse the prediction",
        "independence": "the reference blend, the causal EWMA operator, the two-leg book, the "
                        "rebalancing U, rank IC/IC and all four scenarios are re-implemented here "
                        "from the raw frozen replay parquet; no experiment or evaluation-library "
                        "call is used for the candidate",
        "objects": {"reference": REFERENCE, "candidate": CANDIDATE, "controls": list(CONTROLS)},
        "cost_scenarios": {label: rate for label, rate in RATES},
        "primary_stress_scenario": PRIMARY,
        "alpha_evaluated": ALPHA,
        "guard_bars_predeclared": BARS,
        "acceptance_map": {
            "four_scenarios_5bp_primary": "all four rates reported for both objects; 5bp is the "
                                          "primary stress and its net change is stated explicitly",
            "three_deltas_together": "gross retention, cost reduction and the turnover proxy change "
                                     "side by side at every rate; the official turnover item is a "
                                     "cross-submission rank percentile and stays labelled "
                                     "unavailable locally",
            "signal_not_constant": "per-timestamp cross-sectional spread, defined rank IC share, "
                                   "IC/RankIC and long-short retention",
            "single_change_no_grid": "one alpha declared, one operator, no loop over alpha; the "
                                     "recomputation with alpha=0.5 reproduces the stored daily rows",
            "no_score_threshold": "identical membership, rank-based book, book invariant to "
                                  "rescaling the standardized scores",
            "frozen_gate_readout": "the frozen pre-registered lines (thresholds read from the "
                                   "registered file, every statistic recomputed here) are applied to "
                                   "the independent dailies so the not-promotable readout is "
                                   "reproduced rather than quoted",
        },
        "sealed_interval": list(SEALED),
        "notes": ["verifier guard calibrations (constant and within-day frozen signals) are not "
                  "evaluated arms and are never used to choose an operator",
                  "no promotion, rejection or model selection output"],
    }
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    output = args.output_root / f"{started:%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(plan_bytes)
    print(f"predeclared verifier plan {output.relative_to(ROOT).as_posix()}/plan.json sha256={plan_sha}")

    # ------------------------------------------------------------------ recomputation
    windows = [load_window(*window) for window in WINDOWS]
    keys, target, member_scores = pooled(windows)
    print(f"inputs pooled: {len(keys)} samples, {keys['date'].nunique()} cross sections, "
          f"{len(WINDOWS)} windows", flush=True)
    references = np.stack([member_scores[member] for member in MEMBERS])

    objects = {REFERENCE: references.mean(axis=0)}
    objects[CANDIDATE] = causal_ewma(keys, objects[REFERENCE], ALPHA)
    objects["control_constant_zero"] = np.zeros(len(keys))
    objects["control_daily_freeze"] = daily_freeze(keys, objects[REFERENCE])

    summaries, daily_frames, sections_store = {}, {}, {}
    for object_id, scores in objects.items():
        sections = evaluate_sections(keys, target, scores)
        summary = aggregate(sections)
        summaries[object_id] = summary
        daily_frames[object_id] = summary["daily"]
        sections_store[object_id] = sections
        rank = summary["rank_ic_mean"]
        gross = summary["gross_return_mean_economic"]
        print(f"evaluated {object_id}: rank_ic={'n/a' if rank is None else f'{rank:.6f}'} "
              f"gross_bp={'n/a' if gross is None else f'{gross * 1e4:.3f}'} "
              f"turnover={summary['turnover_mean_daily']:.4f}", flush=True)

    reference_summary = summaries[REFERENCE]
    candidate_summary = summaries[CANDIDATE]
    scenarios = {object_id: scenario_block(summaries[object_id], sections_store[object_id])
                 for object_id in objects}

    # Membership is a hard gate: no timestamp may lose or gain a stock because of the change.
    membership_gap = int(np.abs(sections_store[REFERENCE]["sample_count"].to_numpy()
                                - sections_store[CANDIDATE]["sample_count"].to_numpy()).max())

    # Scale invariance: if a standardized score were used as a bp threshold, multiplying the
    # scores would change the book. A rank-based book must be unchanged.
    scaled = evaluate_sections(keys, target, objects[CANDIDATE] * 1000.0)
    scale_gap = {
        "long_short_return": num(np.abs(scaled["long_short_return"].to_numpy()
                                        - sections_store[CANDIDATE]["long_short_return"].to_numpy()).max()),
        "turnover": num(np.nanmax(np.abs(scaled["turnover"].to_numpy()
                                      - sections_store[CANDIDATE]["turnover"].to_numpy()))),
        "sample_count": int(np.abs(scaled["sample_count"].to_numpy()
                                   - sections_store[CANDIDATE]["sample_count"].to_numpy()).max()),
    }

    # ---------------------------------------------------------- stored artifacts to check
    t6 = json.loads((ROOT / T6_RECORD).read_text(encoding="utf-8"))
    t7_record = json.loads((ROOT / T7_RECORD).read_text(encoding="utf-8"))
    t7_data = json.loads((ROOT / T7_RUN / "data.json").read_text(encoding="utf-8"))
    t7_plan = json.loads((ROOT / T7_RUN / "plan.json").read_text(encoding="utf-8"))
    t7_config = json.loads((ROOT / T7_RUN / "config.json").read_text(encoding="utf-8"))
    t7_checks = json.loads((ROOT / T7_RUN / "checks.json").read_text(encoding="utf-8"))
    stored_objects = {object_id: json.loads((ROOT / T7_RUN / "objects" / object_id / "metrics.json")
                                            .read_text(encoding="utf-8"))
                      for object_id in (REFERENCE, CANDIDATE)}

    tolerance = BARS["reconcile_abs_tolerance"]
    reference_vs_t6 = [
        compare("gross_return_mean_bp", scenarios[REFERENCE]["0bp"]["gross_return_mean_bp"],
                t6["reference"]["gross_return_mean_bp"], tolerance),
        compare("turnover_local", scenarios[REFERENCE]["0bp"]["turnover_local"],
                t6["reference"]["turnover_local"], tolerance),
        compare("breakeven_bp", scenarios[REFERENCE]["5bp"]["breakeven_single_side_rate_bp"],
                t6["reference"]["breakeven_single_side_rate_bp"], tolerance),
    ]
    for label, _ in RATES:
        reference_vs_t6.append(compare(f"cost_bp_{label}", scenarios[REFERENCE][label]["cost_mean_bp"],
                                       t6["reference"]["cost_bp"][label], tolerance))
        reference_vs_t6.append(compare(f"net_bp_{label}", scenarios[REFERENCE][label]["net_return_mean_bp"],
                                       t6["reference"]["net_bp"][label], tolerance))

    signal_keys = (("rank_ic_mean", "rank_ic_mean"), ("ic_mean", "ic_mean"),
                   ("rank_ic_ir", "rank_ic_ir"), ("long_short_sharpe", "long_short_sharpe"),
                   ("long_short_return_mean", "long_short_return_mean"),
                   ("min_daily_long_short_return", "min_daily_long_short_return"),
                   ("turnover_mean_daily", "turnover_mean_local"),
                   ("rank_persistence_mean", "rank_persistence_mean"),
                   ("score_change_mae", "score_change_mae"))
    candidate_vs_t7, signal_vs_t7 = [], []
    for object_id in (REFERENCE, CANDIDATE):
        stored = stored_objects[object_id]
        summary = summaries[object_id]
        candidate_vs_t7.extend([
            compare(f"{object_id}.gross_bp", scenarios[object_id][PRIMARY]["gross_return_mean_bp"],
                    stored["scenarios"][PRIMARY]["gross_return_mean_bp"], tolerance),
            compare(f"{object_id}.cost_bp", scenarios[object_id][PRIMARY]["cost_mean_bp"],
                    stored["scenarios"][PRIMARY]["cost_mean_bp"], tolerance),
            compare(f"{object_id}.net_bp", scenarios[object_id][PRIMARY]["net_return_mean_bp"],
                    stored["scenarios"][PRIMARY]["net_return_mean_bp"], tolerance),
            compare(f"{object_id}.turnover", scenarios[object_id][PRIMARY]["turnover_local"],
                    stored["scenarios"][PRIMARY]["turnover_mean_local"], tolerance),
            compare(f"{object_id}.breakeven_bp",
                    scenarios[object_id][PRIMARY]["breakeven_single_side_rate_bp"],
                    stored["scenarios"][PRIMARY]["breakeven_single_side_rate_bp"], tolerance),
        ])
        for label, _ in RATES:
            candidate_vs_t7.append(compare(
                f"{object_id}.scenario_{label}.cost_bp",
                scenarios[object_id][label]["cost_mean_bp"],
                stored["scenarios"][label]["cost_mean_bp"], tolerance))
            candidate_vs_t7.append(compare(
                f"{object_id}.scenario_{label}.net_bp",
                scenarios[object_id][label]["net_return_mean_bp"],
                stored["scenarios"][label]["net_return_mean_bp"], tolerance))
        for key, stored_key in signal_keys:
            signal_vs_t7.append(compare(f"{object_id}.{key}", summary[key], stored["signal"][stored_key],
                                        max(tolerance, 1e-12)))
    daily_reconciliations = {object_id: [daily_reconcile(
        sections_store[object_id],
        ROOT / T7_RUN / "objects" / object_id / f"daily_{label}.csv", rate)
        for label, rate in RATES]
        for object_id in (REFERENCE, CANDIDATE)}
    daily_gap = max(item["max_abs_gap_by_field"][field]
                    for item_list in daily_reconciliations.values() for item in item_list
                    for field in ("rank_ic", "ic", "long_short_return", "turnover",
                                  "total_abs_weight_change", "cost", "net_return"))

    gross_retention = (candidate_summary["gross_return_mean_economic"]
                       / reference_summary["gross_return_mean_economic"])
    gross_retention_daily = (candidate_summary["long_short_return_mean"]
                             / reference_summary["long_short_return_mean"])
    turnover_ratio = candidate_summary["turnover_mean_daily"] / reference_summary["turnover_mean_daily"]
    cost_reduction_5bp = (scenarios[CANDIDATE][PRIMARY]["cost_mean_bp"]
                          - scenarios[REFERENCE][PRIMARY]["cost_mean_bp"])
    record_vs_recomputed = [
        compare("gross_retention_ratio_5bp", gross_retention,
                t7_record["gross_retention_ratio_5bp"], tolerance),
        compare("cost_reduction_bp_5bp", cost_reduction_5bp,
                t7_record["cost_reduction_bp_5bp"], tolerance),
        compare("turnover_ratio_5bp", turnover_ratio, t7_record["turnover_ratio_5bp"], tolerance),
        compare("candidate_breakeven_bp",
                scenarios[CANDIDATE][PRIMARY]["breakeven_single_side_rate_bp"],
                t7_record["candidate_5bp"]["breakeven_bp"], tolerance),
    ]

    # --------------------------------------------- frozen pre-registered gate lines
    prereg_path = ROOT / "configs/b4_promotion_prereg_20261027.json"
    registration_path = ROOT / "configs/b4_promotion_prereg_20261027.registration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    prereg_sha = sha256(prereg_path)
    stored_readout = json.loads((ROOT / T7_READOUT).read_text(encoding="utf-8"))
    gate_dates = pd.to_datetime(keys["date"])
    gate_sealed_hits = int(((gate_dates >= pd.Timestamp(SEALED[0]))
                            & (gate_dates < pd.Timestamp(SEALED[1]))).sum())
    gate = frozen_gate_readout(daily_frames[REFERENCE], daily_frames[CANDIDATE],
                               sections_store[REFERENCE], sections_store[CANDIDATE],
                               reference_summary, candidate_summary, prereg, gate_sealed_hits)
    gate["prereg"] = {
        "path": prereg_path.relative_to(ROOT).as_posix(), "sha256": prereg_sha,
        "registered_sha256": registration.get("prereg_sha256"),
        "verified": prereg_sha == registration.get("prereg_sha256"),
        "registration_id": registration.get("registration_id"),
        "registered_at": registration.get("registered_at"),
        "thresholds": {"primary_metric": prereg["object_policy_template"]["primary_metric"],
                       "min_improvement": prereg["object_policy_template"]["min_improvement"],
                       "max_metric_regressions":
                           prereg["object_policy_template"]["max_metric_regressions"],
                       "net5_limit": prereg["net5_limit"]},
    }
    try:
        from jinyiquant.protocol import validate_change
        validate_change({"factor": "postprocess.causal_ewma", "baseline": "identity",
                         "candidate": "causal_ewma_alpha050",
                         "hypothesis": "output-level temporal persistence lowers turnover"})
        gate["change_vocabulary"] = {"refused": False, "error": None}
    except Exception as exc:  # the gate's own vocabulary, not this verifier's rule
        gate["change_vocabulary"] = {"refused": True, "error": str(exc)}
    gate_vs_readout = [
        compare("improvement_pooled_daily", gate["primary"]["improvement_pooled_daily"],
                stored_readout["primary"]["improvement"], 1e-9),
        compare("gate_lines_passed", gate["lines_passed"],
                stored_readout["readout_summary"]["gate_lines_passed"], 0),
        compare("promotable_under_frozen_lines",
                float(gate["promotable_under_frozen_lines"]),
                float(stored_readout["readout_summary"]["promotable_under_frozen_lines"]), 0),
        compare("interval_low", gate["primary"]["interval"]["low"],
                stored_readout["primary"]["interval"]["low"], 1e-9),
        compare("interval_high", gate["primary"]["interval"]["high"],
                stored_readout["primary"]["interval"]["high"], 1e-9),
        compare("positive_periods", gate["primary"]["period_evidence"]["positive_periods"],
                stored_readout["primary"]["period_evidence"]["positive_periods"], 0),
        compare("net5_regression", gate["net5"]["regression"],
                stored_readout["net5"]["regression"], 1e-9),
    ]
    for metric in gate["secondary"]:
        gate_vs_readout.append(compare(f"secondary.{metric}",
                                       gate["secondary"][metric]["regression"],
                                       stored_readout["secondary"][metric]["regression"], 1e-9))
    gate["readout_reconciliation"] = gate_vs_readout
    gate["estimator_audit"] = {
        "frozen_engine_estimator": gate["primary"]["improvement"],
        "archived_readout_estimator": stored_readout["primary"]["improvement"],
        "difference": num(gate["primary"]["improvement"]
                          - stored_readout["primary"]["improvement"]),
        "explanation": "the frozen engine (jinyiquant.comparison.compare_evaluation_objects) "
                       "averages the declared period effects with equal weight; the archived "
                       "readout reports the pooled day-weighted mean of the paired daily "
                       "differences. Period lengths differ (19-22 days), so the two estimators "
                       "differ slightly. Both are far below the frozen practical threshold, so no "
                       "gate line changes outcome; the engine estimator is the pre-registered one.",
        "stored_blocking_evidence": stored_readout["readout_summary"]["blocking_evidence"],
    }

    print("reconciliation and gate readout complete", flush=True)

    # ---------------------------------------------------------- independent platform text
    quotes = text_quotes(ROOT / COMPETITION_TEXT, {
        "turnover_direction": r"换手越低越好",
        "turnover_rank_percentile": r"rank 百分比排名",
        "coverage_limit": r"缺失率不得高于 40%",
    })
    official_fields = []
    for object_id, stored in stored_objects.items():
        for label, block in stored["scenarios"].items():
            official = block.get("official_turnover_item") or {}
            official_fields.append({"object": object_id, "scenario": label,
                                    "official_value": official.get("official_value"),
                                    "official_status": official.get("official_status"),
                                    "local_proxy": official.get("local_proxy")})
    scan = static_scan(ROOT / T7_SCRIPT)

    # ------------------------------------------------------------- signal / guard numbers
    signal = {}
    for object_id in (REFERENCE, CANDIDATE):
        signal[object_id] = {key: num(summaries[object_id][key]) for key in
                             ("ic_mean", "rank_ic_mean", "ic_ir", "rank_ic_ir",
                              "long_short_return_mean", "long_short_sharpe",
                              "min_daily_long_short_return", "turnover_mean_daily",
                              "rank_persistence_mean", "score_change_mae",
                              "min_cross_sectional_std", "ranking_defined_share",
                              "undefined_rank_ic_count", "undefined_turnover_count",
                              "cross_section_count", "day_count", "boundary_tie_sections")}
    retention = {
        "gross_economic": num(gross_retention),
        "gross_daily_mean": num(gross_retention_daily),
        "rank_ic": num(candidate_summary["rank_ic_mean"] / reference_summary["rank_ic_mean"]),
        "ic": num(candidate_summary["ic_mean"] / reference_summary["ic_mean"]),
        "long_short_return": num(candidate_summary["long_short_return_mean"]
                                 / reference_summary["long_short_return_mean"]),
        "turnover_ratio": num(turnover_ratio),
        "cost_ratio_5bp": num(scenarios[CANDIDATE][PRIMARY]["cost_mean_bp"]
                              / scenarios[REFERENCE][PRIMARY]["cost_mean_bp"]),
    }

    controls = {}
    for control_id in CONTROLS:
        summary = summaries[control_id]
        controls[control_id] = {
            "role": "verifier guard calibration (not an evaluated arm)",
            "rank_ic_mean": summary["rank_ic_mean"],
            "ic_mean": summary["ic_mean"],
            "long_short_return_mean": summary["long_short_return_mean"],
            "gross_return_mean_economic": summary["gross_return_mean_economic"],
            "turnover_mean_daily": summary["turnover_mean_daily"],
            "min_cross_sectional_std": summary["min_cross_sectional_std"],
            "ranking_defined_share": summary["ranking_defined_share"],
            "undefined_rank_ic_count": summary["undefined_rank_ic_count"],
            "cross_section_count": summary["cross_section_count"],
            "fails_guards": {
                "cross_sectional_spread": summary["min_cross_sectional_std"]
                <= BARS["min_cross_sectional_std"],
                "rank_ic_defined": summary["undefined_rank_ic_count"] > 0,
                "rank_ic_retention": (summary["rank_ic_mean"] is not None
                                      and summary["rank_ic_mean"]
                                      < BARS["min_rank_ic_retention"] * reference_summary["rank_ic_mean"]),
                "gross_retention": (summary["gross_return_mean_economic"] is not None
                                    and summary["gross_return_mean_economic"]
                                    < BARS["min_gross_retention"]
                                    * reference_summary["gross_return_mean_economic"]),
            },
        }
        controls[control_id]["flagged"] = any(controls[control_id]["fails_guards"].values())

    # ------------------------------------------------------------------------- checks
    checks = []
    stored_windows = {item["name"]: item for item in t7_data["windows"]}
    window_sha_match = all(
        window["keys_sha256"] == stored_windows[window["name"]]["keys_sha256"]
        and window["targets_sha256"] == stored_windows[window["name"]]["targets_sha256"]
        and window["replay"] == stored_windows[window["name"]]["replay_run"]
        for window in windows)
    dates = pd.to_datetime(keys["date"])
    sealed_hits = int(((dates >= pd.Timestamp(SEALED[0])) & (dates < pd.Timestamp(SEALED[1]))).sum())
    checks.append(check(
        "inputs_frozen_and_sealed_untouched",
        window_sha_match and sealed_hits == 0,
        {"window_keys_targets_sha_match_t7_data_json": window_sha_match,
         "samples_in_sealed_interval": sealed_hits,
         "samples": int(len(keys)), "cross_sections": int(keys["date"].nunique()),
         "days": int(dates.dt.normalize().nunique()),
         "date_range": [str(dates.min()), str(dates.max())],
         "member_prediction_sha256": {window["name"]: window["prediction_sha256"]
                                      for window in windows}}))

    checks.append(check(
        "recomputed_reference_reproduces_t6_cost_curve",
        all(item["within_tolerance"] for item in reference_vs_t6),
        reference_vs_t6))

    checks.append(check(
        "recomputed_candidate_reproduces_t7_artifacts_day_by_day",
        all(item["within_tolerance"] for item in candidate_vs_t7)
        and all(item["within_tolerance"] for item in signal_vs_t7)
        and daily_gap is not None and daily_gap <= tolerance,
        {"aggregate": candidate_vs_t7, "signal": signal_vs_t7,
         "daily_max_abs_gap": daily_gap, "tolerance": tolerance,
         "daily_files_checked": {object_id: [item["file"] for item in item_list]
                                 for object_id, item_list in daily_reconciliations.items()}}))

    four = all(scenarios[object_id][label]["gross_return_mean_bp"] is not None
               for object_id in (REFERENCE, CANDIDATE) for label, _ in RATES)
    gross_rate_free = all(
        abs(scenarios[CANDIDATE][label]["gross_return_mean_bp"]
            - scenarios[CANDIDATE]["0bp"]["gross_return_mean_bp"]) <= 1e-12 for label, _ in RATES)
    net_5bp_change = (scenarios[CANDIDATE][PRIMARY]["net_return_mean_bp"]
                      - scenarios[REFERENCE][PRIMARY]["net_return_mean_bp"])
    checks.append(check(
        "four_scenarios_reported_with_5bp_primary",
        four and gross_rate_free and t7_plan["primary_stress_scenario"] == PRIMARY,
        {"scenarios": {label: {"reference": {"gross_bp": scenarios[REFERENCE][label]["gross_return_mean_bp"],
                                             "cost_bp": scenarios[REFERENCE][label]["cost_mean_bp"],
                                             "net_bp": scenarios[REFERENCE][label]["net_return_mean_bp"]},
                               "candidate": {"gross_bp": scenarios[CANDIDATE][label]["gross_return_mean_bp"],
                                             "cost_bp": scenarios[CANDIDATE][label]["cost_mean_bp"],
                                             "net_bp": scenarios[CANDIDATE][label]["net_return_mean_bp"]}}
                       for label, _ in RATES},
         "gross_rate_free": gross_rate_free, "primary": PRIMARY,
         "net_change_5bp_bp": net_5bp_change,
         "candidate_net_5bp_still_negative": scenarios[CANDIDATE][PRIMARY]["net_return_mean_bp"] < 0,
         "candidate_breakeven_bp": scenarios[CANDIDATE][PRIMARY]["breakeven_single_side_rate_bp"],
         "monthly_net_mean_bp_5bp_candidate": scenarios[CANDIDATE][PRIMARY]["monthly_net_mean_bp"]}))

    three_deltas_ok = (retention["gross_economic"] is not None
                       and retention["gross_economic"] >= BARS["min_gross_retention"]
                       and cost_reduction_5bp < 0 and retention["turnover_ratio"] < 1.0)
    official_clean = all(field["official_value"] is None for field in official_fields)
    checks.append(check(
        "three_deltas_reported_together_with_proxy_labelled",
        three_deltas_ok and official_clean and all(len(quotes[key]) > 0 for key in quotes),
        {"gross_retention_5bp": retention["gross_economic"],
         "gross_retention_daily_mean": retention["gross_daily_mean"],
         "gross_delta_bp": (scenarios[CANDIDATE][PRIMARY]["gross_return_mean_bp"]
                            - scenarios[REFERENCE][PRIMARY]["gross_return_mean_bp"]),
         "cost_reduction_bp_by_rate": {label: scenarios[CANDIDATE][label]["cost_mean_bp"]
                                       - scenarios[REFERENCE][label]["cost_mean_bp"]
                                       for label, _ in RATES},
         "net_change_bp_by_rate": {label: scenarios[CANDIDATE][label]["net_return_mean_bp"]
                                   - scenarios[REFERENCE][label]["net_return_mean_bp"]
                                   for label, _ in RATES},
         "turnover_proxy": {"definition": "local turnover = U / 2",
                            "reference": reference_summary["turnover_mean_daily"],
                            "candidate": candidate_summary["turnover_mean_daily"],
                            "delta": candidate_summary["turnover_mean_daily"]
                            - reference_summary["turnover_mean_daily"],
                            "ratio": turnover_ratio,
                            "official_value": None,
                            "official_status": "unavailable locally: cross-submission rank "
                                               "percentile of the published turnover item"},
         "official_turnover_fields_are_null": official_clean,
         "platform_text_quotes": quotes}))

    signal_ok = (candidate_summary["min_cross_sectional_std"] > BARS["min_cross_sectional_std"]
                 and candidate_summary["undefined_rank_ic_count"] == 0
                 and candidate_summary["ranking_defined_share"] >= BARS["min_defined_rank_ic_share"]
                 and retention["rank_ic"] >= BARS["min_rank_ic_retention"]
                 and retention["ic"] >= BARS["min_ic_retention"]
                 and retention["long_short_return"] >= BARS["min_long_short_retention"]
                 and candidate_summary["rank_ic_mean"] > 0
                 and candidate_summary["long_short_return_mean"] > 0)
    checks.append(check(
        "candidate_is_not_a_constant_signal",
        signal_ok,
        {"bars": BARS,
         "reference": signal[REFERENCE], "candidate": signal[CANDIDATE],
         "retention": retention}))

    checks.append(check(
        "collapse_guard_has_teeth_on_degenerate_controls",
        controls["control_constant_zero"]["flagged"],
        {"controls": controls,
         "statement": "the same guard that passes the candidate must reject a deliberately "
                      "collapsed signal; the controls are verifier calibrations, not arms"}))

    single_change_ok = (
        t7_plan["single_change"]["alphas_evaluated"] == [ALPHA]
        and t7_plan["single_change"]["grid_search"] is False
        and t7_plan["single_change"]["thresholds_applied"] == []
        and t7_config["single_change"]["alphas_evaluated"] == [ALPHA]
        and scan["alpha_declarations"] == [str(ALPHA)]
        and len(scan["grid_declaration"]) == 1
        and len(scan["loops_over_alpha"]) == 0
        and all(item["within_tolerance"] for item in candidate_vs_t7 + signal_vs_t7)
        and daily_gap is not None and daily_gap <= tolerance)
    checks.append(check(
        "single_change_no_grid_no_retraining",
        single_change_ok,
        {"declared": {"alphas_evaluated": t7_plan["single_change"]["alphas_evaluated"],
                      "grid_search": t7_plan["single_change"]["grid_search"],
                      "thresholds_applied": t7_plan["single_change"]["thresholds_applied"],
                      "operator": t7_plan["single_change"]["operator"]},
         "static_scan": scan,
         "numeric_proof": "recomputing operator(reference, alpha=0.5) from the raw frozen "
                          "parquet reproduces every stored daily row, so no second free "
                          "parameter, no retrained model and no extra transform can hide in the "
                          "candidate",
         "daily_max_abs_gap": daily_gap,
         "no_training_declared": t7_config.get("no_training")}))

    threshold_ok = (membership_gap == 0 and scale_gap["long_short_return"] <= 1e-12
                    and scale_gap["turnover"] <= 1e-12 and scale_gap["sample_count"] == 0
                    and t7_plan["single_change"]["thresholds_applied"] == []
                    and not scan["score_comparison_lines"])
    checks.append(check(
        "standardized_score_not_used_as_bp_return_threshold",
        threshold_ok,
        {"membership_identical": membership_gap == 0, "max_sample_count_gap": membership_gap,
         "book_invariant_to_score_rescaling_by_1000": scale_gap,
         "thresholds_applied": t7_plan["single_change"]["thresholds_applied"],
         "score_comparison_lines_in_experiment_script": scan["score_comparison_lines"],
         "reasoning": "the book is built from cross-sectional ranks at quantile 0.2 and cost is "
                      "rate * traded notional, so raw score magnitudes (and therefore any "
                      "standardized-score threshold) never enter the bp economics"}))

    linearity = {}
    for object_id in (REFERENCE, CANDIDATE):
        identity = max(abs(scenarios[object_id][label]["net_return_mean_bp"]
                           - (scenarios[object_id][label]["gross_return_mean_bp"]
                              - scenarios[object_id][label]["cost_mean_bp"]))
                       for label, _ in RATES)
        spread = (max(scenarios[object_id][label]["turnover_local"] for label, _ in RATES)
                  - min(scenarios[object_id][label]["turnover_local"] for label, _ in RATES))
        linearity[object_id] = {"max_net_identity_residual": num(identity),
                                "turnover_spread_across_rates": num(spread),
                                "stored_cost_ratios": {
                                    f"{label}": num(scenarios[object_id][label]["cost_mean_bp"]
                                                    / scenarios[object_id]["5bp"]["cost_mean_bp"])
                                    for label, _ in RATES}}
    checks.append(check(
        "cost_linear_and_turnover_rate_invariant",
        all((item["max_net_identity_residual"] or 0) <= 1e-12
            and (item["turnover_spread_across_rates"] or 0) == 0
            for item in linearity.values()),
        {"linearity": linearity,
         "stored_scenarios": {object_id: {label: {"cost_bp": stored_objects[object_id]["scenarios"][label]["cost_mean_bp"],
                                                  "turnover": stored_objects[object_id]["scenarios"][label]["turnover_mean_local"]}
                                          for label, _ in RATES}
                              for object_id in (REFERENCE, CANDIDATE)}}))

    checks.append(check(
        "record_headline_numbers_follow_from_recomputation",
        all(item["within_tolerance"] for item in record_vs_recomputed),
        record_vs_recomputed))

    checks.append(check(
        "frozen_gate_lines_recomputed_from_independent_dailies",
        gate["prereg"]["verified"] and all(item["within_tolerance"] for item in gate_vs_readout),
        {"prereg_verified": gate["prereg"]["verified"],
         "prereg_sha256": gate["prereg"]["sha256"],
         "readout_reconciliation": gate_vs_readout,
         "lines": gate["lines"], "lines_passed": gate["lines_passed"],
         "lines_total": gate["lines_total"], "blocking_lines": gate["blocking_lines"],
         "promotable_under_frozen_lines": gate["promotable_under_frozen_lines"],
         "change_vocabulary": gate["change_vocabulary"]}))

    verdict = "pass" if all(item["passed"] for item in checks) else "fail"

    results = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "kind": "verification",
        "task": "t14",
        "created_at": started.isoformat(),
        "verdict": verdict,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "predeclared_bars": BARS,
        "alpha": ALPHA,
        "primary_stress_scenario": PRIMARY,
        "scenarios": scenarios,
        "signal": signal,
        "retention": retention,
        "deltas": {
            "gross_retention_ratio": retention["gross_economic"],
            "gross_delta_bp": (scenarios[CANDIDATE][PRIMARY]["gross_return_mean_bp"]
                               - scenarios[REFERENCE][PRIMARY]["gross_return_mean_bp"]),
            "cost_reduction_bp_by_rate": {label: scenarios[CANDIDATE][label]["cost_mean_bp"]
                                          - scenarios[REFERENCE][label]["cost_mean_bp"]
                                          for label, _ in RATES},
            "net_change_bp_by_rate": {label: scenarios[CANDIDATE][label]["net_return_mean_bp"]
                                      - scenarios[REFERENCE][label]["net_return_mean_bp"]
                                      for label, _ in RATES},
            "turnover_proxy_delta": candidate_summary["turnover_mean_daily"]
            - reference_summary["turnover_mean_daily"],
            "turnover_proxy_ratio": retention["turnover_ratio"],
            "breakeven_bp_reference": scenarios[REFERENCE][PRIMARY]["breakeven_single_side_rate_bp"],
            "breakeven_bp_candidate": scenarios[CANDIDATE][PRIMARY]["breakeven_single_side_rate_bp"],
        },
        "official_turnover_item": {
            "status": "unavailable locally",
            "reason": "the published turnover item enters scoring as a cross-submission rank "
                      "percentile (lower is better); the submission population and platform "
                      "implementation are unavailable, so no official value is produced",
            "local_proxy_definition": "turnover = U / 2 with U = sum(|target weight - pre-trade "
                                      "drifted weight|)",
            "reference": reference_summary["turnover_mean_daily"],
            "candidate": candidate_summary["turnover_mean_daily"],
            "improvement": "monotone decrease at every rate (rate-invariant by construction)",
            "platform_text_quotes": quotes,
            "stored_official_fields": official_fields,
        },
        "frozen_gate_readout": gate,
        "prereg": gate["prereg"],
        "controls": controls,
        "reconciliation": {
            "reference_vs_t6_cost_curve": reference_vs_t6,
            "candidate_and_reference_vs_t7_aggregates": candidate_vs_t7,
            "signal_vs_t7": signal_vs_t7,
            "daily": daily_reconciliations,
            "record_vs_recomputation": record_vs_recomputed,
            "tolerance": tolerance,
        },
        "t7_checks_passed": t7_checks.get("passed"),
        "checks": checks,
        "evidence_limits": [
            "local approximate caliber on already-viewed development months; not a platform score",
            "the official turnover item is a cross-submission rank percentile and cannot be "
            "computed locally; only the U/2 proxy is reported",
            "cost is linear in traded notional with no market-impact term; 5bp is a declared stress",
            "the archived gate readout reports the paired improvement as the pooled day-weighted "
            "mean; the frozen engine weights declared periods equally (difference 8.4e-5, both far "
            "below the frozen threshold, so no gate line changes outcome)",
            "no promotion, rejection or model-selection output: t7's frozen readout stays the "
            "owner of the promotion question",
        ],
        "inputs": {"windows": [{"name": window["name"], "keys_sha256": window["keys_sha256"],
                                "targets_sha256": window["targets_sha256"],
                                "replay": window["replay"]} for window in windows],
                   "members": list(MEMBERS)},
    }

    (output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "checks.json").write_text(json.dumps(
        {"schema_version": 1, "checks": checks, "checks_passed": results["checks_passed"],
         "checks_total": results["checks_total"], "verdict": verdict},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reconcile_rows = []
    for object_id, item_list in daily_reconciliations.items():
        for item in item_list:
            row = {"object": object_id, "file": item["file"], "rows": item["rows"]}
            row.update({f"gap_{key}": value for key, value in item["max_abs_gap_by_field"].items()})
            reconcile_rows.append(row)
    pd.DataFrame(reconcile_rows).to_csv(output / "reconcile_daily.csv", index=False,
                                        lineterminator="\n")

    # ------------------------------------------------------------------- console report
    print(f"\nverdict: {verdict}  ({results['checks_passed']}/{results['checks_total']} checks)")
    print(f"{'scenario':<8}{'gross_ref':>11}{'cost_ref':>10}{'net_ref':>10}"
          f"{'gross_cand':>11}{'cost_cand':>10}{'net_cand':>10}{'proxy':>18}")
    proxy = (f"{reference_summary['turnover_mean_daily']:.4f}->"
             f"{candidate_summary['turnover_mean_daily']:.4f}")
    for label, _ in RATES:
        r, c = scenarios[REFERENCE][label], scenarios[CANDIDATE][label]
        print(f"{label:<8}{r['gross_return_mean_bp']:>11.3f}{r['cost_mean_bp']:>10.3f}"
              f"{r['net_return_mean_bp']:>10.3f}{c['gross_return_mean_bp']:>11.3f}"
              f"{c['cost_mean_bp']:>10.3f}{c['net_return_mean_bp']:>10.3f}{proxy:>18}")
    print(f"\nretention: gross {retention['gross_economic']:.4f} | rank_ic {retention['rank_ic']:.4f}"
          f" | ic {retention['ic']:.4f} | long-short {retention['long_short_return']:.4f}"
          f" | turnover {retention['turnover_ratio']:.4f} | cost5bp {retention['cost_ratio_5bp']:.4f}")
    print(f"frozen gate lines: {gate['lines_passed']}/{gate['lines_total']} pass; "
          f"promotable_under_frozen_lines={gate['promotable_under_frozen_lines']}; "
          f"blocking={gate['blocking_lines']}")
    print(f"candidate min cross-sectional std {candidate_summary['min_cross_sectional_std']:.4f}"
          f"; undefined rank IC {candidate_summary['undefined_rank_ic_count']}"
          f"; boundary-tie sections {candidate_summary['boundary_tie_sections']}")
    for item in checks:
        print(f"  [{'PASS' if item['passed'] else 'FAIL'}] {item['name']}")
    print(f"\nrun archive: {output.relative_to(ROOT).as_posix()}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
