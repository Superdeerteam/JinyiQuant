"""Development-only measurement of the IC-turnover frontier under causal EWMA smoothing.

Reads only the archived replay predictions of the ten development windows (all strictly before
the sealed interval). No training, no inference, no sealed read, no re-selection.

For every window this script

  1. loads ``windows/<w>/targets.parquet`` and each member's ``predictions.parquet``, inner-merges
     on ``(date, instrument)`` and asserts the key ordering is identical across every version,
  2. builds the equal-weight candidate blend (3 checkpoints) and incumbent blend (9 checkpoints),
  3. sweeps the causal EWMA alpha over the requested grid, evaluating every alpha with the project
     evaluation function (eval-v2: quantile 0.2, 252 periods, min_cross_section 3, undefined
     constant portfolios, single_side_rate 0.0 exactly as ``rung_candidate_compare.py`` uses it),
  4. pools the ten windows by trading day (``pd.concat`` of the daily tables, the same day-weighted
     aggregation the archived numbers use) and answers the three frontier questions.

Alpha semantics (load-bearing, verified by the reproduce gate)
--------------------------------------------------------------
``causal_ewma`` implements ``s_t = alpha * raw_t + (1 - alpha) * s_{t-1}`` per instrument with the
state reset every trading day, so alpha = 1 is the identity (modulo the operator's own per-timestamp
z-score, a positive affine map that leaves RankIC, IC, portfolio weights and turnover unchanged)
and alpha = 0 holds each instrument's first intraday value. The reference workflow never smooths
the incumbent, so every archived number labelled "alpha = 0" is the *unsmoothed* blend. This script
therefore carries two extra rows per window/model beside the grid: ``alpha = 1.0``
(``identity_unsmoothed`` = the raw blend, i.e. what the reference work calls alpha = 0) and
``alpha = 0.9111677303420447`` (``current_choice``). The grid row ``alpha = 0.0`` is the genuine
hold-first operator and is reported as such.

Usage
    /root/py312 scripts/alpha_frontier.py --mode sweep     # compute + cache the alpha sweep
    /root/py312 scripts/alpha_frontier.py --mode report    # write the four deliverables
    /root/py312 scripts/alpha_frontier.py                  # both
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _path in (str(REPO / "src"), str(REPO / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from compare_blend_replay import block_interval  # noqa: E402
from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from verify_turnover_change import causal_ewma  # noqa: E402

# --------------------------------------------------------------------------- constants

# rung_candidate_compare.py evaluates the unsmoothed and EWMA series with single_side_rate=0.0.
RATE = 0.0
CURRENT_ALPHA = 0.9111677303420447
IDENTITY_ALPHA = 1.0

# Exactly the requested grid: 0.00..0.85 step 0.05, then 0.90, then 0.91..0.99 step 0.01.
GRID = tuple(round(0.05 * i, 2) for i in range(18)) + tuple(round(0.90 + 0.01 * i, 2) for i in range(10))

CANDIDATE = ("cand-s11-264", "cand-s29-264", "cand-s47-264")
INCUMBENT = (
    "inc-tlob-s11", "inc-tlob-s29", "inc-tlob-s47",
    "inc-mlp-s11", "inc-mlp-s29", "inc-mlp-s47",
    "inc-patch-s11", "inc-patch-s29", "inc-patch-s47",
)
MODELS = {"candidate": CANDIDATE, "incumbent": INCUMBENT}
VERSIONS = CANDIDATE + INCUMBENT

# The five windows that overlap the original alpha study: the reproduce gate.
GATE_WINDOWS = ("nov2023", "jan2024", "mar2024", "apr2024", "jun2024")
# The two archived levels come from two different evaluator calibers, and the gate has to say so
# instead of pretending they are comparable:
#   candidate 1.0996  -- per-window replay caliber, the same one this script computes;
#   incumbent 1.10195 -- continuous 104-day object caliber (registered as `local_proxy`
#                        "drift-aware pre-trade weights" in configs/b4_promotion_prereg_*.json).
# A per-window replay restarts the book at every window, so it drops exactly the window-boundary
# rebalances. The size of that drop is MEASURED from the archived daily candidate series below,
# never assumed.
GATE = {
    ("candidate", "current_choice"): {"rank_ic": 0.078845, "turnover": 1.0996,
                                      "caliber": "per_window",
                                      "label": "candidate blend at alpha = 0.9111677303420447"},
    ("incumbent", "identity_unsmoothed"): {"rank_ic": 0.070687, "turnover": 1.10195,
                                           "caliber": "continuous_object",
                                           "label": "incumbent blend at alpha = 0 (unsmoothed)"},
}
GATE_TOLERANCE = 0.0005

# Archived daily candidate series from the original 441-day study: the same object this script
# rebuilds, evaluated in the continuous-object caliber. Used to measure the caliber offset.
ARCHIVED_DAILY = {
    "current_choice": "/root/runs/archived-441/daily_0bp_ewma_match.csv",
    "identity_unsmoothed": "/root/runs/archived-441/daily_0bp_blend.csv",
}
CALIBER_IC_TOLERANCE = 1e-5

# Fixed turnovers to compare across windows. 1.20 is above the candidate curve's reach in most
# windows (the raw blend tops out near 1.17-1.19), so the grid stops at 1.15; the per-window
# maximum turnover is reported alongside.
FIXED_TURNOVERS = (1.00, 1.05, 1.10, 1.15)
TARGET_TURNOVERS = (1.05, 1.00)

# Pre-declared materiality thresholds, fixed before looking at any result.
MATERIAL_RANK_IC_SPREAD = 0.005   # max-min of RankIC across windows at a fixed turnover
MATERIAL_CHEAP_LOSS = 0.001       # |pooled RankIC loss| of a fixed turnover target
MATERIAL_TURNOVER_SD = 0.02       # incumbent turnover sd across windows

DAILY_METRICS = ("rank_ic", "turnover", "long_short_return")
BOOTSTRAP_BLOCK = 5
BOOTSTRAP_DRAWS = 20000


# --------------------------------------------------------------------------- primitives

def evaluation_config(rate: float | None = RATE) -> EvaluationConfig:
    return EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=rate)


def evaluate_scores(keys: pd.DataFrame, targets: pd.DataFrame, scores: np.ndarray,
                    rate: float | None = RATE) -> tuple[dict, pd.DataFrame]:
    predictions = keys.assign(score=scores)[["date", "instrument", "score"]]
    metrics, daily, _ = evaluate(predictions, targets, evaluation_config(rate),
                                 return_sections=True, undefined_constant_portfolios=True)
    return metrics, daily


def smooth(keys: pd.DataFrame, blend: np.ndarray, alpha: float) -> np.ndarray:
    """alpha = 1.0 is this operator's exact identity, so hand back the raw blend there."""
    if alpha >= IDENTITY_ALPHA:
        return np.asarray(blend, dtype=float)
    return causal_ewma(keys, blend, alpha)


def alpha_kind(alpha: float) -> str:
    if alpha >= IDENTITY_ALPHA:
        return "identity_unsmoothed"
    if abs(alpha - CURRENT_ALPHA) < 1e-15:
        return "current_choice"
    return "grid"


def alphas() -> list[float]:
    return sorted(set(GRID) | {CURRENT_ALPHA, IDENTITY_ALPHA})


# --------------------------------------------------------------------------- loading

def key_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exact dtype-insensitive representation of a (date, instrument) key sequence."""
    dates = frame["date"].to_numpy(dtype="datetime64[ns]").astype("int64")
    instruments = frame["instrument"].to_numpy(dtype=str)
    return dates, instruments


def same_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right):
        return False
    left_dates, left_instruments = key_arrays(left)
    right_dates, right_instruments = key_arrays(right)
    return bool(np.array_equal(left_dates, right_dates)
                and np.array_equal(left_instruments, right_instruments))


def load_window(run_dir: Path, window: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Load one window and return (keys, targets, {model: blend}) with hard alignment checks."""
    base = run_dir / "windows" / window
    targets = pd.read_parquet(base / "targets.parquet")
    targets["date"] = pd.to_datetime(targets["date"]).astype("datetime64[ns]")
    targets = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    keys = targets[["date", "instrument"]].reset_index(drop=True)
    if len(keys) == 0:
        raise SystemExit(f"{window}: no target keys")
    if keys.duplicated().any():
        raise SystemExit(f"{window}: duplicate target keys")

    scores: dict[str, np.ndarray] = {}
    reference: pd.DataFrame | None = None
    for version in VERSIONS:
        predictions = pd.read_parquet(base / version / "predictions.parquet")
        if not {"date", "instrument", "score"}.issubset(predictions.columns):
            raise SystemExit(f"{window}/{version}: predictions missing date/instrument/score")
        predictions["date"] = pd.to_datetime(predictions["date"]).astype("datetime64[ns]")
        merged = predictions.merge(keys, on=["date", "instrument"], how="inner",
                                   validate="one_to_one")
        if merged.duplicated(["date", "instrument"]).any():
            raise SystemExit(f"{window}/{version}: duplicate merged keys")
        merged = merged.sort_values(["date", "instrument"]).reset_index(drop=True)
        merged_keys = merged[["date", "instrument"]]
        if not same_keys(merged_keys, keys):
            raise SystemExit(f"{window}/{version}: coverage or key-order mismatch "
                             f"({len(merged)} rows vs {len(keys)} targets)")
        values = merged["score"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SystemExit(f"{window}/{version}: non-finite scores")
        scores[version] = values
        if reference is None:
            reference = merged_keys
        elif not same_keys(merged_keys, reference):
            raise SystemExit(f"{window}/{version}: key ordering differs between versions")

    blends = {model: np.mean(np.stack([scores[v] for v in members]), axis=0)
              for model, members in MODELS.items()}
    return keys, targets[["date", "instrument", "target"]], blends


# --------------------------------------------------------------------------- sweep

def sweep(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    plan = json.loads((run_dir / "plan.json").read_text())
    windows = [entry["name"] for entry in plan["windows"]]
    cache = run_dir / "analysis"
    cache.mkdir(parents=True, exist_ok=True)
    grid = alphas()
    print(f"windows={len(windows)} versions={len(VERSIONS)} alphas={len(grid)} rate={RATE}",
          flush=True)

    for window in windows:
        metrics_path = cache / f"metrics_{window}.parquet"
        daily_path = cache / f"daily_{window}.parquet"
        if metrics_path.is_file() and daily_path.is_file() and not args.force:
            print(f"[{window}] cached, skipping", flush=True)
            continue
        keys, targets, blends = load_window(run_dir, window)
        days = int(keys["date"].dt.normalize().nunique())
        print(f"[{window}] keys={len(keys)} days={days}", flush=True)
        metric_rows, daily_rows = [], []
        for model, blend in blends.items():
            for alpha in grid:
                smoothed = smooth(keys, blend, alpha)
                metrics, daily = evaluate_scores(keys, targets, smoothed)
                kind = alpha_kind(alpha)
                for column in ("rank_ic_mean", "ic_mean", "ic_ir", "turnover_mean",
                               "long_short_return_mean"):
                    if column in metrics and metrics[column] is not None:
                        if not np.isfinite(metrics[column]):
                            raise SystemExit(f"{window}/{model}/{alpha}: non-finite {column}")
                metric_rows.append({
                    "window": window, "model": model, "alpha": float(alpha), "kind": kind,
                    "rank_ic_mean": metrics["rank_ic_mean"], "ic_mean": metrics["ic_mean"],
                    "ic_ir": metrics["ic_ir"], "turnover_mean": metrics["turnover_mean"],
                    "long_short_mean": metrics["long_short_return_mean"],
                    "rank_ic_ir": metrics["rank_ic_ir"],
                    "days": int(metrics["day_count"]),
                    "cross_sections": int(metrics["cross_section_count"]),
                    "samples": int(metrics["sample_count"]),
                    "undefined_turnover": int(metrics["undefined_turnover_count"]),
                })
                frame = daily[["date", *DAILY_METRICS]].copy()
                frame.insert(0, "window", window)
                frame.insert(1, "model", model)
                frame.insert(2, "alpha", float(alpha))
                frame.insert(3, "kind", kind)
                daily_rows.append(frame)
        pd.DataFrame(metric_rows).to_parquet(metrics_path, index=False)
        pd.concat(daily_rows, ignore_index=True).to_parquet(daily_path, index=False)
        print(f"[{window}] swept {len(grid) * len(MODELS)} curves", flush=True)


# --------------------------------------------------------------------------- analysis helpers

def load_cache(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = run_dir / "analysis"
    metric_files = sorted(cache.glob("metrics_*.parquet"))
    if not metric_files:
        raise SystemExit(f"no cached sweep under {cache}; run --mode sweep first")
    metrics = pd.concat([pd.read_parquet(p) for p in metric_files], ignore_index=True)
    daily = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("daily_*.parquet"))],
                      ignore_index=True)
    return metrics, daily


def curve(metrics: pd.DataFrame, window: str, model: str) -> pd.DataFrame:
    frame = metrics[(metrics.window == window) & (metrics.model == model)]
    return frame.sort_values("alpha").reset_index(drop=True)


def point(metrics: pd.DataFrame, window: str, model: str, kind: str) -> pd.Series | None:
    frame = metrics[(metrics.window == window) & (metrics.model == model) & (metrics.kind == kind)]
    if len(frame) != 1:
        return None
    return frame.iloc[0]


def pooled_daily_series(daily: pd.DataFrame, windows, model: str, kind: str,
                        column: str = "rank_ic") -> pd.Series:
    """Concatenate every window's daily series: the day-weighted pooling the archive uses."""
    frame = daily[(daily.model == model) & (daily.kind == kind) & daily.window.isin(list(windows))]
    frame = frame[["window", "date", column]].dropna()
    if frame.duplicated(["window", "date"]).any():
        raise SystemExit(f"duplicate daily rows for {model}/{kind}")
    return frame.reset_index(drop=True)[column]


def pooled_mean(metrics: pd.DataFrame, windows, model: str, kind: str, column: str) -> float | None:
    values = []
    for window in windows:
        row = point(metrics, window, model, kind)
        if row is None:
            continue
        values.append(float(row[column]))
    return float(np.mean(values)) if values else None


def gate_daily_frame(daily: pd.DataFrame, windows, model: str, kind: str) -> pd.DataFrame:
    frame = daily[(daily.model == model) & (daily.kind == kind)
                  & daily.window.isin(list(windows))].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[["window", "date", "rank_ic", "turnover"]].reset_index(drop=True)
    if frame.duplicated(["window", "date"]).any():
        raise SystemExit(f"duplicate daily rows for {model}/{kind}")
    if frame[["rank_ic", "turnover"]].isna().any().any():
        raise SystemExit(f"undefined daily metric rows for {model}/{kind}")
    return frame


def caliber_reconciliation(daily: pd.DataFrame, windows) -> dict:
    """Measure, do not assume, the offset between the archive's continuous-object caliber and
    this script's per-window caliber. IC has to match day by day; if it does, any turnover gap is
    accounting and it can only sit on window-boundary rebalances (at most one per window after the
    first). The measured offset then becomes a prediction we carry to the incumbent."""
    report = {"archived_daily": dict(ARCHIVED_DAILY), "levels": {}, "single_offset": None,
              "note": "archive = one continuous object; this replay restarts the book per window, "
                      "so only the window-boundary rebalances can differ"}
    offsets = {}
    for kind, path in ARCHIVED_DAILY.items():
        archived = pd.read_csv(path)
        archived["date"] = pd.to_datetime(archived["date"])
        if archived["date"].duplicated().any():
            raise SystemExit("duplicate dates in archived series " + path)
        mine = gate_daily_frame(daily, windows, "candidate", kind)
        merged = archived[["date", "rank_ic", "turnover"]].merge(
            mine.rename(columns={"rank_ic": "rank_ic_mine", "turnover": "turnover_mine"}),
            on="date", how="inner", validate="one_to_one")
        if len(merged) != len(mine):
            raise SystemExit("archived series does not cover the same days: "
                             + str(len(merged)) + " of " + str(len(mine)))
        delta = merged["turnover"] - merged["turnover_mine"]
        ic_gap = (merged["rank_ic"] - merged["rank_ic_mine"]).abs()
        differing = merged.loc[delta.abs() > 1e-9]
        allowed = max(len(windows) - 1, 1)
        offsets[kind] = float(delta.mean())
        report["levels"][kind] = {
            "days": int(len(merged)),
            "rank_ic_archived": float(merged["rank_ic"].mean()),
            "rank_ic_mine": float(merged["rank_ic_mine"].mean()),
            "rank_ic_max_abs_daily_gap": float(ic_gap.max()),
            "turnover_archived": float(merged["turnover"].mean()),
            "turnover_mine": float(merged["turnover_mine"].mean()),
            "turnover_offset": float(delta.mean()),
            "differing_days": int(len(differing)),
            "differing_dates": [str(value.date()) for value in differing["date"]],
            "differing_turnover_sum": float(delta.sum()),
            "max_differing_days_allowed": int(allowed),
            "ic_pipeline_identical": bool(ic_gap.max() <= CALIBER_IC_TOLERANCE),
            "only_boundary_days_differ": bool(len(differing) <= allowed),
        }
    report["single_offset"] = offsets.get("current_choice")
    report["passed"] = bool(all(level["ic_pipeline_identical"]
                                and level["only_boundary_days_differ"]
                                for level in report["levels"].values()))
    return report


def alpha_at_turnover(rows: pd.DataFrame, target: float) -> float | None:
    frame = rows[np.isfinite(rows.turnover_mean.to_numpy(dtype=float))]
    frame = frame.sort_values("turnover_mean")
    turnover = frame.turnover_mean.to_numpy(dtype=float)
    alpha = frame.alpha.to_numpy(dtype=float)
    if len(turnover) < 2 or target < turnover[0] or target > turnover[-1]:
        return None
    return float(np.interp(target, turnover, alpha))


def jsonable(value):
    """Recursively convert numpy scalars to Python and non-finite floats to null."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def rank_ic_at_turnover(rows: pd.DataFrame, target: float) -> float | None:
    frame = rows[np.isfinite(rows.turnover_mean.to_numpy(dtype=float))]
    frame = frame.sort_values("turnover_mean")
    turnover = frame.turnover_mean.to_numpy(dtype=float)
    rank_ic = frame.rank_ic_mean.to_numpy(dtype=float)
    if len(turnover) < 2 or target < turnover[0] or target > turnover[-1]:
        return None
    return float(np.interp(target, turnover, rank_ic))


# --------------------------------------------------------------------------- report

def build_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run_dir / "plan.json").read_text())
    windows = [entry["name"] for entry in plan["windows"]]
    metrics, daily = load_cache(run_dir)
    grid = alphas()
    for window in windows:
        for model in MODELS:
            rows = curve(metrics, window, model)
            if len(rows) != len(grid):
                raise SystemExit(f"{window}/{model}: cached {len(rows)} curves, "
                                 f"expected {len(grid)}")

    # --- 1. frontier table -------------------------------------------------------
    frontier = metrics[["window", "model", "alpha", "rank_ic_mean", "ic_mean", "ic_ir",
                        "turnover_mean", "long_short_mean"]]
    frontier = frontier.sort_values(["window", "model", "alpha"]).reset_index(drop=True)
    frontier.to_csv(out / "frontier.csv", index=False)

    # --- 2. identity-operator self check ----------------------------------------
    # alpha = 1.0 must be the exact identity: re-evaluate one window's raw blend through
    # causal_ewma at 1.0 and confirm the portfolio metrics are unchanged.
    keys, targets, blends = load_window(run_dir, windows[0])
    identity_metrics, _ = evaluate_scores(keys, targets, causal_ewma(keys, blends["candidate"], 1.0))
    raw_row = point(metrics, windows[0], "candidate", "identity_unsmoothed")
    identity_check = {
        "window": windows[0],
        "rank_ic_delta": abs(identity_metrics["rank_ic_mean"] - float(raw_row["rank_ic_mean"])),
        "turnover_delta": abs(identity_metrics["turnover_mean"] - float(raw_row["turnover_mean"])),
        "note": "alpha=1.0 through causal_ewma vs the raw blend; z-scoring is a per-timestamp "
                "positive affine map, so both must agree exactly",
    }
    identity_check["passed"] = bool(identity_check["rank_ic_delta"] < 1e-9
                                    and identity_check["turnover_delta"] < 1e-9)
    del keys, targets, blends

    # --- 3. reproduce gate -------------------------------------------------------
    caliber = caliber_reconciliation(daily, GATE_WINDOWS)
    gate = {"windows": list(GATE_WINDOWS), "tolerance": GATE_TOLERANCE,
            "primary_aggregation": "pooled daily, day-weighted (what the archive reports)",
            "caliber_reconciliation": caliber,
            "checks": []}
    for (model, kind), expected in GATE.items():
        pooled_rank_ic = float(pooled_daily_series(daily, GATE_WINDOWS, model, kind).mean())
        pooled_turnover = float(pooled_daily_series(daily, GATE_WINDOWS, model, kind,
                                                    "turnover").mean())
        window_rank_ic = pooled_mean(metrics, GATE_WINDOWS, model, kind, "rank_ic_mean")
        window_turnover = pooled_mean(metrics, GATE_WINDOWS, model, kind, "turnover_mean")
        gap_ic = abs(pooled_rank_ic - expected["rank_ic"])
        offset = (float(caliber["single_offset"] or 0.0)
                  if expected["caliber"] == "continuous_object" else 0.0)
        expected_turnover = expected["turnover"] - offset
        gap_turnover = abs(pooled_turnover - expected_turnover)
        gate["checks"].append({
            "label": expected["label"], "model": model, "kind": kind,
            "rank_ic": pooled_rank_ic, "expected_rank_ic": expected["rank_ic"],
            "rank_ic_gap": gap_ic,
            "turnover": pooled_turnover, "expected_turnover": expected["turnover"],
            "expected_turnover_in_this_caliber": expected_turnover,
            "caliber": expected["caliber"], "caliber_offset_applied": offset,
            "turnover_gap": gap_turnover,
            "diagnostic_window_mean_rank_ic": window_rank_ic,
            "diagnostic_window_mean_turnover": window_turnover,
            "passed": bool(gap_ic <= GATE_TOLERANCE and gap_turnover <= GATE_TOLERANCE),
        })
    gate["passed"] = (all(check["passed"] for check in gate["checks"])
                      and identity_check["passed"] and caliber["passed"])
    gate["days"] = int(len(pooled_daily_series(daily, GATE_WINDOWS, "candidate", "current_choice")))
    if not gate["passed"]:
        (out / "gate_failure.json").write_text(json.dumps(
            {"gate": gate, "identity_check": identity_check}, indent=2) + "\n")
        print("REPRODUCE GATE FAILED")
        for check in gate["checks"]:
            print(f"  {check['label']}: rank_ic={check['rank_ic']:.6f} "
                  f"(expected {check['expected_rank_ic']}, gap {check['rank_ic_gap']:.6f}) "
                  f"turnover={check['turnover']:.5f} "
                  f"(expected {check['expected_turnover']}, gap {check['turnover_gap']:.5f})")
        print(f"  identity check: {identity_check}")
        print("  caliber reconciliation: " + json.dumps(caliber, indent=2))
        return 2

    (out / "gate.json").write_text(json.dumps(
        {"gate": gate, "identity_check": identity_check}, indent=2) + chr(10))
    print("REPRODUCE GATE PASSED (%d days, tolerance %s)" % (gate["days"], GATE_TOLERANCE))
    for check in gate["checks"]:
        print("  %s: rank_ic gap %.2e | turnover %.5f vs %.5f in caliber %s (gap %.2e)"
              % (check["label"], check["rank_ic_gap"], check["turnover"],
                 check["expected_turnover_in_this_caliber"], check["caliber"],
                 check["turnover_gap"]))
    for kind, level in caliber["levels"].items():
        print("  archive caliber [%s]: rank_ic max daily gap %.2e | turnover offset %+.6f "
              "on %d/%d days %s"
              % (kind, level["rank_ic_max_abs_daily_gap"], level["turnover_offset"],
                 level["differing_days"], level["days"], level["differing_dates"]))

    # --- 4. Q1: does the frontier shift across windows? --------------------------
    q1_curves, q1_fixed = {}, {}
    for window in windows:
        rows = curve(metrics, window, "candidate")
        incumbent = point(metrics, window, "incumbent", "identity_unsmoothed")
        q1_curves[window] = {
            "alpha": [float(v) for v in rows.alpha],
            "turnover": [float(v) for v in rows.turnover_mean],
            "rank_ic": [float(v) for v in rows.rank_ic_mean],
            "ic_mean": [float(v) for v in rows.ic_mean],
            "incumbent_turnover": float(incumbent["turnover_mean"]),
            "incumbent_rank_ic": float(incumbent["rank_ic_mean"]),
        }
        q1_fixed[window] = {f"{t:.2f}": rank_ic_at_turnover(rows, t) for t in FIXED_TURNOVERS}

    q1_spread = {}
    for target in FIXED_TURNOVERS:
        key = f"{target:.2f}"
        values = np.array([q1_fixed[w][key] for w in windows if q1_fixed[w][key] is not None],
                          dtype=float)
        q1_spread[key] = {
            "n_windows": int(len(values)),
            "mean": float(values.mean()) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "range": float(values.max() - values.min()) if len(values) else None,
            "by_window": {w: q1_fixed[w][key] for w in windows},
        }

    # A pure level shift moves every fixed-turnover point by the same amount; a shape change
    # does not. Remove each window's own level and look at what survives.
    residuals = {}
    for window in windows:
        values = [q1_fixed[window][f"{t:.2f}"] for t in FIXED_TURNOVERS]
        if any(value is None for value in values):
            continue
        level = float(np.mean(values))
        residuals[window] = {f"{t:.2f}": value - level
                             for t, value in zip(FIXED_TURNOVERS, values)}
    residual_spread = {}
    for target in FIXED_TURNOVERS:
        key = f"{target:.2f}"
        values = np.array([residuals[w][key] for w in residuals], dtype=float)
        residual_spread[key] = {
            "n_windows": int(len(values)),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "range": float(values.max() - values.min()) if len(values) else None,
        }

    displacement = []
    for window in windows:
        rows = curve(metrics, window, "candidate")
        incumbent = point(metrics, window, "incumbent", "identity_unsmoothed")
        turnover = float(incumbent["turnover_mean"])
        value = rank_ic_at_turnover(rows, turnover)
        displacement.append({
            "window": window, "turnover": turnover,
            "candidate_rank_ic_at_that_turnover": value,
            "incumbent_rank_ic": float(incumbent["rank_ic_mean"]),
            "gap": None if value is None else value - float(incumbent["rank_ic_mean"]),
        })
    gaps = np.array([row["gap"] for row in displacement if row["gap"] is not None], dtype=float)
    max_range = max(float(entry["range"]) for entry in q1_spread.values()
                    if entry["range"] is not None)
    q1_verdict = ("regime effect present" if max_range >= MATERIAL_RANK_IC_SPREAD
                  else "no material regime effect")
    q1 = {
        "question": "Does the IC-turnover frontier shift across windows?",
        "materiality_threshold_range": MATERIAL_RANK_IC_SPREAD,
        "curves": q1_curves,
        "rank_ic_at_fixed_turnover": q1_fixed,
        "across_window_spread_at_fixed_turnover": q1_spread,
        "level_removed_residual_spread_at_fixed_turnover": residual_spread,
        "residual_note": "each window's fixed-turnover points minus that window's own mean over "
                         "them; a pure vertical displacement leaves zeros here",
        "max_turnover_by_window": {w: float(point(metrics, w, "candidate",
                                                 "identity_unsmoothed")["turnover_mean"])
                                   for w in windows},
        "incumbent_operating_turnover_displacement": displacement,
        "mean_gap_candidate_to_incumbent_at_incumbent_turnover":
            float(gaps.mean()) if len(gaps) else None,
        "max_across_window_range": max_range,
        "verdict": q1_verdict,
    }

    # --- 5. Q2: cost of a fixed absolute turnover target -------------------------
    # alpha* per window is interpolated from the cached curve; the RankIC at alpha* is then
    # evaluated exactly rather than interpolated, so the reported value is a real measurement.
    solved = {}
    for window in windows:
        keys, targets, blends = load_window(run_dir, window)
        solved[window] = {}
        for model in ("candidate",):
            rows = curve(metrics, window, model)
            solved[window][model] = {}
            for target in TARGET_TURNOVERS:
                alpha_star = alpha_at_turnover(rows, target)
                entry = {"alpha_star": alpha_star, "target_turnover": target}
                if alpha_star is None:
                    entry.update({"rank_ic_at_target": None, "turnover_at_target_alpha": None,
                                  "daily": None})
                else:
                    star_metrics, star_daily = evaluate_scores(
                        keys, targets, smooth(keys, blends[model], alpha_star))
                    entry.update({
                        "rank_ic_at_target": star_metrics["rank_ic_mean"],
                        "turnover_at_target_alpha": star_metrics["turnover_mean"],
                        "long_short_at_target": star_metrics["long_short_return_mean"],
                        "daily": star_daily[["date", "rank_ic", "turnover"]],
                    })
                solved[window][model][f"{target:.2f}"] = entry
        del keys, targets, blends

    q2_windows = []
    for window in windows:
        row = {"window": window}
        for target in TARGET_TURNOVERS:
            key = f"{target:.2f}"
            entry = solved[window]["candidate"][key]
            row[key] = {
                "alpha_star": entry["alpha_star"],
                "rank_ic_at_target": entry["rank_ic_at_target"],
                "realized_turnover": entry["turnover_at_target_alpha"],
            }
        current = point(metrics, window, "candidate", "current_choice")
        unsmoothed = point(metrics, window, "candidate", "identity_unsmoothed")
        row["alpha_current"] = {"alpha": CURRENT_ALPHA,
                                "rank_ic": float(current["rank_ic_mean"]),
                                "turnover": float(current["turnover_mean"])}
        row["alpha_zero_unsmoothed"] = {"rank_ic": float(unsmoothed["rank_ic_mean"]),
                                        "turnover": float(unsmoothed["turnover_mean"])}
        freeze = metrics[(metrics.window == window) & (metrics.model == "candidate")
                         & (metrics.alpha == 0.0)].iloc[0]
        row["alpha_zero_operator_hold_first"] = {"rank_ic": float(freeze["rank_ic_mean"]),
                                                 "turnover": float(freeze["turnover_mean"])}
        q2_windows.append(row)

    current_mask = ((daily.model == "candidate") & (daily.kind == "current_choice")
                    & daily.window.isin(list(windows)))
    current = daily.loc[current_mask, ["window", "date", "rank_ic"]].dropna()
    current = current.rename(columns={"rank_ic": "current_rank_ic"})
    current = current.reset_index(drop=True)
    if current.duplicated(["window", "date"]).any():
        raise SystemExit("duplicate current-alpha daily rows")
    q2_boot = {}
    for target in TARGET_TURNOVERS:
        key = f"{target:.2f}"
        parts = []
        for window in windows:
            entry = solved[window]["candidate"][key]
            if entry["daily"] is None:
                continue
            frame = entry["daily"].copy()
            frame.insert(0, "window", window)
            parts.append(frame)
        if not parts:
            q2_boot[key] = {"note": "target unreachable in every window", "reachable": False}
            continue
        star = pd.concat(parts, ignore_index=True).dropna(subset=["rank_ic"])
        merged = star[["window", "date", "rank_ic"]].merge(
            current, on=["window", "date"], how="inner")
        if merged.empty:
            q2_boot[key] = {"note": "no overlapping days"}
            continue
        delta = (merged["rank_ic"].to_numpy(dtype=float)
                 - merged["current_rank_ic"].to_numpy(dtype=float))
        interval = block_interval(delta, block=BOOTSTRAP_BLOCK, draws=BOOTSTRAP_DRAWS)
        pooled_star = float(merged["rank_ic"].mean())
        pooled_current = float(merged["current_rank_ic"].mean())
        q2_boot[key] = {
            "reachable": True,
            "pooled_rank_ic_at_target_alpha": pooled_star,
            "pooled_rank_ic_at_current_alpha": pooled_current,
            "pooled_delta": pooled_star - pooled_current,
            "delta_mean": interval["mean"],
            "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "days": interval["days"], "block": BOOTSTRAP_BLOCK, "draws": BOOTSTRAP_DRAWS,
            "ci_excludes_zero": bool(interval["ci_low"] is not None
                                     and (interval["ci_low"] > 0 or interval["ci_high"] < 0)),
        }
        q2_boot[key]["cheap"] = bool(q2_boot[key]["pooled_delta"] >= -MATERIAL_CHEAP_LOSS)
        alphas_star = [row[key]["alpha_star"] for row in q2_windows
                       if row[key]["alpha_star"] is not None]
        q2_boot[key]["alpha_star_mean"] = float(np.mean(alphas_star)) if alphas_star else None
        q2_boot[key]["alpha_star_min"] = float(np.min(alphas_star)) if alphas_star else None
        q2_boot[key]["alpha_star_max"] = float(np.max(alphas_star)) if alphas_star else None
        q2_boot[key]["alpha_star_sd"] = (float(np.std(alphas_star, ddof=1))
                                         if len(alphas_star) > 1 else None)

    keys_of = {t: f"{t:.2f}" for t in TARGET_TURNOVERS}
    reachable = {t: bool(q2_boot.get(keys_of[t], {}).get("reachable")) for t in TARGET_TURNOVERS}
    cheap = {t: bool(q2_boot.get(keys_of[t], {}).get("cheap")) for t in TARGET_TURNOVERS}
    cheap_targets = [t for t in TARGET_TURNOVERS if reachable[t] and cheap[t]]
    if len(cheap_targets) == len(TARGET_TURNOVERS):
        q2_verdict = ("every tested fixed turnover target is cheap (pooled RankIC loss below "
                      "%.3f) and should replace matching the incumbent's turnover"
                      % MATERIAL_CHEAP_LOSS)
    elif cheap_targets:
        q2_verdict = ("a fixed turnover target is cheap for %s only"
                      % ", ".join("%.2f" % t for t in cheap_targets))
    else:
        q2_verdict = "no tested fixed turnover target is cheap enough to replace matching"
    q2 = {
        "question": "What does a fixed absolute turnover target cost?",
        "materiality_threshold_loss": MATERIAL_CHEAP_LOSS,
        "targets": list(TARGET_TURNOVERS),
        "per_window": q2_windows,
        "pooled_bootstrap": q2_boot,
        "reachable": {keys_of[t]: reachable[t] for t in TARGET_TURNOVERS},
        "cheap": {keys_of[t]: cheap[t] for t in TARGET_TURNOVERS},
        "cheap_targets": cheap_targets,
        "verdict": q2_verdict,
    }

    # --- 6. Q3: how much does the incumbent's own turnover drift? ----------------
    def drift(model: str, kind: str) -> dict:
        values = np.array([float(point(metrics, w, model, kind)["turnover_mean"])
                           for w in windows], dtype=float)
        rank_ic = np.array([float(point(metrics, w, model, kind)["rank_ic_mean"])
                            for w in windows], dtype=float)
        return {
            "by_window": {w: float(v) for w, v in zip(windows, values)},
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max()),
            "range": float(values.max() - values.min()),
            "rank_ic_by_window": {w: float(v) for w, v in zip(windows, rank_ic)},
            "rank_ic_mean": float(rank_ic.mean()), "rank_ic_sd": float(rank_ic.std(ddof=1)),
        }

    incumbent_turnover = drift("incumbent", "identity_unsmoothed")
    candidate_current = drift("candidate", "current_choice")
    candidate_unsmoothed = drift("candidate", "identity_unsmoothed")
    ratio = (float(incumbent_turnover["sd"]) / float(incumbent_turnover["mean"])
             if incumbent_turnover["mean"] else None)
    q3_verdict = ("the incumbent's turnover was a moving target, not a stable one"
                  if incumbent_turnover["sd"] >= MATERIAL_TURNOVER_SD
                  else "the incumbent's turnover was a stable target")
    q3 = {
        "question": "How much does the incumbent's own turnover drift across windows?",
        "materiality_threshold_sd": MATERIAL_TURNOVER_SD,
        "alpha_zero_note": "alpha = 0 here is the unsmoothed incumbent blend, the quantity the "
                           "candidate alpha was matched to; the alpha=0.0 hold-first operator row "
                           "is reported separately in per_window_hold_first",
        "incumbent_at_alpha_zero_unsmoothed": incumbent_turnover,
        "candidate_at_current_alpha": candidate_current,
        "candidate_at_alpha_zero_unsmoothed": candidate_unsmoothed,
        "incumbent_relative_sd": ratio,
        "per_window_hold_first": {w: {
            "incumbent_turnover": float(metrics[(metrics.window == w)
                                                & (metrics.model == "incumbent")
                                                & (metrics.alpha == 0.0)].iloc[0]["turnover_mean"]),
            "candidate_turnover": float(metrics[(metrics.window == w)
                                                & (metrics.model == "candidate")
                                                & (metrics.alpha == 0.0)].iloc[0]["turnover_mean"]),
        } for w in windows},
        "verdict": q3_verdict,
    }

    # --- 7. pooled summaries -----------------------------------------------------
    pooled_block = {}

    def pooled_for_alpha(model: str, alpha: float, column: str) -> pd.Series:
        frame = daily[(daily.model == model) & np.isclose(daily.alpha.to_numpy(dtype=float), alpha,
                                                          rtol=0.0, atol=1e-12)]
        frame = frame[["window", "date", column]].dropna()
        if frame.duplicated(["window", "date"]).any():
            raise SystemExit(f"duplicate daily rows for {model}@{alpha}")
        return frame[column]

    for model in MODELS:
        pooled_block[model] = {}
        for alpha in grid:
            rank_ic = pooled_for_alpha(model, alpha, "rank_ic")
            turnover = pooled_for_alpha(model, alpha, "turnover")
            long_short = pooled_for_alpha(model, alpha, "long_short_return")
            frame = metrics[(metrics.model == model)
                            & np.isclose(metrics.alpha.to_numpy(dtype=float), alpha,
                                         rtol=0.0, atol=1e-12)]
            pooled_block[model][f"{alpha:.16g}"] = {
                "kind": alpha_kind(alpha),
                "rank_ic_mean": float(rank_ic.mean()),
                "turnover_mean": float(turnover.mean()),
                "long_short_mean": float(long_short.mean()),
                "ic_mean": float(frame["ic_mean"].mean()),
                "ic_ir": float(frame["ic_ir"].mean()),
                "days": int(len(rank_ic)),
                "aggregation": "pooled daily, day-weighted",
            }

    window_meta = {}
    for window in windows:
        sample = metrics[(metrics.window == window)].iloc[0]
        window_meta[window] = {
            "days": int(sample["days"]), "samples": int(sample["samples"]),
            "cross_sections": int(sample["cross_sections"]),
        }

    # --- 8. frontier.json --------------------------------------------------------
    frontier_json = {
        "schema_version": 1,
        "experiment_id": plan.get("experiment_id"),
        "run_id": run_dir.name,
        "plan": str(run_dir / "plan.json"),
        "evaluation": {
            "version": "eval-v2 (local re-implementation)",
            "quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
            "constant_portfolios": "undefined", "single_side_rate": RATE,
            "aggregation": "daily equal timestamp weights; pooled across windows by trading day",
        },
        "alpha_semantics": {
            "operator": "causal_ewma: s_t = alpha * raw_t + (1 - alpha) * s_{t-1}, per instrument, "
                        "state reset every trading day, plus a per-timestamp cross-sectional z-score",
            "alpha_1_0": "identity (raw blend); the reference workflow's 'alpha = 0' incumbent",
            "alpha_0_0": "hold each instrument's first intraday value (the grid's low-turnover end)",
            "current_choice": CURRENT_ALPHA,
            "rows_labelled_identity_unsmoothed": "the unsmoothed blend the reference work calls "
                                                 "alpha = 0",
        },
        "models": {model: list(members) for model, members in MODELS.items()},
        "blend": "equal weight per member",
        "grid": [{"alpha": float(a), "kind": alpha_kind(a)} for a in grid],
        "windows": window_meta,
        "window_bounds": {entry["name"]: {"start": entry["start"], "end": entry["end"]}
                          for entry in plan["windows"]},
        "frontier": json.loads(frontier.to_json(orient="records")),
        "pooled": pooled_block,
        "gate": gate,
        "identity_operator_check": identity_check,
        "q1": q1, "q2": q2, "q3": q3,
        "materiality_thresholds": {
            "rank_ic_spread_range": MATERIAL_RANK_IC_SPREAD,
            "cheap_loss": MATERIAL_CHEAP_LOSS,
            "turnover_sd": MATERIAL_TURNOVER_SD,
        },
        "bootstrap": {"scheme": "circular moving-block", "block": BOOTSTRAP_BLOCK,
                      "draws": BOOTSTRAP_DRAWS},
        "per_day_series_cache": {"path": str(run_dir / "analysis" / "daily_<window>.parquet"),
                                 "columns": ["window", "model", "alpha", "kind", "date",
                                             "rank_ic", "turnover", "long_short_return"]},
    }
    (out / "frontier.json").write_text(json.dumps(jsonable(frontier_json), indent=2) + chr(10))
    (out / "answers.json").write_text(json.dumps(jsonable({
        "schema_version": 1,
        "experiment_id": plan.get("experiment_id"),
        "run_id": run_dir.name,
        "gate": gate,
        "identity_operator_check": identity_check,
        "q1_frontier_shift": q1,
        "q2_fixed_turnover_target_cost": q2,
        "q3_incumbent_turnover_drift": q3,
        "materiality_thresholds": frontier_json["materiality_thresholds"],
    }), indent=2) + chr(10))

    plot(out / "frontier.png", windows, metrics, q1_curves)
    print(f"wrote {out/'frontier.csv'}, {out/'frontier.json'}, {out/'answers.json'}, "
          f"{out/'frontier.png'}")
    print(f"gate passed: {gate['passed']}")
    print(f"Q1 {q1_verdict} (max range {max_range:.5f})")
    for target in TARGET_TURNOVERS:
        key = f"{target:.2f}"
        entry = q2_boot[key]
        if entry.get("pooled_delta") is None:
            print(f"Q2 target {key}: {entry.get('note', 'unreachable')}")
        else:
            print(f"Q2 target {key}: delta {entry['pooled_delta']:+.6f} "
                  f"[{entry['ci_low']:+.6f}, {entry['ci_high']:+.6f}]")
    print(f"Q3 {q3_verdict} (sd {incumbent_turnover['sd']:.5f}, "
          f"range {incumbent_turnover['range']:.5f})")
    return 0


# --------------------------------------------------------------------------- plot

def plot(path: Path, windows, metrics: pd.DataFrame, curves: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(11.5, 8.0), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, window in enumerate(windows):
        color = colors(index % 10)
        rows = curve(metrics, window, "candidate").sort_values("turnover_mean")
        axes.plot(rows.turnover_mean, rows.rank_ic_mean, "-", color=color, linewidth=1.5,
                  label=window, alpha=0.95)
        axes.plot(rows.turnover_mean, rows.rank_ic_mean, ".", color=color, markersize=3)
        inc = point(metrics, window, "incumbent", "identity_unsmoothed")
        axes.plot([inc.turnover_mean], [inc.rank_ic_mean], "s", markerfacecolor="none",
                  markeredgecolor=color, markersize=10, markeredgewidth=1.6)
        cur = point(metrics, window, "candidate", "current_choice")
        axes.plot([cur.turnover_mean], [cur.rank_ic_mean], "*", color=color, markersize=16)

    # Annotate alpha along a representative curve: the median window at the current alpha.
    levels = [(w, float(point(metrics, w, "candidate", "current_choice")["rank_ic_mean"]))
              for w in windows]
    representative = sorted(levels, key=lambda item: item[1])[len(levels) // 2][0]
    rows = curve(metrics, representative, "candidate")
    for alpha in (0.0, 0.5, 0.9, IDENTITY_ALPHA):
        match = rows[np.isclose(rows.alpha, alpha)]
        if match.empty:
            continue
        turnover = float(match.iloc[0]["turnover_mean"])
        rank_ic = float(match.iloc[0]["rank_ic_mean"])
        label = "alpha=1.0\n(unsmoothed)" if alpha >= IDENTITY_ALPHA else f"alpha={alpha:.2f}"
        axes.annotate(label, xy=(turnover, rank_ic), xytext=(16, 14),
                      textcoords="offset points", fontsize=7.5, color="0.25",
                      arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.7})

    axes.set_xlabel("turnover  (traded notional / 2, mean over trading days)")
    axes.set_ylabel("RankIC  (cross-sectional, 30-minute forward VWAP label)")
    axes.set_title("IC–turnover frontier by monthly development window\n"
                   f"candidate = equal-weight 3-checkpoint 441d@2.64 blend;  incumbent = "
                   f"unsmoothed 9-checkpoint blend\n(annotated alphas follow {representative})",
                   fontsize=10)
    axes.grid(True, linewidth=0.4, alpha=0.5)
    handles, labels = axes.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], linestyle="none", marker="s", markerfacecolor="none",
                              markeredgecolor="0.3", markersize=9))
    labels.append("incumbent (unsmoothed) per window")
    handles.append(plt.Line2D([], [], linestyle="none", marker="*", color="0.3", markersize=14))
    labels.append("candidate @ alpha=0.9111677303420447")
    axes.legend(handles, labels, ncol=2, fontsize=8, loc="lower right", framealpha=0.92)
    axes.text(0.01, 0.01, "alpha increases along each curve toward higher turnover; "
                          "alpha=0.0 is the hold-first operator",
              transform=axes.transAxes, fontsize=7.5, color="0.4", va="bottom")
    figure.savefig(path, dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------- entry

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", choices=["sweep", "report", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.run_dir is None:
        root = Path("/root/runs/alpha-frontier-20261028")
        args.run_dir = str(sorted(p for p in root.iterdir() if p.is_dir())[-1])
    if args.mode in ("sweep", "both"):
        sweep(args)
    if args.mode in ("report", "both"):
        return build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
