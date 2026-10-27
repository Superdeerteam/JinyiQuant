"""One single temporal-consistency change on the submittable factor, verified under eval-v2.

The platform only accepts factor values, so turnover can only be moved by making the
submitted factor more persistent in time (see ``runs/cost-curve-20261027``). This script
applies exactly one such change and measures it against the unchanged reference:

* change: per-timestamp cross-sectional z-score, then a causal EWMA along the intraday
  cross sections with a fixed ``ALPHA = 0.5`` (half-life = one cross section = the 30-minute
  label horizon), reset at each trading day. The factor at a timestamp therefore uses only
  that timestamp and earlier ones;
* no search: ``ALPHA`` is the only value ever evaluated by this run (``GRID`` has one entry);
  no other operator, threshold, weighting or model is touched, and nothing is retrained;
* guards predeclared before any metric: the candidate must stay far from a constant signal
  (cross-sectional spread > 0 everywhere, at least 99% of cross sections with a defined rank
  IC, and at least half of the reference rank IC and gross spread retained). These are
  collapse guards, not promotion thresholds — promotion belongs to the statistics and
  verification tasks;
* reporting: gross retention, cost reduction and the turnover change are reported side by
  side at 0/2/5/10bp single-side cost with 5bp as the primary stress scenario, using paired
  daily differences with 5-day moving-block bootstrap intervals inside each month.

Single metric entry point: ``jinyiquant.evaluation.evaluate`` (eval-v2). Nothing is fitted,
no standardized score is ever used as a bp trading threshold, and no sealed data is read.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from cost_curve_eval import (
    MEMBERS,
    PRIMARY_RATE,
    RATES,
    SEALED,
    WINDOWS,
    economics_headroom,
    evaluation_config,
    load_window,
    month_key,
    monthly_table,
    num,
)

from jinyiquant.comparison import _block_indices
from jinyiquant.evaluation import CONTRACT, EVALUATION_VERSION, evaluate
from jinyiquant.io import canonical_json, digest, file_sha256, utc_now, write_json
from jinyiquant.tracking import Recorder

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "turnover-stability-20261027"
ALPHA = 0.5
GRID = (ALPHA,)
BOOTSTRAP = {"block_days": 5, "draws": 4000, "seed": 20261027, "confidence": 0.95}
GUARDS = {"min_cross_sectional_std": 1e-12, "min_defined_rank_ic_share": 0.99,
          "min_rank_ic_retention": 0.5, "min_gross_retention": 0.5}
PAIRED_METRICS = ("rank_ic", "ic", "long_short_return", "turnover", "cost", "net_return")
REFERENCE = "reference_nine_blend"
CANDIDATE = "candidate_ewma_alpha050_nine_blend"


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def cross_sectional_z(keys: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Population z-score inside each timestamp; the platform standardizes the same way."""
    frame = keys.assign(score=scores)
    grouped = frame.groupby("date")["score"]
    mean = grouped.transform("mean").to_numpy(dtype=float)
    scale = grouped.transform("std", ddof=0).to_numpy(dtype=float)
    if not np.isfinite(scale).all():
        raise ValueError("cross-sectional scale is not finite")
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(scale > 0, (scores - mean) / scale, 0.0)
    return values


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def ewma_panel(day_codes: np.ndarray, observations: pd.DataFrame, alpha: float,
               limit: np.ndarray | None = None) -> np.ndarray:
    """Causal EWMA along rows; the state resets whenever ``day_codes`` changes."""
    matrix = observations.to_numpy(dtype=float)
    size = len(matrix)
    rows_allowed = np.full(size, size + 1) if limit is None else np.asarray(limit)
    result = np.full_like(matrix, np.nan)
    state = np.full(matrix.shape[1], np.nan)
    for row in range(size):
        if row == 0 or day_codes[row] != day_codes[row - 1]:
            state = np.full(matrix.shape[1], np.nan)
        if row > rows_allowed[row]:
            continue
        value = matrix[row]
        observed = np.isfinite(value)
        blended = np.where(observed,
                           np.where(np.isfinite(state), alpha * value + (1 - alpha) * state, value),
                           state)
        state = blended
        result[row] = np.where(observed, blended, np.nan)
    return result


def score_panel(keys: pd.DataFrame, scores: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Cross-sectionally standardized (timestamp x instrument) panel plus day codes."""
    z = cross_sectional_z(keys, scores)
    frame = pd.DataFrame({"date": keys["date"].to_numpy(), "instrument": keys["instrument"].to_numpy(),
                          "z": z})
    panel = frame.pivot(index="date", columns="instrument", values="z").sort_index()
    day_codes = pd.factorize(pd.to_datetime(panel.index).normalize())[0].astype(np.int64)
    return panel, day_codes


def smooth_scores(keys: pd.DataFrame, scores: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    """Apply the single predeclared change and return scores plus operator diagnostics."""
    panel, day_codes = score_panel(keys, scores)
    smoothed_panel = ewma_panel(day_codes, panel, alpha)
    flat = pd.DataFrame(smoothed_panel, index=panel.index, columns=panel.columns).stack(future_stack=True)
    values = flat.reindex(pd.MultiIndex.from_frame(keys[["date", "instrument"]])).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("smoothed scores do not cover every reference key")
    diagnostics = {
        "alpha": alpha,
        "operator": "per-timestamp z-score, then causal per-day EWMA along the intraday cross sections",
        "reset": "state resets at each trading day; only the current and earlier timestamps are used",
        "panel_shape": [int(panel.shape[0]), int(panel.shape[1])],
        "sections": int(panel.shape[0]),
        "day_count": int(len(np.unique(day_codes))),
    }
    return values, diagnostics


def causality_check(keys: pd.DataFrame, scores: np.ndarray, alpha: float) -> dict:
    """Recompute every timestamp from data truncated there; a future-leak must change nothing."""
    panel, day_codes = score_panel(keys, scores)
    full = ewma_panel(day_codes, panel, alpha)
    worst = 0.0
    failing = []
    for limit_row in range(len(day_codes)):
        rows = ewma_panel(day_codes[:limit_row + 1], panel.iloc[:limit_row + 1], alpha,
                          np.arange(limit_row + 1))
        left, right = full[limit_row], rows[limit_row]
        finite = np.isfinite(left) | np.isfinite(right)
        difference = float(np.nanmax(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        worst = max(worst, difference)
        if difference > 1e-15:
            failing.append({"row": limit_row, "difference": difference})
    return {"truncated_recompute_max_abs_difference": worst, "passed": worst <= 1e-15,
            "rows_verified": int(len(day_codes)), "failing_rows": failing[:5],
            "method": "for every timestamp, rerun the operator on the panel truncated at that timestamp "
                      "and compare the value it produces"}


def identity_check() -> dict:
    """alpha == 1 must reproduce its own input; a code invariant, not an evaluated scenario."""
    sample = pd.DataFrame({"a": [1.0, np.nan, 3.0, 2.0], "b": [0.5, 1.5, np.nan, 2.5]})
    day_codes = np.asarray([0, 0, 1, 1])
    out = ewma_panel(day_codes, sample, 1.0)
    expected = sample.to_numpy(dtype=float)
    difference = float(np.nanmax(np.abs(out - expected)))
    return {"alpha_1_reproduces_input_max_abs_difference": difference, "passed": difference <= 1e-15}


def build_objects(windows: list[dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    keys = pd.concat([window["keys"] for window in windows], ignore_index=True)
    targets = pd.concat([window["targets"] for window in windows], ignore_index=True)
    keys = keys.sort_values(["date", "instrument"]).reset_index(drop=True)
    targets = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    if not keys.equals(targets[["date", "instrument"]]):
        raise ValueError("pooled target keys changed")
    if keys.duplicated().any():
        raise ValueError("pooled keys are not unique")
    reference = np.stack([np.concatenate([window["scores"][member] for window in windows])
                          for member in MEMBERS]).mean(axis=0)
    candidate, diagnostics = smooth_scores(keys, reference, ALPHA)
    return {REFERENCE: reference, CANDIDATE: candidate}, keys, targets, diagnostics


def evaluate_object(payload):
    """Worker: one object evaluated at every predeclared single-side rate."""
    object_id, keys, scores, targets = payload
    predictions = keys.assign(score=scores)[["date", "instrument", "score"]]
    results = {}
    for label, rate in RATES:
        started = time.time()
        metrics, daily, _ = evaluate(predictions, targets, evaluation_config(rate),
                                     return_sections=True, undefined_constant_portfolios=True)
        results[label] = {"single_side_rate": rate, "metrics": metrics, "daily": daily,
                          "seconds": round(time.time() - started, 3)}
    return object_id, results


def block_bootstrap(differences: pd.Series, months: pd.Series) -> dict | None:
    frame = pd.DataFrame({"value": differences.to_numpy(dtype=float), "month": months.to_numpy()})
    frame = frame[np.isfinite(frame["value"])]
    if frame.empty:
        return None
    groups = [(name, block["value"].to_numpy(dtype=float))
              for name, block in frame.groupby("month", sort=True)]
    if any(len(values) < 2 for _, values in groups):
        return None
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = []
    for _ in range(BOOTSTRAP["draws"]):
        total, count = 0.0, 0
        for _, values in groups:
            indices = _block_indices(len(values), min(BOOTSTRAP["block_days"], len(values)), rng)
            total += float(values[indices].sum())
            count += len(indices)
        draws.append(total / count)
    sampled = np.asarray(draws)
    alpha = (1 - BOOTSTRAP["confidence"]) / 2
    low, high = np.quantile(sampled, [alpha, 1 - alpha])
    return {"mean": num(frame["value"].mean()), "low": num(low), "high": num(high),
            "share_positive": num(float((sampled > 0).mean())),
            "defined_days": int(len(frame)), "draws": BOOTSTRAP["draws"],
            "block_days": BOOTSTRAP["block_days"], "seed": BOOTSTRAP["seed"],
            "confidence": BOOTSTRAP["confidence"]}


def signal_block(metrics: dict) -> dict:
    return {
        "rank_ic_mean": num(metrics.get("rank_ic_mean")),
        "rank_ic_ir": num(metrics.get("rank_ic_ir")),
        "ic_mean": num(metrics.get("ic_mean")),
        "long_short_return_mean": num(metrics.get("long_short_return_mean")),
        "long_short_sharpe": num(metrics.get("long_short_sharpe")),
        "min_daily_long_short_return": num(metrics.get("min_daily_long_short_return")),
        "turnover_mean_local": num(metrics.get("turnover_mean")),
        "rank_persistence_mean": num(metrics.get("rank_persistence_mean")),
        "score_change_mae": num(metrics.get("score_change_mae")),
        "undefined_rank_ic_count": metrics.get("undefined_rank_ic_count"),
        "undefined_turnover_count": metrics.get("undefined_turnover_count"),
        "cross_section_count": metrics.get("cross_section_count"),
        "day_count": metrics.get("day_count"),
    }


def scenario_summary(metrics: dict, daily: pd.DataFrame, label: str, rate: float) -> dict:
    economic = metrics["economic_metrics"]
    return {
        "label": label,
        "single_side_rate": rate,
        "single_side_rate_bp": rate * 1e4,
        "gross_return_mean": num(economic.get("gross_return_mean")),
        "gross_return_mean_bp": None if economic.get("gross_return_mean") is None
        else num(economic["gross_return_mean"] * 1e4),
        "cost_mean": num(economic.get("cost_mean")),
        "cost_mean_bp": None if economic.get("cost_mean") is None else num(economic["cost_mean"] * 1e4),
        "net_return_mean": num(economic.get("net_return_mean")),
        "net_return_mean_bp": None if economic.get("net_return_mean") is None
        else num(economic["net_return_mean"] * 1e4),
        "total_abs_weight_change_mean": num(economic.get("total_abs_weight_change_mean")),
        "turnover_mean_local": num(metrics.get("turnover_mean")),
        "breakeven_single_side_rate_bp": None if economic.get("breakeven_single_side_rate") is None
        else num(economic["breakeven_single_side_rate"] * 1e4),
        "share_days_net_positive": num((daily["net_return"] > 0).sum()
                                       / max(int(daily["net_return"].notna().sum()), 1)),
        "rank_ic_mean": num(metrics.get("rank_ic_mean")),
        "long_short_sharpe": num(metrics.get("long_short_sharpe")),
        "min_daily_long_short_return": num(metrics.get("min_daily_long_short_return")),
        "undefined_rebalance_count": int(economic.get("undefined_rebalance_count") or 0),
        "economic_status": economic.get("status"),
        "official_turnover_item": {
            "official_value": None,
            "official_status": "unavailable locally",
            "official_definition": "turnover is one of the five published metrics, scored as a rank "
                                   "percentile over all valid submissions, lower is better, weight 0.2",
            "local_proxy": num(metrics.get("turnover_mean")),
            "rate_invariant": True,
        },
        "monthly": monthly_table(daily),
    }


def build_plan(windows: list[dict], controllability_path: str) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "hypothesis": "Because the platform accepts only factor values, the turnover of the frozen "
                      "nine-model reference can be reduced by exactly one temporal-consistency change to "
                      "the submittable output without retraining; a causal EWMA with a one-cross-section "
                      "half-life lowers traded notional while keeping the rank signal away from a constant.",
        "dependency": {"cost_curve": "runs/cost-curve-20261027", "controllability": controllability_path,
                       "required_turnover_for_5bp_breakeven": 0.5291153514246137},
        "no_training": True,
        "sealed_interval": list(SEALED),
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "replay_run": window["replay"], "keys_sha256": window["keys_sha256"],
                     "targets_sha256": window["targets_sha256"]} for window in windows],
        "single_change": {
            "id": "causal_ewma_alpha050",
            "applied_at": "submittable factor output; no other component is touched",
            "operator": "per-timestamp cross-sectional z-score, then causal per-day EWMA along the "
                        "intraday cross sections, state reset at each trading day",
            "alpha": ALPHA,
            "half_life_sections": 1,
            "rationale": "the label is the forward 30-minute VWAP return and consecutive cross sections are "
                         "30 minutes apart, so a one-cross-section half-life is the minimal non-trivial "
                         "persistence operator: it halves the weight of every older observation without "
                         "betting on horizons the label never measures",
            "alphas_evaluated": list(GRID),
            "grid_search": False,
            "thresholds_applied": [],
            "threshold_policy": "no trading threshold is applied; standardized scores are not bp returns, so "
                                "no score threshold is used anywhere",
            "unchanged": ["members and blend weights", "data windows and labels", "quantile 0.2 two-leg book",
                          "cost model and cost scenarios", "sampling clock", "evaluation contract"],
        },
        "objects": [{"id": REFERENCE, "role": "reference", "definition": "unchanged nine-model equal blend"},
                    {"id": CANDIDATE, "role": "candidate",
                     "definition": "same nine-model blend after the single temporal-consistency change"}],
        "cost_scenarios": {label: rate for label, rate in RATES},
        "primary_stress_scenario": PRIMARY_RATE,
        "guards_predeclared": {
            **GUARDS,
            "statement": "collapse guards only: the candidate must not become a constant or lose most of its "
                         "ranking ability; they are not promotion thresholds and this run never promotes",
        },
        "bootstrap": {**BOOTSTRAP, "unit": "paired daily differences of candidate minus reference; moving "
                                            "blocks of consecutive trading days inside each month"},
        "reporting_rules": [
            "gross retention, cost reduction and the turnover change are reported together at every rate",
            "5bp is the primary stress scenario",
            "the official turnover item is a cross-submission rank percentile; only the local U/2 proxy is "
            "computable and it is never presented as an official score",
            "net return at any rate is a local economic lens, never the official score",
            "no promotion, rejection or model selection output",
        ],
        "checks_predeclared": [
            "sealed interval never read and evaluation stays inside the five declared windows",
            "operator is causal: recomputing each timestamp from truncated data changes nothing",
            "operator identity at alpha=1 (code invariant, not an evaluated scenario)",
            "only one alpha is evaluated and no threshold is applied",
            "cost is linear in the single-side rate and turnover is rate-invariant",
            "candidate is not a constant signal and keeps the guards",
            "reference 0bp evaluated twice gives an identical digest",
        ],
        "research_area": "five already-viewed development months; sealed interval [2024-10-08, 2025-01-01) "
                         "is never read",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/turnover-stability-20261027")
    parser.add_argument("--record", type=Path,
                        default=ROOT / "modified_tlob/turnover_stability_20261027.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    started = utc_now()
    timer = time.time()
    rss_samples = []

    def rss() -> float:
        value = float(psutil.Process().memory_info().rss / 1e6)
        rss_samples.append(value)
        return value

    windows = [load_window(*window) for window in WINDOWS]
    objects, keys, targets, operator_diagnostics = build_objects(windows)
    causal = causality_check(keys, objects[REFERENCE], ALPHA)
    identity = identity_check()
    plan = build_plan(windows, "runs/cost-curve-20261027")
    plan_bytes = (canonical_json(plan) + "\n").encode("utf-8")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    output = args.output_root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{plan_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_bytes(plan_bytes)
    recorder = Recorder(output, EXPERIMENT_ID)
    recorder.event("plan", "predeclared", plan_sha256=plan_sha, single_change=plan["single_change"]["id"],
                   alpha=ALPHA, alphas_evaluated=list(GRID), rates={k: v for k, v in RATES},
                   primary=PRIMARY_RATE, windows=[window["name"] for window in windows])
    print(f"predeclared plan {rel(output / 'plan.json')} sha256={plan_sha}")
    sections = int(keys["date"].nunique())
    days = int(pd.to_datetime(keys["date"]).dt.normalize().nunique())
    recorder.event("operator", "applied", alpha=ALPHA, sections=sections, days=days,
                   causality_passed=causal["passed"], rss_mb=rss())

    payloads = [(object_id, keys, scores, targets) for object_id, scores in objects.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        evaluated = {object_id: results for object_id, results in executor.map(evaluate_object, payloads)}
    repeat_metrics, _, _ = evaluate(
        keys.assign(score=objects[REFERENCE])[["date", "instrument", "score"]], targets,
        evaluation_config(0.0), return_sections=True, undefined_constant_portfolios=True)
    recorder.event("evaluation", "scenarios_complete", rss_mb=rss())

    scenario_blocks = {}
    for object_id in (REFERENCE, CANDIDATE):
        object_dir = output / "objects" / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        scenario_blocks[object_id] = {}
        for label, _ in RATES:
            daily = evaluated[object_id][label]["daily"]
            daily.to_csv(object_dir / f"daily_{label}.csv", index=False, lineterminator="\n")
            scenario_blocks[object_id][label] = scenario_summary(
                evaluated[object_id][label]["metrics"], daily, label, evaluated[object_id][label]["single_side_rate"])
        write_json(object_dir / "metrics.json", {
            "schema_version": 1, "object_id": object_id, "evaluation_version": EVALUATION_VERSION,
            "single_change": plan["single_change"]["id"] if object_id == CANDIDATE else None,
            "signal": signal_block(evaluated[object_id][PRIMARY_RATE]["metrics"]),
            "scenarios": scenario_blocks[object_id],
            "daily_sha256": {label: file_sha256(object_dir / f"daily_{label}.csv") for label, _ in RATES},
        })

    comparisons = {}
    paired_frames = []
    for label, _ in RATES:
        reference_daily = evaluated[REFERENCE][label]["daily"].set_index("date")
        candidate_daily = evaluated[CANDIDATE][label]["daily"].set_index("date")
        joined = reference_daily[list(PAIRED_METRICS)].join(
            candidate_daily[list(PAIRED_METRICS)], lsuffix="_reference", rsuffix="_candidate", how="inner")
        paired = joined.reset_index()
        paired["month"] = month_key(paired["date"])
        paired["scenario"] = label
        for metric in PAIRED_METRICS:
            paired[f"diff_{metric}"] = paired[f"{metric}_candidate"] - paired[f"{metric}_reference"]
        paired_frames.append(paired)
        block = {}
        for metric in PAIRED_METRICS:
            block[metric] = block_bootstrap(pd.to_numeric(paired[f"diff_{metric}"], errors="coerce"),
                                            paired["month"])
        comparisons[label] = block
    (output / "comparison").mkdir(exist_ok=True)
    write_json(output / "comparison/paired_intervals.json",
               {"scenarios": comparisons, "bootstrap": BOOTSTRAP,
                "sign": "candidate minus reference; negative cost/turnover difference means less trading"})
    pd.concat(paired_frames, ignore_index=True).to_csv(output / "comparison/paired_daily.csv",
                                                       index=False, lineterminator="\n")

    reference_primary = scenario_blocks[REFERENCE][PRIMARY_RATE]
    candidate_primary = scenario_blocks[CANDIDATE][PRIMARY_RATE]
    reference_signal = signal_block(evaluated[REFERENCE][PRIMARY_RATE]["metrics"])
    candidate_signal = signal_block(evaluated[CANDIDATE][PRIMARY_RATE]["metrics"])
    gross_ratio = (None if (reference_primary["gross_return_mean"] in (None, 0)
                            or candidate_primary["gross_return_mean"] is None)
                   else num(candidate_primary["gross_return_mean"] / reference_primary["gross_return_mean"]))
    turnover_ratio = (None if reference_primary["turnover_mean_local"] in (None, 0)
                      else num(candidate_primary["turnover_mean_local"]
                               / reference_primary["turnover_mean_local"]))
    rank_ic_ratio = (None if reference_signal["rank_ic_mean"] in (None, 0)
                     else num(candidate_signal["rank_ic_mean"] / reference_signal["rank_ic_mean"]))
    candidate_headroom = economics_headroom(candidate_primary["gross_return_mean"],
                                            candidate_primary["total_abs_weight_change_mean"],
                                            candidate_primary["turnover_mean_local"], 0.0005)
    defined_share = None
    if candidate_signal["cross_section_count"]:
        defined = (candidate_signal["cross_section_count"]
                   - int(candidate_signal["undefined_rank_ic_count"] or 0))
        defined_share = num(defined / candidate_signal["cross_section_count"])
    minimum_std = float(np.min([np.nanstd(block["score"].to_numpy(dtype=float))
                                for _, block in keys.assign(score=objects[CANDIDATE]).groupby("date")]))

    checks = []
    evaluated_dates = pd.to_datetime(keys["date"])
    sealed_hits = int(((evaluated_dates >= pd.Timestamp(SEALED[0]))
                       & (evaluated_dates < pd.Timestamp(SEALED[1]))).sum())
    checks.append({"name": "sealed_interval_untouched", "passed": sealed_hits == 0,
                   "evidence": f"{len(keys)} samples in {sections} cross sections over {days} days, "
                               f"{evaluated_dates.min():%Y-%m-%d}..{evaluated_dates.max():%Y-%m-%d}; "
                               f"samples inside {list(SEALED)}: {sealed_hits}"})
    window_hits = [{"window": name, "rows": int(((evaluated_dates >= pd.Timestamp(start))
                                                 & (evaluated_dates < pd.Timestamp(end))).sum())}
                   for name, start, end, _ in WINDOWS]
    checks.append({"name": "evaluation_inside_declared_windows",
                   "passed": all(item["rows"] > 0 for item in window_hits)
                   and sum(item["rows"] for item in window_hits) == len(keys),
                   "evidence": window_hits})
    checks.append({"name": "operator_is_causal", "passed": causal["passed"], "evidence": causal})
    checks.append({"name": "operator_identity_at_alpha_1", "passed": identity["passed"], "evidence": identity})
    checks.append({"name": "single_change_no_grid_no_threshold",
                   "passed": len(GRID) == 1 and plan["single_change"]["thresholds_applied"] == [],
                   "evidence": {"alphas_evaluated": list(GRID), "alphas_declared_in_plan": 1,
                                "thresholds_applied": [],
                                "operators_applied": [plan["single_change"]["id"]]}})
    linearity = {}
    rate_free = {}
    for object_id in (REFERENCE, CANDIDATE):
        residuals = {}
        for label, rate in RATES:
            block = scenario_blocks[object_id][label]
            expected = None if block["cost_mean"] is None or block["total_abs_weight_change_mean"] is None \
                else rate * block["total_abs_weight_change_mean"]
            residuals[label] = None if expected is None or block["cost_mean"] is None \
                else abs(block["cost_mean"] - expected)
        linearity[object_id] = {"cost_residuals": residuals,
                                "passed": all(value is not None and value <= 1e-12
                                              for value in residuals.values())}
        changes = [scenario_blocks[object_id][label]["total_abs_weight_change_mean"] for label, _ in RATES]
        turnovers = [scenario_blocks[object_id][label]["turnover_mean_local"] for label, _ in RATES]
        rate_free[object_id] = {
            "change_spread": None if any(value is None for value in changes)
            else num(max(changes) - min(changes)),
            "turnover_spread": None if any(value is None for value in turnovers)
            else num(max(turnovers) - min(turnovers))}
    checks.append({"name": "cost_linear_and_turnover_rate_invariant",
                   "passed": all(linearity[object_id]["passed"] for object_id in linearity)
                   and all(block["change_spread"] is not None and block["change_spread"] <= 1e-12
                           and block["turnover_spread"] is not None and block["turnover_spread"] <= 1e-12
                           for block in rate_free.values()),
                   "evidence": {"linearity": linearity, "rate_free": rate_free}})
    guards = {
        "min_cross_sectional_std": {"value": num(minimum_std), "required": GUARDS["min_cross_sectional_std"],
                                    "passed": bool(minimum_std > GUARDS["min_cross_sectional_std"])},
        "defined_rank_ic_share": {"value": defined_share,
                                  "required": GUARDS["min_defined_rank_ic_share"],
                                  "passed": bool(defined_share is not None
                                                 and defined_share >= GUARDS["min_defined_rank_ic_share"])},
        "rank_ic_retention": {"value": rank_ic_ratio, "required": GUARDS["min_rank_ic_retention"],
                              "passed": bool(rank_ic_ratio is not None
                                             and rank_ic_ratio >= GUARDS["min_rank_ic_retention"])},
        "gross_retention": {"value": gross_ratio, "required": GUARDS["min_gross_retention"],
                            "passed": bool(gross_ratio is not None
                                           and gross_ratio >= GUARDS["min_gross_retention"])},
        "candidate_rank_ic_positive": {"value": candidate_signal["rank_ic_mean"],
                                       "passed": bool(candidate_signal["rank_ic_mean"] is not None
                                                      and candidate_signal["rank_ic_mean"] > 0)},
        "candidate_gross_positive": {"value": candidate_signal["long_short_return_mean"],
                                     "passed": bool(candidate_signal["long_short_return_mean"] is not None
                                                    and candidate_signal["long_short_return_mean"] > 0)},
    }
    checks.append({"name": "candidate_is_not_a_constant_signal", "passed": all(item["passed"] for item in guards.values()),
                   "evidence": guards,
                   "note": "collapse guards predeclared in plan.json; they are not promotion thresholds"})
    checks.append({"name": "reproducible_evaluation",
                   "passed": digest(evaluated[REFERENCE]["0bp"]["metrics"]) == digest(repeat_metrics),
                   "evidence": "reference 0bp evaluated twice produces identical canonical metrics JSON"})
    checks_passed = all(item["passed"] for item in checks)
    write_json(output / "checks.json", {"schema_version": 1, "passed": checks_passed, "checks": checks})
    recorder.event("checks", "completed", passed=checks_passed,
                   failed=[item["name"] for item in checks if not item["passed"]])

    five = comparisons[PRIMARY_RATE]
    conclusion = {
        "single_change": plan["single_change"]["id"],
        "alpha": ALPHA,
        "alphas_evaluated": list(GRID),
        "gross_retention_5bp": {"ratio": gross_ratio,
                                "absolute_bp": None if (candidate_primary["gross_return_mean_bp"] is None
                                                        or reference_primary["gross_return_mean_bp"] is None)
                                else num(candidate_primary["gross_return_mean_bp"]
                                         - reference_primary["gross_return_mean_bp"]),
                                "paired_interval": five["long_short_return"]},
        "cost_reduction_5bp": {"absolute_bp": None if (candidate_primary["cost_mean_bp"] is None
                                                       or reference_primary["cost_mean_bp"] is None)
                               else num(candidate_primary["cost_mean_bp"] - reference_primary["cost_mean_bp"]),
                               "paired_interval": five["cost"]},
        "official_turnover_change": {"local_proxy_ratio": turnover_ratio,
                                     "local_proxy_absolute": None if (candidate_primary["turnover_mean_local"] is None
                                                                      or reference_primary["turnover_mean_local"] is None)
                                     else num(candidate_primary["turnover_mean_local"]
                                              - reference_primary["turnover_mean_local"]),
                                     "paired_interval": five["turnover"],
                                     "official_value": None,
                                     "official_note": "rank percentile across submissions; only the monotone "
                                                      "direction of the local proxy is informative"},
        "net_5bp": {"reference_bp": reference_primary["net_return_mean_bp"],
                    "candidate_bp": candidate_primary["net_return_mean_bp"],
                    "paired_interval": five["net_return"]},
        "signal_retained": {"reference": reference_signal, "candidate": candidate_signal,
                            "rank_ic_ratio": rank_ic_ratio,
                            "rank_persistence_gain": None if (reference_signal["rank_persistence_mean"] is None
                                                              or candidate_signal["rank_persistence_mean"] is None)
                            else num(candidate_signal["rank_persistence_mean"]
                                     - reference_signal["rank_persistence_mean"])},
        "residual_gap": {"turnover_local_now": candidate_primary["turnover_mean_local"],
                         "turnover_required_for_5bp_breakeven": candidate_headroom[
                             "turnover_required_for_zero_net"],
                         "reduction_still_required_share": candidate_headroom[
                             "turnover_reduction_required_share"],
                         "candidate_breakeven_single_side_rate_bp": candidate_primary[
                             "breakeven_single_side_rate_bp"]},
        "statements": [
            f"单一改动 {plan['single_change']['id']}（alpha={ALPHA}，半衰期 1 个截面）把本地换手代理从 "
            f"{fmt(reference_primary['turnover_mean_local'], 4)} 变到 "
            f"{fmt(candidate_primary['turnover_mean_local'], 4)}（{fmt(turnover_ratio)} 倍），"
            f"5bp 成本从 {fmt(reference_primary['cost_mean_bp'], 2)}bp 降到 "
            f"{fmt(candidate_primary['cost_mean_bp'], 2)}bp。",
            f"毛收益保留 {fmt(gross_ratio)} 倍（{fmt(reference_primary['gross_return_mean_bp'], 2)}bp → "
            f"{fmt(candidate_primary['gross_return_mean_bp'], 2)}bp），RankIC 保留 {fmt(rank_ic_ratio)} 倍，"
            f"多空夏普 {fmt(candidate_primary['long_short_sharpe'], 2)}，未出现常数信号"
            f"（最小截面标准差 {minimum_std:.3e}）。",
            f"5bp 主压力情景净收益 {fmt(reference_primary['net_return_mean_bp'], 2)}bp → "
            f"{fmt(candidate_primary['net_return_mean_bp'], 2)}bp；候选盈亏平衡单边成本 "
            f"{fmt(candidate_primary['breakeven_single_side_rate_bp'], 2)}bp。",
            f"残余缺口：5bp 打平仍需换手从 {fmt(candidate_primary['turnover_mean_local'])} 降到 "
            f"{fmt(candidate_headroom['turnover_required_for_zero_net'])}"
            f"（再降 {fmt(candidate_headroom['turnover_reduction_required_share'])} 倍比例）；"
            f"本次只评估一个 alpha，没有搜索，更强的持续性必须另立预登记。",
            "以上净收益/成本是本地经济性口径，官方换手项是相对排名，两者都不是官方得分。",
        ],
    }
    comparison_document = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_version": EVALUATION_VERSION,
        "single_change": plan["single_change"],
        "operator_diagnostics": operator_diagnostics,
        "sealed_interval": list(SEALED),
        "primary_stress_scenario": PRIMARY_RATE,
        "cost_scenarios": {label: rate for label, rate in RATES},
        "data": {"rows": int(len(keys)), "days": days, "cross_sections": sections,
                 "windows": [{"name": window["name"], "keys_sha256": window["keys_sha256"],
                              "targets_sha256": window["targets_sha256"]} for window in windows]},
        "reference": {label: scenario_blocks[REFERENCE][label] for label, _ in RATES},
        "candidate": {label: scenario_blocks[CANDIDATE][label] for label, _ in RATES},
        "paired_intervals": comparisons,
        "guards": guards,
        "conclusion": conclusion,
        "checks": {"passed": checks_passed, "path": "checks.json"},
        "reporting_rules": plan["reporting_rules"],
    }
    write_json(output / "comparison.json", comparison_document)
    write_json(output / "controllability_reference.json",
               {"source": "runs/cost-curve-20261027 (verified quotes)",
                "answer": "仅可提交因子；持仓与调仓时钟不可控"})
    primary_daily = evaluated[CANDIDATE][PRIMARY_RATE]["daily"]
    primary_daily.to_csv(output / "daily.csv", index=False, lineterminator="\n")
    run_metrics = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "evaluation_version": EVALUATION_VERSION,
        "single_change": plan["single_change"]["id"], "alpha": ALPHA,
        "primary_stress_scenario": PRIMARY_RATE,
        "rows": int(len(keys)), "days": days, "cross_sections": sections, "day_count": days,
        "valid_days": int(primary_daily["long_short_return"].notna().sum()),
        "reference": {label: {key: scenario_blocks[REFERENCE][label][key]
                              for key in ("gross_return_mean_bp", "cost_mean_bp", "net_return_mean_bp",
                                          "turnover_mean_local", "rank_ic_mean")}
                      for label, _ in RATES},
        "candidate": {label: {key: scenario_blocks[CANDIDATE][label][key]
                              for key in ("gross_return_mean_bp", "cost_mean_bp", "net_return_mean_bp",
                                          "turnover_mean_local", "rank_ic_mean")}
                      for label, _ in RATES},
        "gross_retention_ratio_5bp": gross_ratio,
        "turnover_ratio_5bp": turnover_ratio,
        "cost_reduction_bp_5bp": conclusion["cost_reduction_5bp"]["absolute_bp"],
    }
    write_json(output / "metrics.json", run_metrics)
    data_document = {
        "schema_version": 1, "source": "archived multiscale replay predictions and vwap labels",
        "version": "cpt_jyc_2026", "evaluation_version": EVALUATION_VERSION,
        "windows": [{"name": window["name"], "start": window["start"], "end": window["end"],
                     "replay_run": window["replay"], "keys_sha256": window["keys_sha256"],
                     "targets_sha256": window["targets_sha256"]} for window in windows],
    }
    write_json(output / "data.json", data_document)
    config = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "hypothesis": plan["hypothesis"],
        "output_root": "runs",
        "data": {"source": data_document["source"], "version": data_document["version"]},
        "evaluation": {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
                       "constant_portfolios": "undefined", "single_side_rate": None,
                       "scenarios": [{"label": label, "single_side_rate": rate} for label, rate in RATES]},
        "windows": [{"name": name, "start": start, "end": end} for name, start, end, _ in WINDOWS],
        "splits": {"train": {"start": "2023-08-01T00:00:00", "end": "2023-11-01T00:00:00"},
                   "validation": {"start": "2023-11-01T00:00:00", "end": "2024-07-01T00:00:00"},
                   "final": None},
        "evaluation_split": "validation",
        "single_change": plan["single_change"],
        "no_training": True,
    }
    write_json(output / "config.json", config)

    record = {
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "status": "completed",
        "kind": "evaluation", "created_at": utc_now(), "run_dir": rel(output),
        "comparison": rel(output / "comparison.json"), "checks": rel(output / "checks.json"),
        "plan": rel(output / "plan.json"), "evaluation_version": EVALUATION_VERSION,
        "single_change": plan["single_change"]["id"], "alpha": ALPHA,
        "upstream": {"cost_curve": "runs/cost-curve-20261027",
                     "record": "modified_tlob/cost_curve_20261027.json"},
        "gross_retention_ratio_5bp": gross_ratio,
        "turnover_ratio_5bp": turnover_ratio,
        "cost_reduction_bp_5bp": conclusion["cost_reduction_5bp"]["absolute_bp"],
        "reference_5bp": {"gross_bp": reference_primary["gross_return_mean_bp"],
                          "cost_bp": reference_primary["cost_mean_bp"],
                          "net_bp": reference_primary["net_return_mean_bp"],
                          "turnover_local": reference_primary["turnover_mean_local"]},
        "candidate_5bp": {"gross_bp": candidate_primary["gross_return_mean_bp"],
                          "cost_bp": candidate_primary["cost_mean_bp"],
                          "net_bp": candidate_primary["net_return_mean_bp"],
                          "turnover_local": candidate_primary["turnover_mean_local"],
                          "breakeven_bp": candidate_primary["breakeven_single_side_rate_bp"]},
        "residual_gap": conclusion["residual_gap"],
        "checks_passed": checks_passed,
        "evidence_limits": [
            "single alpha only; no grid search was run and no stronger persistence is claimed",
            "local approximate caliber on already-viewed development months; not a platform score",
            "the official turnover item is a rank percentile that cannot be computed locally",
            "cost is linear in traded notional with no market-impact term; 5bp is a declared stress assumption",
            "no promotion or rejection: promotion needs the separately pre-registered statistics round",
        ],
    }
    write_json(args.record, record)

    artifacts = [{"path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
                  "bytes": path.stat().st_size}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "schema_version": 1, "run_id": output.name, "experiment_id": EXPERIMENT_ID,
        "kind": "evaluation", "status": "completed", "started_at": started, "ended_at": utc_now(),
        "hypothesis": plan["hypothesis"], "evaluation_version": EVALUATION_VERSION,
        "evaluation_split": "validation", "plan_sha256": plan_sha,
        "config_sha256": digest(config), "exit_code": 0, "seed": None,
        "splits": config["splits"],
        "single_change": {"id": plan["single_change"]["id"], "alpha": ALPHA,
                          "alphas_evaluated": list(GRID), "grid_search": False},
        "evaluation_source": {"evaluation_py_sha256": file_sha256(ROOT / "src/jinyiquant/evaluation.py"),
                              "script_sha256": file_sha256(Path(__file__)),
                              "upstream_script_sha256": file_sha256(ROOT / "scripts/cost_curve_eval.py"),
                              "contract_sha256": digest(CONTRACT)},
        "code": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "files": {"scripts/turnover_stability_change.py": file_sha256(Path(__file__)),
                           "scripts/cost_curve_eval.py": file_sha256(ROOT / "scripts/cost_curve_eval.py"),
                           "src/jinyiquant/evaluation.py": file_sha256(ROOT / "src/jinyiquant/evaluation.py")}},
        "data": {"rows": int(len(keys)), "days": days, "cross_sections": sections,
                 "windows": [w["name"] for w in windows],
                 "sha256": digest([{name: window[name] for name in
                                    ("name", "keys_sha256", "targets_sha256")} for window in windows])},
        "metrics": {"path": "metrics.json", "sha256": file_sha256(output / "metrics.json")},
        "record": rel(args.record),
        "cost_scenarios": {label: rate for label, rate in RATES},
        "primary_stress_scenario": PRIMARY_RATE,
        "checks": rel(output / "checks.json"),
        "resources": {"max_rss_mb": max(rss_samples), "wall_seconds": round(time.time() - timer, 3),
                      "workers": args.workers,
                      "rss_note": "max of sampled RSS points in the parent process only"},
        "artifacts": artifacts,
    }
    write_json(output / "manifest.json", manifest)
    recorder.event("run", "ended", status="completed", checks_passed=checks_passed,
                   wall_seconds=manifest["resources"]["wall_seconds"], record=rel(args.record))
    print(canonical_json({"run_dir": rel(output), "checks_passed": checks_passed,
                          "turnover_ratio": turnover_ratio,
                          "gross_retention_ratio": gross_ratio,
                          "cost_reduction_bp_5bp": conclusion["cost_reduction_5bp"]["absolute_bp"],
                          "net_5bp_candidate": candidate_primary["net_return_mean_bp"],
                          "breakeven_bp_candidate": candidate_primary["breakeven_single_side_rate_bp"]}))
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
