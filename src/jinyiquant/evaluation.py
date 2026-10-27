"""Single versioned evaluation contract (eval-v2) for local competition research.

The contract below is copied into every metrics product, so a result is
self-describing without reading this docstring.

* Return unit: the official forward 30-minute VWAP label return
  (``vwap_return``), a price fraction, not a cash-account P&L.  The label window
  is the same-day forward 30 trading minutes; the 11:30 cross section ends at
  13:30 (lunch break skipped) and overnight returns are never predicted.
* Portfolio: ``max(1, floor(n * quantile))`` stocks per side of each cross
  section, boundary ties share the boundary weight.  Each leg carries 1.0
  notional: gross exposure 2.0, net exposure 0.0, and the long-short spread
  return is measured per 1.0 equity (return on gross notional is spread / 2).
* Cost: ``U = sum(|target weight - pre-trade drifted weight|)`` and
  ``cost = single_side_rate * U``.  ``turnover = U / 2`` with the conversion
  factor 0.5 stated in the output.  The single-side rate must be configured;
  when it is absent cost/net are undefined and counted, never zero-filled.
* Pre-trade weights drift by the previous label return when the previous label
  ends exactly at the current timestamp.  Across overnight or missing gaps the
  intervening return is unknown: previous target weights are carried forward and
  the count is reported.  Consecutive overlapping labels cannot be independent
  trades; their cost is left undefined and counted.
* Daily aggregation: timestamps inside a day are equally weighted; a daily value
  is a descriptive arithmetic mean of label-period values, never compounded or
  summed into a cumulative return.
* ``IC_IR = mean(daily IC) / sample std(daily IC, ddof=1)``, not annualized.
  ``Sharpe = mean(daily spread) / sample std(ddof=1) * sqrt(periods_per_year)``,
  252 by default.
* Competition and economic blocks are reported side by side.  The competition
  block only reconstructs published rules and marks every unavailable component
  as approximate/unavailable.  No cross-submission relative ranking and no
  weighted total score are produced; this contract never promotes a model.
* Undefined values remain ``None`` in JSON metrics and ``NaN`` in the
  per-timestamp and daily frames, with explicit counts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .io import digest

EVALUATION_VERSION = "eval-v2"
LABEL_HORIZON_MINUTES = 30
QUANTILE = 0.2
PERIODS_PER_YEAR = 252
TURNOVER_CONVERSION_FACTOR = 0.5
LEG_CAPITAL = 1.0
GROSS_EXPOSURE = 2.0

KEYS = ["date", "instrument"]
METRIC_SPEC = {
    "rank_ic_mean": ("rank_ic", "mean", "maximize"),
    "rank_ic_ir": ("rank_ic", "ir", "maximize"),
    "ic_mean": ("ic", "mean", "maximize"),
    "ic_ir": ("ic", "ir", "maximize"),
    "long_short_return_mean": ("long_short_return", "mean", "maximize"),
    "long_short_sharpe": ("long_short_return", "sharpe", "maximize"),
    "long_return_mean": ("long_return", "mean", "maximize"),
    "short_return_mean": ("short_return", "mean", "minimize"),
    "turnover_mean": ("turnover", "mean", "minimize"),
    "extreme_return": ("long_short_return", "lower_tail", "maximize"),
    "min_daily_long_short_return": ("long_short_return", "min", "maximize"),
}

CONTRACT: dict[str, Any] = {
    "evaluation_version": EVALUATION_VERSION,
    "label": {
        "return_unit": "fraction of price: official forward 30-minute VWAP label return "
                       "(vwap_return); consumed unchanged, not a cash-account P&L",
        "horizon_minutes": LABEL_HORIZON_MINUTES,
        "clock": "same-day forward window only; the 11:30 cross section ends at 13:30 "
                 "(lunch break skipped); overnight returns are never predicted or used",
        "overlap": "two consecutive cross sections overlap when the next timestamp is earlier "
                   "than the previous label end; overlapping label returns are never treated as "
                   "independent trades and their rebalance cost/turnover is undefined and counted",
        "daily_aggregation": "timestamps inside a day are equally weighted; a daily value is a "
                             "descriptive arithmetic mean of label-period values, never compounded "
                             "into an equity curve and never summed across days",
        "undefined": "undefined values stay None in JSON metrics and NaN in frames with explicit "
                     "counts; they are never zero-filled",
    },
    "portfolio": {
        "default_quantile": QUANTILE,
        "tail_size": "max(1, floor(n * quantile)) stocks per side per cross section; equals "
                     "floor(n * quantile) whenever n * quantile >= 1",
        "boundary_ties": "stocks tied at the boundary share the boundary weight equally",
        "leg_capital": LEG_CAPITAL,
        "gross_exposure": GROSS_EXPOSURE,
        "net_exposure": 0.0,
        "weights": "net weight vector per 1.0 equity: long leg sums to +1, short leg to -1; "
                   "spread return = long leg mean label return - short leg mean label return",
        "return_on_gross_notional": "spread / gross_exposure = spread / 2",
        "constant_scores": "equal scores still form equal-weight legs; the neutral spread is not "
                           "stock-selection ability and IC is undefined and counted",
        "selection": "larger score is long, smaller score is short",
    },
    "cost": {
        "single_side_rate": "decimal one-side rate c applied to traded notional; configured "
                            "explicitly, otherwise cost/net are undefined and counted",
        "cost_model": "cost = single_side_rate * U",
        "u_definition": "U = sum(|target_weight - pre_trade_drifted_weight|) over the union of "
                        "stocks; absent positions count as zero",
        "turnover": "U is the two-sided absolute notional change per 1.0 equity; turnover = U / 2 "
                    "with conversion_factor = 0.5 is the one-way share turnover of the two-leg book",
        "units": "cost is a return fraction per 1.0 equity, the same basis as the spread return",
        "pretrade_weights": "adjacent labels: drift each weight by the previous label return "
                            "w_i * (1 + r_i) / (1 + sum(w * r)); overnight or missing timestamp "
                            "gaps: carry previous target weights forward because the intervening "
                            "return is unknown, and count it",
        "breakeven": "single-side rate c* such that mean(net spread) = 0: c* = mean(gross) / "
                     "mean(U) over defined rebalances",
    },
    "aggregation": {
        "cross_section": "one prediction timestamp; IC is Pearson correlation of raw scores and "
                         "label returns, rank IC is Pearson correlation of average ranks",
        "ic_ir": "mean(daily IC) / sample std(daily IC, ddof=1); not annualized",
        "sharpe": "mean(daily long-short spread) / sample std(ddof=1) * sqrt(periods_per_year); "
                  "periods_per_year defaults to 252",
        "daily": "same-day timestamps equally weighted; undefined timestamps are skipped in that "
                 "day's mean and counted; aggregates equally weight days",
        "economic": "economic means are equal-weighted over defined rebalance timestamps (an "
                    "observation is one timestamp with defined U); the sample's first timestamp "
                    "and overlapping pairs are undefined and counted",
    },
    "competition": {
        "published_rules": "five raw metrics with fixed 0.2 weights in the published description: "
                           "IC mean, IC_IR, long-short Sharpe, stress stability, turnover (lower "
                           "is better); official aggregation uses relative rank across all valid "
                           "submissions before weighting",
        "local_status": "approximate: platform winsorization, standardization, BARRA style "
                        "residualization, stress definitions and cross-submission ranking are "
                        "unavailable locally",
        "not_produced": "no cross-submission relative ranking and no weighted total score are "
                        "produced; metrics are descriptive and never promote a model",
    },
}

_MISSING = object()


def _option(config: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(config, dict):
        if name in config:
            return config[name]
    else:
        value = getattr(config, name, _MISSING)
        if value is not _MISSING:
            return value
    if default is _MISSING:
        raise KeyError(name)
    return default


def _validated(frame: pd.DataFrame, value: str, name: str) -> pd.DataFrame:
    required = KEYS + [value]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"{name} must contain {required}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError(f"{name} contains no rows or missing keys/values")
    result = frame[required].copy()
    if result["instrument"].map(lambda x: not isinstance(x, str) or not x.strip() or x != x.strip()).any():
        raise ValueError(f"{name} instrument keys must be nonempty strings")
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError(f"{name} contains invalid dates")
    if isinstance(result["date"].dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} requires timezone-naive exchange-local dates")
    if result.duplicated(KEYS).any():
        raise ValueError(f"{name} contains duplicate keys")
    result[value] = pd.to_numeric(result[value], errors="raise")
    if not np.isfinite(result[value].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains nonfinite {value}")
    return result


def align_predictions(predictions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Require exact key sets; neither inner-join loss nor positional alignment."""
    if set(predictions.columns) != {*KEYS, "score"}:
        raise ValueError("predictions must contain only date, instrument, score")
    left = _validated(predictions, "score", "predictions")
    right = _validated(targets, "target", "targets")
    joined = left.merge(right, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if (joined["_merge"] != "both").any():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"prediction/target keys differ: {counts}")
    return joined.drop(columns="_merge").sort_values(KEYS, ignore_index=True)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    # Scaling before centering avoids overflow for otherwise valid large inputs.
    a = a / max(float(np.max(np.abs(a))), 1.0)
    b = b / max(float(np.max(np.abs(b))), 1.0)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def label_end(timestamp, horizon_minutes: int = LABEL_HORIZON_MINUTES) -> pd.Timestamp:
    """Official forward-window end; the 30-minute 11:30 label ends at 13:30."""
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError("label horizon must be a positive number of minutes")
    moment = pd.Timestamp(timestamp)
    if horizon == LABEL_HORIZON_MINUTES and (moment.hour, moment.minute) == (11, 30):
        return moment + pd.Timedelta(hours=2)
    return moment + pd.Timedelta(minutes=horizon)


def tail_weights(scores, count: int, *, largest: bool) -> np.ndarray:
    """Equal weights on ``count`` extreme scores; boundary ties share the weight."""
    values = np.asarray(scores, dtype=float)
    ordered = np.sort(values)
    boundary = ordered[-count] if largest else ordered[count - 1]
    strictly = values > boundary if largest else values < boundary
    tied = values == boundary
    weights = strictly.astype(float)
    weights[tied] = (count - int(strictly.sum())) / int(tied.sum())
    return weights / count


def neutral_weights(scores: pd.Series, quantile: float = QUANTILE) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    """Return long, short and net weights plus the per-side stock count."""
    values = pd.Series(scores, dtype=float)
    if values.empty:
        raise ValueError("portfolio requires a nonempty cross section")
    if values.index.has_duplicates:
        raise ValueError("portfolio requires unique instrument keys")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("portfolio scores must be finite")
    if not 0 < float(quantile) <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    tail_size = max(1, int(np.floor(float(quantile) * len(values))))
    raw_long = tail_weights(values.to_numpy(dtype=float), tail_size, largest=True)
    raw_short = tail_weights(values.to_numpy(dtype=float), tail_size, largest=False)
    long = pd.Series(raw_long, index=values.index)
    short = pd.Series(raw_short, index=values.index)
    return long, short, long - short, tail_size


def drift_weights(weights: pd.Series, period_returns: pd.Series) -> pd.Series:
    """Drift weights by the realized period return on the same notional base."""
    weight = pd.Series(weights, dtype=float)
    returns = pd.Series(period_returns, dtype=float).reindex(weight.index)
    if weight.isna().any() or returns.isna().any():
        raise ValueError("drift requires finite weights and period returns")
    if not np.isfinite(weight.to_numpy()).all() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("drift requires finite weights and period returns")
    spread = float(np.dot(weight.to_numpy(), returns.to_numpy()))
    if 1.0 + spread <= 0:
        raise ValueError("period return invalidates the notional base")
    return weight * (1.0 + returns) / (1.0 + spread)


def total_abs_weight_change(target_weights: pd.Series, pretrade_weights: pd.Series) -> float:
    """U = sum(|target - pre-trade drifted|) over the union of stocks."""
    target = pd.Series(target_weights, dtype=float)
    pretrade = pd.Series(pretrade_weights, dtype=float)
    if target.index.has_duplicates or pretrade.index.has_duplicates:
        raise ValueError("weights require unique instrument keys")
    if target.isna().any() or pretrade.isna().any():
        raise ValueError("weights must be finite")
    union = target.index.union(pretrade.index)
    delta = target.reindex(union, fill_value=0.0) - pretrade.reindex(union, fill_value=0.0)
    return float(np.abs(delta.to_numpy(dtype=float)).sum())


def rebalance_economics(target_weights: pd.Series, pretrade_weights: pd.Series, *,
                        gross_return: float, single_side_rate: float) -> dict[str, Any]:
    """Calibration entry point: gross, U, turnover, cost and net for one rebalance."""
    rate = float(single_side_rate)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("single_side_rate must be a finite nonnegative decimal rate")
    gross = float(gross_return)
    if not np.isfinite(gross):
        raise ValueError("gross_return must be finite")
    change = total_abs_weight_change(target_weights, pretrade_weights)
    cost = rate * change
    return {
        "evaluation_version": EVALUATION_VERSION,
        "gross_return": gross,
        "total_abs_weight_change": change,
        "turnover": change * TURNOVER_CONVERSION_FACTOR,
        "turnover_conversion_factor": TURNOVER_CONVERSION_FACTOR,
        "turnover_definition": "turnover = U / 2",
        "single_side_rate": rate,
        "cost": cost,
        "cost_model": "cost = single_side_rate * U",
        "net_return": gross - cost,
    }


def breakeven_single_side_rate(gross_returns, total_abs_weight_changes) -> float | None:
    """Single-side rate that zeroes mean net return; None when undefined."""
    gross = np.asarray(gross_returns, dtype=float)
    changes = np.asarray(total_abs_weight_changes, dtype=float)
    valid = np.isfinite(gross) & np.isfinite(changes)
    if not valid.any():
        return None
    mean_change = float(changes[valid].mean())
    if mean_change <= 0:
        return None
    return float(gross[valid].mean() / mean_change)


def aggregate_metric(values: np.ndarray, method: str, periods_per_year: float = PERIODS_PER_YEAR) -> float | None:
    """Aggregate finite daily observations; callers must disclose missing counts."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    if method == "mean":
        value = float(np.mean(values))
    elif method in {"ir", "sharpe"}:
        if len(values) < 2 or float(np.std(values, ddof=1)) == 0:
            return None
        value = float(np.mean(values) / np.std(values, ddof=1))
        if method == "sharpe":
            value *= float(np.sqrt(periods_per_year))
    elif method == "min":
        value = float(np.min(values))
    elif method == "lower_tail":
        # Empirical expected shortfall: average of worst ceil(1% * days) days.
        count = max(1, int(np.ceil(len(values) * 0.01)))
        value = float(np.sort(values)[:count].mean())
    else:
        raise ValueError(f"unsupported aggregate method: {method}")
    return value if np.isfinite(value) else None


def _economic_metrics(sections: pd.DataFrame, single_side_rate: float | None,
                      overlap_count: int) -> dict[str, Any]:
    defined = sections.loc[sections["total_abs_weight_change"].notna()]
    gross_mean = float(defined["long_short_return"].mean()) if len(defined) else None
    change_mean = float(defined["total_abs_weight_change"].mean()) if len(defined) else None
    turnover_mean = None if change_mean is None else change_mean * TURNOVER_CONVERSION_FACTOR
    cost_mean = None
    net_mean = None
    if single_side_rate is not None and len(defined):
        cost_mean = float(defined["cost"].mean())
        net_mean = float(gross_mean - cost_mean)
    breakeven = breakeven_single_side_rate(defined["long_short_return"], defined["total_abs_weight_change"])
    if overlap_count:
        status = "descriptive_returns_undefined_overlapping_labels"
        net_mean = None
        breakeven = None
        gross_mean_use = None
    elif single_side_rate is None:
        status = "undefined_single_side_rate_not_configured"
        gross_mean_use = gross_mean
    elif not len(defined):
        status = "undefined_no_defined_rebalance"
        gross_mean_use = gross_mean
    else:
        status = "computed"
        gross_mean_use = gross_mean
    return {
        "evaluation_version": EVALUATION_VERSION,
        "status": status,
        "gross_return_mean": gross_mean_use,
        "cost_mean": cost_mean,
        "net_return_mean": net_mean,
        "turnover_mean": turnover_mean,
        "total_abs_weight_change_mean": change_mean,
        "turnover_conversion_factor": TURNOVER_CONVERSION_FACTOR,
        "turnover_definition": "turnover = total_abs_weight_change / 2 = 0.5 * U",
        "single_side_rate": single_side_rate,
        "cost_model": "cost = single_side_rate * U; U = sum(|target weight - pre-trade drifted weight|)",
        "breakeven_single_side_rate": breakeven,
        "breakeven_definition": "single-side rate c* with mean(net) = 0: c* = mean(gross) / mean(U) "
                                "over defined rebalances",
        "leg_capital": LEG_CAPITAL,
        "gross_exposure": GROSS_EXPOSURE,
        "net_exposure": 0.0,
        "return_unit": "long-short spread per 1.0 equity; return on gross notional is spread / 2",
        "population": "one equally weighted observation per defined rebalance timestamp",
        "comparison_weighting": "the top-level turnover_mean metric averages daily means as the "
                                "comparison key; this block is equal-weighted over defined "
                                "rebalances; both use turnover = U / 2",
        "defined_rebalance_count": int(len(defined)),
        "undefined_rebalance_count": int(len(sections) - len(defined)),
        "overlap_count": int(overlap_count),
        "overlapping_labels_treated_as_independent_trades": False,
        "returns_compounded": False,
        "daily_returns_summed_across_days": False,
        "drift_unknown_rebalance_count": int(sections["gap_drift_unknown"].sum()),
        "undefined_counts": {
            "total_abs_weight_change": int(sections["total_abs_weight_change"].isna().sum()),
            "turnover": int(sections["turnover"].isna().sum()),
            "cost": int(sections["cost"].isna().sum()),
            "net_return": int(sections["net_return"].isna().sum()),
        },
    }


def _competition_item(value: float | None, local_definition: str, caveat: str) -> dict[str, Any]:
    return {
        "value": value,
        "status": "approximate",
        "local_definition": local_definition,
        "missing_official_detail": caveat,
    }


def _competition_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    caveat = ("platform winsorization, standardization, BARRA style residualization, missing "
              "handling and cross-section weighting are unavailable locally")
    return {
        "evaluation_version": EVALUATION_VERSION,
        "scope": "local_approximation",
        "approximate": True,
        "target": "forward 30-minute VWAP label return",
        "published_rules": {
            "raw_metrics": ["IC mean", "IC_IR", "long-short Sharpe", "stress stability", "turnover"],
            "weights": "fixed 0.2 each in the published description",
            "aggregation": "relative rank across all valid submissions, then weighted sum; not "
                           "reproduced locally because the submission population and platform "
                           "implementation are unavailable",
            "turnover_direction": "lower is better",
        },
        "metrics": {
            "ic_mean": _competition_item(metrics["ic_mean"], "mean cross-sectional Pearson IC", caveat),
            "ic_ir": _competition_item(metrics["ic_ir"], "mean(daily IC) / sample std(ddof=1), "
                                                          "not annualized", caveat),
            "long_short_sharpe": _competition_item(metrics["long_short_sharpe"],
                                                   "daily spread mean / sample std(ddof=1) * "
                                                   "sqrt(periods_per_year)", caveat),
            "turnover": _competition_item(metrics["turnover_mean"], "daily mean of turnover = U / 2",
                                          "official turnover construction and rebalancing "
                                          "assumptions are unavailable"),
        },
        "unavailable_metrics": ["stress_stability"],
        "unavailable_reasons": {
            "stress_stability": "official stress scenario definitions (volatility, volume, time "
                                "bucket) are not published",
        },
        "local_status": "approximate; no cross-submission relative ranking and no weighted total "
                        "score are produced",
        "usage": "descriptive only; this contract never promotes, rejects or selects a model",
    }


def evaluate(predictions: pd.DataFrame, targets: pd.DataFrame, evaluation_config: Any, *, return_sections=False,
             undefined_constant_portfolios=False):
    """Return JSON-safe metrics plus daily rows, retaining undefined statistics."""
    quantile = float(_option(evaluation_config, "quantile"))
    periods_per_year = float(_option(evaluation_config, "periods_per_year"))
    min_cross_section = int(_option(evaluation_config, "min_cross_section"))
    horizon = int(_option(evaluation_config, "label_horizon_minutes", LABEL_HORIZON_MINUTES))
    single_side_rate = _option(evaluation_config, "single_side_rate", None)
    if not 0 < quantile <= 0.5 or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("quantile must be in (0, 0.5], periods_per_year must be positive")
    if min_cross_section < 2:
        raise ValueError("min_cross_section must be at least 2")
    if horizon <= 0:
        raise ValueError("label_horizon_minutes must be positive")
    if single_side_rate is not None:
        single_side_rate = float(single_side_rate)
        if not np.isfinite(single_side_rate) or single_side_rate < 0:
            raise ValueError("single_side_rate must be a finite nonnegative decimal rate")
    aligned = align_predictions(predictions, targets)
    rows = []
    previous_weights: pd.Series | None = None
    previous_returns: pd.Series | None = None
    previous_timestamp: pd.Timestamp | None = None
    previous_scores: pd.Series | None = None
    for timestamp, section in aligned.groupby("date", sort=True):
        if len(section) < min_cross_section:
            raise ValueError(f"cross section {timestamp} has fewer than {min_cross_section} samples")
        instruments = section["instrument"].to_numpy()
        score = pd.Series(section["score"].to_numpy(dtype=float), index=instruments)
        target = pd.Series(section["target"].to_numpy(dtype=float), index=instruments)
        target_values = target.to_numpy(dtype=float)
        target_ranks = target.rank(method="average").to_numpy()
        eligible_rank_ic = bool(np.isfinite(_correlation(target_ranks, target_ranks)))
        eligible_ic = bool(np.isfinite(_correlation(target_values, target_values)))
        long, short, net_weights, tail_size = neutral_weights(score, quantile)
        current_scores = score
        score_change = float("nan")
        rank_persistence = float("nan")
        common_count = 0
        if previous_scores is not None:
            common = current_scores.index.intersection(previous_scores.index)
            common_count = len(common)
            if common_count:
                score_change = float(np.abs(current_scores.loc[common] - previous_scores.loc[common]).mean())
            if common_count >= 2:
                rank_persistence = _correlation(current_scores.loc[common].rank().to_numpy(),
                                                previous_scores.loc[common].rank().to_numpy())
        previous_scores = current_scores
        long_return = float(np.dot(long.to_numpy(dtype=float), target_values))
        short_return = float(np.dot(short.to_numpy(dtype=float), target_values))
        if not np.isfinite([long_return, short_return, long_return - short_return]).all():
            raise ValueError("portfolio return overflow; target units are not representable")
        spread = long_return - short_return
        change = float("nan")
        turnover = float("nan")
        cost = float("nan")
        net_return = float("nan")
        label_overlap = False
        if previous_timestamp is None:
            drift_state = "first_timestamp"
            pretrade_source = "none"
        else:
            previous_end = label_end(previous_timestamp, horizon)
            label_overlap = bool(timestamp < previous_end)
            if previous_weights is None or previous_returns is None:
                drift_state = "previous_portfolio_undefined"
                pretrade_source = "none"
            elif label_overlap:
                drift_state = "overlapping_labels"
                pretrade_source = "none"
            elif timestamp == previous_end:
                pretrade = drift_weights(previous_weights, previous_returns)
                change = total_abs_weight_change(net_weights, pretrade)
                drift_state = "adjacent_label"
                pretrade_source = "label_implied_drift"
            else:
                change = total_abs_weight_change(net_weights, previous_weights)
                drift_state = "gap_drift_unknown"
                pretrade_source = "previous_target_carry_forward"
        if np.isfinite(change):
            turnover = change * TURNOVER_CONVERSION_FACTOR
            if single_side_rate is not None:
                cost = single_side_rate * change
                net_return = spread - cost
        if undefined_constant_portfolios and score.nunique() == 1:
            long_return = short_return = float("nan")
            spread = float("nan")
            change = turnover = cost = net_return = float("nan")
            drift_state = "undefined_constant_portfolio"
            pretrade_source = "none"
            net_weights = pd.Series(0.0, index=instruments)
        previous_weights = None if not np.isfinite(turnover) and drift_state == "undefined_constant_portfolio" \
            else net_weights
        previous_returns = target
        previous_timestamp = timestamp
        rows.append({
            "date": timestamp.normalize(),
            "timestamp": timestamp.isoformat(),
            "rank_ic": _correlation(score.rank(method="average").to_numpy(), target_ranks)
                       if eligible_rank_ic else float("nan"),
            "ic": _correlation(score.to_numpy(dtype=float), target_values) if eligible_ic else float("nan"),
            "eligible_rank_ic": eligible_rank_ic,
            "eligible_ic": eligible_ic,
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": spread,
            "turnover": turnover,
            "score_change_mae": score_change,
            "rank_persistence": rank_persistence,
            "signal_common_count": common_count,
            "ranking_defined": bool(score.nunique() > 1),
            "mse": float(np.mean(np.square(score.to_numpy(dtype=float) - target_values))),
            "mae": float(np.mean(np.abs(score.to_numpy(dtype=float) - target_values))),
            "sample_count": len(section),
            "long_position_count": int((long > 0).sum()),
            "short_position_count": int((short > 0).sum()),
            "tail_size": int(tail_size),
            "gross_exposure": float(long.sum() + short.sum()),
            "net_exposure": float(net_weights.sum()),
            "net_weight_gross": float(np.abs(net_weights.to_numpy(dtype=float)).sum()),
            "total_abs_weight_change": change,
            "cost": cost,
            "net_return": net_return,
            "label_overlap": label_overlap,
            "gap_drift_unknown": drift_state == "gap_drift_unknown",
            "drift_state": drift_state,
            "pretrade_weight_source": pretrade_source,
            "evaluation_version": EVALUATION_VERSION,
        })
    sections = pd.DataFrame(rows)
    columns = ["rank_ic", "ic", "long_return", "short_return", "long_short_return", "turnover",
               "total_abs_weight_change", "cost", "net_return", "mse", "mae", "score_change_mae",
               "rank_persistence"]
    grouped = sections.groupby("date", sort=True)
    daily = grouped[columns].mean()
    daily["sample_count"] = grouped["sample_count"].sum()
    daily["cross_section_count"] = grouped.size()
    daily["valid_rank_ic_count"] = grouped["rank_ic"].count()
    daily["valid_ic_count"] = grouped["ic"].count()
    daily["eligible_rank_ic_count"] = grouped["eligible_rank_ic"].sum()
    daily["eligible_ic_count"] = grouped["eligible_ic"].sum()
    daily["defined_turnover_count"] = grouped["turnover"].count()
    daily["undefined_turnover_count"] = grouped.size() - grouped["turnover"].count()
    daily["undefined_cost_count"] = grouped.size() - grouped["cost"].count()
    daily["overlapping_label_section_count"] = grouped["label_overlap"].sum()
    daily["drift_unknown_section_count"] = grouped["gap_drift_unknown"].sum()
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["evaluation_version"] = EVALUATION_VERSION
    metrics = {
        name: aggregate_metric(daily[column].to_numpy(), method, periods_per_year)
        for name, (column, method, _) in METRIC_SPEC.items()
    }
    # Eligibility depends only on targets. Hash timestamps before daily grouping
    # so equal counts at different prediction times cannot justify promotion.
    correlation_coverage = {"schema_version": 1}
    for column in ("rank_ic", "ic"):
        eligible = sections[f"eligible_{column}"]
        valid = eligible & sections[column].notna()
        correlation_coverage[column] = {
            "eligible_section_count": int(eligible.sum()),
            "valid_section_count": int(valid.sum()),
            "eligible_sections_sha256": digest(sections.loc[eligible, "timestamp"].tolist()),
            "valid_sections_sha256": digest(sections.loc[valid, "timestamp"].tolist()),
        }
    overlap_count = int(sections["label_overlap"].sum())
    metrics.update({
        "evaluation_version": EVALUATION_VERSION,
        "contract": deepcopy(CONTRACT),
        "metric_scope": "local_approximation",
        "local_approximation": True,
        "schema_version": 1,
        "mse": aggregate_metric(daily.mse.to_numpy(), "mean"),
        "mae": aggregate_metric(daily.mae.to_numpy(), "mean"),
        "score_change_mae": aggregate_metric(daily.score_change_mae.to_numpy(), "mean"),
        "rank_persistence_mean": aggregate_metric(daily.rank_persistence.to_numpy(), "mean"),
        "sample_count": len(aligned),
        "day_count": len(daily),
        "cross_section_count": len(sections),
        "sample_keys_sha256": digest(aligned[KEYS].assign(date=aligned["date"].astype(str)).to_dict("records")),
        "targets_sha256": digest(aligned[KEYS + ["target"]].assign(date=aligned["date"].astype(str))
                                 .to_dict("records")),
        "undefined_rank_ic_count": int(sections["rank_ic"].isna().sum()),
        "undefined_ic_count": int(sections["ic"].isna().sum()),
        "undefined_turnover_count": int(sections["turnover"].isna().sum()),
        "undefined_total_abs_weight_change_count": int(sections["total_abs_weight_change"].isna().sum()),
        "undefined_cost_count": int(sections["cost"].isna().sum()),
        "undefined_net_return_count": int(sections["net_return"].isna().sum()),
        "rebalance_count": int(sections["turnover"].notna().sum()),
        "ranking_defined_section_count": int(sections.ranking_defined.sum()),
        "overlapping_label_section_count": overlap_count,
        "drift_unknown_section_count": int(sections["gap_drift_unknown"].sum()),
        "correlation_coverage": correlation_coverage,
        "valid_days": {column: int(daily[column].notna().sum()) for column in columns},
        "aggregation": {
            "daily_equal_timestamp_weights": True,
            "ic_ir": "mean(daily IC) / sample std(daily IC, ddof=1); not annualized",
            "ic_ir_periods": "daily",
            "sharpe": "mean(daily spread) / sample std(ddof=1) * sqrt(periods_per_year)",
            "periods_per_year": periods_per_year,
            "sharpe_annualization_factor": float(np.sqrt(periods_per_year)),
            "extreme_return": "mean of the worst ceil(1% * days) daily spreads",
            "undefined_policy": "None in JSON, NaN in frames, explicit counts; never zero-filled",
        },
        "portfolio": {
            "quantile": quantile,
            "tail_size_rule": "max(1, floor(n * quantile)) per side",
            "tail_size_min": int(sections["tail_size"].min()),
            "tail_size_max": int(sections["tail_size"].max()),
            "leg_capital": LEG_CAPITAL,
            "gross_exposure": GROSS_EXPOSURE,
            "net_exposure": 0.0,
            "net_weight_gross_min": float(sections["net_weight_gross"].min()),
            "net_weight_gross_max": float(sections["net_weight_gross"].max()),
            "spread_return_definition": "long leg mean label return - short leg mean label return "
                                        "per 1.0 equity",
            "return_on_gross_notional_definition": "spread / 2.0",
        },
        "label_overlap": {
            "label_horizon_minutes": horizon,
            "overlapping_pair_count": overlap_count,
            "independent_label_periods": overlap_count == 0,
            "overlapping_labels_treated_as_independent_trades": False,
            "definition": "a consecutive pair overlaps when the next timestamp is earlier than the "
                          "previous label end; cost/turnover of that pair is undefined and counted",
        },
        "conventions": {
            "version": EVALUATION_VERSION,
            "cross_section": "one prediction timestamp; average ranks for ties",
            "daily": "equal timestamp mean; undefined correlations remain missing and are counted",
            "correlation_comparison": "each policy correlation requires all target-eligible timestamps; "
                                      "incomplete baseline or absent evidence is inconclusive; "
                                      "incomplete candidate rejected",
            "portfolio": "max(1, floor(n*quantile)) per side; boundary ties share weight; each leg "
                         "1.0 notional, gross 2.0, net 0.0; larger score is long",
            "constant_portfolio": ("undefined: signal cannot rank stocks" if undefined_constant_portfolios
                                   else "constant signals give identical full-market long/short weights; "
                                        "neutral spread is not evidence of stock ranking ability"),
            "signal_change": "mean absolute raw score change and rank correlation on common stocks; "
                             "first timestamp undefined; consecutive sampled timestamps, including overnight",
            "turnover": "turnover = U/2; U = sum(abs(target net weight - pre-trade drifted weight)); "
                        "first timestamp undefined; drift uses the previous label when adjacent, "
                        "carry-forward with a count when the gap return is unknown",
            "cost": "cost = single_side_rate * U; single_side_rate must be configured; gross label "
                    "returns are not a cash-account P&L",
            "rank_ic_ir": "daily mean / sample std (ddof=1), not annualized",
            "ic_ir": "daily mean / sample std (ddof=1), not annualized",
            "long_short_sharpe": "daily mean / sample std * sqrt(periods_per_year); zero variance undefined",
            "extreme_return": "mean of worst ceil(1% * days) daily long-short returns",
            "platform_metrics": "not implemented: official complete evaluation rules and platform "
                                "preprocessing are unavailable; no cross-submission ranking or "
                                "weighted total is produced",
        },
    })
    metrics["economic_metrics"] = _economic_metrics(sections, single_side_rate, overlap_count)
    metrics["competition_metrics"] = _competition_metrics(metrics)
    return (metrics, daily, sections) if return_sections else (metrics, daily)
