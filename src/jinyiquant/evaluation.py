"""Strict alignment and transparent *local approximate* research metrics.

Rank IC is Pearson correlation of average ranks (ties included); constant inputs
are undefined, never zero-filled. Each timestamp is one cross section and daily
statistics equally weight that day's timestamps. Returns are forward target
returns, not a cash-account backtest: overlapping horizons, fees and execution
are not simulated. No official competition aggregate score is produced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io import digest

KEYS = ["date", "instrument"]
METRIC_SPEC = {
    "rank_ic_mean": ("rank_ic", "mean", "maximize"),
    "rank_ic_ir": ("rank_ic", "ir", "maximize"),
    "ic_mean": ("ic", "mean", "maximize"),
    "long_short_return_mean": ("long_short_return", "mean", "maximize"),
    "long_short_sharpe": ("long_short_return", "sharpe", "maximize"),
    "long_return_mean": ("long_return", "mean", "maximize"),
    "short_return_mean": ("short_return", "mean", "minimize"),
    "turnover_mean": ("turnover", "mean", "minimize"),
    "extreme_return": ("long_short_return", "lower_tail", "maximize"),
    "min_daily_long_short_return": ("long_short_return", "min", "maximize"),
}


def _option(config: Any, name: str) -> Any:
    return config[name] if isinstance(config, dict) else getattr(config, name)


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
    # Scaling before centering avoids overflow for otherwise valid large inputs.
    a = a / max(float(np.max(np.abs(a))), 1.0)
    b = b / max(float(np.max(np.abs(b))), 1.0)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def _tail_weights(scores: pd.Series, count: int, *, largest: bool) -> np.ndarray:
    values = scores.to_numpy(dtype=float)
    ordered = np.sort(values)
    boundary = ordered[-count] if largest else ordered[count - 1]
    strictly = values > boundary if largest else values < boundary
    tied = values == boundary
    weights = strictly.astype(float)
    weights[tied] = (count - int(strictly.sum())) / int(tied.sum())
    return weights / count


def aggregate_metric(values: np.ndarray, method: str, periods_per_year: float = 252) -> float | None:
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


def evaluate(predictions: pd.DataFrame, targets: pd.DataFrame, evaluation_config: Any) -> tuple[dict, pd.DataFrame]:
    """Return JSON-safe metrics plus daily rows, retaining undefined statistics."""
    quantile = float(_option(evaluation_config, "quantile"))
    periods_per_year = float(_option(evaluation_config, "periods_per_year"))
    min_cross_section = int(_option(evaluation_config, "min_cross_section"))
    if not 0 < quantile <= 0.5 or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("quantile must be in (0, 0.5], periods_per_year must be positive")
    if min_cross_section < 2:
        raise ValueError("min_cross_section must be at least 2")
    aligned = align_predictions(predictions, targets)
    rows = []
    previous_weights: pd.Series | None = None
    for timestamp, section in aligned.groupby("date", sort=True):
        if len(section) < min_cross_section:
            raise ValueError(f"cross section {timestamp} has fewer than {min_cross_section} samples")
        score = section["score"]
        target = section["target"]
        tail_size = max(1, int(np.floor(quantile * len(section))))
        long = _tail_weights(score, tail_size, largest=True)
        short = _tail_weights(score, tail_size, largest=False)
        weights = pd.Series(long - short, index=section["instrument"].to_numpy())
        turnover = float("nan")
        if previous_weights is not None:
            union = weights.index.union(previous_weights.index)
            turnover = float(0.5 * np.abs(
                weights.reindex(union, fill_value=0) - previous_weights.reindex(union, fill_value=0)
            ).sum())
        previous_weights = weights
        long_return = float(np.dot(long, target.to_numpy(dtype=float)))
        short_return = float(np.dot(short, target.to_numpy(dtype=float)))
        if not np.isfinite([long_return, short_return, long_return - short_return]).all():
            raise ValueError("portfolio return overflow; target units are not representable")
        rows.append({
            "date": timestamp.normalize(),
            "rank_ic": _correlation(score.rank(method="average").to_numpy(), target.rank(method="average").to_numpy()),
            "ic": _correlation(score.to_numpy(dtype=float), target.to_numpy(dtype=float)),
            "long_return": long_return,
            "short_return": short_return,
            "long_short_return": long_return - short_return,
            "turnover": turnover,
            "sample_count": len(section),
        })
    sections = pd.DataFrame(rows)
    columns = ["rank_ic", "ic", "long_return", "short_return", "long_short_return", "turnover"]
    daily = sections.groupby("date", sort=True)[columns].mean()
    daily["sample_count"] = sections.groupby("date")["sample_count"].sum()
    daily["cross_section_count"] = sections.groupby("date").size()
    daily["valid_rank_ic_count"] = sections.groupby("date")["rank_ic"].count()
    daily["valid_ic_count"] = sections.groupby("date")["ic"].count()
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    metrics = {
        name: aggregate_metric(daily[column].to_numpy(), method, periods_per_year)
        for name, (column, method, _) in METRIC_SPEC.items()
    }
    metrics.update({
        "schema_version": 1,
        "metric_scope": "local_approximation",
        "local_approximation": True,
        "sample_count": len(aligned),
        "day_count": len(daily),
        "cross_section_count": len(sections),
        "sample_keys_sha256": digest(aligned[KEYS].assign(date=aligned["date"].astype(str)).to_dict("records")),
        "targets_sha256": digest(aligned[KEYS + ["target"]].assign(date=aligned["date"].astype(str)).to_dict("records")),
        "undefined_rank_ic_count": int(sections["rank_ic"].isna().sum()),
        "undefined_ic_count": int(sections["ic"].isna().sum()),
        "valid_days": {column: int(daily[column].notna().sum()) for column in columns},
        "conventions": {
            "version": "local-v1",
            "cross_section": "one prediction timestamp; average ranks for ties",
            "daily": "equal timestamp mean; undefined correlations remain missing and are counted",
            "portfolio": "equal tail weights; floor(n*quantile), minimum one; boundary ties share weight",
            "turnover": "0.5*sum(abs(net_weight_t-net_weight_previous)); first timestamp undefined; no drift",
            "rank_ic_ir": "daily mean / sample std (ddof=1), not annualized",
            "long_short_sharpe": "daily mean / sample std * sqrt(periods_per_year); zero variance undefined",
            "extreme_return": "mean of worst ceil(1% * days) daily long-short returns",
            "costs": "gross forward label returns; no fees, execution, overlap or compounding adjustment",
            "official_score": "not implemented: official complete evaluation rules unavailable",
        },
    })
    return metrics, daily
