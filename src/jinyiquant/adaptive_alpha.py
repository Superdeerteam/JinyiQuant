"""Turnover-targeting controller for causal EWMA smoothing of factor outputs.

One smoothing strength is chosen per trading day.  The choice for day ``d`` reads only
days strictly before ``d``: the controller measures, for every candidate ``alpha``, the
intraday churn that each past day's raw cross sections would have produced, forecasts
the day-opening churn from realized history, and inverts the predicted next-day
turnover curve at a predeclared target.  No labels, no returns and no future section
enter the decision, so the loop is causal by construction.

The operators here are the ones already used by the recorded fixed-alpha experiment
(``scripts/turnover_stability_change.py``): a per-timestamp population z-score followed
by a causal EWMA that resets every trading day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import neutral_weights

MIN_CROSS_SECTION = 3


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    return value


@dataclass(frozen=True)
class ControllerConfig:
    """Predeclared controller constants; all windows are counted in trading days."""

    target_turnover: float
    quantile: float = 0.2
    window_days: int = 20
    min_history_days: int = 10
    warmup_alpha: float = 0.5
    alpha_grid: tuple[float, ...] = tuple(round(0.10 + 0.05 * step, 2) for step in range(19))

    def __post_init__(self) -> None:
        if not (np.isfinite(self.target_turnover) and self.target_turnover > 0):
            raise ValueError("target_turnover must be positive and finite")
        if not 0 < float(self.quantile) <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        if self.window_days < 1 or self.min_history_days < 1:
            raise ValueError("window_days and min_history_days must be positive")
        if self.min_history_days > self.window_days:
            raise ValueError("min_history_days cannot exceed window_days")
        _validate_alpha(self.warmup_alpha)
        grid = tuple(_validate_alpha(value) for value in self.alpha_grid)
        if len(grid) < 2 or len(set(grid)) != len(grid) or list(grid) != sorted(grid):
            raise ValueError("alpha_grid needs at least two distinct increasing alphas in (0, 1]")
        object.__setattr__(self, "alpha_grid", grid)


@dataclass(frozen=True)
class SectionPanel:
    """Cross-sectionally standardized panel plus the trading-day layout."""

    stamps: pd.DatetimeIndex
    instruments: pd.Index
    z: np.ndarray
    day_codes: np.ndarray
    day_starts: np.ndarray
    day_ends: np.ndarray
    day_labels: np.ndarray
    day_section_counts: np.ndarray

    @property
    def day_count(self) -> int:
        return int(len(self.day_labels))

    def slice_rows(self, stop: int) -> "SectionPanel":
        """Prefix of the panel; day codes are re-derived so the daily EWMA reset still holds."""
        if stop <= 0 or stop > len(self.z):
            raise ValueError("slice stop is outside the panel")
        stamps = self.stamps[:stop]
        normalized = stamps.normalize()
        day_codes = pd.factorize(normalized)[0].astype(np.int64)
        labels = np.unique(normalized.to_numpy())
        starts = np.searchsorted(day_codes, np.arange(len(labels)), side="left")
        ends = np.searchsorted(day_codes, np.arange(len(labels)), side="right") - 1
        return SectionPanel(stamps=stamps, instruments=self.instruments, z=self.z[:stop].copy(),
                            day_codes=day_codes, day_starts=starts, day_ends=ends,
                            day_labels=labels, day_section_counts=ends - starts + 1)


def build_panel(keys: pd.DataFrame, scores: np.ndarray) -> SectionPanel:
    """Cross-sectionally z-score raw scores and lay them out as (section x instrument)."""
    values = np.asarray(scores, dtype=float)
    if len(keys) != len(values):
        raise ValueError("keys and scores have different lengths")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    frame = pd.DataFrame({"date": keys["date"].to_numpy(), "instrument": keys["instrument"].to_numpy(),
                          "score": values})
    if frame.duplicated().any():
        raise ValueError("(date, instrument) keys must be unique")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("keys must be sorted by date")
    grouped = frame.groupby("date", sort=True)["score"]
    mean = grouped.transform("mean").to_numpy(dtype=float)
    scale = grouped.transform("std", ddof=0).to_numpy(dtype=float)
    if not np.isfinite(scale).all():
        raise ValueError("cross-sectional scale is not finite")
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(scale > 0, (values - mean) / scale, 0.0)
    table = frame.assign(z=z).pivot(index="date", columns="instrument", values="z").sort_index()
    stamps = pd.DatetimeIndex(table.index)
    normalized = stamps.normalize()
    day_codes = pd.factorize(normalized)[0].astype(np.int64)
    labels = np.unique(normalized.to_numpy())
    starts = np.searchsorted(day_codes, np.arange(len(labels)), side="left")
    ends = np.searchsorted(day_codes, np.arange(len(labels)), side="right") - 1
    return SectionPanel(stamps=stamps, instruments=pd.Index(table.columns), z=table.to_numpy(dtype=float),
                        day_codes=day_codes, day_starts=starts, day_ends=ends,
                        day_labels=labels, day_section_counts=ends - starts + 1)


def _normalized_alphas(alphas_by_day: dict) -> dict:
    return {pd.Timestamp(label): _validate_alpha(alpha) for label, alpha in alphas_by_day.items()}


def _alpha_rows(panel: SectionPanel, alphas_by_day: dict) -> np.ndarray:
    lookup = _normalized_alphas(alphas_by_day)
    per_day = []
    for label in panel.day_labels:
        try:
            per_day.append(lookup[pd.Timestamp(label)])
        except KeyError as error:
            raise ValueError(f"no alpha declared for trading day {pd.Timestamp(label).date()}") from error
    return np.asarray(per_day)[panel.day_codes]


def _ewma_rows(z: np.ndarray, alphas: np.ndarray, day_codes: np.ndarray) -> np.ndarray:
    """Row-wise causal EWMA with a state reset at every trading-day boundary."""
    result = np.full_like(z, np.nan, dtype=float)
    state = np.full(z.shape[1], np.nan)
    for row in range(len(z)):
        if row == 0 or day_codes[row] != day_codes[row - 1]:
            state = np.full(z.shape[1], np.nan)
        value = z[row]
        observed = np.isfinite(value)
        blended = np.where(observed,
                           np.where(np.isfinite(state), alphas[row] * value + (1.0 - alphas[row]) * state,
                                    value),
                           state)
        state = blended
        result[row] = np.where(observed, blended, np.nan)
    return result


def smoothed_values(panel: SectionPanel, alphas_by_day: dict) -> np.ndarray:
    """Causal per-day EWMA; the alpha is constant inside a day and the state resets daily."""
    return _ewma_rows(panel.z, _alpha_rows(panel, alphas_by_day), panel.day_codes)


def _section_weight(panel: SectionPanel, row_values: np.ndarray, quantile: float):
    series = pd.Series(row_values, index=panel.instruments, dtype=float).dropna()
    if len(series) < MIN_CROSS_SECTION:
        return None
    return neutral_weights(series, quantile)[2]


def section_weights(panel: SectionPanel, smoothed: np.ndarray, quantile: float) -> list:
    """Net long-short weights of the top/bottom quantile book for every section."""
    return [_section_weight(panel, row, quantile) for row in smoothed]


def weight_change(target: pd.Series | None, pretrade: pd.Series | None) -> float:
    """Half the gross traded notional between consecutive target books (no return drift)."""
    if target is None or pretrade is None:
        return float("nan")
    index = target.index.union(pretrade.index)
    traded = (target.reindex(index, fill_value=0.0) - pretrade.reindex(index, fill_value=0.0)).abs().sum()
    return float(traded) / 2.0


def _day_weights(panel: SectionPanel, day: int, alpha: float, quantile: float) -> list:
    """Weights of every section of one day, resetting the EWMA at the day's first section."""
    start, end = int(panel.day_starts[day]), int(panel.day_ends[day])
    state = np.full(panel.z.shape[1], np.nan)
    weights = []
    for row in range(start, end + 1):
        value = panel.z[row]
        observed = np.isfinite(value)
        state = np.where(observed,
                         np.where(np.isfinite(state), alpha * value + (1.0 - alpha) * state, value),
                         state)
        weights.append(_section_weight(panel, np.where(observed, state, np.nan), quantile))
    return weights


def intraday_churn(panel: SectionPanel, weights: list) -> np.ndarray:
    """Mean of the intraday rebalance churn of each trading day."""
    out = np.full(panel.day_count, np.nan)
    for day in range(panel.day_count):
        start, end = int(panel.day_starts[day]), int(panel.day_ends[day])
        changes = [weight_change(weights[row], weights[row - 1]) for row in range(start + 1, end + 1)]
        changes = [value for value in changes if np.isfinite(value)]
        if changes:
            out[day] = float(np.mean(changes))
    return out


def churn_curve(panel: SectionPanel, grid: tuple[float, ...], quantile: float) -> np.ndarray:
    """Per-day realized intraday churn for every candidate alpha (days x grid)."""
    curve = np.full((panel.day_count, len(grid)), np.nan)
    for column, alpha in enumerate(grid):
        smoothed = smoothed_values(panel, {label: alpha for label in panel.day_labels})
        curve[:, column] = intraday_churn(panel, section_weights(panel, smoothed, quantile))
    return curve


def solve_alpha(curve_mean: np.ndarray, grid: tuple[float, ...], required_churn: float) -> tuple[float, str]:
    """Invert an increasing predicted-churn curve; clamp at the grid edges."""
    x = np.asarray(curve_mean, dtype=float)
    y = np.asarray(grid, dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("forecast curve and alpha grid shapes differ")
    if not np.isfinite(x).all():
        raise ValueError("forecast churn curve is not finite")
    if np.any(np.diff(x) < -1e-9):
        raise ValueError("forecast churn curve is not increasing in alpha")
    if required_churn <= x[0]:
        return float(y[0]), "below_grid"
    if required_churn >= x[-1]:
        return float(y[-1]), "above_grid"
    return float(np.interp(required_churn, x, y)), "ok"


@dataclass(frozen=True)
class ControlPath:
    """Per-day decisions and the diagnostic churn accounting."""

    day_labels: np.ndarray
    alphas: np.ndarray
    section_counts: np.ndarray
    forecast_intraday: np.ndarray
    forecast_turnover: np.ndarray
    realized_intraday: np.ndarray
    realized_overnight: np.ndarray
    realized_turnover: np.ndarray
    ready: np.ndarray
    solver_status: tuple[str, ...]

    @property
    def alpha_lookup(self) -> dict:
        return {pd.Timestamp(label): float(alpha) for label, alpha in zip(self.day_labels, self.alphas)}


def control_path(panel: SectionPanel, curve: np.ndarray, config: ControllerConfig) -> ControlPath:
    """Walk the trading days; every alpha decision reads only already-realized days."""
    if curve.shape != (panel.day_count, len(config.alpha_grid)):
        raise ValueError("churn curve shape does not match the panel and alpha grid")
    grid = config.alpha_grid
    days = panel.day_count
    alphas = np.full(days, np.nan)
    forecast_intraday = np.full(days, np.nan)
    forecast_turnover = np.full(days, np.nan)
    realized_intraday = np.full(days, np.nan)
    realized_overnight = np.full(days, np.nan)
    realized_turnover = np.full(days, np.nan)
    decided = []
    history: list[int] = []
    overnight_history: list[float] = []
    previous_last = None
    for day in range(days):
        start = int(panel.day_starts[day])
        opening = _section_weight(panel, panel.z[start], config.quantile)
        overnight = weight_change(opening, previous_last)
        realized_overnight[day] = overnight
        if len(history) < config.min_history_days:
            alpha = config.warmup_alpha
            decision = "warmup"
        else:
            window = history[-config.window_days:]
            # Realized churn cannot fall as smoothing weakens; project the noisy window mean
            # onto that monotone shape before inverting it.
            forecast = np.maximum.accumulate(np.nanmean(curve[window], axis=0))
            overnight_forecast = float(np.nanmean([overnight_history[index] for index in window]))
            sections_forecast = float(np.median([panel.day_section_counts[index] for index in window]))
            required = (config.target_turnover * sections_forecast - overnight_forecast) / (
                sections_forecast - 1.0)
            alpha, decision = solve_alpha(forecast, grid, required)
            forecast_intraday[day] = float(np.interp(alpha, grid, forecast))
            forecast_turnover[day] = (
                overnight_forecast + (sections_forecast - 1.0) * forecast_intraday[day]
            ) / sections_forecast
        alphas[day] = alpha
        decided.append(decision)
        weights = _day_weights(panel, day, alpha, config.quantile)
        changes = [weight_change(weights[row], weights[row - 1])
                   for row in range(1, len(weights))]
        changes = [value for value in changes if np.isfinite(value)]
        realized_intraday[day] = float(np.mean(changes)) if changes else np.nan
        transitions = [overnight] + changes
        transitions = [value for value in transitions if np.isfinite(value)]
        realized_turnover[day] = float(np.mean(transitions)) if transitions else np.nan
        previous_last = weights[-1]
        history.append(day)
        overnight_history.append(overnight)
    return ControlPath(day_labels=panel.day_labels, alphas=alphas,
                       section_counts=panel.day_section_counts,
                       forecast_intraday=forecast_intraday, forecast_turnover=forecast_turnover,
                       realized_intraday=realized_intraday, realized_overnight=realized_overnight,
                       realized_turnover=realized_turnover,
                       ready=np.arange(days) >= config.min_history_days,
                       solver_status=tuple(decided))


def smooth_scores(keys: pd.DataFrame, scores: np.ndarray, alphas_by_day: dict) -> tuple[np.ndarray, dict]:
    """Apply the controller's per-day alphas and return values in the caller's key order."""
    panel = build_panel(keys, scores)
    lookup = _normalized_alphas(alphas_by_day)
    smoothed = smoothed_values(panel, lookup)
    flat = pd.DataFrame(smoothed, index=panel.stamps, columns=panel.instruments).stack(future_stack=True)
    values = flat.reindex(pd.MultiIndex.from_frame(keys[["date", "instrument"]])).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("smoothed scores do not cover every reference key")
    diagnostics = {
        "operator": "per-timestamp z-score, then causal per-day EWMA along the intraday cross "
                    "sections with one alpha per trading day",
        "reset": "state resets at each trading day",
        "sections": int(len(panel.stamps)),
        "day_count": panel.day_count,
        "alpha_min": float(min(lookup.values())),
        "alpha_max": float(max(lookup.values())),
    }
    return values, diagnostics


def operator_causality(panel: SectionPanel, alphas_by_day: dict) -> dict:
    """Recompute every section from the data truncated there; a future leak would show up."""
    alphas = _alpha_rows(panel, alphas_by_day)
    full = _ewma_rows(panel.z, alphas, panel.day_codes)
    worst = 0.0
    failing = []
    for stop in range(1, len(panel.z) + 1):
        truncated = _ewma_rows(panel.z[:stop], alphas[:stop], panel.day_codes[:stop])
        left, right = full[stop - 1], truncated[-1]
        finite = np.isfinite(left) | np.isfinite(right)
        difference = float(np.nanmax(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        worst = max(worst, difference)
        if difference > 1e-15:
            failing.append({"row": stop - 1, "difference": difference})
    return {"truncated_recompute_max_abs_difference": worst, "passed": worst <= 1e-15,
            "rows_verified": int(len(panel.z)), "failing_rows": failing[:5],
            "method": "for every section, rerun the alpha-augmented EWMA on the panel truncated at "
                      "that section and compare the last value"}
