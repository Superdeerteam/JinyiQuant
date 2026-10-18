"""Frozen local risk-residual pipeline: previous-day exposure, equal-weight OLS.

The competition text states that a submitted factor is winsorized, standardized and
then regressed on BARRA risk factors, with the residual used for scoring. The
official implementation is not available locally, so this module is the *local
approximation* frozen in ``configs/risk_neutralization_20261018_plan.json``:

* a signal date uses the most recent exposure snapshot with a strictly earlier
  date; the official ``cpt_jyc_2026_exposure`` rows are end-of-day snapshots
  (their ``ret`` equals the same day's return and ``date`` carries no publication
  time), so same-day rows are never used for an intraday signal;
* scores are winsorized at median +/- 3 * 1.4826 * MAD and z-scored per section;
* the regression is equal-weight OLS on the ten style factors plus all 32
  industry dummies, with no intercept (the dummies span it);
* ``ret``, ``weights``, ``float_market_cap`` and the industry label never enter
  the design, and a stock without a previous-day exposure row is dropped from
  that timestamp for every scheme (no forward fill).

Rank-based metrics are invariant to monotone score transforms, so the
winsorization and z-score affect only the regression fit, not the definition of
ranking quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jinyiquant.evaluation import tail_weights

STYLE_FACTORS = ("SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
                 "EARNYILD", "GROWTH", "LEVERAGE")
INDUSTRY_DUMMIES = ("AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL", "ELECTRONICS",
                    "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE", "LIGHTINDUS", "HEALTH",
                    "UTILITIES", "TRANSPORTATION", "REALESTATE", "COMMETRADE", "LEISERVICE", "BANK",
                    "NONBANKFINAN", "CONGLOMERATES", "CONMAT", "BUILDDECO", "ELECEQP", "MACHIEQUIP",
                    "AERODEF", "COMPUTER", "MEDIA", "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY")
REGRESSORS = STYLE_FACTORS + INDUSTRY_DUMMIES
EXCLUDED = ("ret", "weights", "float_market_cap", "industry_level1_code")


def previous_exposure_date(exposure_dates, signal_date) -> pd.Timestamp:
    """Latest exposure date strictly before the signal date (midnight-normalized)."""
    dates = np.asarray(exposure_dates, dtype="datetime64[ns]")
    if dates.ndim != 1 or not len(dates) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("exposure dates must be a nonempty strictly increasing 1-D array")
    moment = np.datetime64(pd.Timestamp(signal_date).normalize(), "ns")
    position = int(np.searchsorted(dates, moment, side="left")) - 1
    if position < 0:
        raise ValueError(f"no exposure snapshot strictly before {signal_date}")
    return pd.Timestamp(dates[position])


def section_panel(scores, instruments, snapshot, *, columns=REGRESSORS) -> pd.DataFrame:
    """Join one timestamp's scores with a lagged exposure snapshot.

    Only ``columns`` leave the snapshot, so ``ret``/``weights``/market cap cannot
    reach the regression; names without a row are dropped from the timestamp.
    """
    scores = np.asarray(scores, dtype=float)
    instruments = np.asarray(instruments, dtype=object)
    if scores.ndim != 1 or len(scores) != len(instruments) or not len(scores):
        raise ValueError("scores and instruments must be equal-length nonempty vectors")
    if not np.isfinite(scores).all():
        raise ValueError("section scores must be finite")
    missing = set(columns) - set(snapshot.columns)
    if missing:
        raise ValueError(f"exposure snapshot lacks {sorted(missing)}")
    available = snapshot.set_index("instrument")
    if available.index.has_duplicates:
        raise ValueError("exposure snapshot contains duplicate instruments")
    keep = pd.Index(instruments).isin(available.index)
    if not keep.any():
        raise ValueError("no stock in this section has a lagged exposure snapshot")
    panel = pd.DataFrame({"instrument": instruments[keep], "score": scores[keep]})
    factors = available.loc[panel["instrument"], list(columns)].reset_index(drop=True)
    if factors.isna().any().any() or not np.isfinite(factors.to_numpy(dtype=float)).all():
        raise ValueError("exposure snapshot contains undefined regressors")
    return pd.concat([panel.reset_index(drop=True), factors], axis=1).sort_values(
        "instrument", ignore_index=True)


def winsorized_zscore(values, *, k: float = 3.0) -> tuple[np.ndarray, int]:
    """Section winsorization at median +/- k robust sigma, then ddof=0 z-score."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("winsorization requires a finite nonempty vector")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive and finite")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale == 0.0:
        scale = float(values.std(ddof=0))
    clipped = 0
    if scale > 0.0:
        lower, upper = median - k * scale, median + k * scale
        clipped = int(np.count_nonzero((values < lower) | (values > upper)))
        values = np.clip(values, lower, upper)
    spread = float(values.std(ddof=0))
    if spread == 0.0:
        return np.zeros_like(values), clipped
    return (values - values.mean()) / spread, clipped


def residualize(standardized, design) -> tuple[np.ndarray, float]:
    """Equal-weight OLS residual of the standardized score and its explained share."""
    standardized = np.asarray(standardized, dtype=float)
    design = np.asarray(design, dtype=float)
    if (design.ndim != 2 or len(design) != len(standardized)
            or not np.isfinite(design).all() or not np.isfinite(standardized).all()):
        raise ValueError("residualization requires finite aligned scores and design matrix")
    coefficients, *_ = np.linalg.lstsq(design, standardized, rcond=None)
    residual = standardized - design @ coefficients
    total = float(np.sum(np.square(standardized - standardized.mean())))
    explained = 0.0 if total == 0.0 else 1.0 - float(np.sum(np.square(residual))) / total
    return residual, explained


def portfolio_net_weights(scores, *, quantile: float) -> np.ndarray:
    """Long-minus-short net weights with the local-v1 tail convention."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("portfolio weights require a finite section of at least two scores")
    if not np.isfinite(quantile) or not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    count = max(1, int(np.floor(quantile * len(values))))
    return tail_weights(values, count, largest=True) - tail_weights(values, count, largest=False)


def trailing_median(series, dates, *, before, window: int = 20) -> float:
    """Median of a daily panel series over the ``window`` dates strictly before ``before``."""
    if type(window) is not int or window < 2:
        raise ValueError("window must be an integer of at least two dates")
    dates = np.asarray(dates, dtype="datetime64[ns]")
    position = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(before).normalize(), "ns"),
                                   side="left"))
    if position < window:
        raise ValueError(f"need {window} exposure dates before {before}, have {position}")
    chosen = pd.DatetimeIndex(dates[position - window:position])
    values = series.reindex(chosen).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("trailing baseline contains undefined days")
    return float(np.median(values))
