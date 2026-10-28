#!/usr/bin/env python
"""LightGBM tree arm + house fusion operator, packaged for submission inference.

Two responsibilities:

1. ``load_booster`` / ``candidate_scores`` — run the 46-feature LightGBM tree arm
   (``runs/r6r8-20261029/screen/model.txt``) on a feature frame.
2. ``smoothed_arm_score`` / ``fused_scores`` — the exact fusion operator the increment
   gate measured: per-arm equally-weighted member mean -> per-timestamp cross-sectional
   z-score -> causal EWMA (alpha 0.9111677303420447) -> convex blend with weight 0.5.

``cross_sectional_z`` and ``causal_ewma`` are byte-for-byte copies of
``scripts/verify_turnover_change.py`` (the gate imports that module), so the packaged
path is the gate's path; ``scripts/tree_package_selfcheck.py`` proves equality by
importing both and comparing outputs elementwise.

The function shape mirrors ``scripts/fusion_increment_gate.py``:
``smoothed_arm_score(members, ...)`` averages raw member scores once, then smooths;
both the incumbent trunk arm and the candidate tree arm go through it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EWMA_ALPHA = 0.9111677303420447
DEFAULT_WEIGHT = 0.5
KEYS = ["date", "instrument"]

FEATURES = [
    "spread1_rel", "spread2_rel", "spread3_rel",
    "depth1_log", "depth3_log", "imb1", "imb3", "micro1_rel", "micro3_rel",
    "ret_1", "ret_5", "ret_15", "ret_30", "ret_60", "ret_since_open",
    "rv_30", "pos_in_range", "minutes_since_open",
    "ofi_5", "ofi_15", "ofi_30", "dlogdepth_1", "dlogdepth_5", "dlogdepth_15",
    "dimb3_5", "depth1_vs_30m", "vol_ratio_5_30", "log_vol_30", "log_deals_30",
    "mean_spread_30",
    "prev_ret", "prev_rv", "prev_spread", "prev_imb", "prev_range",
    "prev_last30", "prev_last60", "gap",
    "xs_spread1", "xs_imb1", "xs_imb3", "xs_micro1", "xs_ret15",
    "xs_ofi15", "xs_depth1", "xs_volratio",
]


# ----------------------------------------------------------------------------- tree arm
def load_booster(model_txt: str, num_threads: int = 4):
    """Load the frozen LightGBM text model (trained with lightgbm 4.4.0).

    The text is read with Python IO and parsed from a string: LightGBM's C++ reader
    cannot open non-ASCII paths on Windows (this workspace lives under 进益资本杯).
    """
    import lightgbm as lgb

    model = lgb.Booster(model_str=Path(model_txt).read_text(encoding="utf-8"))
    model.num_threads = num_threads
    return model


def candidate_scores(features, booster=None, num_threads: int = 4) -> np.ndarray:
    """Tree-arm score for every row of ``features`` (float64, training order)."""
    if booster is None:
        raise ValueError("booster is required: pass load_booster(model_txt)")
    if isinstance(features, pd.DataFrame):
        matrix = features[FEATURES].to_numpy(dtype=np.float32)
    else:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURES):
            raise ValueError(f"expected (n, {len(FEATURES)}) features, got {matrix.shape}")
    score = booster.predict(matrix, num_threads=num_threads).astype(np.float64)
    if not np.isfinite(score).all():
        raise ValueError("tree arm produced non-finite scores")
    return score


# ----------------------------------------------------------------------------- fusion ops
# copied verbatim from scripts/verify_turnover_change.py (gate import)
def cross_sectional_z(keys: pd.DataFrame, scores) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    frame = pd.DataFrame({"date": keys["date"].to_numpy(), "score": values})
    grouped = frame.groupby("date")["score"]
    mean = grouped.transform("mean").to_numpy(dtype=float)
    scale = grouped.transform("std", ddof=0).to_numpy(dtype=float)
    safe = np.where(scale > 0, scale, 1.0)
    return np.where(scale > 0, (values - mean) / safe, 0.0)


def causal_ewma(keys: pd.DataFrame, scores, alpha: float = EWMA_ALPHA) -> np.ndarray:
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
    smoothed = pd.DataFrame(output, index=panel.index, columns=panel.columns).stack(
        future_stack=True)
    values = smoothed.reindex(
        pd.MultiIndex.from_frame(keys[["date", "instrument"]])).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("the operator does not cover every reference key")
    return values


def smoothed_arm_score(members, keys: pd.DataFrame, alpha: float = EWMA_ALPHA) -> np.ndarray:
    """Average raw member scores once, then z-score + causal EWMA (gate semantics)."""
    stack = [np.asarray(member, dtype=float) for member in members]
    if not stack:
        raise ValueError("arm has no members")
    blended = np.mean(np.stack(stack, axis=0), axis=0)
    return causal_ewma(keys, blended, alpha)


def fuse(incumbent: np.ndarray, candidate: np.ndarray, weight: float = DEFAULT_WEIGHT):
    return (1.0 - weight) * incumbent + weight * candidate


def fused_scores(keys: pd.DataFrame, trunk_members, candidate: np.ndarray,
                 weight: float = DEFAULT_WEIGHT, alpha: float = EWMA_ALPHA) -> np.ndarray:
    """Full gate path: smooth each arm, then convex-combine (weight on the tree arm)."""
    incumbent = smoothed_arm_score(trunk_members, keys, alpha)
    tree = smoothed_arm_score([candidate], keys, alpha)
    return fuse(incumbent, tree, weight)
