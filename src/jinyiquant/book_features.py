#!/usr/bin/env python
"""46-feature limit-order-book table (packaged copy of ``scripts/r6r8_book_features.py``).

This module is the in-package, submission-runnable copy of the feature builder that
produced ``runs/r6r8-20261029/features/features_<YYYY-MM>.parquet``. The numerics,
column order and NaN handling are copied verbatim from the extractor script; the only
additions are small wrappers that take an in-memory bar frame instead of reading the
monthly feather files. Do not "improve" the formulas: the LightGBM model was trained
on exactly these values.

Causality contract (unchanged)
------------------------------
* A row stamped t uses only 1-minute bars with timestamp <= t (as-of semantics; the
  09:30 row uses the 09:25 call-auction bar because there is no 09:30 bar).
* All differences/rolling windows look backwards. Forward fill is limited to 5 steps.
  Cross-sectional z-scores use only same-timestamp peers. ``prev_*``/``gap`` use the
  previous trading day only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
SAMPLE_MIN = [570, 600, 630, 660, 690, 810, 840, 870]

BAR_COLS = [
    "date", "instrument_id", "open", "high", "low", "volume", "deal_number",
    "ask_price1", "ask_price2", "ask_price3", "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3",
]
NUM_COLS = [c for c in BAR_COLS if c not in ("date", "instrument_id")]

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
XS_OF = {
    "xs_spread1": "spread1_rel", "xs_imb1": "imb1", "xs_imb3": "imb3",
    "xs_micro1": "micro1_rel", "xs_ret15": "ret_15", "xs_ofi15": "ofi_15",
    "xs_depth1": "depth1_log", "xs_volratio": "vol_ratio_5_30",
}


# ----------------------------------------------------------------------------- helpers
def _ffill(a: np.ndarray, limit: int = 5) -> np.ndarray:
    n, t = a.shape
    idx = np.where(np.isfinite(a), np.arange(t, dtype=np.int32), np.int32(0))
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, idx, axis=1).copy()
    age = np.arange(t, dtype=np.int32)[None, :] - idx
    out[age > limit] = np.nan
    return out


def _shift1(a: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan)
    out[:, 1:] = a[:, :-1]
    return out


def _last_finite(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a)
    idx = np.where(ok, np.arange(a.shape[1]), -1).max(axis=1)
    out = np.full(a.shape[0], np.nan)
    m = idx >= 0
    out[m] = a[np.arange(a.shape[0])[m], idx[m]]
    return out


def _first_positive(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & (a > 0)
    any_ = ok.any(axis=1)
    idx = np.argmax(ok, axis=1)
    out = np.full(a.shape[0], np.nan)
    out[any_] = a[np.arange(a.shape[0])[any_], idx[any_]]
    return out


def _asof_first_positive(a: np.ndarray, idx8: np.ndarray) -> np.ndarray:
    """First strictly positive value per row among slots <= each sample slot."""
    ok = np.isfinite(a) & (a > 0)
    has = ok.any(axis=1)
    fidx = np.argmax(ok, axis=1)
    first = np.full(a.shape[0], np.nan)
    first[has] = a[np.arange(a.shape[0])[has], fidx[has]]
    known = has[:, None] & (fidx[:, None] <= idx8[None, :])
    return np.where(known, first[:, None], np.nan)


def _xs_z(x: np.ndarray) -> np.ndarray:
    """Cross-sectional (axis 0 = instruments) z-score per timestamp column."""
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        z = (x - mu) / np.where(sd > 0, sd, np.nan)
    return np.clip(z, -5.0, 5.0)


# ----------------------------------------------------------------------------- day builder
def build_day(day_cols: dict[str, np.ndarray], inst_ids: np.ndarray, minutes: np.ndarray,
              prev: pd.DataFrame | None, sample_min: list[int] = SAMPLE_MIN):
    """Compute all features for one trading day.

    Returns (feat: dict[str, (n,8) float32], summary: DataFrame indexed by
    instrument_id, ids: sorted instrument ids).
    """
    ids = np.unique(inst_ids)
    n = len(ids)
    ns = len(sample_min)
    if n == 0:
        return {}, None, ids
    rows = np.searchsorted(ids, inst_ids)
    umin = np.unique(minutes)
    t = len(umin)
    cidx = np.searchsorted(umin, minutes)

    def M(name: str) -> np.ndarray:
        a = np.full((n, t), np.nan)
        a[rows, cidx] = day_cols[name]
        return a

    bp1, ap1 = M("bid_price1"), M("ask_price1")
    bp2, ap2 = M("bid_price2"), M("ask_price2")
    bp3, ap3 = M("bid_price3"), M("ask_price3")
    bv1, av1 = M("bid_volume1"), M("ask_volume1")
    bv2, av2 = M("bid_volume2"), M("ask_volume2")
    bv3, av3 = M("bid_volume3"), M("ask_volume3")
    op, hi, lo = M("open"), M("high"), M("low")
    vol, dn = M("volume"), M("deal_number")

    ok1 = (bp1 > 0) & (ap1 > 0) & (ap1 >= bp1)
    mid = np.where(ok1, (bp1 + ap1) * 0.5, np.nan)
    ok2 = ok1 & (bp2 > 0) & (ap2 >= bp2)
    ok3 = ok2 & (bp3 > 0) & (ap3 >= bp3)
    with np.errstate(invalid="ignore"):
        sp1 = np.where(ok1, (ap1 - bp1) / mid, np.nan)
        sp2 = np.where(ok2, (ap2 - bp2) / mid, np.nan)
        sp3 = np.where(ok3, (ap3 - bp3) / mid, np.nan)
        d1 = np.where(ok1, bv1 + av1, np.nan)
        d3s = np.nansum(np.stack([bv1, bv2, bv3, av1, av2, av3]), axis=0)
        d3 = np.where(ok1, d3s, np.nan)
        imb1 = np.where(ok1 & (d1 > 0), (bv1 - av1) / d1, np.nan)
        imb3 = np.where(ok1 & (d3s > 0),
                        (bv1 + bv2 + bv3 - av1 - av2 - av3) / np.where(d3s > 0, d3s, np.nan), np.nan)
        vv1 = np.where(ok1, bv1 + av1, np.nan)
        micro1 = np.where(ok1 & (vv1 > 0), (bp1 * av1 + ap1 * bv1) / vv1, np.nan)
        v3 = np.where(ok1, bv1 + bv2 + bv3 + av1 + av2 + av3, np.nan)
        micro3 = np.where(ok1 & (v3 > 0),
                          (bp1 * av1 + bp2 * av2 + bp3 * av3 + ap1 * bv1 + ap2 * bv2 + ap3 * bv3)
                          / np.where(v3 > 0, v3, np.nan), np.nan)
        mic1 = (micro1 - mid) / mid
        mic3 = (micro3 - mid) / mid

    midf = _ffill(mid)
    sp1f, sp2f, sp3f = _ffill(sp1), _ffill(sp2), _ffill(sp3)
    d1f, d3f = _ffill(d1), _ffill(d3)
    imb1f, imb3f = _ffill(imb1), _ffill(imb3)
    mic1f, mic3f = _ffill(mic1), _ffill(mic3)
    ret1 = np.full_like(midf, np.nan)
    with np.errstate(invalid="ignore"):
        ret1[:, 1:] = np.log(midf[:, 1:] / midf[:, :-1])

    idx8 = np.searchsorted(umin, np.array(sample_min), side="right") - 1
    valid8 = idx8 >= 0
    idx8c = np.clip(idx8, 0, t - 1)

    def at(a: np.ndarray, lag: int = 0) -> np.ndarray:
        ok = idx8 >= lag
        vals = a[:, np.clip(idx8c - lag, 0, t - 1)]
        return np.where(ok[None, :], vals, np.nan)

    def win(a: np.ndarray, k: int) -> np.ndarray:
        s = idx8c[:, None] - np.arange(k)[None, :]
        ok = (s >= 0) & valid8[:, None]
        vals = a[:, np.clip(s, 0, t - 1)]
        return np.where(ok[None, :, :], vals, np.nan)

    def nsum(a: np.ndarray, k: int) -> np.ndarray:
        w = win(a, k)
        cnt = np.isfinite(w).sum(axis=2)
        return np.where(cnt > 0, np.nansum(w, axis=2), np.nan)

    def nmean(a: np.ndarray, k: int) -> np.ndarray:
        return np.nanmean(win(a, k), axis=2)

    bpf, apf = _ffill(bp1), _ffill(ap1)
    bvf, avf = _ffill(bv1), _ffill(av1)
    pb, pa, pbv, pav = _shift1(bpf), _shift1(apf), _shift1(bvf), _shift1(avf)
    with np.errstate(invalid="ignore"):
        e = ((bpf >= pb) * bvf - (bpf <= pb) * pbv
             - (apf <= pa) * avf + (apf >= pa) * pav)
    e = np.where(np.isfinite(e), e, 0.0)

    m0 = at(midf, 0)
    feat: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        feat["spread1_rel"] = at(sp1f, 0)
        feat["spread2_rel"] = at(sp2f, 0)
        feat["spread3_rel"] = at(sp3f, 0)
        feat["depth1_log"] = np.log1p(at(d1f, 0))
        feat["depth3_log"] = np.log1p(at(d3f, 0))
        feat["imb1"] = at(imb1f, 0)
        feat["imb3"] = at(imb3f, 0)
        feat["micro1_rel"] = at(mic1f, 0)
        feat["micro3_rel"] = at(mic3f, 0)
        for k in (1, 5, 15, 30, 60):
            feat[f"ret_{k}"] = np.log(m0 / at(midf, k))
        open0 = _asof_first_positive(op, idx8)
        feat["ret_since_open"] = np.log(m0 / open0)
        feat["rv_30"] = np.nanstd(win(ret1, 30), axis=2)
        hi_sofar = np.fmax.accumulate(np.where(np.isfinite(hi), hi, np.nan), axis=1)
        lo_sofar = np.fmin.accumulate(np.where(np.isfinite(lo), lo, np.nan), axis=1)
        hi8, lo8 = at(hi_sofar, 0), at(lo_sofar, 0)
        rng = hi8 - lo8
        feat["pos_in_range"] = np.clip((m0 - lo8) / np.where(rng > 0, rng, np.nan), 0.0, 1.0)
        feat["minutes_since_open"] = np.broadcast_to(
            (np.array(sample_min) - 570).astype(float)[None, :], (n, ns)).copy()
        for k in (5, 15, 30):
            feat[f"ofi_{k}"] = nsum(e, k) / (nmean(d1f, k) + 1e-9)
        for k in (1, 5, 15):
            feat[f"dlogdepth_{k}"] = np.log(at(d3f, 0) / at(d3f, k))
        feat["dimb3_5"] = at(imb3f, 0) - at(imb3f, 5)
        feat["depth1_vs_30m"] = np.log(at(d1f, 0) / (nmean(d1f, 30) + 1e-9))
        feat["vol_ratio_5_30"] = nsum(vol, 5) / (nsum(vol, 30) + 1.0)
        feat["log_vol_30"] = np.log1p(nsum(vol, 30))
        feat["log_deals_30"] = np.log1p(nsum(dn, 30))
        feat["mean_spread_30"] = nmean(sp1f, 30)

    prev_al = prev.reindex(ids) if (prev is not None and len(prev)) else None
    for name, col in (("prev_ret", "ret"), ("prev_rv", "rv"), ("prev_spread", "spread"),
                      ("prev_imb", "imb"), ("prev_range", "range"),
                      ("prev_last30", "last30"), ("prev_last60", "last60")):
        if prev_al is None:
            feat[name] = np.full((n, ns), np.nan)
        else:
            v = prev_al[col].to_numpy(dtype=float)[:, None]
            feat[name] = np.broadcast_to(v, (n, ns)).copy()
    if prev_al is None:
        feat["gap"] = np.full((n, ns), np.nan)
    else:
        with np.errstate(invalid="ignore"):
            g = np.log(open0 / prev_al["last_mid"].to_numpy(dtype=float)[:, None])
            feat["gap"] = np.ascontiguousarray(np.broadcast_to(g, (n, ns))).astype(np.float32)

    for out, src in XS_OF.items():
        feat[out] = _xs_z(feat[src])

    with np.errstate(invalid="ignore"):
        last_mid = _last_finite(midf)
        l30 = np.log(last_mid / midf[:, np.clip(t - 31, 0, t - 1)])
        l60 = np.log(last_mid / midf[:, np.clip(t - 61, 0, t - 1)])
        summary = pd.DataFrame({
            "ret": np.log(last_mid / _first_positive(op)),
            "rv": np.nanstd(ret1, axis=1),
            "spread": np.nanmean(sp1f, axis=1),
            "imb": np.nanmean(imb1f, axis=1),
            "range": (np.nanmax(hi, axis=1) - np.nanmin(lo, axis=1)) / last_mid,
            "last30": l30, "last60": l60, "last_mid": last_mid,
        }, index=pd.Index(ids, name="instrument_id"))

    return {k: np.asarray(v, dtype=np.float32) for k, v in feat.items()}, summary, ids


# ----------------------------------------------------------------------------- row layout
def rows_for_day(feat: dict[str, np.ndarray], ids: np.ndarray, day: np.datetime64,
                 idm: dict, sample_min: list[int] = SAMPLE_MIN) -> pd.DataFrame:
    """Long frame for one day: timestamp-major, instrument-minor (extractor order)."""
    ns = len(sample_min)
    n = len(ids)
    df = pd.DataFrame({k: v.T.reshape(-1) for k, v in feat.items()})
    df.insert(0, "instrument_id", np.tile(ids, ns))
    df.insert(0, "instrument", np.tile([idm[x] for x in ids], ns))
    stamps = (day.astype("datetime64[m]") + np.array(sample_min, dtype="timedelta64[m]"))
    df.insert(0, "date", np.repeat(stamps.astype("datetime64[ns]"), n))
    return df


def day_inputs(bars: pd.DataFrame) -> dict:
    """Split a bar frame into per-day (cols, ids, minutes) inputs in stable order."""
    ts = bars["date"].to_numpy()
    arr = {c: bars[c].to_numpy() for c in NUM_COLS}
    ids = bars["instrument_id"].to_numpy()
    minute = ((ts.astype("datetime64[m]") - ts.astype("datetime64[D]"))
              / np.timedelta64(1, "m")).astype(np.int16)
    day = ts.astype("datetime64[D]")
    order = np.argsort(day, kind="stable")
    uniq, starts = np.unique(day[order], return_index=True)
    starts = np.append(starts, len(order))
    out = {}
    for i, d in enumerate(uniq):
        pos = order[starts[i]:starts[i + 1]]
        out[d] = ({c: arr[c][pos].astype(np.float64) for c in NUM_COLS}, ids[pos], minute[pos])
    return out


def build_table(bars: pd.DataFrame, idm: dict, prev: pd.DataFrame | None = None,
                sample_min: list[int] = SAMPLE_MIN, targets: pd.DataFrame | None = None):
    """Build the 46-feature long table for every day in ``bars`` (chronological).

    ``bars`` must carry :data:`BAR_COLS` (19 columns); ``idm`` maps instrument_id ->
    instrument string; ``prev`` is the previous trading day's summary (from the prior
    call/month), needed for the ``prev_*``/``gap`` block. Returns (frame, last_summary).
    """
    days = day_inputs(bars)
    frames = []
    for d in sorted(days):
        cols, dids, mins = days[d]
        feat, summary, sids = build_day(cols, dids, mins, prev, sample_min)
        if len(sids):
            frames.append(rows_for_day(feat, sids, d, idm, sample_min))
        prev = summary
    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame({c: [] for c in FEATURES}))
    if targets is not None:
        frame = frame.merge(targets, on=["date", "instrument"], how="left")
    return frame, prev
