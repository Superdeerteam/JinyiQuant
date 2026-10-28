#!/usr/bin/env python
"""R6+R8 market-book feature table v1 (independent of the deep window builder).

Purpose
-------
Build a causal, month-by-month feature table of explicit limit-order-book
statistics at the 8 official sample times (09:30/10:00/10:30/11:00/11:30/
13:30/14:00/14:30) for the LightGBM screening experiment (task t1).

Causality contract
------------------
* A feature row stamped t uses ONLY 1-minute bars with timestamp <= t
  (as-of semantics: the last bar at or before t; 09:30 therefore uses the
  09:25 call-auction bar because there is no 09:30 bar).
* Backward differences / rolling windows only. Forward fill (limit 5 steps)
  is backward-looking. Cross-sectional z-scores use only same-timestamp
  peers (known at t). `prev_*`/`gap` use the previous trading day only.
* Proven by `--check-causality`: (a) truncating every day's bars at t must
  leave the feature row at t bitwise identical; (b) scrambling all bars
  after t, and the whole next day, must not change any row <= t.

Outputs (per month parquet under --out):
  date, instrument, instrument_id, <46 features>, target (vwap_return, may be NaN)
One-minute snapshot differences are labelled OFI *proxies* (ofi_*), not true
event-level order-flow imbalance.

Usage
-----
  python scripts/r6r8_book_features.py --out runs/r6r8-20261029/features
  python scripts/r6r8_book_features.py --check-causality 2024-01
  python scripts/r6r8_book_features.py --months 2024-01 --out /tmp/x
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "数据"
BARS = DATA / "cpt_jyc_2026_e2e_bar1m"
VWAP = DATA / "cpt_jyc_2026_vwap"
MAPPING = DATA / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"

SAMPLE_TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
SAMPLE_MIN = [570, 600, 630, 660, 690, 810, 840, 870]

BAR_COLS = [
    "date", "instrument_id", "open", "high", "low", "volume", "deal_number",
    "ask_price1", "ask_price2", "ask_price3", "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3",
]
NUM_COLS = [c for c in BAR_COLS if c not in ("date", "instrument_id")]

FEATURES = [
    # book state at t
    "spread1_rel", "spread2_rel", "spread3_rel",
    "depth1_log", "depth3_log", "imb1", "imb3", "micro1_rel", "micro3_rel",
    # mid-price history (backward log returns)
    "ret_1", "ret_5", "ret_15", "ret_30", "ret_60", "ret_since_open",
    "rv_30", "pos_in_range", "minutes_since_open",
    # depth / flow changes ("OFI proxy" = 1-min snapshot differences)
    "ofi_5", "ofi_15", "ofi_30", "dlogdepth_1", "dlogdepth_5", "dlogdepth_15",
    "dimb3_5", "depth1_vs_30m", "vol_ratio_5_30", "log_vol_30", "log_deals_30",
    "mean_spread_30",
    # previous trading day (strictly before today)
    "prev_ret", "prev_rv", "prev_spread", "prev_imb", "prev_range",
    "prev_last30", "prev_last60", "gap",
    # same-timestamp cross-sectional z-scores of the above
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
    """First strictly positive value per row among slots <= each sample slot.

    Used for the day's opening price: at sample slot j only bars up to idx8[j]
    are known, so an instrument whose first bar appears later has no opening
    price yet (NaN) instead of leaking the future first print.
    """
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

    # as-of slot for each sample time: last bar at or before the sample minute
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

    # order-flow imbalance proxy (Cont/Kukanov/Stoikov on 1-min snapshots)
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


# ----------------------------------------------------------------------------- month runner
def _mapping() -> pd.DataFrame:
    return pd.read_csv(MAPPING)


def _month_arrays(month: str, mapp: pd.DataFrame):
    bars = pd.read_feather(BARS / f"{month.replace('-', '')}.0.feather", columns=BAR_COLS)
    bars = bars[bars["instrument_id"].isin(mapp["instrument_id"].to_numpy())]
    ts = bars["date"].to_numpy()
    idm = mapp.set_index("instrument_id")["instrument"].to_dict()
    arr = {c: bars[c].to_numpy() for c in NUM_COLS}
    ids = bars["instrument_id"].to_numpy()
    del bars
    minute = ((ts.astype("datetime64[m]") - ts.astype("datetime64[D]"))
              / np.timedelta64(1, "m")).astype(np.int16)
    day = ts.astype("datetime64[D]")
    order = np.argsort(day, kind="stable")
    uniq, starts = np.unique(day[order], return_index=True)
    starts = np.append(starts, len(order))
    pos_by_day = {d: order[starts[i]:starts[i + 1]] for i, d in enumerate(uniq)}
    return arr, ids, minute, day, pos_by_day, idm


def _rows(feat: dict[str, np.ndarray], ids: np.ndarray, day: np.datetime64,
          idm: dict, sample_min: list[int]) -> pd.DataFrame:
    ns = len(sample_min)
    n = len(ids)
    df = pd.DataFrame({k: v.T.reshape(-1) for k, v in feat.items()})
    df.insert(0, "instrument_id", np.tile(ids, ns))
    df.insert(0, "instrument", np.tile([idm[x] for x in ids], ns))
    stamps = (day.astype("datetime64[m]") + np.array(sample_min, dtype="timedelta64[m]"))
    df.insert(0, "date", np.repeat(stamps.astype("datetime64[ns]"), n))
    return df


def process_month(month: str, mapp: pd.DataFrame, prev: pd.DataFrame | None,
                  sample_min: list[int] = SAMPLE_MIN, emit: bool = True):
    t0 = time.time()
    arr, ids, minute, day, pos_by_day, idm = _month_arrays(month, mapp)
    frames, n_rows = [], 0
    for d in sorted(pos_by_day):
        pos = pos_by_day[d]
        cols = {c: arr[c][pos].astype(np.float64) for c in NUM_COLS}
        feat, summary, dids = build_day(cols, ids[pos], minute[pos], prev, sample_min)
        if emit and len(dids):
            frames.append(_rows(feat, dids, d, idm, sample_min))
            n_rows += len(dids) * len(sample_min)
        prev = summary
    del arr, ids, minute
    if not emit:
        return None, prev, {"month": month, "rows": 0, "seconds": round(time.time() - t0, 1)}
    feats = pd.concat(frames, ignore_index=True)
    del frames
    lpath = VWAP / f"cpt_jyc_2026_vwap_{month}.parquet"
    if lpath.exists():
        labels = pd.read_parquet(lpath, columns=["date", "instrument", "vwap_return"])
        feats = feats.merge(labels, on=["date", "instrument"], how="left")
        feats = feats.rename(columns={"vwap_return": "target"})
    else:
        feats["target"] = np.nan
    dt = time.time() - t0
    cov = float(feats["target"].notna().mean())
    print(f"[{month}] rows={len(feats):,} target_cov={cov:.4f} "
          f"nan_rate={feats[FEATURES].isna().mean().mean():.3f} {dt:.1f}s", flush=True)
    meta = {"month": month, "rows": int(len(feats)), "target_coverage": cov,
            "seconds": round(dt, 1), "features": len(FEATURES)}
    return feats, prev, meta


# ----------------------------------------------------------------------------- causality proof
def _run_days(day_inputs: dict, order: list, start_prev=None):
    """day_inputs[d] = (cols dict, ids, minutes); returns {d: (feat, ids, summary)}."""
    prev = start_prev
    out = {}
    for d in order:
        cols, dids, mins = day_inputs[d]
        feat, summary, sids = build_day(cols, dids, mins, prev)
        out[d] = (feat, sids, summary)
        prev = summary
    return out


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        d = np.where(both_nan, 0.0, np.abs(a - b))
    return float(np.nanmax(d)) if np.isfinite(d).any() else (0.0 if both_nan.all() else float("nan"))


def causality_check(month: str) -> dict:
    """Prove every feature at t depends only on bars <= t (and the previous day)."""
    mapp = _mapping()
    arr, ids, minute, day, pos_by_day, idm = _month_arrays(month, mapp)
    days = sorted(pos_by_day)
    if len(days) < 3:
        raise SystemExit("need >= 3 days")
    dprev, dtgt = days[-3], days[-2]
    day_inputs = {}
    for d in (dprev, dtgt):
        pos = pos_by_day[d]
        day_inputs[d] = ({c: arr[c][pos].astype(np.float64) for c in NUM_COLS},
                         ids[pos], minute[pos])
    # baseline: dprev then target day
    base = _run_days(day_inputs, [dprev, dtgt])
    prev_summary = base[dprev][2]
    base_feat, base_ids = base[dtgt][0], base[dtgt][1]
    report: dict = {"month": month, "prev_day": str(dprev), "target_day": str(dtgt),
                    "n_instruments": int(len(base_ids))}

    # (a) truncation: bars after each sample time removed -> row at t must be identical
    cols_t, ids_t, mins_t = day_inputs[dtgt]
    trunc = {}
    for j, tmin in enumerate(SAMPLE_MIN):
        m = mins_t <= tmin
        feat_j, _, ids_j = build_day({k: v[m] for k, v in cols_t.items()}, ids_t[m], mins_t[m],
                                     prev_summary)
        if len(ids_j) != len(base_ids) or not np.array_equal(ids_j, base_ids):
            trunc[SAMPLE_TIMES[j]] = {"identical": False, "reason": "instrument set changed"}
            continue
        worst = {name: _diff(feat_j[name][:, j], base_feat[name][:, j]) for name in FEATURES}
        bad = {k: v for k, v in worst.items() if not (v == 0.0)}
        trunc[SAMPLE_TIMES[j]] = {"columns": len(FEATURES), "identical": not bad,
                                  "max_abs_diff": max(worst.values()) if worst else 0.0,
                                  "mismatch_columns": bad}
    report["truncation"] = trunc

    # (b) scramble all bars of the target day after 10:00 -> rows <= 10:00 unchanged
    cols_s = {k: v.copy() for k, v in cols_t.items()}
    fut = np.where(mins_t > 600)[0]
    for k in cols_s:
        cols_s[k][fut] = cols_s[k][fut][::-1]
    day_inputs_b = dict(day_inputs)
    day_inputs_b[dtgt] = (cols_s, ids_t, mins_t)
    pert = _run_days(day_inputs_b, [dprev, dtgt])
    pf = pert[dtgt][0]
    early = [j for j, tmin in enumerate(SAMPLE_MIN) if tmin <= 600]
    late = [j for j, tmin in enumerate(SAMPLE_MIN) if tmin > 600]
    report["future_perturb_1000"] = {
        "rows_le_1000_max_abs_diff": max(_diff(pf[n][:, early], base_feat[n][:, early])
                                         for n in FEATURES),
        "rows_after_1000_max_abs_diff": max(_diff(pf[n][:, late], base_feat[n][:, late])
                                            for n in FEATURES),
    }

    # (c) scramble the whole next day -> target-day rows unchanged
    if len(days) >= 3:
        dnext = days[-1]
        pos = pos_by_day[dnext]
        nc = {c: arr[c][pos].astype(np.float64) for c in NUM_COLS}
        for k in nc:
            nc[k] = nc[k][::-1]
        day_inputs_c = dict(day_inputs)
        day_inputs_c[dnext] = (nc, ids[pos], minute[pos])
        pert2 = _run_days(day_inputs_c, [dprev, dtgt, dnext])
        pf2 = pert2[dtgt][0]
        report["next_day_perturb"] = {
            "target_day_max_abs_diff": max(_diff(pf2[n], base_feat[n]) for n in FEATURES)}

    report["ok"] = (all(v.get("identical") for v in trunc.values())
                    and report["future_perturb_1000"]["rows_le_1000_max_abs_diff"] == 0.0
                    and report["next_day_perturb"]["target_day_max_abs_diff"] == 0.0
                    and report["future_perturb_1000"]["rows_after_1000_max_abs_diff"] > 0.0)
    return report


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--out", default="runs/r6r8-20261029/features")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-causality", default=None, metavar="YYYY-MM")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.check_causality:
        result = causality_check(args.check_causality)
        (out / f"causality_{args.check_causality}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "truncation"}, indent=2, default=str))
        return

    if args.months:
        months = args.months
    else:
        train = [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13) if f"{y}-{m:02d}" <= "2023-10"]
        screen = ["2023-11", "2024-01", "2024-02", "2024-03", "2024-04",
                  "2024-05", "2024-07", "2024-08"]
        months = ["2021-12"] + train + screen  # 2021-12 only to seed prev-day summaries
    summary_only = {"2021-12"}
    meta_path = out / "extract_meta.jsonl"
    mapp = _mapping()
    prev = None
    for month in months:
        emit = month not in summary_only
        fpath = out / f"features_{month}.parquet"
        spath = out / f"prev_{month}.parquet"
        if fpath.exists() and not args.overwrite:
            print(f"[{month}] exists, skip", flush=True)
            prev = pd.read_parquet(spath) if spath.exists() else None
            continue
        feats, prev, meta = process_month(month, mapp, prev, emit=emit)
        if emit:
            feats.to_parquet(fpath, index=False)
            del feats
        if prev is not None:
            prev.to_parquet(spath, index=True)
        with meta_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- feature audit
FEATURE_NOTES = {
    "spread1_rel": ("at t", "level-1 relative spread (ask1-bid1)/mid"),
    "spread2_rel": ("at t", "level-2 relative spread (ask2-bid2)/mid"),
    "spread3_rel": ("at t", "level-3 relative spread (ask3-bid3)/mid"),
    "depth1_log": ("at t", "log1p(bid_volume1+ask_volume1)"),
    "depth3_log": ("at t", "log1p(bid+ask volumes, levels 1-3)"),
    "imb1": ("at t", "(bid_volume1-ask_volume1)/depth1"),
    "imb3": ("at t", "(sum bid vol - sum ask vol)/sum depth, levels 1-3"),
    "micro1_rel": ("at t", "(level-1 microprice - mid)/mid"),
    "micro3_rel": ("at t", "(level-1..3 volume-weighted microprice - mid)/mid"),
    "ret_1": ("1 bar", "log(mid_t/mid_{t-1})"),
    "ret_5": ("5 bars", "log(mid_t/mid_{t-5})"),
    "ret_15": ("15 bars", "log(mid_t/mid_{t-15})"),
    "ret_30": ("30 bars", "log(mid_t/mid_{t-30})"),
    "ret_60": ("60 bars", "log(mid_t/mid_{t-60})"),
    "ret_since_open": ("since open, <= t", "log(mid_t / first open known at t)"),
    "rv_30": ("30 bars", "std of 1-min mid log returns over last 30 bars"),
    "pos_in_range": ("running, <= t", "(mid-running low)/(running high-running low)"),
    "minutes_since_open": ("a priori", "sample time minus 09:30 in clock minutes"),
    "ofi_5": ("5 bars", "Cont-Kukanov-Stoikov snapshot-difference sum / mean depth1"),
    "ofi_15": ("15 bars", "Cont-Kukanov-Stoikov snapshot-difference sum / mean depth1"),
    "ofi_30": ("30 bars", "Cont-Kukanov-Stoikov snapshot-difference sum / mean depth1"),
    "dlogdepth_1": ("1 bar", "log(depth3_t/depth3_{t-1})"),
    "dlogdepth_5": ("5 bars", "log(depth3_t/depth3_{t-5})"),
    "dlogdepth_15": ("15 bars", "log(depth3_t/depth3_{t-15})"),
    "dimb3_5": ("5 bars", "imb3_t - imb3_{t-5}"),
    "depth1_vs_30m": ("30 bars", "log(depth1_t / mean depth1 over last 30 bars)"),
    "vol_ratio_5_30": ("5/30 bars", "volume last 5 bars / (volume last 30 bars + 1)"),
    "log_vol_30": ("30 bars", "log1p(volume over last 30 bars)"),
    "log_deals_30": ("30 bars", "log1p(deal_number over last 30 bars)"),
    "mean_spread_30": ("30 bars", "mean relative spread1 over last 30 bars"),
    "prev_ret": ("previous day", "previous day log(last mid/first open)"),
    "prev_rv": ("previous day", "previous day std of 1-min mid returns"),
    "prev_spread": ("previous day", "previous day mean spread1_rel"),
    "prev_imb": ("previous day", "previous day mean imb1"),
    "prev_range": ("previous day", "previous day (high-low)/last mid"),
    "prev_last30": ("previous day", "previous day log(last mid/mid 30 bars before close)"),
    "prev_last60": ("previous day", "previous day log(last mid/mid 60 bars before close)"),
    "gap": ("<= t + previous day", "log(first open known at t / previous day last mid)"),
}
for _name in [f"xs_{k}" for k in ("spread1", "imb1", "imb3", "micro1", "ret15", "ofi15",
                                  "depth1", "volratio")]:
    FEATURE_NOTES[_name] = ("same timestamp peers",
                            "cross-sectional z-score across instruments at the same timestamp "
                            f"(base field {XS_OF[_name]}), clipped to +-5")


def feature_spec() -> list[dict]:
    """Auditable feature list: what each column reads and how far back it looks."""
    out = []
    for name in FEATURES:
        lookback, note = FEATURE_NOTES[name]
        out.append({"feature": name, "lookback": lookback, "definition": note})
    assert len(out) == len(FEATURES), (len(out), len(FEATURES))
    return out
