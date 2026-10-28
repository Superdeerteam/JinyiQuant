#!/usr/bin/env python
"""R12: cross-day same-clock ("same time of day, past N trading days") memory features.

Hypothesis
----------
The production window is 128 one-minute steps starting at the sample time; it
carries no memory of "how this stock usually trades at this clock time over the
past several trading days".  R12 adds that memory as explicit features:

  for every (trading day d, sample slot s, instrument i), summarise the SAME
  slot s on the previous N completed trading days (N in {5, 20}):
  realised return (official vwap_return label), intraday drift, relative
  spread, log depth3, depth imbalance, activity.

Causality contract
------------------
* A feature row stamped (d, s, i) uses ONLY panel rows of trading days
  STRICTLY before d.  The panel itself uses only 1-minute bars with
  minute <= sample_min[s] (as-of semantics, 09:30 uses the 09:25 auction bar).
* No row at day d is ever touched by the aggregation for day d: the window is
  [j-N, j) over the day axis, proved by --check-causality.
* Only dev months are read; the sealed interval is guarded (see SEALED_START).

Stages
------
  panel     one compact parquet per month: (day, slot, instrument) x
            {mid, spread, depth3_log, imb1, drift, logvol30, ret_fwd}
  features  memory features for the 8 screen windows
  probe     coverage + univariate IC + correlation with the trunk blend +
            residual IC after orthogonalising to the blend (NO model training)
  --check-causality   panel-truncation + future-scramble + past-scramble tests

Usage
-----
  python scripts/r12_memory_features.py --stage all
  python scripts/r12_memory_features.py --check-causality 2024-01-15
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
REPLAY = ROOT / "runs/dev10-replay-20261028/20261028T094439.842764Z_0854d2d5ca3d/windows"
OUT = ROOT / "runs/r12-20261029"

SAMPLE_TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
SAMPLE_MIN = [570, 600, 630, 660, 690, 810, 840, 870]

# sealed interval [2024-10-08, 2025-01-01): never read. Guard on the month list.
SEALED_START = pd.Timestamp("2024-10-01")
MATERIAL_MONTHS = [f"2023-{m:02d}" for m in range(9, 13)] + [f"2024-{m:02d}" for m in range(1, 9)]
WINDOWS = ["nov2023", "jan2024", "feb2024", "mar2024", "apr2024", "may2024", "jul2024", "aug2024"]
WINDOW_MONTHS = {"nov2023": "2023-11", "jan2024": "2024-01", "feb2024": "2024-02",
                 "mar2024": "2024-03", "apr2024": "2024-04", "may2024": "2024-05",
                 "jul2024": "2024-07", "aug2024": "2024-08"}
LOOKBACKS = (5, 20)
MIN_COUNT = {5: 3, 20: 10}

BAR_COLS = ["date", "instrument_id", "open", "volume",
            "ask_price1", "bid_price1", "ask_volume1", "bid_volume1",
            "ask_volume2", "bid_volume2", "ask_volume3", "bid_volume3"]
NUM_COLS = [c for c in BAR_COLS if c not in ("date", "instrument_id")]
PANEL_VARS = ["mid", "spread", "depth3_log", "imb1", "drift", "logvol30", "ret_fwd"]


# --------------------------------------------------------------------------- helpers
def _ffill(a: np.ndarray, limit: int = 5) -> np.ndarray:
    n, t = a.shape
    idx = np.where(np.isfinite(a), np.arange(t, dtype=np.int32), np.int32(0))
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, idx, axis=1).copy()
    age = np.arange(t, dtype=np.int32)[None, :] - idx
    out[age > limit] = np.nan
    return out


def _asof_first_positive(a: np.ndarray, idx8: np.ndarray) -> np.ndarray:
    """First strictly positive value per row among slots <= each sample slot."""
    ok = np.isfinite(a) & (a > 0)
    has = ok.any(axis=1)
    fidx = np.argmax(ok, axis=1)
    first = np.full(a.shape[0], np.nan)
    first[has] = a[np.arange(a.shape[0])[has], fidx[has]]
    return np.where(has[:, None] & (fidx[:, None] <= idx8[None, :]), first[:, None], np.nan)


def mapping() -> pd.DataFrame:
    return pd.read_csv(MAPPING)


# --------------------------------------------------------------------------- panel
def build_day_panel(cols: dict[str, np.ndarray], ids: np.ndarray, minutes: np.ndarray,
                    cap_min: int | None = None):
    """Panel row values for one trading day: (ids_sorted, {var: (n, 8)})."""
    if cap_min is not None:
        m = minutes <= cap_min
        cols = {k: v[m] for k, v in cols.items()}
        ids = ids[m]
        minutes = minutes[m]
    ids_u = np.unique(ids)
    n = len(ids_u)
    ns = len(SAMPLE_MIN)
    if n == 0:
        return ids_u, {}
    umin = np.unique(minutes)
    t = len(umin)
    rows = np.searchsorted(ids_u, ids)
    cidx = np.searchsorted(umin, minutes)

    def M(name: str) -> np.ndarray:
        a = np.full((n, t), np.nan)
        a[rows, cidx] = cols[name]
        return a

    bp1, ap1 = M("bid_price1"), M("ask_price1")
    bv1, av1 = M("bid_volume1"), M("ask_volume1")
    bv2, av2 = M("ask_volume2"), M("bid_volume2")
    bv3, av3 = M("ask_volume3"), M("bid_volume3")
    op, vol = M("open"), M("volume")

    ok1 = (bp1 > 0) & (ap1 > 0) & (ap1 >= bp1)
    mid = np.where(ok1, (bp1 + ap1) * 0.5, np.nan)
    depth = np.where(ok1, bv1 + bv2 + bv3 + av1 + av2 + av3, np.nan)
    d1 = np.where(ok1, bv1 + av1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        sp1 = np.where(ok1, (ap1 - bp1) / mid, np.nan)
        imb1 = np.where(ok1 & (d1 > 0), (bv1 - av1) / d1, np.nan)
    midf, spf, depf, imf = _ffill(mid), _ffill(sp1), _ffill(depth), _ffill(imb1)

    idx8 = np.searchsorted(umin, np.array(SAMPLE_MIN), side="right") - 1
    valid8 = idx8 >= 0
    idx8c = np.clip(idx8, 0, t - 1)

    def at(a: np.ndarray) -> np.ndarray:
        return np.where(valid8[None, :], a[:, idx8c], np.nan)

    m8 = at(midf)
    open8 = _asof_first_positive(op, idx8)
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = np.log(m8 / open8)
    # activity: volume of the 30 bars ending at the as-of bar (backward looking)
    vf = np.where(np.isfinite(vol), vol, np.nan)
    s = idx8c[:, None] - np.arange(30)[None, :]
    okw = (s >= 0) & valid8[:, None]
    w = np.where(okw[None, :, :], vf[:, np.clip(s, 0, t - 1)], np.nan)
    cnt = np.isfinite(w).sum(axis=2)
    v30 = np.where(cnt >= 15, np.nansum(w, axis=2), np.nan)

    out = {
        "mid": m8,
        "spread": at(spf),
        "depth3_log": np.log1p(at(depf)),
        "imb1": at(imf),
        "drift": drift,
        "logvol30": np.log1p(v30),
    }
    return ids_u, out


def panel_month(month: str, out_dir: Path, force: bool = False) -> Path:
    path = out_dir / f"panel_{month}.parquet"
    if path.exists() and not force:
        return path
    mapp = mapping()
    mid_ok = mapp["instrument_id"].to_numpy()
    id2name = mapp.set_index("instrument_id")["instrument"].to_dict()
    bars = pd.read_feather(BARS / f"{month.replace('-', '')}.0.feather", columns=BAR_COLS)
    bars = bars[bars["instrument_id"].isin(mid_ok)]
    ts = bars["date"].to_numpy()
    if ts.max() >= np.datetime64(SEALED_START):
        raise SystemExit(f"refusing to read sealed window row in {month}")
    arr = {c: bars[c].to_numpy() for c in NUM_COLS}
    ids = bars["instrument_id"].to_numpy()
    del bars
    minute = ((ts.astype("datetime64[m]") - ts.astype("datetime64[D]"))
              / np.timedelta64(1, "m")).astype(np.int16)
    day = ts.astype("datetime64[D]")
    order = np.argsort(day, kind="stable")
    uniq, starts = np.unique(day[order], return_index=True)
    starts = np.append(starts, len(order))
    frames = []
    for i, d in enumerate(uniq):
        pos = order[starts[i]:starts[i + 1]]
        cols = {c: arr[c][pos].astype(np.float64) for c in NUM_COLS}
        dids, vals = build_day_panel(cols, ids[pos], minute[pos])
        if not len(dids):
            continue
        ns = len(SAMPLE_MIN)
        stamps = (d.astype("datetime64[m]") + np.array(SAMPLE_MIN, dtype="timedelta64[m]"))
        frame = {"date": np.repeat(stamps.astype("datetime64[ns]"), len(dids)),
                 "slot": np.repeat(np.arange(ns, dtype=np.int8), len(dids)),
                 "instrument_id": np.tile(dids, ns),
                 "instrument": np.tile([id2name[x] for x in dids], ns)}
        for k, v in vals.items():
            frame[k] = v.T.reshape(-1)
        frames.append(pd.DataFrame(frame))
    panel = pd.concat(frames, ignore_index=True)
    del frames
    lpath = VWAP / f"cpt_jyc_2026_vwap_{month}.parquet"
    if lpath.exists():
        lab = pd.read_parquet(lpath, columns=["date", "instrument", "vwap_return"])
        lab = lab.drop_duplicates(["date", "instrument"], keep="first")
        panel = panel.merge(lab.rename(columns={"vwap_return": "ret_fwd"}),
                            on=["date", "instrument"], how="left")
    else:
        panel["ret_fwd"] = np.nan
    panel = panel.sort_values(["date", "instrument_id"], ignore_index=True)
    if panel.duplicated(["date", "slot", "instrument"]).any():
        raise SystemExit(f"duplicate (date, slot, instrument) rows in {month}")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    print(f"[panel {month}] rows={len(panel):,} ret_fwd_cov={panel['ret_fwd'].notna().mean():.4f} "
          f"mid_nan={panel['mid'].isna().mean():.4f}", flush=True)
    return path


# --------------------------------------------------------------------------- memory
def _window_stats(a: np.ndarray, n_win: int, min_count: int):
    """(D,S,I) -> past-n_win-day mean / std / count for target-day index j over [j-n, j)."""
    d = a.shape[0]
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)
    pad = np.zeros((1,) + x.shape[1:])
    cs = np.concatenate([pad, np.cumsum(x, axis=0)], axis=0)
    cc = np.concatenate([pad, np.cumsum(ok, axis=0).astype(np.float64)], axis=0)
    cs2 = np.concatenate([pad, np.cumsum(x * x, axis=0)], axis=0)
    hi = np.arange(d)
    lo = np.maximum(hi - n_win, 0)
    s, c, s2 = cs[hi] - cs[lo], cc[hi] - cc[lo], cs2[hi] - cs2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c >= min_count, s / np.where(c > 0, c, np.nan), np.nan)
        var = s2 / np.where(c > 0, c, np.nan) - mean * mean
    return mean, np.sqrt(np.clip(var, 0.0, None)), c


def memory_features(tensors: dict[str, np.ndarray], day_list: list, inst_list: list,
                    lookbacks=LOOKBACKS, min_count=MIN_COUNT):
    """Return {feature: (D, S, I)} memory tensors."""
    feats: dict[str, np.ndarray] = {}
    ret = tensors["ret_fwd"]
    hit = np.where(np.isfinite(ret), (ret > 0).astype(np.float64), np.nan)
    for n in lookbacks:
        mc = min_count[n]
        mean, std, cnt = _window_stats(ret, n, mc)
        feats[f"mem_ret_mean{n}"] = mean
        feats[f"mem_ret_std{n}"] = std
        feats[f"mem_ret_tstat{n}"] = np.where(std > 0, mean / std, np.nan)
        m_hit, _, _ = _window_stats(hit, n, mc)
        feats[f"mem_ret_hit{n}"] = m_hit
        for name, src in (("drift", "drift"), ("spread", "spread"), ("depth", "depth3_log"),
                          ("imb", "imb1"), ("logvol", "logvol30")):
            feats[f"mem_{name}_mean{n}"] = _window_stats(tensors[src], n, mc)[0]
        feats[f"mem_days{n}"] = cnt  # history depth diagnostic, not a signal
    last = np.full_like(ret, np.nan)
    last[1:] = ret[:-1]
    feats["mem_ret_last"] = last
    return feats


def tensorise(panel: pd.DataFrame):
    days = np.sort(panel["date"].dt.normalize().unique())
    insts = np.sort(panel["instrument"].unique())
    di = pd.Index(days).get_indexer(panel["date"].dt.normalize())
    ii = pd.Index(insts).get_indexer(panel["instrument"])
    si = panel["slot"].to_numpy()
    shape = (len(days), len(SAMPLE_MIN), len(insts))
    tens = {}
    for v in PANEL_VARS:
        a = np.full(shape, np.nan)
        a[di, si, ii] = panel[v].to_numpy()
        tens[v] = a
    return tens, list(days), list(insts)


def features_stage(months: list[str], out_dir: Path, windows=WINDOWS):
    panels = []
    for m in months:
        panels.append(pd.read_parquet(out_dir / "panel" / f"panel_{m}.parquet"))
    panel = pd.concat(panels, ignore_index=True)
    del panels
    tens, days, insts = tensorise(panel)
    del panel
    feats = memory_features(tens, days, insts)
    feat_names = [k for k in feats if not k.startswith("mem_days")]
    emit_names = list(feats)
    day_arr = np.array(days, dtype="datetime64[D]")
    inst_arr = np.array(insts)
    out_paths = []
    for w in windows:
        month = WINDOW_MONTHS[w]
        mask = pd.DatetimeIndex(day_arr).strftime("%Y-%m").to_numpy() == month
        if not mask.any():
            print(f"[features {w}] no days for {month}")
            continue
        sel = np.where(mask)[0]
        ns, ni = len(SAMPLE_MIN), len(insts)
        stamp = (day_arr[sel].astype("datetime64[m]")[:, None, None]
                 + np.array(SAMPLE_MIN, dtype="timedelta64[m]")[None, :, None])
        frame = {"date": np.broadcast_to(stamp, (len(sel), ns, ni)).reshape(-1),
                 "slot": np.broadcast_to(np.arange(ns)[None, :, None], (len(sel), ns, ni)).reshape(-1),
                 "instrument": np.broadcast_to(inst_arr[None, None, :], (len(sel), ns, ni)).reshape(-1)}
        for k in emit_names:
            frame[k] = feats[k][sel].reshape(-1)
        df = pd.DataFrame(frame)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"features_{w}.parquet"
        df.to_parquet(p, index=False)
        out_paths.append(p)
        print(f"[features {w}] rows={len(df):,} days={len(sel)} "
              f"cov_ret20={df['mem_ret_mean20'].notna().mean():.4f}", flush=True)
    return out_paths, feat_names


# --------------------------------------------------------------------------- probe math
def _group_rank_corr(df: pd.DataFrame, xcol: str, ycol: str = "target") -> pd.Series:
    """Per-timestamp (date) Spearman corr, vectorised through rank algebra."""
    work = df[["date", xcol, ycol]].dropna()
    if work.empty:
        return pd.Series(dtype=float)
    g = work.groupby("date", sort=True)
    ok = g.size() >= 3
    work = work[work["date"].map(ok).to_numpy()]
    if work.empty:
        return pd.Series(dtype=float)
    g = work.groupby("date", sort=True)
    rx = g[xcol].rank(method="average").to_numpy()
    ry = g[ycol].rank(method="average").to_numpy()
    w = pd.DataFrame({"date": work["date"].to_numpy(), "rx": rx, "ry": ry,
                      "rx2": rx * rx, "ry2": ry * ry, "rxy": rx * ry})
    s = w.groupby("date", sort=True).agg(n=("rx", "size"), sx=("rx", "sum"), sy=("ry", "sum"),
                                         sx2=("rx2", "sum"), sy2=("ry2", "sum"), sxy=("rxy", "sum"))
    n = s["n"].to_numpy(dtype=float)
    mx, my = s["sx"].to_numpy() / n, s["sy"].to_numpy() / n
    vx = np.clip(s["sx2"].to_numpy() / n - mx * mx, 0.0, None)
    vy = np.clip(s["sy2"].to_numpy() / n - my * my, 0.0, None)
    cov = s["sxy"].to_numpy() / n - mx * my
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vx * vy)
    r[~np.isfinite(r)] = np.nan
    return pd.Series(r, index=s.index)


def _residual_ic(df: pd.DataFrame, xcol: str, blend: str = "blend", ycol: str = "target") -> pd.Series:
    """IC of the feature rank after removing its within-timestamp blend-rank component."""
    work = df[["date", xcol, blend, ycol]].dropna()
    if work.empty:
        return pd.Series(dtype=float)
    cnt = work.groupby("date", sort=True).size()
    work = work[work["date"].map(cnt >= 3).to_numpy()]
    if work.empty:
        return pd.Series(dtype=float)
    g = work.groupby("date", sort=True)
    rx = g[xcol].rank(method="average").to_numpy()
    rb = g[blend].rank(method="average").to_numpy()
    ry = g[ycol].rank(method="average").to_numpy()
    w = pd.DataFrame({"date": work["date"].to_numpy(), "rx": rx, "rb": rb, "ry": ry})
    w["rx2"] = w["rx"] ** 2
    w["rb2"] = w["rb"] ** 2
    w["rxb"] = w["rx"] * w["rb"]
    w["ry2"] = w["ry"] ** 2
    w["rxry"] = w["rx"] * w["ry"]
    w["rbry"] = w["rb"] * w["ry"]
    s = w.groupby("date", sort=True).agg(
        n=("rx", "size"), sx=("rx", "sum"), sb=("rb", "sum"), sy=("ry", "sum"),
        sx2=("rx2", "sum"), sb2=("rb2", "sum"), sxb=("rxb", "sum"),
        sy2=("ry2", "sum"), sxry=("rxry", "sum"), sbry=("rbry", "sum"))
    n = s["n"].to_numpy(dtype=float)
    mx, mb, my = s["sx"].to_numpy() / n, s["sb"].to_numpy() / n, s["sy"].to_numpy() / n
    vx = np.clip(s["sx2"].to_numpy() / n - mx * mx, 0.0, None)
    vb = np.clip(s["sb2"].to_numpy() / n - mb * mb, 0.0, None)
    vy = np.clip(s["sy2"].to_numpy() / n - my * my, 0.0, None)
    cxb = s["sxb"].to_numpy() / n - mx * mb
    cxy = s["sxry"].to_numpy() / n - mx * my
    cby = s["sbry"].to_numpy() / n - mb * my
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cxb / vb
        cov = cxy - beta * cby
        vres = vx - beta * cxb
        r = cov / np.sqrt(np.clip(vres, 0.0, None) * vy)
    r[~np.isfinite(r)] = np.nan
    return pd.Series(r, index=s.index)


def _block_bootstrap_ci(series: np.ndarray, n_draws: int = 20000, block: int = 5,
                        seed: int = 20261028) -> tuple[float, float]:
    """5-day circular block bootstrap, same scheme as scripts/compare_multiscale_replay.py."""
    s = np.asarray([v for v in np.asarray(series, dtype=float) if np.isfinite(v)])
    if len(s) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(s), size=n_draws)
    idx = (starts[:, None] + np.arange(block)[None, :]) % len(s)
    means = s[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _daily(ts_series: pd.Series) -> pd.Series:
    """Per-timestamp series -> per-trading-day mean (eval-v2 caliber)."""
    s = ts_series.dropna()
    if s.empty:
        return s
    return s.groupby(pd.DatetimeIndex(s.index).normalize(), sort=True).mean()


def load_window(window: str, feat_path: Path) -> pd.DataFrame:
    base = REPLAY / window
    keys = pd.read_parquet(base / "keys.parquet")
    targets = pd.read_parquet(base / "targets.parquet")[["date", "instrument", "target"]]
    scores = []
    for seed in ("s11", "s29", "s47"):
        p = pd.read_parquet(base / f"r441_{seed}" / "predictions.parquet")
        p = p.rename(columns={"score": f"score_{seed}"})
        scores.append(p[["date", "instrument", f"score_{seed}"]])
    blend = scores[0]
    for other in scores[1:]:
        blend = blend.merge(other, on=["date", "instrument"], how="inner")
    blend["blend"] = blend[[f"score_{s}" for s in ("s11", "s29", "s47")]].mean(axis=1)
    df = keys.merge(targets, on=["date", "instrument"], how="left")
    df = df.merge(blend[["date", "instrument", "blend"]], on=["date", "instrument"], how="left")
    feats = pd.read_parquet(feat_path)
    df = df.merge(feats, on=["date", "instrument"], how="left")
    df["window"] = window
    return df


def _evaluator_check(window: str, data: pd.DataFrame) -> dict:
    """Cross-check the fast rank-IC path against src/jinyiquant/evaluation.evaluate."""
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from jinyiquant.config import EvaluationConfig
    from jinyiquant.evaluation import evaluate

    base = REPLAY / window
    targets = pd.read_parquet(base / "targets.parquet")[["date", "instrument", "target", "label_end"]]
    sub = data[["date", "instrument", "blend"]].dropna()
    answer = targets.merge(sub, on=["date", "instrument"], how="inner", validate="one_to_one")
    pred = answer[["date", "instrument", "blend"]].rename(columns={"blend": "score"})
    cfg = EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                           constant_portfolios="undefined", single_side_rate=0.0)
    metrics, daily = evaluate(pred, answer[["date", "instrument", "target"]], cfg,
                              return_sections=False, undefined_constant_portfolios=True)
    return {"window": window, "evaluator_rank_ic_mean": float(metrics["rank_ic_mean"]),
            "rows": int(len(pred))}


def probe_stage(feat_names: list[str], out_dir: Path, windows=WINDOWS):
    frames = []
    coverage = {}
    for w in windows:
        p = out_dir / f"features_{w}.parquet"
        if not p.exists():
            continue
        df = load_window(w, p)
        frames.append(df)
        coverage[w] = {
            "keys": int(len(df)),
            "days": int(df["date"].dt.normalize().nunique()),
            "instruments": int(df["instrument"].nunique()),
            "target_cov": float(df["target"].notna().mean()),
            "blend_cov": float(df["blend"].notna().mean()),
            "feature_cov": {k: float(df[k].notna().mean()) for k in feat_names},
            "mem_days20_mean": float(df["mem_days20"].mean()),
            "mem_days20_min": float(df["mem_days20"].min()),
            "mem_days5_mean": float(df["mem_days5"].mean()),
        }
        print(f"[probe {w}] keys={len(df):,} days={coverage[w]['days']} "
              f"cov_ret20={coverage[w]['feature_cov']['mem_ret_mean20']:.4f} "
              f"days20_mean={coverage[w]['mem_days20_mean']:.1f}", flush=True)
    data = pd.concat(frames, ignore_index=True)
    del frames
    day_window = data.groupby(data["date"].dt.normalize(), sort=True)["window"].first()

    def per_window(daily: pd.Series) -> dict:
        w = day_window.reindex(daily.index)
        out = {}
        for win in windows:
            sel = daily[(w == win).to_numpy()]
            out[win] = float(sel.mean()) if len(sel) else float("nan")
        return out

    # reference: the trunk blend's own daily rank_ic on the same rows/caliber
    ts_blend = _group_rank_corr(data, "blend")
    daily_blend = _daily(ts_blend)
    blend_pooled = float(daily_blend.mean())
    evaluator_check = _evaluator_check(windows[0], data[data["window"] == windows[0]])
    evaluator_check["fast_path_pooled"] = float(
        _daily(_group_rank_corr(data[data["window"] == windows[0]], "blend")).mean())

    rows = []
    for k in feat_names:
        ts_ic = _group_rank_corr(data, k)
        daily = _daily(ts_ic)
        if daily.empty:
            rows.append({"feature": k, "pooled_ic": float("nan")})
            continue
        lo, hi = _block_bootstrap_ci(daily.to_numpy())
        daily_corr = _daily(_group_rank_corr(data, k, ycol="blend"))
        daily_res = _daily(_residual_ic(data, k))
        lo_r, hi_r = _block_bootstrap_ci(daily_res.to_numpy()) if not daily_res.empty else (np.nan, np.nan)
        row = {"feature": k, "pooled_ic": float(daily.mean()), "ic_std": float(daily.std(ddof=1)),
               "ic_ci_lo": lo, "ic_ci_hi": hi,
               "ic_pos_windows": int(sum(1 for v in per_window(daily).values() if v > 0)),
               "n_days": int(daily.shape[0]),
               "corr_blend": float(daily_corr.mean()) if not daily_corr.empty else float("nan"),
               "resid_ic": float(daily_res.mean()) if not daily_res.empty else float("nan"),
               "resid_ci_lo": lo_r, "resid_ci_hi": hi_r,
               "coverage": float(data[k].notna().mean())}
        for win, val in per_window(daily).items():
            row[f"ic_{win}"] = val
        for win, val in per_window(daily_res).items():
            row[f"resid_{win}"] = val
        rows.append(row)
        if not np.isfinite(row["pooled_ic"]):
            continue
        print(f"[ic] {k:22s} pooled={row['pooled_ic']:+.4f} ci=[{lo:+.4f},{hi:+.4f}] "
              f"corr={row['corr_blend']:+.3f} resid={row['resid_ic']:+.4f} "
              f"lo/hi=[{lo_r:+.4f},{hi_r:+.4f}]", flush=True)

    table = pd.DataFrame(rows).sort_values("resid_ic", ascending=False, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "probe_ic_table.csv", index=False)
    report = {"blend_pooled_ic": blend_pooled,
              "blend_per_window": per_window(daily_blend),
              "evaluator_check": evaluator_check,
              "coverage": coverage,
              "ic_table": table.to_dict(orient="records"),
              "n_days_total": int(data["date"].dt.normalize().nunique()),
              "n_windows": len(windows)}
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[probe] blend pooled rank_ic={blend_pooled:.6f} "
          f"(evaluator {evaluator_check['evaluator_rank_ic_mean']:.6f} on {evaluator_check['window']}) "
          f"days={report['n_days_total']} windows={len(windows)}", flush=True)
    return table, report


# --------------------------------------------------------------------------- causality
def _fuse(data: pd.DataFrame, pb: np.ndarray, pf: np.ndarray, w: float):
    """Rank-fuse blend + one feature; return aligned (blend_ic_daily, fused_ic_daily)."""
    ok = np.isfinite(pf)
    d = data[ok]
    b_ic = _daily(_group_rank_corr(d, "blend", "target"))
    fused = (1 - w) * pb[ok] + w * pf[ok]
    f_ic = _daily(_group_rank_corr(d.assign(fused=fused), "fused", "target"))
    return b_ic, f_ic, ok


def fusion_stage(feat_names: list[str], out_dir: Path, windows=WINDOWS,
                 w: float = 0.20):
    """Zero-parameter fusion diagnostic (no fitting anywhere).

    Score space is per-timestamp percentile rank (centred at 0.5), so rank fusion is
        fused = (1-w) * pct(blend) + w * pct(feature)
    with w=0.20 fixed a priori (team fusion grid's a-priori point, NOT selected here).
    For each candidate we report the paired per-day rank_ic delta (fused minus blend on
    the same rows: rows with a NaN feature are dropped from both legs), the pooled mean
    over all screen windows, a 5-day circular block-bootstrap CI (seed 20261028, 20000
    draws) over the concatenated 165-day daily series, and the positive-window count.
    Composites (equal-weight mean of member percentile ranks, membership fixed by the
    R12 hypothesis, not by IC): mem_ret_family6 = mean/tstat/hit at 5 and 20 days;
    mem_all19 = every signal feature.
    """
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from jinyiquant.config import EvaluationConfig
    from jinyiquant.evaluation import evaluate

    fam6 = ["mem_ret_mean5", "mem_ret_mean20", "mem_ret_tstat5", "mem_ret_tstat20",
            "mem_ret_hit5", "mem_ret_hit20"]
    cands = list(feat_names) + ["mem_ret_family6", "mem_all19"]
    acc: dict[str, dict] = {}
    months_seen: list[str] = []
    for wname in windows:
        fp = out_dir / f"features_{wname}.parquet"
        if not fp.exists():
            continue
        data = load_window(wname, fp)
        data = data[np.isfinite(data["blend"].to_numpy()) & np.isfinite(data["target"].to_numpy())]
        g = data.groupby("date", sort=False)
        pb = (g["blend"].rank(pct=True) - 0.5).to_numpy()
        pct = {n: (g[n].rank(pct=True) - 0.5).to_numpy() for n in feat_names}
        pct["mem_ret_family6"] = np.nanmean(np.stack([pct[n] for n in fam6]), axis=0)
        pct["mem_all19"] = np.nanmean(np.stack([pct[n] for n in feat_names]), axis=0)
        for name in cands:
            b_ic, f_ic, ok = _fuse(data, pb, pct[name], w)
            delta = (f_ic - b_ic).dropna()
            a = acc.setdefault(name, {"delta": [], "b": [], "f": [], "rows": 0, "n": 0})
            a["delta"].append(delta)
            a["b"].append(b_ic)
            a["f"].append(f_ic)
            a["rows"] += int(ok.sum())
            a["n"] += len(data)
        months_seen.append(wname)
        ev = _evaluator_check(wname, data) if wname == windows[0] else None
        note = "n/a" if ev is None else f"{ev['evaluator_rank_ic_mean']:.6f}"
        print(f"[fusion {wname}] rows={len(data)} blend_ic="
              f"{_daily(_group_rank_corr(data, 'blend', 'target')).mean():.6f} evaluator={note}",
              flush=True)
    rows = []
    for name in cands:
        a = acc[name]
        delta = pd.concat(a["delta"]).sort_index()
        per_win = [float(pd.concat([s]).mean()) for s in a["delta"]]
        lo, hi = _block_bootstrap_ci(delta.to_numpy())
        rows.append({"candidate": name, "w": w,
                     "blend_ic": round(float(pd.concat(a["b"]).mean()), 6),
                     "fused_ic": round(float(pd.concat(a["f"]).mean()), 6),
                     "delta": round(float(delta.mean()), 6),
                     "delta_lo": round(lo, 6), "delta_hi": round(hi, 6),
                     "days": int(len(delta)),
                     "pos_windows": int(sum(1 for v in per_win if v > 0)),
                     "windows": len(per_win),
                     "worst_window": round(min(per_win), 6),
                     "coverage": round(a["rows"] / max(a["n"], 1), 4)})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(out_dir / "probe_fusion.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def _ts_resid_on(df: pd.DataFrame, xcol: str, zcol: str) -> pd.Series:
    """Timestamp-level rank IC of xcol vs target after removing zcol (rank space)."""
    work = df[[ "date", "target", xcol, zcol]].dropna()
    out = {}
    for ts, grp in work.groupby("date", sort=True):
        if len(grp) < 3:
            continue
        rx = grp[xcol].rank(pct=True).to_numpy() - 0.5
        rz = grp[zcol].rank(pct=True).to_numpy() - 0.5
        ry = grp["target"].rank(pct=True).to_numpy() - 0.5
        denom = float(rz @ rz)
        if denom <= 0:
            continue
        beta = float(rz @ rx) / denom
        resid = rx - beta * rz
        if np.std(resid) <= 0:
            continue
        out[ts] = float(np.corrcoef(resid, ry)[0, 1])
    return pd.Series(out).sort_index()


def slot_split_stage(feat_names: list[str], out_dir: Path, windows=WINDOWS,
                     slot: str = "14:30"):
    """Robustness of the 14:30 memory edge: is it an illiquidity/stale-print artifact?

    (a) family6 IC after removing the cross-sectional depth rank (mem_depth_mean20,
        itself built from strictly prior days), (b) IC inside the liquid / illiquid
        halves of the same-day cross-section split on that causal depth rank.
    """
    fam6 = ["mem_ret_mean5", "mem_ret_mean20", "mem_ret_tstat5", "mem_ret_tstat20",
            "mem_ret_hit5", "mem_ret_hit20"]
    acc: dict[tuple, list] = {}
    for wname in windows:
        fp = out_dir / f"features_{wname}.parquet"
        if not fp.exists():
            continue
        data = load_window(wname, fp)
        data = data[np.isfinite(data["blend"].to_numpy()) & np.isfinite(data["target"].to_numpy())]
        data = data[data["date"].dt.strftime("%H:%M") == slot]
        if data.empty:
            continue
        g = data.groupby("date", sort=False)
        pct = {n: (g[n].rank(pct=True) - 0.5).to_numpy() for n in feat_names}
        pct["mem_ret_family6"] = np.nanmean(np.stack([pct[n] for n in fam6]), axis=0)
        data = data.assign(**pct)
        d = data.dropna(subset=["mem_ret_family6", "mem_depth_mean20", "target"])
        acc.setdefault(("family6", "all"), []).append(
            _group_rank_corr(d, "mem_ret_family6", "target").rename("v"))
        acc.setdefault(("family6", "depth_neutral"), []).append(
            _ts_resid_on(d, "mem_ret_family6", "mem_depth_mean20").rename("v"))
        grp = d.groupby("date", sort=False)["mem_depth_mean20"]
        half = grp.rank(pct=True)
        for label, mask in (("liquid_half", half > 0.5), ("illiquid_half", half <= 0.5)):
            sub = d[mask.to_numpy()]
            acc.setdefault(("family6", label), []).append(
                _group_rank_corr(sub, "mem_ret_family6", "target").rename("v"))
        acc.setdefault(("blend", "all"), []).append(
            _group_rank_corr(d, "blend", "target").rename("v"))
    rows = []
    for (name, kind), series in acc.items():
        pooled = pd.concat(series).sort_index()
        lo, hi = _block_bootstrap_ci(pooled.to_numpy())
        rows.append({"candidate": name, "split": kind, "slot": slot,
                     "ic": round(float(pooled.mean()), 6),
                     "lo": round(lo, 6), "hi": round(hi, 6),
                     "pos_windows": int(sum(1 for s in series if s.mean() > 0)),
                     "windows": len(series), "days": int(len(pooled))})
    df = pd.DataFrame(rows).sort_values(["candidate", "split"])
    df.to_csv(out_dir / "probe_slot_split.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def slot_ic_stage(feat_names: list[str], out_dir: Path, windows=WINDOWS):
    """Per-sample-time univariate IC of the leading R12 candidates vs the blend's own IC
    at that slot, plus their IC after orthogonalising out the blend (no target used in
    the orthogonalisation).  Pooled over the 8 screen windows, day-level block CI."""
    fam6 = ["mem_ret_mean5", "mem_ret_mean20", "mem_ret_tstat5", "mem_ret_tstat20",
            "mem_ret_hit5", "mem_ret_hit20"]
    cands = ["blend", "mem_ret_family6", "mem_ret_hit20", "mem_ret_mean20",
             "mem_ret_last", "mem_depth_mean20"]
    acc: dict[tuple, list] = {}
    for wname in windows:
        fp = out_dir / f"features_{wname}.parquet"
        if not fp.exists():
            continue
        data = load_window(wname, fp)
        data = data[np.isfinite(data["blend"].to_numpy()) & np.isfinite(data["target"].to_numpy())]
        g = data.groupby("date", sort=False)
        pct = {n: (g[n].rank(pct=True) - 0.5).to_numpy() for n in feat_names}
        pct["mem_ret_family6"] = np.nanmean(np.stack([pct[n] for n in fam6]), axis=0)
        pct["mem_ret_family6"] = pct["mem_ret_family6"]
        data = data.assign(**{c: pct[c] for c in cands if c in pct})
        for name in cands:
            ic = _group_rank_corr(data, name, "target").dropna()
            for sv, ser in ic.groupby(ic.index.strftime("%H:%M")):
                acc.setdefault((name, sv, "ic"), []).append(ser.rename("v"))
            if name != "blend":
                res = _residual_ic(data, name).dropna()
                for sv, ser in res.groupby(res.index.strftime("%H:%M")):
                    acc.setdefault((name, sv, "resid"), []).append(ser.rename("v"))
    rows = []
    for (name, sv, kind), series in acc.items():
        pooled = pd.concat(series).sort_index()
        lo, hi = _block_bootstrap_ci(pooled.to_numpy())
        rows.append({"candidate": name, "slot": sv, "kind": kind,
                     "value": round(float(pooled.mean()), 6),
                     "lo": round(lo, 6), "hi": round(hi, 6),
                     "pos_windows": int(sum(1 for s in series if s.mean() > 0)),
                     "windows": len(series), "days": int(len(pooled))})
    df = pd.DataFrame(rows).sort_values(["kind", "candidate", "slot"])
    df.to_csv(out_dir / "probe_slot_ic.csv", index=False)
    print(df[df["kind"] == "ic"].to_string(index=False), flush=True)
    print(df[df["kind"] == "resid"].to_string(index=False), flush=True)
    return df


def sweep_stage(feat_names: list[str], out_dir: Path, windows=WINDOWS,
                weights=(0.10, 0.20, 0.30, 0.40), slot_w: float = 0.20):
    """Diagnostic sweep for the leading candidates: delta vs fixed fusion weight, and
    per-sample-time delta at slot_w.  Nothing here is fitted; the weights are the team
    grid points, reported to show the shape/headroom of the fusion response."""
    fam6 = ["mem_ret_mean5", "mem_ret_mean20", "mem_ret_tstat5", "mem_ret_tstat20",
            "mem_ret_hit5", "mem_ret_hit20"]
    cands = ["mem_ret_family6", "mem_ret_hit20", "mem_ret_mean20", "mem_all19"]
    per_w: dict[tuple, list] = {}
    per_slot: dict[str, dict[str, list]] = {}
    for wname in windows:
        fp = out_dir / f"features_{wname}.parquet"
        if not fp.exists():
            continue
        data = load_window(wname, fp)
        data = data[np.isfinite(data["blend"].to_numpy()) & np.isfinite(data["target"].to_numpy())]
        g = data.groupby("date", sort=False)
        pb = (g["blend"].rank(pct=True) - 0.5).to_numpy()
        pct = {n: (g[n].rank(pct=True) - 0.5).to_numpy() for n in feat_names}
        pct["mem_ret_family6"] = np.nanmean(np.stack([pct[n] for n in fam6]), axis=0)
        pct["mem_all19"] = np.nanmean(np.stack([pct[n] for n in feat_names]), axis=0)
        for name in cands:
            pf = pct[name]
            ok = np.isfinite(pf)
            d = data[ok]
            b_ic = _daily(_group_rank_corr(d, "blend", "target"))
            for wt in weights:
                fused = (1 - wt) * pb[ok] + wt * pf[ok]
                f_ic = _daily(_group_rank_corr(d.assign(fused=fused), "fused", "target"))
                per_w.setdefault((name, wt), []).append((f_ic - b_ic).dropna())
            if name in ("mem_ret_family6", "mem_ret_hit20"):
                fused = (1 - slot_w) * pb[ok] + slot_w * pf[ok]
                col = _group_rank_corr(d.assign(fused=fused), "fused", "target")
                bcol = _group_rank_corr(d, "blend", "target")
                dd = (col - bcol).dropna()
                for sv, series in dd.groupby(dd.index.strftime("%H:%M")):
                    per_slot.setdefault(name, {}).setdefault(sv, []).append(series)
    rows = []
    for (name, wt), series in per_w.items():
        delta = pd.concat(series).sort_index()
        per_win = [float(s.mean()) for s in series]
        lo, hi = _block_bootstrap_ci(delta.to_numpy())
        rows.append({"candidate": name, "w": wt, "delta": round(float(delta.mean()), 6),
                     "delta_lo": round(lo, 6), "delta_hi": round(hi, 6),
                     "pos_windows": int(sum(1 for v in per_win if v > 0)),
                     "windows": len(per_win), "days": int(len(delta))})
    df = pd.DataFrame(rows).sort_values(["candidate", "w"])
    df.to_csv(out_dir / "probe_weight_sweep.csv", index=False)
    print(df.to_string(index=False), flush=True)
    srows = []
    for name, slots in per_slot.items():
        for sv, series in sorted(slots.items()):
            delta = pd.concat(series).sort_index()
            per_win = [float(s.mean()) for s in series]
            lo, hi = _block_bootstrap_ci(delta.to_numpy())
            srows.append({"candidate": name, "slot": sv, "delta": round(float(delta.mean()), 6),
                          "delta_lo": round(lo, 6), "delta_hi": round(hi, 6),
                          "pos_windows": int(sum(1 for v in per_win if v > 0)),
                          "windows": len(per_win), "obs": int(len(delta))})
    sdf = pd.DataFrame(srows).sort_values(["candidate", "slot"])
    sdf.to_csv(out_dir / "probe_slot.csv", index=False)
    print(sdf.to_string(index=False), flush=True)
    return df, sdf


def _arr_diff(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-aware elementwise max |diff|: both-NaN -> 0, one-sided NaN -> inf."""
    nan_a, nan_b = ~np.isfinite(a), ~np.isfinite(b)
    if (nan_a ^ nan_b).any():
        return float("inf")
    both = nan_a & nan_b
    d = np.where(both, 0.0, np.abs(np.where(both, 0.0, a) - np.where(both, 0.0, b)))
    return float(d.max()) if d.size else 0.0


def _tensor_diff(a: np.ndarray, b: np.ndarray) -> float:
    both = np.isnan(a) & np.isnan(b)
    d = np.where(both, 0.0, np.abs(a - b))
    if not np.isfinite(d).any():
        return 0.0
    return float(np.nanmax(d))


def check_causality(target_day: str, out_dir: Path, months: list[str]):
    panels = [pd.read_parquet(out_dir / "panel" / f"panel_{m}.parquet") for m in months]
    panel = pd.concat(panels, ignore_index=True)
    tens, days, insts = tensorise(panel)
    feats = memory_features(tens, days, insts)
    day_arr = np.array(days, dtype="datetime64[D]")
    j0 = int(np.where(day_arr == np.datetime64(target_day))[0][0])
    report: dict = {"target_day": target_day, "day_index": j0,
                    "history_days": int(min(j0, max(LOOKBACKS))), "tests": {}}

    names = [k for k in feats if not k.startswith("mem_days")]
    base = {k: feats[k][j0].copy() for k in names}

    # (1) scramble every panel value of days >= j0 -> features at j0 must not move
    rng = np.random.default_rng(7)
    pert = {}
    for v, a in tens.items():
        b = a.copy()
        sl = b[j0:]
        flat = sl.reshape(-1)
        finite = np.where(np.isfinite(flat))[0]
        vals = flat[finite]
        rng.shuffle(vals)
        flat[finite] = vals
        pert[v] = b
    pf = memory_features(pert, days, insts)
    report["tests"]["future_days_scrambled"] = {k: _tensor_diff(pf[k][j0], base[k]) for k in names}

    # (2) realign days after j0 (shift the whole future block by one day) -> same expectation
    pert2 = {}
    for v, a in tens.items():
        b = a.copy()
        fut = b[j0:]
        if fut.shape[0] > 1:
            b[j0:] = np.concatenate([fut[-1:], fut[:-1]], axis=0)
        pert2[v] = b
    pf2 = memory_features(pert2, days, insts)
    report["tests"]["future_days_rolled"] = {k: _tensor_diff(pf2[k][j0], base[k]) for k in names}

    # (3) power control: scramble the past (days < j0) -> features at j0 MUST change
    pert3 = {}
    for v, a in tens.items():
        b = a.copy()
        sl = b[:j0]
        flat = sl.reshape(-1)
        finite = np.where(np.isfinite(flat))[0]
        vals = flat[finite]
        rng.shuffle(vals)
        flat[finite] = vals
        pert3[v] = b
    pf3 = memory_features(pert3, days, insts)
    report["tests"]["past_days_scrambled_power"] = {k: _tensor_diff(pf3[k][j0], base[k]) for k in names}

    # (4) panel construction: truncating bars at each sample minute leaves that slot identical
    month = pd.Timestamp(target_day).strftime("%Y-%m")
    mapp = mapping()
    bars = pd.read_feather(BARS / f"{month.replace('-', '')}.0.feather", columns=BAR_COLS)
    bars = bars[bars["instrument_id"].isin(mapp["instrument_id"].to_numpy())]
    ts = bars["date"].to_numpy()
    minute = ((ts.astype("datetime64[m]") - ts.astype("datetime64[D]"))
              / np.timedelta64(1, "m")).astype(np.int16)
    day = ts.astype("datetime64[D]")
    pos = np.where(day == np.datetime64(target_day))[0]
    ids = bars["instrument_id"].to_numpy()[pos]
    mins = minute[pos]
    cols = {c: bars[c].to_numpy()[pos].astype(np.float64) for c in NUM_COLS}
    base_ids, base_vals = build_day_panel(cols, ids, mins)
    trunc = {}
    for j, tmin in enumerate(SAMPLE_MIN):
        tid, tvals = build_day_panel(cols, ids, mins, cap_min=tmin)
        if not np.array_equal(tid, base_ids):
            trunc[SAMPLE_TIMES[j]] = {"identical": False, "reason": "instrument set changed"}
            continue
        worst = max(_arr_diff(tvals[k][:, j], base_vals[k][:, j]) for k in base_vals)
        trunc[SAMPLE_TIMES[j]] = {"identical": bool(worst == 0.0), "max_abs_diff": worst}
    report["tests"]["panel_truncation"] = trunc

    ok = (all(v == 0.0 for v in report["tests"]["future_days_scrambled"].values())
          and all(v == 0.0 for v in report["tests"]["future_days_rolled"].values())
          and all(v > 0.0 for v in report["tests"]["past_days_scrambled_power"].values())
          and all(v["identical"] for v in trunc.values()))
    report["passed"] = bool(ok)
    (out_dir / "causality.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"[causality] passed={report['passed']} day={target_day} "
          f"future_max={max(report['tests']['future_days_scrambled'].values()):.3g} "
          f"past_min={min(report['tests']['past_days_scrambled_power'].values()):.3g}", flush=True)
    return report


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["panel", "features", "probe", "fusion", "sweep", "slotic", "slotsplit", "none", "all"])
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--windows", nargs="*", default=WINDOWS)
    ap.add_argument("--check-causality", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--force-panel", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    months = args.months or MATERIAL_MONTHS
    for m in months:
        if pd.Timestamp(m + "-01") >= SEALED_START:
            raise SystemExit(f"month {m} is inside the sealed interval")

    t0 = time.time()
    if args.stage in ("panel", "all"):
        (out_dir / "panel").mkdir(parents=True, exist_ok=True)
        for m in months:
            panel_month(m, out_dir / "panel", force=args.force_panel)
    if args.stage in ("features", "all"):
        _, feat_names = features_stage(months, out_dir, args.windows)
        (out_dir / "feature_names.json").write_text(json.dumps(feat_names, indent=2))
    if args.stage in ("probe", "all"):
        feat_names = json.loads((out_dir / "feature_names.json").read_text())
        probe_stage(feat_names, out_dir, args.windows)
    if args.stage in ("fusion", "all"):
        feat_names = json.loads((out_dir / "feature_names.json").read_text())
        fusion_stage(feat_names, out_dir, args.windows)
    if args.stage in ("sweep", "all"):
        feat_names = json.loads((out_dir / "feature_names.json").read_text())
        sweep_stage(feat_names, out_dir, args.windows)
    if args.stage in ("slotic", "all"):
        feat_names = json.loads((out_dir / "feature_names.json").read_text())
        slot_ic_stage(feat_names, out_dir, args.windows)
    if args.stage in ("slotsplit", "all"):
        feat_names = json.loads((out_dir / "feature_names.json").read_text())
        slot_split_stage(feat_names, out_dir, args.windows)
    if args.check_causality:
        check_causality(args.check_causality, out_dir, months)
    print(f"[done] {time.time() - t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
