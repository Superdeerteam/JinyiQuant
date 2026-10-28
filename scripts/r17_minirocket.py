#!/usr/bin/env python
"""R17 screen: fixed MiniRocket-style convolutional features + ridge on the 128x12 raw window.

Hypothesis (one fixed config, pre-registered before reading screen results)
--------------------------------------------------------------------------
A fixed bank of dilated random-shape convolutions (MiniRocket: the 84 C(9,3)
kernels with +2/-1 weights) followed by a single regularised linear readout is a
different error source from both the MLPLOB deep model and the R6/R8 book-feature
trees.  The question is not whether it beats them alone, but whether a frozen
convex fusion with the existing equal-weight 441-day three-seed blend improves
later-development-window rank_ic without inflating turnover.

Fixed configuration (no tuning on screen windows)
-------------------------------------------------
* input = the canonical deep window: 128 minutes x 12 raw LOB columns, the exact
  ``data`` block of configs/441_lotb_replay_plan.json (ffill<=5, zero fill).
* transform: per-window per-channel standardisation inside the window, channels
  summed, then 21 fixed MiniRocket kernels (every 4th of the 84 C(9,3)
  combinations, fixed a priori by task design "a few kernels") at the MiniRocket
  dilation set for length 128 (dilations [1,2,3,4,6,8,11,15]); one
  golden-ratio quantile bias per (kernel,dilation), feature = mean PPV.
  168 features total.
* readout: ridge on the per-timestamp cross-sectional z-score of vwap_return
  (the same target as the R6 tree), lambda = 1e4 fixed, solved once by normal
  equations accumulated chunk-by-chunk (no window tensor is ever materialised).
* monitored rollout: fused(w) = (1-w)*blend_z + w*rocket_z, w frozen on
  nov2023+jan2024, honest read on feb/mar/apr/may/jul/aug; w=0.20 also reported
  as the a-priori no-selection choice.

Deviations from the MiniRocket paper (documented, deliberate)
-------------------------------------------------------------
1. all 12 channels are summed per kernel/dilation instead of drawing a random
   1..9-channel subset per combination (12x cheaper; kernel/dilation/bias
   structure unchanged; PPV is invariant to a positive channel scale).
2. convolutions are evaluated at valid positions only (no circular padding or
   alternating edge trimming).
3. per-window per-channel standardisation before convolution (the raw window
   mixes prices/100 with share volumes; the deep model normalises internally).
4. 21 of the 84 kernels and num_features = 168 (a few kernels by task design), not 10,000.

Only the eight registered screen windows are read; the confirmation windows
(jun2024, sep2024) and the sealed interval are never touched.

Usage
    python scripts/r17_minirocket.py --mode bench   # timing probe, one chunk
    python scripts/r17_minirocket.py --mode all
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from jinyiquant.data import BarArrays, instrument_codes, window_geometry, window_arrays  # noqa: E402

import r6r8_screen as R6  # noqa: E402

RUN = REPO / "runs" / "r17-20261029"
REPLAY = R6.REPLAY
DATA = REPO / "数据"
BARS = DATA / "cpt_jyc_2026_e2e_bar1m"
LABELS = DATA / "cpt_jyc_2026_vwap"
MAPP_PATH = DATA / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"
FEATURES = ["ask_price1", "ask_volume1", "bid_price1", "bid_volume1",
            "ask_price2", "ask_volume2", "bid_price2", "bid_volume2",
            "ask_price3", "ask_volume3", "bid_price3", "bid_volume3"]
PRICE_COLS = [i for i, c in enumerate(FEATURES) if c.startswith(("ask_price", "bid_price"))]
LOOKBACK = 128
MAX_FFILL = 5
PRICE_SCALE = 100.0
KERNELS = np.array(list(itertools.combinations(range(9), 3)), dtype=np.int64)[::4]  # 21 of 84
DILATIONS = [1, 2, 3, 4, 6, 8, 11, 15]
PHI = (1.0 + 5.0 ** 0.5) / 2.0
LEVELS = np.array([((i + 1) * PHI) % 1.0 for i in range(len(KERNELS) * len(DILATIONS))])
N_FEATURES = len(KERNELS) * len(DILATIONS)
CHUNK = 32768
LAMBDA = 1.0e4
TRAIN_MONTHS = ([f"2022-{m:02d}" for m in range(1, 13)]
                + [f"2023-{m:02d}" for m in range(1, 11)])
WINDOW_MONTH = R6.WINDOW_MONTH
WINDOWS = R6.WINDOWS
MAPP = pd.read_csv(MAPP_PATH)


def log(msg: str) -> None:
    print(f"[r17 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(month + "-01")
    return start, start + pd.offsets.MonthBegin(1)


def _read_part(month: str, start: pd.Timestamp, end: pd.Timestamp):
    """Read one monthly file; return (float32 features, dates, instrument strings)."""
    path = BARS / f"{month.replace('-', '')}.0.feather"
    f = pd.read_feather(path, columns=["date", "instrument_id", *FEATURES])
    f = f[(f["date"] >= start) & (f["date"] < end)]
    f = f.merge(MAPP, on="instrument_id", how="left").dropna(subset=["instrument"])
    values = f[FEATURES].to_numpy(dtype=np.float32)
    values[values == -1.0] = np.nan
    for c in PRICE_COLS:
        values[:, c] /= PRICE_SCALE
    dates = f["date"].to_numpy(dtype="datetime64[ns]")
    inst = f["instrument"].to_numpy(dtype=object)
    del f
    return values, dates, inst


def build_arrays(parts) -> BarArrays:
    values = np.concatenate([p[0] for p in parts], axis=0)
    dates = np.concatenate([p[1] for p in parts], axis=0)
    inst = np.concatenate([p[2] for p in parts], axis=0)
    del parts
    codes, names = pd.factorize(inst)
    del inst
    order = np.lexsort((dates, codes))
    values = values[order]
    dates = dates[order]
    codes = codes[order]
    counts = np.bincount(codes, minlength=len(names)).astype(np.int64)
    return BarArrays(values=values, dates=dates,
                     offsets=np.concatenate((np.zeros(1, np.int64), np.cumsum(counts))),
                     names=np.asarray(names, dtype=object))


def load_arrays(month: str, *, history_days: int = 0) -> BarArrays:
    start, end = month_bounds(month)
    first = (start - pd.Timedelta(days=history_days)) if history_days else start
    months = sorted({first.strftime("%Y-%m"), month})
    parts = [_read_part(m, first, end) for m in months]
    return build_arrays(parts)


def month_keys(month: str) -> pd.DataFrame:
    """Keys to score: window keys for screen months, label rows otherwise."""
    for window, m in WINDOW_MONTH.items():
        if m == month:
            keys = pd.read_parquet(REPLAY / "windows" / window / "keys.parquet")
            return keys.sort_values(["date", "instrument"]).reset_index(drop=True)
    labels = pd.read_parquet(LABELS / f"cpt_jyc_2026_vwap_{month}.parquet")
    return labels[["date", "instrument"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def windows_for(arrays: BarArrays, keys: pd.DataFrame, chunk: int = CHUNK):
    """Yield (row slice, x windows) for one month's keys; geometry computed once."""
    codes = instrument_codes(arrays, keys["instrument"])
    begin, stop = window_geometry(arrays, codes, keys["date"].to_numpy(), LOOKBACK)
    n = len(keys)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        m = sl.stop - sl.start
        x = np.zeros((m, LOOKBACK, len(FEATURES)), np.float32)
        observed = np.zeros((m, LOOKBACK, len(FEATURES)), bool)
        history = np.zeros((m, LOOKBACK), "datetime64[ns]")
        window_arrays(arrays, codes[sl], begin[sl], stop[sl], x, observed, history,
                      lookback=LOOKBACK, minimum=1, policy="ffill", limit=MAX_FFILL,
                      history_feature_count=len(FEATURES), features=FEATURES)
        yield sl, x


def _standardised_channels(x: np.ndarray) -> np.ndarray:
    """Per-window per-channel standardisation, summed over the 12 channels."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mu) / sd).sum(axis=2)


def ppv_features(x: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MiniRocket-style PPV features for a batch of raw windows (n, 128, 12)."""
    n = x.shape[0]
    xs = _standardised_channels(x)
    out = np.empty((n, N_FEATURES), np.float32)
    buf = np.empty((n, LOOKBACK), np.float32)
    col = 0
    for d in DILATIONS:
        width = LOOKBACK - 8 * d
        taps = np.stack([xs[:, j * d:j * d + width] for j in range(9)], axis=0)  # (9,n,width)
        total = taps.sum(axis=0)
        view = buf[:, :width]
        for k in range(len(KERNELS)):
            i0, i1, i2 = KERNELS[k]
            np.multiply(taps[i0], 3.0, out=view)
            np.add(view, taps[i1], out=view)
            np.add(view, taps[i2], out=view)
            np.subtract(view, total, out=view)          # 3*(t0+t1+t2) - sum(all taps)
            out[:, col] = np.count_nonzero(view > biases[col], axis=1) / width
            col += 1
    return out


def conv_for_biases(x: np.ndarray):
    """Channel-summed convolution values per (kernel, dilation), for bias quantiles."""
    xs = _standardised_channels(x)
    cols = []
    for d in DILATIONS:
        width = LOOKBACK - 8 * d
        taps = np.stack([xs[:, j * d:j * d + width] for j in range(9)], axis=0)
        total = taps.sum(axis=0)
        for k in range(len(KERNELS)):
            i0, i1, i2 = KERNELS[k]
            cols.append((3.0 * (taps[i0] + taps[i1] + taps[i2]) - total).ravel())
    return cols


def fit_biases(month: str = "2023-10", sample: int = 8192) -> np.ndarray:
    arrays = load_arrays(month)
    keys = month_keys(month)
    step = max(1, len(keys) // sample)
    keys = keys.iloc[::step].iloc[:sample].reset_index(drop=True)
    x = np.zeros((len(keys), LOOKBACK, len(FEATURES)), np.float32)
    observed = np.zeros_like(x, bool)
    history = np.zeros((len(keys), LOOKBACK), "datetime64[ns]")
    codes = instrument_codes(arrays, keys["instrument"])
    begin, stop = window_geometry(arrays, codes, keys["date"].to_numpy(), LOOKBACK)
    window_arrays(arrays, codes, begin, stop, x, observed, history, lookback=LOOKBACK,
                  minimum=1, policy="ffill", limit=MAX_FFILL,
                  history_feature_count=len(FEATURES), features=FEATURES)
    cols = conv_for_biases(x)
    log(f"biases from {len(keys)} windows on {month}")
    return np.array([np.quantile(c, lvl) for c, lvl in zip(cols, LEVELS)], np.float32)


def fit_ridge(biases: np.ndarray) -> dict:
    d = N_FEATURES
    gram = np.zeros((d, d), np.float64)
    xty = np.zeros(d, np.float64)
    sx = np.zeros(d, np.float64)
    sy = 0.0
    n_rows = 0
    seconds = {}
    for month in TRAIN_MONTHS:
        t0 = time.time()
        arrays = load_arrays(month)
        keys = month_keys(month)
        labels = pd.read_parquet(LABELS / f"cpt_jyc_2026_vwap_{month}.parquet")
        keys = keys.merge(labels[["date", "instrument", "vwap_return"]],
                          on=["date", "instrument"], how="inner")
        keys = keys[np.isfinite(keys["vwap_return"])].reset_index(drop=True)
        keys["target"] = keys["vwap_return"].astype(np.float64)
        y = R6._csz_by_timestamp(keys)
        ok = np.isfinite(y)
        n_month = 0
        for sl, x in windows_for(arrays, keys):
            mask = ok[sl]
            if not mask.any():
                continue
            f = ppv_features(x[mask], biases).astype(np.float64)
            yy = y[sl][mask]
            gram += f.T @ f
            xty += f.T @ yy
            sx += f.sum(axis=0)
            sy += float(yy.sum())
            n_rows += len(yy)
            n_month += len(yy)
        seconds[month] = round(time.time() - t0, 1)
        log(f"train {month} rows={n_month} total={n_rows} {seconds[month]}s")
        del arrays
    mu = sx / n_rows
    sd = np.sqrt(np.maximum(np.diag(gram) / n_rows - mu * mu, 1e-12))
    ybar = sy / n_rows
    a = (gram - n_rows * np.outer(mu, mu)) / np.outer(sd, sd)
    b = (xty - n_rows * mu * ybar) / sd
    beta = np.linalg.solve(a + LAMBDA * np.eye(d), b)
    return {"beta": beta, "mu": mu, "sd": sd, "intercept": float(ybar), "rows": int(n_rows),
            "train_seconds": seconds, "lambda": LAMBDA, "features": N_FEATURES,
            "dilations": DILATIONS, "biases": biases}


def predict_window(window: str, model: dict) -> dict:
    month = WINDOW_MONTH[window]
    t0 = time.time()
    arrays = load_arrays(month, history_days=12)
    keys = month_keys(month)
    beta, mu, sd = model["beta"], model["mu"], model["sd"]
    scores = np.full(len(keys), np.nan, np.float64)
    for sl, x in windows_for(arrays, keys):
        f = ppv_features(x, model["biases"]).astype(np.float64)
        scores[sl] = ((f - mu) / sd) @ beta
    out = RUN / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pred = keys[["date", "instrument"]].assign(score=scores)
    pred.to_parquet(out / f"{window}.parquet", index=False)
    log(f"predict {window} rows={len(pred)} nan={int(np.isnan(scores).sum())} {time.time() - t0:.1f}s")
    del arrays
    return {"rows": int(len(pred)), "seconds": round(time.time() - t0, 1)}


def _spearman_residual(dates: np.ndarray, r: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({"date": dates, "r": r, "y": y})
    ics = []
    for _, g in df.groupby("date", sort=True):
        if len(g) < 5:
            continue
        ics.append(float(g["r"].rank().corr(g["y"].rank())))
    return float(np.nanmean(ics)) if ics else float("nan")


def run_eval(model: dict, predict_meta: dict) -> dict:
    daily_rows = []
    diag = {"correlation": {}, "residual_ic": {}, "rocket_vs_r6": {}}
    tree_pred_dir = REPO / "runs" / "r6r8-20261029" / "screen" / "predictions"
    for window in WINDOWS:
        base = REPLAY / "windows" / window
        keys = pd.read_parquet(base / "keys.parquet").sort_values(["date", "instrument"]).reset_index(drop=True)
        targets = pd.read_parquet(base / "targets.parquet")
        rocket = pd.read_parquet(RUN / "predictions" / f"{window}.parquet")
        rocket = keys[["date", "instrument"]].merge(rocket, on=["date", "instrument"], how="left")
        members = {}
        for name, seeds in R6.BLENDS.items():
            acc = np.zeros(len(keys))
            for seed in seeds:
                p = pd.read_parquet(base / seed / "predictions.parquet")
                p = keys[["date", "instrument"]].merge(p, on=["date", "instrument"], how="left")
                acc += p["score"].to_numpy()
            members[name] = acc / len(seeds)
        frame = keys[["date", "instrument"]].assign(rocket=rocket["score"].to_numpy(),
                                                    r441=members["r441"], r441p=members["r441p"])
        frame["rocket_z"] = R6._zscore(frame, "rocket")
        frame["r441_z"] = R6._zscore(frame, "r441")
        frame["r441p_z"] = R6._zscore(frame, "r441p")
        for name, col in (("rocket", "rocket"), ("blend_r441", "r441"), ("blend_r441p", "r441p")):
            daily_rows.append(R6._daily(keys, targets, frame[col].to_numpy(), name).assign(window=window))
        for w in R6.WEIGHTS:
            fused = (1.0 - w) * frame["r441_z"].to_numpy() + w * frame["rocket_z"].to_numpy()
            daily_rows.append(R6._daily(keys, targets, fused, f"fused_{w:.2f}").assign(window=window))
        ok = np.isfinite(frame["rocket_z"]) & np.isfinite(frame["r441_z"])
        diag["correlation"][window] = {
            "rocket_r441": float(np.corrcoef(frame.loc[ok, "rocket_z"], frame.loc[ok, "r441_z"])[0, 1]),
            "rocket_r441p": float(np.corrcoef(frame.loc[ok, "rocket_z"], frame.loc[ok, "r441p_z"])[0, 1]),
        }
        merged = keys[["date", "instrument"]].assign(rocket=frame["rocket_z"], blend=frame["r441_z"])
        merged = merged.merge(targets, on=["date", "instrument"], how="inner")
        merged = merged[np.isfinite(merged["rocket"]) & np.isfinite(merged["blend"])].reset_index(drop=True)
        resid = np.empty(len(merged))
        for date, g in merged.groupby("date", sort=True):
            b = g["blend"].to_numpy(dtype=float)
            t = g["rocket"].to_numpy(dtype=float)
            beta = float(np.dot(t, b) / max(np.dot(b, b), 1e-12))
            resid[g.index.to_numpy()] = t - beta * b
        diag["residual_ic"][window] = _spearman_residual(merged["date"].to_numpy(), resid,
                                                         merged["target"].to_numpy(dtype=float))
        if (tree_pred_dir / f"{window}.parquet").exists():
            tp = pd.read_parquet(tree_pred_dir / f"{window}.parquet")
            tp = keys[["date", "instrument"]].merge(tp, on=["date", "instrument"], how="left")
            ok2 = np.isfinite(tp["score"].to_numpy()) & np.isfinite(frame["rocket_z"].to_numpy())
            diag["rocket_vs_r6"][window] = float(np.corrcoef(
                pd.Series(tp.loc[ok2, "score"].to_numpy()).rank(pct=True),
                pd.Series(frame.loc[ok2, "rocket_z"].to_numpy()).rank(pct=True))[0, 1])
    daily = pd.concat(daily_rows, ignore_index=True)
    RUN.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(RUN / "daily.parquet", index=False)
    pooled = {name: R6._pooled(daily[daily["variant"] == name], WINDOWS)
              for name in daily["variant"].unique()}
    per_window = {}
    for name in daily["variant"].unique():
        sub = daily[daily["variant"] == name]
        per_window[name] = {w: float(sub.loc[sub["window"] == w, "rank_ic"].mean()) for w in WINDOWS}
    fusion = {}
    for w in R6.WEIGHTS:
        name = f"fused_{w:.2f}"
        fusion[f"{w:.2f}"] = {"frozen": R6._paired(daily, name, "blend_r441", R6.FROZEN),
                              "later": R6._paired(daily, name, "blend_r441", R6.LATER),
                              "all": R6._paired(daily, name, "blend_r441", WINDOWS),
                              "vs_r441p_all": R6._paired(daily, name, "blend_r441p", WINDOWS)}
    frozen_choice = max(R6.WEIGHTS, key=lambda w: fusion[f"{w:.2f}"]["frozen"]["mean_delta"])
    summary = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "hypothesis": "fixed MiniRocket conv features + ridge add a different error source to the 441d MLPLOB three-seed blend",
        "windows": WINDOWS, "frozen_windows": R6.FROZEN, "later_windows": R6.LATER,
        "weights": R6.WEIGHTS, "primary_a_priori_weight": R6.APRIORI_W,
        "chosen_weight_by_frozen": frozen_choice,
        "seed": R6.SEED, "block": R6.BLOCK, "draws": R6.DRAWS,
        "pooled": pooled, "per_window": per_window, "fusion": fusion,
        "diagnostics": diag,
        "model": {k: v for k, v in model.items() if k not in ("beta", "mu", "sd", "biases")},
        "predict": predict_meta,
        "guardrails": {"windows_read": WINDOWS, "confirmation_windows_read": [],
                       "sealed_interval_read": False,
                       "feature_transform_source": "configs/441_lotb_replay_plan.json data block"},
        "deviations": ["all 12 channels summed per kernel/dilation instead of random 1..9-channel subsets",
                       "valid-position convolution only (no circular padding / edge trimming)",
                       "per-window per-channel standardisation before convolution",
                       "num_features=168 (21 kernels x 8 dilations x 1 bias)"],
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def _load_model() -> dict:
    raw = json.loads((RUN / "model.json").read_text(encoding="utf-8"))
    model = {k: np.asarray(raw[k]) for k in ("beta", "mu", "sd", "biases")}
    model["biases"] = model["biases"].astype(np.float32)
    model.update({k: v for k, v in raw.items() if k not in ("beta", "mu", "sd", "biases")})
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["bench", "fit", "predict", "eval", "all"])
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.mode == "bench":
        t0 = time.time()
        arrays = load_arrays("2024-01", history_days=12)
        log(f"load {time.time() - t0:.1f}s rows={arrays.rows} insts={len(arrays.names)}")
        keys = month_keys("2024-01")
        for sl, x in windows_for(arrays, keys.iloc[:CHUNK]):
            t1 = time.time()
            f = ppv_features(x, np.full(N_FEATURES, 0.5, np.float32))
            log(f"chunk rows={len(x)} gather+ppv={time.time() - t1:.2f}s feats={f.shape} "
                f"mean={f.mean():.3f} std={f.std():.3f}")
            break
        return
    if args.mode in ("fit", "all"):
        t0 = time.time()
        biases = fit_biases()
        log(f"biases fitted {time.time() - t0:.1f}s")
        model = fit_ridge(biases)
        model = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}
        (RUN / "model.json").write_text(json.dumps(model), encoding="utf-8")
        log(f"ridge fitted rows={model['rows']} total={time.time() - t0:.1f}s")
    if args.mode in ("predict", "all"):
        model = _load_model()
        meta = {w: predict_window(w, model) for w in WINDOWS}
        (RUN / "predict_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if args.mode in ("eval", "all"):
        model = _load_model()
        predict_meta = json.loads((RUN / "predict_meta.json").read_text(encoding="utf-8"))
        summary = run_eval(model, predict_meta)
        log(json.dumps({k: round(v["rank_ic"], 5) for k, v in summary["pooled"].items()}))
        log("w=0.20 all: " + json.dumps(summary["fusion"]["0.20"]["all"]["mean_delta"]))


if __name__ == "__main__":
    main()
