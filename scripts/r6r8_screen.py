#!/usr/bin/env python
"""R6+R8 screen: fixed LightGBM on the book feature table and its fusion increment.

Only the eight registered *screen* windows are read
(nov2023 jan2024 feb2024 mar2024 apr2024 may2024 jul2024 aug2024 = 165 days).
The confirmation windows (jun2024, sep2024) and the sealed interval are never read.

Protocol (pre-registered before touching screen results)
-------------------------------------------------------
1. Train one fixed LightGBM on 2022-01-04..2023-10-31 (the same era the 441-day
   MLPLOB backbone was trained on), target = per-timestamp cross-sectional
   z-score of vwap_return, all 46 features, no early stopping, no tuning.
2. Score the eight screen windows out of time.
3. Blend with the existing equal-weight r441 3-seed MLPLOB blend.  Both sides
   are z-scored inside each timestamp (a positive affine map, so the baseline
   blend's RankIC / portfolio composition is unchanged), then
       fused(w) = (1 - w) * blend_z + w * tree_z .
   The small weight grid is frozen on the two earliest screen windows
   (nov2023+jan2024 = 44 days) and the honest read is the later six windows
   (feb/mar/apr/may/jul/aug = 121 days).  w = 0.20 is also reported as the
   a-priori choice with no selection at all.
4. Paired daily rank_ic delta vs the untreated blend, pooled over windows,
   5-day moving-block bootstrap (ceil(n/5) blocks, 20000 draws, seed 20261028),
   plus per-window signs, turnover and long-short means.

Usage
    python scripts/r6r8_screen.py --mode all
    python scripts/r6r8_screen.py --mode train|predict|eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO / "src"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compare_blend_replay import block_interval  # noqa: E402
from jinyiquant.config import EvaluationConfig  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from r6r8_book_features import FEATURES, feature_spec  # noqa: E402

RUN = REPO / "runs" / "r6r8-20261029"
FEAT = RUN / "features"
SCREEN = RUN / "screen"
REPLAY = REPO / "runs" / "dev10-replay-20261028" / "20261028T094439.842764Z_0854d2d5ca3d"

TRAIN_MONTHS = ([f"2022-{m:02d}" for m in range(1, 13)]
                + [f"2023-{m:02d}" for m in range(1, 11)])
WINDOW_MONTH = {"nov2023": "2023-11", "jan2024": "2024-01", "feb2024": "2024-02",
                "mar2024": "2024-03", "apr2024": "2024-04", "may2024": "2024-05",
                "jul2024": "2024-07", "aug2024": "2024-08"}
WINDOWS = list(WINDOW_MONTH)
FROZEN = ["nov2023", "jan2024"]
LATER = [w for w in WINDOWS if w not in FROZEN]
BLENDS = {"r441": ["r441_s11", "r441_s29", "r441_s47"],
          "r441p": ["r441p_s11", "r441p_s29", "r441p_s47"]}
WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
APRIORI_W = 0.20
SEED = 20261028
BLOCK = 5
DRAWS = 20000

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63,
              min_child_samples=500, max_bin=63, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              num_threads=4, seed=20261029, verbosity=-1, deterministic=True,
              force_col_wise=True)
ROUNDS = 600

EVAL_CFG = EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                            constant_portfolios="undefined", single_side_rate=0.0)


# ------------------------------------------------------------------ training
def _csz_by_timestamp(frame: pd.DataFrame) -> np.ndarray:
    """Per-timestamp cross-sectional z-score of the target (winsorised at +-5)."""
    codes, uniq = pd.factorize(frame["date"], sort=False)
    y = frame["target"].to_numpy(dtype=np.float64)
    cnt = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
    s1 = np.bincount(codes, weights=y, minlength=len(uniq))
    s2 = np.bincount(codes, weights=y * y, minlength=len(uniq))
    mu = s1 / cnt
    var = np.maximum(s2 / cnt - mu * mu, 0.0)
    sd = np.sqrt(var)
    sd[~(sd > 1e-12)] = np.nan
    z = (y - mu[codes]) / sd[codes]
    return np.clip(z, -5.0, 5.0)


def stage_train() -> dict:
    import lightgbm as lgb

    SCREEN.mkdir(parents=True, exist_ok=True)
    cols = ["date", "target"] + FEATURES
    total = 0
    for month in TRAIN_MONTHS:
        t = pd.read_parquet(FEAT / f"features_{month}.parquet", columns=["target"])
        total += int(t["target"].notna().sum())
    print(f"[train] rows={total:,} features={len(FEATURES)} months={len(TRAIN_MONTHS)}", flush=True)

    X = np.empty((total, len(FEATURES)), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    pos = 0
    for month in TRAIN_MONTHS:
        frame = pd.read_parquet(FEAT / f"features_{month}.parquet", columns=cols)
        frame = frame[frame["target"].notna()]
        n = len(frame)
        X[pos:pos + n] = frame[FEATURES].to_numpy(dtype=np.float32)
        y[pos:pos + n] = _csz_by_timestamp(frame).astype(np.float32)
        pos += n
        del frame
        print(f"[train] loaded {month} pos={pos:,}", flush=True)
    assert pos == total, (pos, total)

    t0 = time.time()
    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURES, free_raw_data=True,
                         params={"max_bin": PARAMS["max_bin"], "verbosity": -1})
    dtrain.construct()
    X_ref = X
    X = None
    del X_ref
    model = lgb.train(PARAMS, dtrain, num_boost_round=ROUNDS)
    train_seconds = time.time() - t0
    # LightGBM's C++ writer cannot open non-ASCII paths on Windows; use Python IO.
    (SCREEN / "model.txt").write_text(model.model_to_string(), encoding="utf-8")
    importance = sorted(zip(FEATURES, model.feature_importance("gain")),
                        key=lambda kv: -kv[1])
    meta = {"rows": int(total), "features": FEATURES, "params": {k: v for k, v in PARAMS.items()},
            "rounds": ROUNDS, "train_seconds": round(train_seconds, 1),
            "train_months": TRAIN_MONTHS, "target": "per-timestamp CSZ(vwap_return), clip +-5",
            "importance_gain": [{"feature": f, "gain": float(g)} for f, g in importance]}
    (SCREEN / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] done in {train_seconds:.0f}s; top: {[f for f, _ in importance[:8]]}", flush=True)
    return meta


# ------------------------------------------------------------------ prediction
def _window_features(window: str) -> pd.DataFrame:
    month = WINDOW_MONTH[window]
    cols = ["date", "instrument"] + FEATURES
    return pd.read_parquet(FEAT / f"features_{month}.parquet", columns=cols)


def stage_predict() -> dict:
    import lightgbm as lgb

    model = lgb.Booster(model_str=(SCREEN / "model.txt").read_text(encoding="utf-8"))
    out = SCREEN / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for window in WINDOWS:
        t0 = time.time()
        keys = pd.read_parquet(REPLAY / "windows" / window / "keys.parquet")
        feats = _window_features(window)
        merged = keys.merge(feats, on=["date", "instrument"], how="left", validate="one_to_one")
        feats_np = merged[FEATURES].to_numpy(dtype=np.float32)
        score = model.predict(feats_np, num_threads=4).astype(np.float64)
        if not np.isfinite(score).all():
            raise SystemExit(f"{window}: non-finite tree scores")
        pred = merged[["date", "instrument"]].assign(score=score)
        pred.to_parquet(out / f"{window}.parquet", index=False)
        feat_cov = float(np.isfinite(feats_np).any(axis=1).mean())
        seconds = time.time() - t0
        meta[window] = {"rows": int(len(pred)), "days": int(keys["date"].dt.normalize().nunique()),
                        "feature_coverage": feat_cov, "predict_seconds": round(seconds, 2)}
        print(f"[predict] {window} rows={len(pred):,} cov={feat_cov:.4f} {seconds:.1f}s", flush=True)
    (SCREEN / "predict_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ------------------------------------------------------------------ evaluation
def _daily(keys: pd.DataFrame, targets: pd.DataFrame, scores: np.ndarray, name: str) -> pd.DataFrame:
    pred = keys[["date", "instrument"]].assign(score=np.asarray(scores, dtype=float))
    pred = pred.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                      how="inner", validate="one_to_one")
    metrics, daily = evaluate(pred, targets, EVAL_CFG, return_sections=False,
                              undefined_constant_portfolios=True)
    daily = daily.copy()
    daily["variant"] = name
    return daily


def _zscore(frame: pd.DataFrame, column: str) -> np.ndarray:
    g = frame.groupby("date")[column]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform("std").to_numpy(dtype=float)
    z = (frame[column].to_numpy(dtype=float) - mu) / np.where(sd > 0, sd, np.nan)
    return z


def _pooled(daily: pd.DataFrame, windows: list[str]) -> dict:
    sub = daily[daily["window"].isin(windows)]
    d = sub["rank_ic"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    return {"days": int(len(d)), "rank_ic": float(np.mean(d)),
            "turnover": float(np.nanmean(sub["turnover"])),
            "long_short": float(np.nanmean(sub["long_short_return"])),
            "rank_ic_ir": float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 2 else None}


def _paired(daily: pd.DataFrame, variant: str, base: str, windows: list[str]) -> dict:
    left = daily[(daily["variant"] == variant) & daily["window"].isin(windows)]
    right = daily[(daily["variant"] == base) & daily["window"].isin(windows)]
    merged = left.merge(right, on=["window", "date"], suffixes=("", "_base"))
    delta = (merged["rank_ic"].to_numpy(dtype=float)
             - merged["rank_ic_base"].to_numpy(dtype=float))
    ok = np.isfinite(delta)
    delta = delta[ok]
    interval = block_interval(delta, block=BLOCK, draws=DRAWS, seed=SEED)
    per_window = {}
    for w in windows:
        m = merged["window"] == w
        v = (merged.loc[m, "rank_ic"].to_numpy(dtype=float)
             - merged.loc[m, "rank_ic_base"].to_numpy(dtype=float))
        v = v[np.isfinite(v)]
        per_window[w] = {"days": int(len(v)), "delta": float(np.mean(v)) if len(v) else None}
    signs = sum(1 for v in per_window.values() if v["delta"] is not None and v["delta"] > 0)
    return {"mean_delta": interval["mean"], "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"], "days": interval["days"],
            "positive_windows": int(signs), "windows": len(windows), "per_window": per_window}


def stage_eval() -> dict:
    SCREEN.mkdir(parents=True, exist_ok=True)
    rows = []
    diag = {"correlation": {}, "residual_ic": {}}
    for window in WINDOWS:
        keys = pd.read_parquet(REPLAY / "windows" / window / "keys.parquet")
        targets = pd.read_parquet(REPLAY / "windows" / window / "targets.parquet")
        tree = pd.read_parquet(SCREEN / "predictions" / f"{window}.parquet")
        frame = keys.merge(tree, on=["date", "instrument"], how="left", validate="one_to_one")
        for tag, members in BLENDS.items():
            parts = []
            for member in members:
                p = pd.read_parquet(REPLAY / "windows" / window / member / "predictions.parquet")
                parts.append(p["score"].to_numpy(dtype=float))
            frame[f"blend_{tag}"] = np.mean(np.stack(parts, axis=0), axis=0)
        frame["tree"] = frame["score"]
        frame["tree_z"] = _zscore(frame, "tree")
        for tag in BLENDS:
            frame[f"blend_{tag}_z"] = _zscore(frame, f"blend_{tag}")
        # diagnostics: pooled correlation of the per-timestamp z-scores
        for tag in BLENDS:
            b = frame[f"blend_{tag}_z"].to_numpy(dtype=float)
            t = frame["tree_z"].to_numpy(dtype=float)
            m = np.isfinite(b) & np.isfinite(t)
            diag["correlation"].setdefault(tag, {})[window] = float(np.corrcoef(b[m], t[m])[0, 1])
        # residual IC (tree orthogonalised to the blend inside each timestamp)
        b = frame["blend_r441_z"].to_numpy(dtype=float)
        t = frame["tree_z"].to_numpy(dtype=float)
        codes = pd.factorize(frame["date"], sort=False)[0]
        ok = np.isfinite(b) & np.isfinite(t)
        ones = np.ones(len(frame), dtype=np.float64)
        bb = np.where(ok, b, 0.0)
        tt = np.where(ok, t, 0.0)
        n_ok = np.bincount(codes, weights=ones * ok, minlength=codes.max() + 1)
        s_bb = np.bincount(codes, weights=bb * bb, minlength=codes.max() + 1)
        s_bt = np.bincount(codes, weights=bb * tt, minlength=codes.max() + 1)
        beta = np.where((n_ok >= 10) & (s_bb > 1e-12), s_bt / np.where(s_bb > 1e-12, s_bb, 1.0), np.nan)
        resid = np.where(ok, t - beta[codes] * b, np.nan)
        d = _daily(keys, targets, resid, "tree_resid")
        rows.append(d.assign(window=window))
        diag["residual_ic"][window] = float(np.nanmean(d["rank_ic"]))
        # variants
        for name, scores in (("blend_r441", frame["blend_r441"]), ("blend_r441p", frame["blend_r441p"]),
                             ("tree", frame["tree"])):
            d = _daily(keys, targets, scores, name)
            rows.append(d.assign(window=window))
        for w in WEIGHTS:
            fused = (1.0 - w) * frame["blend_r441_z"] + w * frame["tree_z"]
            d = _daily(keys, targets, fused, f"fused_{w:.2f}")
            rows.append(d.assign(window=window))
            if abs(w - APRIORI_W) < 1e-9:
                pred = keys[["date", "instrument"]].assign(score=fused.to_numpy(dtype=float))
                out = SCREEN / "predictions_fused"
                out.mkdir(exist_ok=True)
                pred.to_parquet(out / f"{window}.parquet", index=False)
        print(f"[eval] {window} done", flush=True)

    daily = pd.concat(rows, ignore_index=True)
    daily.to_parquet(SCREEN / "daily.parquet", index=False)

    pooled = {name: _pooled(daily[daily["variant"] == name], WINDOWS)
              for name in list(daily["variant"].unique())}
    per_window = {}
    for name in daily["variant"].unique():
        per_window[name] = {w: _pooled(daily[daily["variant"] == name], [w]) for w in WINDOWS}

    fusion = {}
    for w in WEIGHTS:
        name = f"fused_{w:.2f}"
        fusion[f"{w:.2f}"] = {
            "frozen": _paired(daily, name, "blend_r441", FROZEN),
            "later": _paired(daily, name, "blend_r441", LATER),
            "all": _paired(daily, name, "blend_r441", WINDOWS),
            "pooled": pooled[name],
        }
    chosen = max(WEIGHTS, key=lambda w: fusion[f"{w:.2f}"]["frozen"]["mean_delta"])
    primary = fusion[f"{APRIORI_W:.2f}"]

    result = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "windows": WINDOWS, "frozen_windows": FROZEN, "later_windows": LATER,
        "blend_members": BLENDS, "weights": WEIGHTS,
        "chosen_weight_by_frozen": chosen,
        "primary_a_priori_weight": APRIORI_W,
        "seed": SEED, "block": BLOCK, "draws": DRAWS,
        "pooled": pooled, "per_window": per_window, "fusion": fusion,
        "primary": {"weight": APRIORI_W, "frozen": primary["frozen"],
                    "later": primary["later"], "all": primary["all"]},
        "chosen": {"weight": chosen, "frozen": fusion[f"{chosen:.2f}"]["frozen"],
                   "later": fusion[f"{chosen:.2f}"]["later"],
                   "all": fusion[f"{chosen:.2f}"]["all"]},
        "diagnostics": diag,
        "feature_spec": feature_spec(),
        "evaluation": {"config": {"quantile": EVAL_CFG.quantile,
                                  "periods_per_year": EVAL_CFG.periods_per_year,
                                  "min_cross_section": EVAL_CFG.min_cross_section,
                                  "constant_portfolios": EVAL_CFG.constant_portfolios,
                                  "single_side_rate": EVAL_CFG.single_side_rate},
                       "definition": "src/jinyiquant/evaluation.py eval-v2, mean over the 8 "
                                     "timestamp cross-sections per day; project convention "
                                     "single_side_rate=0.0"},
        "train_meta": json.loads((SCREEN / "train_meta.json").read_text(encoding="utf-8")),
        "predict_meta": json.loads((SCREEN / "predict_meta.json").read_text(encoding="utf-8")),
        "causality_check": str(FEAT / "causality_2024-01.json"),
        "guardrails": {"windows_read": WINDOWS, "confirmation_windows_read": [],
                       "sealed_interval_read": False,
                       "note": "jun2024/sep2024 (confirmation) and 2024-10-08..2025-01-01 "
                               "(sealed) were never read"},
    }
    (SCREEN / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"pooled": pooled, "chosen_w": chosen,
                      "primary_later": primary["later"],
                      "primary_all": primary["all"]}, indent=2), flush=True)
    return result


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "train", "predict", "eval"])
    args = ap.parse_args()
    if args.mode in ("all", "train"):
        stage_train()
    if args.mode in ("all", "predict"):
        stage_predict()
    if args.mode in ("all", "eval"):
        stage_eval()


if __name__ == "__main__":
    main()
