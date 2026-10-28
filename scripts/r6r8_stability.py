"""Descriptive stability replays for the R6 market-book tree (no selection, no verdict).

Two questions from the captain, both answered descriptively on the 8 screen windows only:

  a) three LightGBM seeds on the 441-day training window (2022-01-02..2023-10-31);
  b) one seed trained on the true 927-day window (2020-01-02..2023-10-31) -- 486 more
     trading days of training data than the backbone's 441-day rung.

Everything else is frozen: the 46 features, the 600 fixed rounds, the LightGBM
hyper-parameters, the 8 screen windows.  Only the seed and the training-month list vary.
Predictions and per-day metrics are written under runs/r6r8-20261029/stability/.

    python scripts/r6r8_stability.py --mode seeds441
    python scripts/r6r8_stability.py --mode train927
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r6r8_screen as R6  # noqa: E402

KEYS = ["date", "instrument"]
STAB = R6.RUN / "stability"
MONTHS_927 = [f"{y}-{m:02d}" for y in (2020, 2021) for m in range(1, 13)] + R6.TRAIN_MONTHS
SEEDS = (20261030, 20261031)  # 20261029 already exists as the frozen screen model


def _count_rows(months: list[str]) -> int:
    total = 0
    for month in months:
        t = pd.read_parquet(R6.FEAT / f"features_{month}.parquet", columns=["target"])
        total += int(t["target"].notna().sum())
    return total


def _train(name: str, months: list[str], seed: int, use_memmap: bool) -> dict:
    import lightgbm as lgb

    out = STAB / name
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.txt"
    meta_path = out / "train_meta.json"
    if model_path.exists() and meta_path.exists():
        print(f"[train:{name}] exists, skip", flush=True)
        return json.loads(meta_path.read_text(encoding="utf-8"))
    total = _count_rows(months)
    print(f"[train:{name}] rows={total:,} months={len(months)} seed={seed} memmap={use_memmap}",
          flush=True)
    if use_memmap:
        X = np.lib.format.open_memmap(out / "X.f32.npy", mode="w+", dtype=np.float32,
                                      shape=(total, len(R6.FEATURES)))
    else:
        X = np.empty((total, len(R6.FEATURES)), dtype=np.float32)
    y = np.empty(total, dtype=np.float32)
    pos = 0
    for month in months:
        frame = pd.read_parquet(R6.FEAT / f"features_{month}.parquet",
                                columns=["date", "target"] + R6.FEATURES)
        frame = frame[frame["target"].notna()]
        n = len(frame)
        X[pos:pos + n] = frame[R6.FEATURES].to_numpy(dtype=np.float32)
        y[pos:pos + n] = R6._csz_by_timestamp(frame).astype(np.float32)
        pos += n
        del frame
    assert pos == total, (pos, total)
    params = dict(R6.PARAMS, seed=seed)
    t0 = time.time()
    dtrain = lgb.Dataset(X, label=y, feature_name=R6.FEATURES, free_raw_data=True,
                         params={"max_bin": params["max_bin"], "verbosity": -1})
    dtrain.construct()
    del X
    model = lgb.train(params, dtrain, num_boost_round=R6.ROUNDS)
    train_seconds = time.time() - t0
    model_path.write_text(model.model_to_string(), encoding="utf-8")
    meta = {"name": name, "rows": int(total), "months": months, "seed": seed,
            "params": params, "rounds": R6.ROUNDS, "train_seconds": round(train_seconds, 1),
            "memmap": use_memmap}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if (out / "X.f32.npy").exists():
        (out / "X.f32.npy").unlink()
    print(f"[train:{name}] done in {train_seconds:.0f}s", flush=True)
    return meta


def _predict(name: str) -> dict:
    import lightgbm as lgb

    out = STAB / name
    booster = lgb.Booster(model_str=(out / "model.txt").read_text(encoding="utf-8"))
    pdir = out / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for window in R6.WINDOWS:
        t0 = time.time()
        month = R6.WINDOW_MONTH[window]
        keys = pd.read_parquet(R6.REPLAY / "windows" / window / "keys.parquet")
        feats = pd.read_parquet(R6.FEAT / f"features_{month}.parquet",
                                columns=KEYS + R6.FEATURES)
        merged = keys.merge(feats, on=KEYS, how="left", validate="one_to_one")
        score = booster.predict(merged[R6.FEATURES].to_numpy(dtype=np.float32),
                                num_threads=4).astype(np.float64)
        if not np.isfinite(score).all():
            raise SystemExit(f"{name}/{window}: non-finite scores")
        merged[KEYS].assign(score=score).to_parquet(pdir / f"{window}.parquet", index=False)
        meta[window] = {"rows": int(len(merged)), "predict_seconds": round(time.time() - t0, 2)}
        print(f"[predict:{name}] {window} rows={len(merged):,} {meta[window]['predict_seconds']}s",
              flush=True)
    (out / "predict_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _mean3() -> Path:
    """Equal-weight mean of the three seed scores per window (same target scale)."""
    out = STAB / "mean3" / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    dirs = [R6.SCREEN / "predictions"] + [STAB / f"seed{s}" / "predictions" for s in SEEDS]
    for window in R6.WINDOWS:
        frames = [pd.read_parquet(d / f"{window}.parquet") for d in dirs]
        base = frames[0][KEYS]
        scores = [f["score"].to_numpy() for f in frames]
        for f in frames:
            assert np.array_equal(f["date"].to_numpy(), base["date"].to_numpy()) and \
                np.array_equal(f["instrument"].to_numpy(), base["instrument"].to_numpy())
        base.assign(score=np.mean(scores, axis=0)).to_parquet(out / f"{window}.parquet",
                                                              index=False)
    return out


def _eval(tag: str, variants: dict[str, Path]) -> dict:
    rows = []
    for window in R6.WINDOWS:
        keys = pd.read_parquet(R6.REPLAY / "windows" / window / "keys.parquet")
        targets = pd.read_parquet(R6.REPLAY / "windows" / window / "targets.parquet")
        for name, pdir in variants.items():
            pred = pd.read_parquet(pdir / f"{window}.parquet")
            merged = keys.merge(pred, on=KEYS, how="left", validate="one_to_one")
            if not np.isfinite(merged["score"]).all():
                raise SystemExit(f"{tag}/{window}/{name}: missing scores")
            daily = R6._daily(keys, targets, merged["score"].to_numpy(dtype=float), name)
            rows.append(daily.assign(window=window))
    daily = pd.concat(rows, ignore_index=True)
    daily.to_parquet(STAB / f"daily_{tag}.parquet", index=False)
    pooled = {name: R6._pooled(daily[daily["variant"] == name], R6.WINDOWS) for name in variants}
    per_window = {}
    for name in variants:
        sub = daily[daily["variant"] == name]
        per_window[name] = {w: float(sub[sub["window"] == w]["rank_ic"].mean())
                            for w in R6.WINDOWS}
    paired = {}
    names = list(variants)
    ref = names[0]
    for name in names[1:]:
        paired[f"{name}_minus_{ref}"] = {
            "all8": R6._paired(daily, name, ref, R6.WINDOWS),
            "later6": R6._paired(daily, name, ref, R6.LATER),
        }
    return {"schema": "r6r8-stability/1", "tag": tag, "generated_utc": pd.Timestamp.utcnow().isoformat(),
            "windows": R6.WINDOWS, "variants": {k: str(v) for k, v in variants.items()},
            "pooled": pooled, "per_window": per_window, "paired": paired}


def _seed_spread(per_window: dict) -> dict:
    out = {}
    for w in R6.WINDOWS:
        vals = [per_window[f"seed{s}"][w] for s in (20261029,) + SEEDS]
        out[w] = {"min": min(vals), "max": max(vals), "spread": max(vals) - min(vals)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seeds441", "train927", "all"], default="all")
    args = ap.parse_args()
    if args.mode in ("seeds441", "all"):
        timing = {}
        for seed in SEEDS:
            name = f"seed{seed}"
            timing[name] = _train(name, R6.TRAIN_MONTHS, seed, use_memmap=False)
            timing[name]["predict"] = _predict(name)
        mean3 = _mean3()
        variants = {"seed20261029": R6.SCREEN / "predictions",
                    "seed20261030": STAB / f"seed{SEEDS[0]}" / "predictions",
                    "seed20261031": STAB / f"seed{SEEDS[1]}" / "predictions",
                    "mean3": mean3}
        summary = _eval("seeds441", variants)
        summary["seed_train_seconds"] = {k: v["train_seconds"] for k, v in timing.items()}
        summary["spread_across_seeds"] = _seed_spread(summary["per_window"])
        (STAB / "seeds441_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps({"pooled": summary["pooled"],
                          "paired_mean3": summary["paired"]["mean3_minus_seed20261029"]["all8"],
                          "later6": summary["paired"]["mean3_minus_seed20261029"]["later6"]},
                         indent=1), flush=True)
    if args.mode in ("train927", "all"):
        missing = [m for m in MONTHS_927 if not (R6.FEAT / f"features_{m}.parquet").exists()]
        if missing:
            raise SystemExit(f"missing feature months for 927d: {missing}")
        meta = _train("seed927", MONTHS_927, 20261029, use_memmap=True)
        pred_meta = _predict("seed927")
        variants = {"seed20261029_441d": R6.SCREEN / "predictions",
                    "seed20261029_927d": STAB / "seed927" / "predictions"}
        summary = _eval("train927", variants)
        summary["train_meta"] = meta
        summary["predict_meta"] = pred_meta
        summary["training_days_927"] = len(MONTHS_927)
        (STAB / "train927_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps({"pooled": summary["pooled"],
                          "paired": summary["paired"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
