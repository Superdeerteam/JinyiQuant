"""Sealed scoring for the two untouched confirmation windows (jun2024, sep2024).

Generates candidate-side artifacts only. No metric of any kind is computed, no target
value is read: the vwap label files are never opened, and the replay `targets.parquet`
is read with ``columns=["date", "instrument"]`` for key alignment alone.

Feature computation reuses the frozen module `scripts/r6r8_book_features.py`
(sha a3feada6...) by calling its own `_month_arrays` / `build_day` / `_rows` on the
month, so the values are produced by the identical code path while the label-merge step
of `process_month` is skipped entirely. Scoring reuses `runs/r6r8-20261029/screen/model.txt`
(sha 98f58e81...) unchanged; nothing is refitted.

Outputs
  runs/r6r8-20261029/sealed-jun-sep/
    features/features_<month>.parquet        (sealed months kept out of the screen dir)
    windows/<window>/predictions.parquet      [date, instrument, score]
    sealed_manifest.json
  runs/r6r8-20261029/review/sealed_evidence.json  (manifest sha + model/feature shas + UTC)

Logs print only row counts, key counts, finiteness and sha256 values.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r6r8_book_features as F  # noqa: E402
import r6r8_screen as R6  # noqa: E402

SEALED = ROOT / "runs/r6r8-20261029/sealed-jun-sep"
REVIEW = ROOT / "runs/r6r8-20261029/review"
WINDOWS = {"jun2024": "2024-06", "sep2024": "2024-09"}
PREV_SEED = {"2024-06": "2024-05", "2024-09": "2024-08"}
# screen windows are only used by --dry-run, which never touches the confirmation windows
DRY_RUN_PREV_SEED = {"2023-11": "2023-10", "2024-01": "2023-12",
                     "2024-02": "2024-01", "2024-03": "2024-02", "2024-04": "2024-03",
                     "2024-05": "2024-04", "2024-07": "2024-05", "2024-08": "2024-07"}
MODEL_SHA_EXPECTED = "98f58e81"
FEATURE_SHA_EXPECTED = "a3feada6"
KEYS = ["date", "instrument"]
FEATURES = R6.FEATURES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log(msg: str) -> None:
    print(f"[sealed {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def features_for(month: str) -> pd.DataFrame:
    """Same computation as r6r8_book_features.process_month, without the label merge."""
    mapp = F._mapping()
    seed = PREV_SEED.get(month) or DRY_RUN_PREV_SEED[month]
    prev = pd.read_parquet(R6.RUN / "features" / f"prev_{seed}.parquet")
    arr, ids, minute, day, pos_by_day, idm = F._month_arrays(month, mapp)
    frames = []
    for d in sorted(pos_by_day):
        pos = pos_by_day[d]
        cols = {c: arr[c][pos].astype(np.float64) for c in F.NUM_COLS}
        feat, summary, dids = F.build_day(cols, ids[pos], minute[pos], prev, F.SAMPLE_MIN)
        if len(dids):
            frames.append(F._rows(feat, dids, d, idm, F.SAMPLE_MIN))
        prev = summary
    feats = pd.concat(frames, ignore_index=True)
    return feats


def _score(booster, feats: pd.DataFrame, keys: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    merged = keys.merge(feats[KEYS + FEATURES], on=KEYS, how="left", validate="one_to_one")
    feats_np = merged[FEATURES].to_numpy(dtype=np.float32)
    score = booster.predict(feats_np, num_threads=4).astype(np.float64)
    return merged, score


def dry_run(window: str) -> int:
    """Exercise the exact sealed path on ONE screen window; write nothing into SEALED/windows."""
    assert window in R6.WINDOW_MONTH, f"{window} is not a screen window"
    month = R6.WINDOW_MONTH[window]
    model_path = R6.SCREEN / "model.txt"
    model_sha = _sha(model_path)
    feature_sha = _sha(ROOT / "scripts/r6r8_book_features.py")
    assert model_sha.startswith(MODEL_SHA_EXPECTED), model_sha
    assert feature_sha.startswith(FEATURE_SHA_EXPECTED), feature_sha
    import lightgbm as lgb
    booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    t0 = time.time()
    _log(f"dry-run window={window} month={month} (screen window; confirmation windows untouched)")
    feats = features_for(month)
    _log(f"features rows={len(feats)} cols={len(feats.columns)} sha={_sha_frame(feats)[:16]}")
    keys = pd.read_parquet(R6.REPLAY / "windows" / window / "keys.parquet")
    target_columns_read = KEYS
    tkeys = pd.read_parquet(R6.REPLAY / "windows" / window / "targets.parquet",
                            columns=target_columns_read)
    _log(f"targets read columns={target_columns_read} rows={len(tkeys)} (no target value read)")
    merged, score = _score(booster, feats, keys)
    finite = int(np.isfinite(score).sum())
    if finite != len(score):
        raise SystemExit(f"{window}: non-finite scores {len(score) - finite}")
    pred = merged[KEYS].assign(score=score)
    aligned = pred.merge(tkeys, on=KEYS, how="inner", validate="one_to_one")
    key_diff = {
        "keys_total": int(len(keys)),
        "target_keys_total": int(len(tkeys)),
        "rows_aligned": int(len(aligned)),
        "keys_minus_target_keys": int(len(keys) - len(aligned)),
        "target_keys_minus_keys": int(len(tkeys) - len(aligned)),
        "keyset_equal": bool(pred[KEYS].reset_index(drop=True).equals(
            aligned[KEYS].reset_index(drop=True))),
    }
    out = SEALED / "dry-run" / window
    out.mkdir(parents=True, exist_ok=True)
    ppath = out / "predictions.parquet"
    aligned.reset_index(drop=True).to_parquet(ppath, index=False)
    screen = pd.read_parquet(R6.SCREEN / "predictions" / f"{window}.parquet")
    cmp = aligned.merge(screen, on=KEYS, suffixes=("_dry", "_screen"))
    max_abs = float(np.max(np.abs(cmp.score_dry - cmp.score_screen)))
    stored = pd.read_parquet(R6.RUN / "features" / f"features_{month}.parquet",
                             columns=KEYS + FEATURES)
    m = feats.merge(stored, on=KEYS, suffixes=("_new", "_stored"))
    col_diff = {}
    for c in FEATURES:
        a = m[c + "_new"].to_numpy()
        b = m[c + "_stored"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        with np.errstate(invalid="ignore"):
            d = np.where(both_nan, 0.0, np.abs(a - b))
        col_diff[c] = float(np.nanmax(d)) if len(d) else 0.0
    report = {
        "schema": "r6r8-sealed-dry-run/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "window": window, "month": month,
        "model_sha256": model_sha, "feature_script_sha256": feature_sha,
        "target_columns_read": target_columns_read,
        "metrics_computed": False,
        "key_diff": key_diff,
        "features_compared_rows": int(len(m)),
        "feature_max_abs_diff_vs_screen_table": max(col_diff.values()) if col_diff else None,
        "feature_columns_with_diff": [c for c, v in col_diff.items() if v > 0.0],
        "prediction_max_abs_diff_vs_screen": max_abs,
        "prediction_rows": int(len(aligned)),
        "prediction_sha256": _sha(ppath),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "dry_run_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _log(f"rows={len(aligned)} keys={len(keys)} key_diff={key_diff}")
    _log(f"feature max|diff| vs screen table = {report['feature_max_abs_diff_vs_screen_table']}")
    _log(f"prediction max|diff| vs screen = {max_abs}")
    _log(f"prediction sha={report['prediction_sha256'][:16]} seconds={report['seconds']}")
    return 0


def _sha_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", default=None, metavar="WINDOW",
                    help="run the sealed path on one SCREEN window (e.g. aug2024)")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run(args.dry_run)
    model_path = R6.SCREEN / "model.txt"
    model_sha = _sha(model_path)
    feature_sha = _sha(ROOT / "scripts/r6r8_book_features.py")
    assert model_sha.startswith(MODEL_SHA_EXPECTED), model_sha
    assert feature_sha.startswith(FEATURE_SHA_EXPECTED), feature_sha
    import lightgbm as lgb
    booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    SEALED.mkdir(parents=True, exist_ok=True)
    (SEALED / "features").mkdir(parents=True, exist_ok=True)
    (SEALED / "windows").mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {"schema": "r6r8-sealed/1", "generated_utc": generated,
                "sealed": True, "metrics_computed": False,
                "model_file": str(model_path.relative_to(ROOT)), "model_sha256": model_sha,
                "feature_script": "scripts/r6r8_book_features.py",
                "feature_script_sha256": feature_sha,
                "target_read_policy": "targets.parquet read with columns=['date','instrument'] only",
                "windows": {}}
    for window, month in WINDOWS.items():
        t0 = time.time()
        fpath = SEALED / "features" / f"features_{month}.parquet"
        if fpath.exists():
            feats = pd.read_parquet(fpath)
        else:
            feats = features_for(month)
            feats.to_parquet(fpath, index=False)
        keys = pd.read_parquet(R6.REPLAY / "windows" / window / "keys.parquet")
        tkeys = pd.read_parquet(R6.REPLAY / "windows" / window / "targets.parquet",
                                columns=KEYS)
        merged = keys.merge(feats[KEYS + FEATURES], on=KEYS, how="left", validate="one_to_one")
        score = booster.predict(merged[FEATURES].to_numpy(dtype=np.float32),
                                num_threads=4).astype(np.float64)
        finite = int(np.isfinite(score).sum())
        if finite != len(score):
            raise SystemExit(f"{window}: non-finite scores {len(score) - finite}")
        pred = merged[KEYS].assign(score=score)
        aligned = pred.merge(tkeys, on=KEYS, how="inner", validate="one_to_one")
        out = SEALED / "windows" / window
        out.mkdir(parents=True, exist_ok=True)
        ppath = out / "predictions.parquet"
        aligned.reset_index(drop=True).to_parquet(ppath, index=False)
        digest = _sha(ppath)
        manifest["windows"][window] = {
            "month": month, "keys_total": int(len(keys)), "rows": int(len(aligned)),
            "dropped_unlabeled": int(len(keys) - len(aligned)),
            "days": int(aligned["date"].dt.normalize().nunique()),
            "finite_scores": finite, "columns": ["date", "instrument", "score"],
            "predictions_sha256": digest,
            "features_file": str(fpath.relative_to(ROOT)),
            "features_sha256": _sha(fpath),
            "seconds": round(time.time() - t0, 1)}
        _log(f"{window} rows={len(aligned)} keys={len(keys)} dropped={len(keys)-len(aligned)} "
             f"finite={finite} sha={digest[:16]}")
    mpath = SEALED / "sealed_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    evidence = {"schema": "r6r8-sealed-evidence/1", "generated_utc": generated,
                "statement": ("Sealed scores for jun2024/sep2024 were generated from the frozen "
                              "model and frozen feature code; no IC, correlation, return or "
                              "turnover was computed and no target value was read."),
                "manifest_path": str(mpath.relative_to(ROOT)),
                "manifest_sha256": _sha(mpath),
                "model_sha256": model_sha, "feature_script_sha256": feature_sha,
                "windows": {w: {k: v for k, v in d.items()
                                if k in ("rows", "keys_total", "predictions_sha256")}
                            for w, d in manifest["windows"].items()}}
    REVIEW.mkdir(parents=True, exist_ok=True)
    epath = REVIEW / "sealed_evidence.json"
    epath.write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    _log(f"manifest sha={_sha(mpath)[:16]} evidence sha={_sha(epath)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
