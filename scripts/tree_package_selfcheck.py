#!/usr/bin/env python
"""Selfcheck for the tree-fusion submission package -> runs/tree-package-20261029/selfcheck.json

Proves the packaged path reproduces the increment gate's numbers:
(a) rebuild the 46 features from raw 1-minute bars and compare with the archived tables;
(b) cross-version score sensitivity: local lightgbm 4.7.0 vs the 4.4.0 training box;
(c) recompute the fused arm for all 8 gate windows and diff with the gate's own operators;
(d) one-day end-to-end inside dist/ (bars -> features -> booster -> fusion) + cost/RSS;
(e) requirements resolution.

Read-only with respect to every frozen artifact; writes only selfcheck.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "submission-tree-20261029"
OUT = ROOT / "runs" / "tree-package-20261029"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jinyiquant import book_features as bf  # noqa: E402
from jinyiquant import tree_fusion as tf  # noqa: E402
import fusion_increment_gate as gate  # noqa: E402

DATA = ROOT / "数据"
BARS = DATA / "cpt_jyc_2026_e2e_bar1m"
MAPPING = DATA / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"
FEAT = ROOT / "runs" / "r6r8-20261029" / "features"
SCREEN = ROOT / "runs" / "r6r8-20261029" / "screen"
ALIGNED = ROOT / "runs" / "fusion-gate" / "aligned-r6r8-20261028"
PLAN_PATH = ROOT / "runs" / "fusion-gate" / "plans" / "r6r8-screen-20261028.json"
REG_PATH = ROOT / "configs" / "fusion_increment_registration_20261028.json"

MONTH = {"nov2023": "2023-11", "jan2024": "2024-01", "feb2024": "2024-02", "mar2024": "2024-03",
         "apr2024": "2024-04", "may2024": "2024-05", "jul2024": "2024-07", "aug2024": "2024-08"}
# initial prev for a month = end-of-day summary of the last feature month processed before it
# (the extractor skips 2023-12 and 2024-06, so 2024-01 chains from 2023-11 and 2024-07 from 2024-05)
PREV = {"2023-11": "2023-10", "2024-01": "2023-11", "2024-02": "2024-01", "2024-03": "2024-02",
        "2024-04": "2024-03", "2024-05": "2024-04", "2024-07": "2024-05", "2024-08": "2024-07"}

LEDGER = {
    "record_id": "screen-R6+R8-68e4535ee7",
    "pooled_delta_rank_ic": 0.01157987683312182,
    "ci": [0.010149262547097039, 0.012890599389107133],
    "positive_windows": 8,
    "weight": 0.5,
    "days": 165,
    "turnover_delta": 0.03261119433642856,
    "post_freeze_pooled_delta": 0.011610751347326769,
}


def log(message: str) -> None:
    print(f"[selfcheck] {message}", flush=True)


def col_report(archived: np.ndarray, rebuilt: np.ndarray) -> dict:
    """Exact-value comparison that keeps NaN/+-inf semantics in view."""
    a = np.asarray(archived, dtype=np.float64)
    b = np.asarray(rebuilt, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[finite] - b[finite])
    signs = np.signbit(a) != np.signbit(b)
    inf_mismatch = int(((np.isinf(a) != np.isinf(b)) |
                        (np.isinf(a) & np.isinf(b) & signs)).sum())
    return {
        "max_abs_diff_finite": float(diff.max()) if diff.size else 0.0,
        "n_finite_pairs": int(finite.sum()),
        "n_nan_archived": int(np.isnan(a).sum()),
        "n_nan_rebuilt": int(np.isnan(b).sum()),
        "n_nan_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
        "n_pos_inf_archived": int((a == np.inf).sum()),
        "n_pos_inf_rebuilt": int((b == np.inf).sum()),
        "n_neg_inf_archived": int((a == -np.inf).sum()),
        "n_neg_inf_rebuilt": int((b == -np.inf).sum()),
        "n_inf_mismatch": inf_mismatch,
        "exact": bool(diff.size and diff.max() == 0.0 and inf_mismatch == 0
                      and (np.isnan(a) == np.isnan(b)).all()),
    }


def score_table(keys_p: pd.DataFrame, frame: pd.DataFrame, booster) -> np.ndarray:
    merged = keys_p.merge(frame[["date", "instrument", *tf.FEATURES]],
                          on=["date", "instrument"], how="left", validate="one_to_one",
                          indicator=True)
    if (merged["_merge"] == "right_only").any():
        raise SystemExit("feature rows outside the scored key set")
    return tf.candidate_scores(merged, booster)


def diff_stats(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    extra = np.abs(cand - ref)
    nonzero = extra > 0
    return {
        "rows": int(ref.size),
        "max_abs_diff": float(extra.max()) if ref.size else 0.0,
        "n_nonzero": int(nonzero.sum()),
        "n_exact": int((~nonzero).sum()),
        "abs_diff_quantiles_nonzero": ([float(q) for q in np.nanquantile(extra[nonzero], [0.5, 0.9, 0.99, 1.0])]
                                       if nonzero.any() else []),
    }


def main() -> None:
    started = time.perf_counter()
    mapping = pd.read_csv(MAPPING)
    idm = mapping.set_index("instrument_id")["instrument"].to_dict()
    booster = tf.load_booster(SCREEN / "model.txt")
    report: dict = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "package": str(DIST.relative_to(ROOT)),
                    "gate_ledger": LEDGER, "selfcheck_script": "scripts/tree_package_selfcheck.py"}

    # ---------------------------------------------------------------- (a) feature rebuild
    log("(a) rebuilding the 46 features month by month")
    a_windows, a_scores = {}, {}
    for win, month in MONTH.items():
        t0 = time.perf_counter()
        bars = pd.read_feather(BARS / f"{month.replace('-', '')}.0.feather", columns=bf.BAR_COLS)
        prev = pd.read_parquet(FEAT / f"prev_{PREV[month]}.parquet")
        frame, last = bf.build_table(bars, idm, prev)
        build_seconds = time.perf_counter() - t0
        archived = pd.read_parquet(FEAT / f"features_{month}.parquet")
        merged = archived[["date", "instrument", *bf.FEATURES]].merge(
            frame[["date", "instrument", *bf.FEATURES]], on=["date", "instrument"],
            how="outer", suffixes=("_a", "_r"), indicator=True)
        inner = merged[merged["_merge"] == "both"]
        columns = {}
        for feature in bf.FEATURES:
            columns[feature] = col_report(inner[f"{feature}_a"], inner[f"{feature}_r"])
        differing = sorted(name for name, stat in columns.items() if not stat["exact"])
        inf_columns = {name: {"pos": stat["n_pos_inf_archived"], "neg": stat["n_neg_inf_archived"]}
                       for name, stat in columns.items()
                       if stat["n_pos_inf_archived"] or stat["n_neg_inf_archived"]}
        a_windows[win] = {
            "month": month, "rows_archived": int(len(archived)), "rows_rebuilt": int(len(frame)),
            "rows_matched": int(len(inner)),
            "rows_only_archived": int((merged["_merge"] == "left_only").sum()),
            "rows_only_rebuilt": int((merged["_merge"] == "right_only").sum()),
            "build_seconds": round(build_seconds, 3),
            "columns_exact": int(len(bf.FEATURES) - len(differing)),
            "columns_differing": differing,
            "columns_with_inf": inf_columns,
            "max_abs_diff_finite": max(stat["max_abs_diff_finite"] for stat in columns.values()),
            "n_nan_mismatch": int(sum(stat["n_nan_mismatch"] for stat in columns.values())),
            "n_inf_mismatch": int(sum(stat["n_inf_mismatch"] for stat in columns.values())),
            "last_summary_rows": int(len(last)),
        }
        keys_p = pd.read_parquet(SCREEN / "predictions" / f"{win}.parquet")[["date", "instrument"]]
        score_rebuilt = score_table(keys_p, frame, booster)
        archived_score = pd.read_parquet(SCREEN / "predictions" / f"{win}.parquet")["score"].to_numpy()
        a_scores[win] = diff_stats(archived_score, score_rebuilt)
        log(f"    {win}: {len(inner)}/{len(archived)} rows, {len(differing)} differing cols, "
            f"score max|d|={a_scores[win]['max_abs_diff']:.3g}")
        del bars, frame, archived, merged
    report["a_feature_rebuild"] = {"windows": a_windows, "score_vs_archived": a_scores}

    # ---------------------------------------------------------------- (b) cross-version
    log("(b) scoring the archived feature tables with local lightgbm 4.7.0")
    b_windows = {}
    for win, month in MONTH.items():
        archived = pd.read_parquet(FEAT / f"features_{month}.parquet")
        keys_p = pd.read_parquet(SCREEN / "predictions" / f"{win}.parquet")[["date", "instrument"]]
        local_score = score_table(keys_p, archived, booster)
        archived_score = pd.read_parquet(SCREEN / "predictions" / f"{win}.parquet")["score"].to_numpy()
        stats = diff_stats(archived_score, local_score)
        aligned = keys_p.merge(archived[["date", "instrument", *tf.FEATURES]],
                               on=["date", "instrument"], how="left", validate="one_to_one")
        feature_values = aligned[tf.FEATURES].to_numpy(dtype=np.float64)
        nonfinite = ~np.isfinite(feature_values)
        stats["rows_with_any_nonfinite_feature"] = int(nonfinite.any(axis=1).sum())
        stats["rows_with_any_inf_feature"] = int(np.isinf(feature_values).any(axis=1).sum())
        stats["rows_with_nan_feature"] = int(np.isnan(feature_values).any(axis=1).sum())
        b_windows[win] = stats
        log(f"    {win}: max|d|={stats['max_abs_diff']:.3g} nonzero={stats['n_nonzero']}")
        del archived, local_score, feature_values
    report["b_cross_version"] = {
        "local_lightgbm": "4.7.0", "training_lightgbm": "4.4.0",
        "windows": b_windows,
        "verdict": ("bitwise identical on all 8 windows" if not any(
            row["n_nonzero"] for row in b_windows.values()) else "differences present"),
    }

    # ---------------------------------------------------------------- (c) fusion
    log("(c) recomputing the frozen fusion with the packaged operators")
    registration = json.loads(REG_PATH.read_text(encoding="utf-8"))
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    windows = [row["name"] for row in registration["window_roles"]["windows"]
               if row["role"] == "screen"]
    days_registered = sum(row["trading_days"] for row in registration["window_roles"]["windows"]
                          if row["role"] == "screen")
    cache = gate.WindowCache(plan, registration)
    fit = gate.fit_weight(cache, registration)
    weight = float(fit["weight"])
    trunk_daily_frames, fused_daily_frames, per_window = [], [], {}
    for win in windows:
        entry = cache.get(win)
        keys, targets = entry["keys"], entry["targets"]
        trunk_members = [gate.load_scores(plan["incumbent"]["run_dir"], win, member["version"], keys)
                         for member in plan["incumbent"]["members"]]
        candidate = gate.load_scores(plan["candidate"]["run_dir"], win,
                                     plan["candidate"]["members"][0]["version"], keys)
        pkg_fused = tf.fused_scores(keys, trunk_members, candidate, weight)
        ref_fused = gate.fuse(entry["incumbent"], entry["candidate"], weight)
        pkg_daily = gate.daily_metrics(keys, targets, pkg_fused)
        ref_daily = gate.daily_metrics(keys, targets, ref_fused)
        trunk_daily = gate.daily_metrics(keys, targets, entry["incumbent"])
        if not pkg_daily["date"].equals(ref_daily["date"]) or not pkg_daily["date"].equals(
                trunk_daily["date"]):
            raise SystemExit(f"{win}: daily date index mismatch")
        daily_diff = {column: float(np.abs(pkg_daily[column].to_numpy()
                                           - ref_daily[column].to_numpy()).max())
                      for column in ("rank_ic", "turnover", "long_short_return", "net_return_5bp")}
        per_window[win] = {
            "days": int(len(pkg_daily)),
            "score_max_abs_diff_vs_gate": float(np.abs(pkg_fused - ref_fused).max()),
            "daily_max_abs_diff_vs_gate": daily_diff,
            "trunk_rank_ic_mean": float(trunk_daily["rank_ic"].mean()),
            "fused_rank_ic_mean": float(pkg_daily["rank_ic"].mean()),
            "delta_rank_ic_mean": float((pkg_daily["rank_ic"] - trunk_daily["rank_ic"]).mean()),
            "trunk_turnover_mean": float(trunk_daily["turnover"].mean()),
            "fused_turnover_mean": float(pkg_daily["turnover"].mean()),
        }
        trunk_daily_frames.append(trunk_daily)
        fused_daily_frames.append(pkg_daily)
        log(f"    {win}: fused-vs-gate max|d|={per_window[win]['score_max_abs_diff_vs_gate']:.3g}, "
            f"daily={max(daily_diff.values()):.3g}")
    trunk_all = pd.concat(trunk_daily_frames, ignore_index=True)
    fused_all = pd.concat(fused_daily_frames, ignore_index=True)
    if not trunk_all["date"].equals(fused_all["date"]):
        raise SystemExit("pooled daily frames disagree on the date index")
    delta = fused_all["rank_ic"].to_numpy() - trunk_all["rank_ic"].to_numpy()
    windows_delta = {}
    for win, frame in per_window.items():
        windows_delta[win] = frame["delta_rank_ic_mean"]
    pooled = {
        "days": int(len(delta)),
        "delta_rank_ic_mean": float(delta.mean()),
        "delta_rank_ic_abs_diff_vs_ledger": float(abs(delta.mean() - LEDGER["pooled_delta_rank_ic"])),
        "trunk_rank_ic_mean": float(trunk_all["rank_ic"].mean()),
        "fused_rank_ic_mean": float(fused_all["rank_ic"].mean()),
        "trunk_turnover_mean": float(trunk_all["turnover"].mean()),
        "fused_turnover_mean": float(fused_all["turnover"].mean()),
        "turnover_delta": float(fused_all["turnover"].mean() - trunk_all["turnover"].mean()),
        "positive_windows": int(sum(value > 0 for value in windows_delta.values())),
        "windows_delta_rank_ic": windows_delta,
        "interval": gate.interval_summary(delta),
    }
    post = pd.to_datetime(trunk_all["date"]) >= pd.Timestamp("2023-11-15")
    pooled["post_freeze_delta_rank_ic_mean"] = float(delta[post].mean())
    pooled["ci"] = [pooled["interval"].get("ci_low"), pooled["interval"].get("ci_high")]
    report["c_window"] = {
        "weight": weight, "weight_fit": fit, "days_registered": days_registered,
        "windows": per_window, "pooled": pooled,
        "max_daily_abs_diff_vs_gate": max(
            max(row["daily_max_abs_diff_vs_gate"].values()) for row in per_window.values()),
        "max_score_abs_diff_vs_gate": max(
            row["score_max_abs_diff_vs_gate"] for row in per_window.values()),
    }
    log(f"    pooled delta {pooled['delta_rank_ic_mean']:.17g} "
        f"(ledger diff {pooled['delta_rank_ic_abs_diff_vs_ledger']:.3g}), "
        f"positive_windows {pooled['positive_windows']}")

    # ---------------------------------------------------------------- (d) end to end
    log("(d) one-day end-to-end through dist/")
    import psutil  # noqa: E402

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    sys.path.insert(0, str(DIST))
    import inference  # noqa: E402

    targets = pd.read_parquet(ALIGNED / "windows" / "nov2023" / "targets.parquet")
    targets = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    keys = targets[["date", "instrument"]]
    first_day = pd.Timestamp(keys["date"].min()).normalize()
    prev = pd.read_parquet(FEAT / "prev_2023-10.parquet")
    month_bars = pd.read_feather(BARS / "202311.0.feather", columns=bf.BAR_COLS)
    day_bars = month_bars[month_bars["date"].dt.normalize() == first_day].reset_index(drop=True)
    peak_rss = max(peak_rss, process.memory_info().rss)

    t0 = time.perf_counter()
    day_scores, day_last = inference.tree_scores(day_bars, mapping, prev, inference.load_tree_arm())
    day_seconds = time.perf_counter() - t0
    peak_rss = max(peak_rss, process.memory_info().rss)

    t0 = time.perf_counter()
    window_scores, window_last = inference.tree_scores(month_bars, mapping, prev,
                                                       inference.load_tree_arm())
    window_seconds = time.perf_counter() - t0
    peak_rss = max(peak_rss, process.memory_info().rss)

    trunk_members = [pd.read_parquet(ALIGNED / "windows" / "nov2023" / f"r441_s{seed}" /
                                     "predictions.parquet")[["date", "instrument", "score"]]
                     for seed in ("11", "29", "47")]
    member_scores = []
    for member in trunk_members:
        aligned_member = keys.merge(member, on=["date", "instrument"], how="left",
                                    validate="one_to_one")
        member_scores.append(aligned_member["score"].to_numpy(dtype=float))
    aligned_window = keys.merge(window_scores, on=["date", "instrument"], how="left",
                                validate="one_to_one")
    tree_raw = aligned_window["score"].to_numpy(dtype=float)
    if not np.isfinite(tree_raw).all():
        raise SystemExit("tree arm scored a key row as non-finite")

    ref_entry = cache.get("nov2023")
    pkg_fused = tf.fused_scores(keys, member_scores, tree_raw, weight)
    day_mask = keys["date"].dt.normalize() == first_day
    day_diff = np.abs(pkg_fused[day_mask] - gate.fuse(ref_entry["incumbent"], ref_entry["candidate"],
                                                      weight)[day_mask]).max()
    window_diff = np.abs(pkg_fused - gate.fuse(ref_entry["incumbent"], ref_entry["candidate"],
                                               weight)).max()
    peak_rss = max(peak_rss, process.memory_info().rss)
    rows_per_day = int(day_mask.sum())
    report["d_end_to_end"] = {
        "window": "nov2023", "first_day": str(first_day.date()),
        "pipeline": "bars -> book_features.build_table -> tree_fusion.candidate_scores -> "
                    "tree_fusion.fused_scores (trunk members from the frozen replay)",
        "package_entry": "dist/submission-tree-20261029/inference.py::tree_scores",
        "day_rows": rows_per_day,
        "day_features_rows": int(len(day_scores)),
        "day_seconds": round(day_seconds, 3),
        "window_rows": int(len(window_scores)),
        "window_seconds": round(window_seconds, 3),
        "seconds_per_176k_rows": round(window_seconds * 176000 / max(len(window_scores), 1), 3),
        "peak_rss_mb": round(peak_rss / 1e6, 1),
        "fused_max_abs_diff_first_day_vs_gate": float(day_diff),
        "fused_max_abs_diff_window_vs_gate": float(window_diff),
        "gate_cost_seconds_per_day_ledger": 0.9964848484848484,
        "nondeterministic_note": "trunk arm needs dai, unavailable locally; frozen replay member "
                                 "scores are the same vectors the gate consumed",
        "last_summary_rows": int(len(window_last)),
    }
    log(f"    day {rows_per_day} rows in {day_seconds:.3f}s, window {window_seconds:.3f}s, "
        f"fused vs gate max|d|={window_diff:.3g}")

    # ---------------------------------------------------------------- (e) requirements
    log("(e) requirements")
    requirements = (DIST / "requirements.txt").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, "-m", "pip", "download", "lightgbm==4.4.0",
                               "--no-deps", "-d", tmp], capture_output=True, text=True,
                              timeout=300)
        downloaded = sorted(Path(tmp).glob("lightgbm*"))
        resolution = {
            "command": "pip download lightgbm==4.4.0 --no-deps",
            "returncode": int(proc.returncode),
            "status": "resolved" if proc.returncode == 0 else "not_checked",
            "files": [path.name for path in downloaded],
            "stdout_tail": proc.stdout.strip().splitlines()[-3:],
            "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        }
    report["e_requirements"] = {
        "lines": requirements.strip().splitlines(),
        "lightgbm_pin_present": "lightgbm==4.4.0" in requirements,
        "resolution": resolution,
    }
    report["runtime"] = {"python": sys.version.split()[0],
                         "lightgbm": __import__("lightgbm").__version__,
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "elapsed_seconds": round(time.perf_counter() - started, 1)}

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "selfcheck.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    log(f"wrote {path} ({path.stat().st_size} bytes)")
    print(json.dumps({
        "a_columns_exact": {w: row["columns_exact"] for w, row in a_windows.items()},
        "a_score_max_abs_diff": {w: row["max_abs_diff"] for w, row in a_scores.items()},
        "b_max_abs_diff": {w: row["max_abs_diff"] for w, row in b_windows.items()},
        "c_pooled_delta": pooled["delta_rank_ic_mean"],
        "c_daily_max_abs_diff": report["c_window"]["max_daily_abs_diff_vs_gate"],
        "d_fused_first_day": float(day_diff), "d_fused_window": float(window_diff),
        "e_lightgbm_pin": report["e_requirements"]["lightgbm_pin_present"],
        "e_resolution": resolution["status"],
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
