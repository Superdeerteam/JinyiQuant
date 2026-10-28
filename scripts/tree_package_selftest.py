#!/usr/bin/env python
"""Simulated 60-trading-day request through ``dist/submission-tree-20261029/inference.py``.

No platform here, so the data access hook (``inference._run_query``) is backed by the local
parquet/feather files and ``modified_tlob_flat.main`` is replaced by a documented stand-in
trunk (the frozen trunk needs ``dai`` + torch and is unchanged code). Everything this test
is about - chunked loading, feature build, tree scoring, whole-range EWMA, fusion, row
wiring - runs the packaged code verbatim.

    python scripts/tree_package_selftest.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "submission-tree-20261029"
OUT = ROOT / "runs" / "tree-package-20261029"
DATA = ROOT / "数据"
BARS_DIR = DATA / "cpt_jyc_2026_e2e_bar1m"
MAPPING = DATA / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv"
FEAT = ROOT / "runs" / "r6r8-20261029" / "features"
SCREEN = ROOT / "runs" / "r6r8-20261029" / "screen"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DIST))
from jinyiquant import book_features as bf  # noqa: E402
from jinyiquant import tree_fusion as tf  # noqa: E402
import inference  # noqa: E402  (the packaged entry, resolved from DIST)

START = pd.Timestamp("2023-09-04")   # 60 trading days ending with the nov2023 gate window
END = pd.Timestamp("2023-11-30")
MONTHS = ("202308", "202309", "202310", "202311")


def log(message: str) -> None:
    print(f"[selftest] {message}", flush=True)


class LocalTables:
    """Stands in for DAI: bounded date slices straight from the local files."""

    def __init__(self):
        self.cache = {}
        self.calls = 0
        self.rows = 0

    def __call__(self, table, columns, start, end):
        self.calls += 1
        if "instrument_map" in str(table):
            frame = pd.read_csv(MAPPING)[list(columns)]
            self.rows += len(frame)
            return frame
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        pieces = []
        for month in MONTHS:
            stamp = pd.Timestamp(f"{month[:4]}-{month[4:]}-01")
            if stamp >= end or stamp + pd.offsets.MonthBegin(1) <= start:
                continue
            if month not in self.cache:
                self.cache[month] = pd.read_feather(BARS_DIR / f"{month}.0.feather")
            frame = self.cache[month]
            picked = frame.loc[(frame["date"] >= start) & (frame["date"] < end)]
            if len(picked):
                pieces.append(picked[list(columns)])
        if not pieces:
            return pd.DataFrame({column: [] for column in columns})
        frame = pd.concat(pieces, ignore_index=True)
        self.rows += len(frame)
        return frame


def turnover_proxy(keys: pd.DataFrame, scores) -> float:
    """Mean |score change| between consecutive intraday timestamps, averaged over days."""
    frame = keys[["date", "instrument"]].assign(score=np.asarray(scores, dtype=float))
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["instrument", "date"])
    frame["step"] = frame.groupby("instrument")["date"].diff().dt.total_seconds()
    frame["delta"] = frame.groupby("instrument")["score"].diff().abs()
    intraday = frame[(frame["step"] > 0) & (frame["step"] <= 3600)]
    return float(intraday.groupby(frame["date"].dt.normalize())["delta"].mean().mean())


def stand_in_trunk(keys: pd.DataFrame) -> np.ndarray:
    """Documented trunk stand-in: EWMA of the cross-sectional z of an intraday quote signal."""
    ordered = keys.sort_values(["date", "instrument"]).reset_index(drop=True)
    rng = np.random.default_rng(20261029)
    signal = rng.standard_normal(len(ordered))
    return tf.causal_ewma(ordered, signal, tf.EWMA_ALPHA)


def main() -> None:
    started = time.perf_counter()
    tables = LocalTables()
    inference._run_query = tables
    booster = inference.load_tree_arm()
    mapping = pd.read_csv(MAPPING)

    keys = pd.concat([pd.read_parquet(FEAT / f"features_{month}.parquet")[["date", "instrument"]]
                      for month in ("2023-09", "2023-10", "2023-11")], ignore_index=True)
    keys["date"] = pd.to_datetime(keys["date"])
    keys = (keys[(keys["date"].dt.normalize() >= START) & (keys["date"].dt.normalize() <= END)]
            .sort_values(["date", "instrument"]).reset_index(drop=True))
    trunk_factor = stand_in_trunk(keys)

    calls = {}
    captured = []

    def stub_trunk(datasources, start_date, end_date):
        calls["start"] = str(start_date)
        calls["end"] = str(end_date)
        return keys.assign(factor=trunk_factor)

    original_tree_scores = inference.tree_scores

    def recorded(bars, mapping_frame, prev, model, num_threads=4):
        scores, summary = original_tree_scores(bars, mapping_frame, prev, model, num_threads)
        captured.append(scores)
        return scores, summary

    inference.tree_scores = recorded
    sys.modules["modified_tlob_flat"] = SimpleNamespace(main=stub_trunk)

    peak = {"rss": psutil.Process().memory_info().rss}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], psutil.Process().memory_info().rss)
            time.sleep(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    fused_frame = inference.main({"bar1m": "cpt_jyc_2026_e2e_bar1m",
                                  "instrument_map": "cpt_jyc_2026_stock_bar1m_instrument_mapping"},
                                 START.date(), END.date())
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join()
    inference.tree_scores = original_tree_scores
    peak["rss"] = max(peak["rss"], psutil.Process().memory_info().rss)

    chunked = pd.concat(captured, ignore_index=True)
    chunked = chunked[(chunked["date"] >= START) & (chunked["date"] < END + pd.Timedelta(days=1))]
    in_range = keys.merge(chunked, on=["date", "instrument"], how="left", validate="one_to_one")
    n_missing = int(in_range["score"].isna().sum())
    key_set = set(map(tuple, keys[["date", "instrument"]].to_numpy()))
    fused_keys = set(map(tuple, fused_frame[["date", "instrument"]].to_numpy()))

    # the same tree arm rebuilt in one month-at-once pass, for the nov2023 overlap check
    month_bars = pd.read_feather(BARS_DIR / "202311.0.feather", columns=bf.BAR_COLS)
    prev = pd.read_parquet(FEAT / "prev_2023-10.parquet")
    whole_frame, _ = bf.build_table(month_bars, mapping.set_index("instrument_id")["instrument"]
                                    .to_dict(), prev)
    whole_scores = tf.candidate_scores(whole_frame, booster)
    nov_keys = keys[pd.to_datetime(keys["date"]).dt.strftime("%Y-%m") == "2023-11"]
    whole_merged = nov_keys.merge(whole_frame[["date", "instrument", *tf.FEATURES]],
                                  on=["date", "instrument"], how="left", validate="one_to_one")
    whole = pd.DataFrame({"date": whole_merged["date"], "instrument": whole_merged["instrument"],
                          "score": whole_scores})
    chunked_nov = chunked[pd.to_datetime(chunked["date"]).dt.strftime("%Y-%m") == "2023-11"]
    overlap = nov_keys.merge(chunked_nov, on=["date", "instrument"], how="left",
                             validate="one_to_one").merge(
        whole, on=["date", "instrument"], how="left", suffixes=("_chunked", "_whole"),
        validate="one_to_one")
    chunked_vs_whole = float(np.abs(overlap["score_chunked"].to_numpy()
                                    - overlap["score_whole"].to_numpy()).max())
    archived = pd.read_parquet(SCREEN / "predictions" / "nov2023.parquet")[["date", "instrument", "score"]]
    archived = archived.merge(nov_keys, on=["date", "instrument"], how="inner")
    chunked_nov2 = chunked_nov.merge(nov_keys, on=["date", "instrument"], how="inner")
    merged_arch = archived.merge(chunked_nov2, on=["date", "instrument"], how="inner",
                                 suffixes=("_archived", "_local"), validate="one_to_one")
    vs_archived = float(np.abs(merged_arch["score_archived"].to_numpy()
                               - merged_arch["score_local"].to_numpy()).max())

    tree_raw = in_range["score"].to_numpy(dtype=float)
    tree_z = tf.cross_sectional_z(keys, tree_raw)
    tree_smoothed = tf.causal_ewma(keys, tree_raw, tf.EWMA_ALPHA)
    report = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "package_entry": "dist/submission-tree-20261029/inference.py::main",
        "request": {"start": str(START.date()), "end": str(END.date()),
                    "trading_days": int(keys["date"].dt.normalize().nunique()),
                    "rows": int(len(keys))},
        "data_access": {"mode": "local files behind inference._run_query",
                        "queries": int(tables.calls), "bar_rows_read": int(tables.rows),
                        "chunk_days": inference.CHUNK_DAYS,
                        "warmup_days": inference.WARMUP_DAYS},
        "trunk_stand_in": "modified_tlob_flat.main replaced by a documented stand-in "
                          "(EWMA of the cross-sectional z of a fixed-seed signal); the real "
                          "trunk path is unchanged frozen code and cannot run without dai",
        "trunk_call": calls,
        "rows_out": int(len(fused_frame)),
        "keys_missing_vs_trunk": int(len(key_set - fused_keys)),
        "keys_extra_vs_trunk": int(len(fused_keys - key_set)),
        "tree_rows_missing_vs_trunk": n_missing,
        "rows_missing_after_merge": n_missing,
        "fused_finite": bool(np.isfinite(fused_frame["factor"].to_numpy()).all()),
        "tree_arm_vs_archived_nov2023_max_abs_diff": vs_archived,
        "chunked_vs_month_at_once_nov2023_max_abs_diff": chunked_vs_whole,
        "turnover_proxy": {
            "definition": "mean over days of the mean |score change| across consecutive "
                          "intraday timestamps, per instrument",
            "tree_raw_z_units": turnover_proxy(keys, tree_z),
            "tree_smoothed": turnover_proxy(keys, tree_smoothed),
            "trunk_stand_in": turnover_proxy(keys, trunk_factor),
            "fused": turnover_proxy(keys, fused_frame["factor"].to_numpy(dtype=float)),
            "note": "tree_raw is reported in the tree's own score units (std~0.07), every "
                    "other row is in cross-sectional z units, so only same-unit rows compare; "
                    "smoothing must lower z-unit turnover, not raise it",
        },
        "cost": {"wall_seconds": round(wall, 2),
                 "seconds_per_day": round(wall / max(keys["date"].dt.normalize().nunique(), 1), 3),
                 "peak_rss_mb": round(peak["rss"] / 1e6, 1)},
        "budget": {
            "wall_seconds_from_submission_config": int(json.loads(
                (DIST / "submission_config.json").read_text(encoding="utf-8"))["wall_seconds"]),
            "wall_fraction_of_budget": round(wall / float(json.loads(
                (DIST / "submission_config.json").read_text(encoding="utf-8"))["wall_seconds"]), 4),
            "peak_rss_limit_mb": 8000,
            "frozen_gate_cost_seconds_per_day": 0.9964848484848484,
            "note": "end-to-end wall here includes reading 17M bar rows through the local-file "
                    "data hook; the gate's 0.9965 s/day is feature+score only (selfcheck (d))",
        },
    }
    report["verdict"] = ("PASS" if report["keys_missing_vs_trunk"] == 0
                         and report["keys_extra_vs_trunk"] == 0 and n_missing == 0
                         and report["fused_finite"] and vs_archived == 0.0
                         and chunked_vs_whole == 0.0
                         and report["cost"]["peak_rss_mb"] < 8000 else "FAIL")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "selftest.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    log(f"total {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
