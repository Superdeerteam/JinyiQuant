"""Extend the frozen R6 feature table to earlier months (for the 927-day training replay).

Calls the frozen feature module `scripts/r6r8_book_features.py` unchanged
(`process_month`), so the feature values are produced by the identical code path.
Nothing is refitted and no model is touched; only the feature table gains months.

    python scripts/r6r8_extend_features.py --months 2020-01 .. 2021-12

Writes `runs/r6r8-20261029/features/features_<month>.parquet` and `prev_<month>.parquet`
(already-existing months are skipped unless --overwrite) and appends per-month timing to
`runs/r6r8-20261029/features/extract_meta_ext.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r6r8_book_features as F  # noqa: E402
import r6r8_screen as R6  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", required=True, metavar="YYYY-MM")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = R6.FEAT
    out.mkdir(parents=True, exist_ok=True)
    mapp = F._mapping()
    prev = None
    meta_path = out / "extract_meta_ext.jsonl"
    for month in args.months:
        fpath = out / f"features_{month}.parquet"
        spath = out / f"prev_{month}.parquet"
        if fpath.exists() and not args.overwrite:
            print(f"[{month}] exists, skip", flush=True)
            prev = pd.read_parquet(spath) if spath.exists() else None
            continue
        t0 = time.time()
        feats, prev, meta = F.process_month(month, mapp, prev, emit=True)
        feats.to_parquet(fpath, index=False)
        meta["emit"] = True
        del feats
        if prev is not None:
            prev.to_parquet(spath, index=True)
        with meta_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(meta) + "\n")
        print(f"[{month}] rows={meta['rows']} target_coverage={meta['target_coverage']} {meta['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
