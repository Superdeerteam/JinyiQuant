"""Reproduce a budget point's stored predictions from the exported text weights.

The hand-off contract for R14 is a single ``modified_tlob_weights-<point>.json`` file: the
scoring host loads it with the existing ``TLOBAdapter.load`` and replays the screen windows.
This script closes that loop on the arm's own machine -- it loads the exported file, predicts
the same validation keys and reports the difference against the predictions the run stored, so
"the export is re-loadable and prediction-identical" is a measured number, not a claim.

The window read starts ``--history-days`` before the checked days, so every checked key sees the
same 128-bar history the run's pool gave it; the last days of the validation split are the only
ones this checks, which is exactly the part a replay can reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.data import build_batch, load_data  # noqa: E402
from jinyiquant.io import write_json  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402

SCHEMA = "r14-export-roundtrip/1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_keys(predictions: pd.DataFrame, *, cutoff) -> pd.DataFrame:
    dates = pd.to_datetime(predictions["date"])
    return predictions.loc[dates >= pd.Timestamp(cutoff), ["date", "instrument"]].reset_index(drop=True)


def roundtrip(config, *, run_dir: Path, point: str, days: int) -> dict:
    split = getattr(config.splits, config.evaluation_split)
    stored = pd.read_parquet(run_dir / "budget_points" / f"{point}_predictions.parquet")
    weights = run_dir / f"modified_tlob_weights-{point}.json"
    if not weights.is_file():
        raise SystemExit(f"missing exported weights: {weights}")
    cutoff = pd.Timestamp(split.end) - pd.Timedelta(days=days)
    keys = checked_keys(stored, cutoff=cutoff)
    if len(keys) == 0:
        raise SystemExit(f"no stored predictions on or after {cutoff}")
    start = cutoff - pd.Timedelta(days=config.data.history_days)
    bundle = load_data(config.data, start=start, end=split.end)
    batch = build_batch(bundle.bars, keys, config.data)
    adapter = TLOBAdapter.load(weights, device="cpu")
    replay = adapter.predict(batch)
    frame = batch.keys[["date", "instrument"]].copy()
    frame["replay"] = np.asarray(replay, dtype=np.float64)
    merged = frame.merge(stored[["date", "instrument", "score"]], on=["date", "instrument"],
                         how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise SystemExit(f"the stored predictions are missing {len(frame) - len(merged)} keys")
    difference = np.abs(merged["replay"].to_numpy() - merged["score"].to_numpy())
    return {"schema_version": SCHEMA, "run": str(run_dir), "point": point,
            "recipe": str(config.data.bars_path), "evaluation_split": config.evaluation_split,
            "weights": {"path": weights.name, "sha256": file_sha256(weights)},
            "checked_keys": int(len(merged)), "cutoff": str(cutoff.date()),
            "days": days, "max_abs_difference": float(difference.max()),
            "bitwise_equal": bool(difference.max() == 0.0),
            "mean_abs_difference": float(difference.mean()),
            "stored_score_sha256": hashlib.sha256(merged["score"].to_numpy().tobytes()).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    config = load_config(Path(args.recipe))
    record = roundtrip(config, run_dir=Path(args.run), point=args.point, days=args.days)
    if args.output:
        write_json(Path(args.output), record)
    print(json.dumps({key: record[key] for key in ("point", "checked_keys", "cutoff",
                                                   "max_abs_difference", "bitwise_equal")},
                     ensure_ascii=False, indent=1))
    return 0 if record["bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
