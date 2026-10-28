#!/usr/bin/env python
"""Assemble ``dist/submission-tree-20261029/``: trunk (r441 x3 seeds) + LightGBM tree arm.

The package keeps the frozen flat layout that ``artifacts/submission-20261028`` uses, so
the platform entry is unchanged; the only additions are the tree arm and the fusion.

Builds the directory and MANIFEST only. No zip, no upload, and the frozen
``artifacts/submission-20261028/`` tree is never written to (it is read, as source).

    python scripts/build_submission_v2.py
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "submission-tree-20261029"
FROZEN = ROOT / "artifacts" / "submission-20261028" / "flat"
SCREEN = ROOT / "runs" / "r6r8-20261029" / "screen"
SEEDS = {
    "s11": ROOT / "runs" / "dose-replay-20261027" / "weights" / "441d-seed11" / "modified_tlob_weights-scaling-1.32.json",
    "s29": ROOT / "runs" / "dose-replay-20261027" / "weights" / "441d-seed29" / "modified_tlob_weights-scaling-1.32.json",
    "s47": ROOT / "runs" / "dose-replay-20261027" / "weights" / "441d-seed47" / "modified_tlob_weights-scaling-1.32.json",
}
SEED_NAMES = {"s11": "tlob_441d_s11", "s29": "tlob_441d_s29", "s47": "tlob_441d_s47"}

INFERENCE_PY = '''"""Submission entry: 441d trunk (equal-weight 3 seeds) x 46-feature LightGBM tree.

The trunk arm is the frozen TLOB blend (three seed checkpoints, ``score_mean``) with the
registered per-section causal EWMA; that part is ``modified_tlob_flat.predict_range``.
The candidate arm is the LightGBM tree over the packaged 46 book features, scored from the
same 1-minute bars in bounded daily chunks, then put through the identical operator.
Final factor = 0.5 * trunk_smoothed + 0.5 * tree_smoothed (the frozen gate weight).
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path

import book_features
import tree_fusion

ROOT = Path(__file__).resolve().parent
WEIGHT = 0.5
ALPHA = 0.9111677303420447
# prev_*/gap on the first scored day need the previous trading day's summary: read this much
# calendar history before start_date to seed it, and emit no scores for the warm-up block.
WARMUP_DAYS = 15
CHUNK_DAYS = 5
KEYS = ["date", "instrument"]


def load_tree_arm(model_txt=None, num_threads: int = 4):
    return tree_fusion.load_booster(model_txt or ROOT / "model.txt", num_threads=num_threads)


def tree_scores(bars: pd.DataFrame, mapping: pd.DataFrame, prev: pd.DataFrame | None,
                booster, num_threads: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bars -> 46 book features (8 sample rows per day) -> tree score.

    One row per (sample timestamp, instrument); raw bars are dropped by the caller.
    """
    idm = mapping.set_index("instrument_id")["instrument"].to_dict()
    frame, last = book_features.build_table(bars, idm, prev, book_features.SAMPLE_MIN)
    score = tree_fusion.candidate_scores(frame, booster, num_threads=num_threads)
    return frame[["date", "instrument"]].assign(score=score), last


def fuse_arms(keys: pd.DataFrame, trunk_smoothed: np.ndarray, tree_raw: np.ndarray,
              weight: float = WEIGHT, alpha: float = ALPHA) -> np.ndarray:
    """Blend the request-aligned trunk score with the tree arm under one EWMA pass."""
    tree = tree_fusion.smoothed_arm_score([np.asarray(tree_raw, dtype=float)], keys, alpha)
    return tree_fusion.fuse(np.asarray(trunk_smoothed, dtype=float), tree, weight)


def _run_query(table, columns, start, end) -> pd.DataFrame:
    """Single data-access hook: one bounded date slice of a platform table."""
    import dai
    from modified_tlob_flat import table_name

    frame = dai.query(f"SELECT {', '.join(columns)} FROM {table_name(table)}",
                      filters={"date": [str(start), str(end)]}).df()
    return frame[list(columns)]


def _mapping(datasources) -> pd.DataFrame:
    columns = ["instrument", "instrument_id"]
    return _run_query(datasources.get("instrument_map", "cpt_jyc_2026_stock_bar1m_instrument_mapping"),
                      columns, "1990-01-01", "2100-01-01")[columns]


def _bar_chunks(datasources, start: pd.Timestamp, end: pd.Timestamp):
    """Yield 1-minute bars in slices of CHUNK_DAYS; each slice is consumed and dropped."""
    cursor = pd.Timestamp(start).normalize()
    stop_date = pd.Timestamp(end).normalize()
    while cursor <= stop_date:
        upper = min(cursor + pd.Timedelta(days=CHUNK_DAYS), stop_date + pd.Timedelta(days=1))
        bars = _run_query(datasources["bar1m"], book_features.BAR_COLS, cursor, upper)
        if len(bars):
            bars = bars.copy()
            bars["date"] = pd.to_datetime(bars["date"])
            bars = bars.loc[(bars["date"] >= cursor) & (bars["date"] < upper)]
            if len(bars):
                yield bars
        cursor = upper


def main(datasources, start_date, end_date):
    """Platform entry: trunk scores plus a chunked tree arm, fused once over the range."""
    from modified_tlob_flat import main as trunk_main  # dai-dependent, unchanged

    trunk = trunk_main(datasources, start_date, end_date)
    keys = trunk[KEYS].reset_index(drop=True)
    mapping = _mapping(datasources)
    booster = load_tree_arm()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    prev, pieces = None, []
    for bars in _bar_chunks(datasources, start - pd.Timedelta(days=WARMUP_DAYS), end):
        scores, prev = tree_scores(bars, mapping, prev, booster)
        del bars
        pieces.append(scores.loc[scores["date"] >= start])
    tree = (pd.concat(pieces, ignore_index=True) if pieces
            else pd.DataFrame({"date": [], "instrument": [], "score": []}))
    merged = keys.merge(tree, on=KEYS, how="left", validate="one_to_one")
    if len(merged) != len(keys):
        raise ValueError(f"tree arm covered {len(merged)} rows for {len(keys)} keys")
    if merged["score"].isna().any():
        missing = int(merged["score"].isna().sum())
        raise ValueError(f"tree arm left {missing} request keys unscored")
    raw = merged["score"].to_numpy(dtype=float)
    fused = fuse_arms(keys, trunk["factor"].to_numpy(dtype=float), raw)
    if not np.isfinite(fused).all():
        raise ValueError("fused score has non-finite entries")
    return keys.assign(factor=fused)
'''

NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# 441d trunk x LightGBM tree fusion\n",
                    "Trunk: three frozen TLOB seeds (equal-weight score_mean) + registered causal EWMA.\n",
                    "Tree: 46 explicit book features -> LightGBM 4.4.0 model -> same EWMA.\n",
                    "Final factor = 0.5 * trunk + 0.5 * tree.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["from inference import main as _predict\n", "\n", "\n",
                    "def main(datasources, start_date, end_date):\n",
                    "    return _predict(datasources, start_date, end_date)\n"]},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}

PROVENANCE = {
    "schema_version": 1,
    "package": "submission-tree-20261029",
    "trunk": {
        "model": "TLOB 441d, equal-weight mean of seeds 11/29/47 (score_mean)",
        "weights": {name: str(path.relative_to(ROOT)) for name, path in SEEDS.items()},
        "training_window_trading_days": 441,
        "smoothing": {"id": "causal_ewma_turnover_matched", "alpha": 0.9111677303420447,
                      "state_reset": "trading_day", "operator_source": "scripts/verify_turnover_change.py"},
    },
    "candidate": {
        "model": "LightGBM 4.4.0 regressor, 600 rounds, num_leaves 63, 46 features",
        "model_file": "model.txt",
        "model_sha256_prefix": "98f58e81",
        "feature_module": "book_features.py",
        "features": "runs/r6r8-20261029/features (46 columns, unchanged column order)",
        "score_calibration": "per-timestamp cross-sectional z of vwap_return, clip +-5",
    },
    "fusion": {
        "weight_on_candidate": 0.5,
        "formula": "fused = 0.5 * smoothed(trunk) + 0.5 * smoothed(candidate)",
        "operator": "tree_fusion.fused_scores / smoothed_arm_score",
        "smoothing_alpha": 0.9111677303420447,
    },
    "gate_evidence": {
        "plan": "runs/fusion-gate/plans/r6r8-screen-20261028.json",
        "ledger_pooled_delta_rank_ic": 0.01157987683312182,
        "ledger_ci": [0.010149262547097039, 0.012890599389107133],
        "positive_windows": 8,
        "selfcheck": "runs/tree-package-20261029/selfcheck.json",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_requirements() -> str:
    text = (FROZEN / "requirements.txt").read_text(encoding="utf-8")
    if "lightgbm" in text:
        return text
    lines = text.splitlines()
    body = [line for line in lines if line.strip() and not line.startswith("#")]
    body.append("lightgbm==4.4.0")
    body.sort()
    header = [line for line in lines if line.startswith("#")]
    return "\n".join(header + body) + "\n"


def build_sharded_manifest() -> Path:
    """Sharded-blend manifest: the three seed checkpoints load one at a time at runtime."""
    members, weights = [], {}
    for key, path in SEEDS.items():
        name = SEED_NAMES[key]
        filename = f"modified_tlob_weights_seed{key[1:]}.json"
        members.append({"name": name, "file": filename, "weight": 1.0 / 3.0,
                        "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        weights[name] = 1.0 / 3.0
    payload = {"schema_version": 1, "kind": "modified_tlob_sharded_blend",
               "members": members,
               "blend": {"method": "score_mean", "weights": weights, "kind": "mlplob",
                         "member_kinds": {SEED_NAMES[k]: "mlplob" for k in SEEDS},
                         "member_reduction":
                         "sum the three member score arrays in member order, then one division "
                         "by the member count (np.mean over the stacked scores)",
                         "post_processing": {
                             "id": "causal_ewma_turnover_matched",
                             "alpha": 0.9111677303420447,
                             "applied_at": "platform inference, per intraday section",
                             "operator_source": "scripts/verify_turnover_change.py::causal_ewma"}}}
    out = DIST / "modified_tlob_weights.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    copied = {}

    for name in ("modified_tlob_flat.py", "jinyiquant_flat.py", "submission_config.json"):
        shutil.copyfile(FROZEN / name, DIST / name)
        copied[name] = "frozen flat layout (artifacts/submission-20261028)"

    for module in ("book_features.py", "tree_fusion.py"):
        shutil.copyfile(ROOT / "src" / "jinyiquant" / module, DIST / module)
        copied[module] = "src/jinyiquant/" + module

    for key, path in SEEDS.items():
        target = DIST / f"modified_tlob_weights_seed{key[1:]}.json"
        shutil.copyfile(path, target)
        copied[target.name] = str(path.relative_to(ROOT))

    shutil.copyfile(SCREEN / "model.txt", DIST / "model.txt")
    copied["model.txt"] = "runs/r6r8-20261029/screen/model.txt"

    merged = build_sharded_manifest()
    copied[merged.name] = "sharded-blend manifest over the three seed checkpoints"

    (DIST / "inference.py").write_text(INFERENCE_PY, encoding="utf-8")
    (DIST / "inference.ipynb").write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    (DIST / "provenance.json").write_text(json.dumps(PROVENANCE, indent=2), encoding="utf-8")

    weights_sha = sha256_file(merged)
    config = json.loads((DIST / "submission_config.json").read_text(encoding="utf-8"))
    config["weights_sha256"] = weights_sha
    config["smoothing"]["alpha"] = 0.9111677303420447
    (DIST / "submission_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    requirements = build_requirements()
    (DIST / "requirements.txt").write_text(requirements, encoding="utf-8")

    rows = []
    for path in sorted(DIST.iterdir()):
        rows.append({"path": path.name, "role": copied.get(path.name, ""),
                     "sha256": sha256_file(path), "size": path.stat().st_size})
    with (DIST / "MANIFEST.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(row["size"] for row in rows)
    print(f"built {DIST} ({len(rows)} files, {total / 1e6:.1f} MB)")
    for row in rows:
        print(f"  {row['size']:>10,}  {row['path']}")
    print(f"weights sha256 {weights_sha}")


if __name__ == "__main__":
    sys.exit(main())
