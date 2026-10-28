"""Pipeline-level invariance audit for the market-context experiment (t3, auditor-owned).

``scripts/market_context.py`` needs the ``jinyiquant`` data layer, which is not installed in
the local audit venv (``.venv-tlob``).  The functions audited here are pure numerics, so they
are exercised with inert stubs standing in for the data layer: no bar, label or split on disk is
ever read.  Four properties are measured, each against the 1e-6 acceptance threshold:

1. ``trunk_forward(..., slice_rows=k)``      - device slicing must not change h or s_base.
2. ``_extract(..., part_rows=k)``            - part checkpointing must be concatenation only.
3. ``_context_table(..., block=k)``          - the block-of-timestamps table must be block-free.
4. ``_context_table`` with shocked future    - earlier timestamps must be bitwise unchanged.
5. ``split_of``                              - the sealed ``final`` split must be unreachable.

Run:  .venv-tlob/Scripts/python.exe scripts/audit_market_pipeline.py [--out PATH]
Exit: 0 if every measured deviation is <= 1e-6, else 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

THRESHOLD = 1e-6
DEFAULT_OUT = ROOT / "runs" / "mktctx-20261028" / "pipeline.json"
POSITIONS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")


class _AnyStub(types.ModuleType):
    """A module whose every attribute exists and is an inert callable (or empty tuple)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = () if name.isupper() else (lambda *a, **k: None)
        setattr(self, name, value)
        return value


def stub_jinyiquant():
    """Register an inert ``jinyiquant`` package so the pipeline module can be imported.

    Nothing from the data layer is ever called: the audited functions are pure numerics, and the
    audit must not touch bars, labels or splits.
    """
    package = _AnyStub("jinyiquant")
    for sub in ("config", "data", "io", "pool", "runner", "tracking", "checks", "comparison",
                "evaluation"):
        module = _AnyStub(f"jinyiquant.{sub}")
        setattr(package, sub, module)
        sys.modules[f"jinyiquant.{sub}"] = module
    sys.modules["jinyiquant"] = package


class StubAdapter:
    """Only the two attributes ``trunk_forward`` touches."""

    device = "cpu"
    target_scale = 1.0

    def __init__(self, network, x):
        self.network = network
        self.x = np.asarray(x, np.float32)

    def _input(self, batch):
        return self.x[np.asarray(batch, np.int64)]


class StubPool:
    def __init__(self, rows):
        self.keys = np.arange(rows)

    def batch(self, idx):
        return np.asarray(idx, np.int64)


def deviation(a, b):
    left = np.asarray(a, np.float64)
    right = np.asarray(b, np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch {left.shape} vs {right.shape}")
    bad = ~(np.isfinite(left) & np.isfinite(right))
    nan_mismatch = int((bad & ~(np.isnan(left) & np.isnan(right))).sum())
    diff = np.abs(np.where(bad, 0.0, left - right))
    finite = np.isfinite(left) & np.isfinite(right)
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff[finite].mean()) if finite.any() else 0.0,
        "nan_mismatch": nan_mismatch,
        "compared": int(diff.size),
        "nonfinite_mismatch": nan_mismatch > 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args(argv)

    import pandas as pd
    import torch

    torch.set_num_threads(1)
    stub_jinyiquant()
    import market_context as pipeline  # noqa: E402  (import after the stubs)

    table_fn = getattr(pipeline, "context_table", None) or getattr(pipeline, "_context_table")
    from modified_tlob.model import MLPLOB  # noqa: E402

    rng = np.random.default_rng(args.seed)
    features = json.loads((ROOT / "configs" / "scaling_441d_csz_seed11.json")
                          .read_text(encoding="utf-8"))["data"]["feature_columns"]
    torch.manual_seed(args.seed)
    network = MLPLOB(hidden_dim=40, num_layers=3, seq_size=128,
                     num_features=len(features)).eval()
    rows = int(args.rows)
    x = rng.normal(0.0, 1.0, size=(rows, 128, len(features))).astype(np.float32)
    adapter = StubAdapter(network, x)
    idx = np.arange(rows)
    worst = 0.0
    import hashlib

    module_path = Path(pipeline.__file__).resolve()
    result = {"schema_version": 1, "audit": "t3 pipeline invariance", "threshold": THRESHOLD,
              "module": str(module_path.relative_to(ROOT)),
              "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
              "module_mtime": module_path.stat().st_mtime_ns,
              "context_table_entry": table_fn.__name__,
              "stubs": "inert jinyiquant package (no data, label or split is read)"}

    # 1) device slicing inside one forward
    base_h, base_s = pipeline.trunk_forward(adapter, idx, slice_rows=rows)
    slices = {}
    for slice_rows in (32, 256, 8192):
        hidden, score = pipeline.trunk_forward(adapter, idx, slice_rows=slice_rows)
        d_h, d_s = deviation(base_h, hidden), deviation(base_s, score)
        slices[str(slice_rows)] = {"representation": d_h, "score": d_s}
        worst = max(worst, d_h["max_abs"], d_s["max_abs"])
    result["trunk_forward_slice_rows"] = slices

    # 2) part_rows checkpointing
    parts = {}
    with tempfile.TemporaryDirectory() as tmp:
        extracted = {}
        for part_rows in (rows, 400, 111):
            directory = Path(tmp) / f"part{part_rows}"
            directory.mkdir()
            extracted[part_rows] = pipeline._extract(adapter, StubPool(rows), key="audit",
                                                     parts=directory, part_rows=part_rows,
                                                     slice_rows=256)
        for part_rows, (hidden, score) in extracted.items():
            d_h, d_s = deviation(base_h, hidden), deviation(base_s, score)
            parts[str(part_rows)] = {"representation": d_h, "score": d_s}
            worst = max(worst, d_h["max_abs"], d_s["max_abs"])
    result["extract_part_rows"] = parts

    # 3) the per-timestamp context table: block size and future rows
    dates = ("2023-11-01", "2023-11-02", "2023-11-03")
    stamps = [f"{d} {t}" for d in dates for t in POSITIONS]
    per_stamp = 100
    pool_keys = pd.DataFrame({
        "instrument": [f"{i:04d}.SZ" for _ in stamps for i in range(per_stamp)],
        "date": pd.to_datetime([stamp for stamp in stamps for _ in range(per_stamp)]),
    })
    count = len(pool_keys)
    hidden = np.maximum(rng.normal(0.0, 0.8, size=(count, 80)).astype(np.float32), 0.0)
    valid = rng.random(count) > 0.1
    timestamps = np.sort(pd.unique(pool_keys["date"].to_numpy()))
    base_context, base_counts = table_fn(pool_keys, hidden, valid, timestamps,
                                                       block=64)
    blocks = {}
    for block in (1, 7, 1000):
        context, counts = table_fn(pool_keys, hidden, valid, timestamps,
                                                  block=block)
        d_context, d_counts = deviation(base_context, context), deviation(base_counts, counts)
        blocks[str(block)] = {"context": d_context, "counts": d_counts}
        worst = max(worst, d_context["max_abs"], d_counts["max_abs"])
    result["context_table_block"] = blocks

    cutoff = len(timestamps) // 2
    shocked = hidden.copy()
    future_rows = np.flatnonzero(np.isin(pool_keys["date"].to_numpy(), timestamps[cutoff:]))
    shocked[future_rows] = rng.normal(0.0, 5.0, size=(future_rows.size, hidden.shape[1]))
    future_context, _ = table_fn(pool_keys, shocked, valid, timestamps, block=64)
    future = deviation(base_context[:cutoff], future_context[:cutoff])
    result["context_table_future_shock"] = {"cutoff": int(cutoff),
                                            "shocked_rows": int(future_rows.size),
                                            "deviation": future}
    worst = max(worst, future["max_abs"])

    # 4) the sealed final window must stay unreachable
    from types import SimpleNamespace

    splits = SimpleNamespace(train="t", validation="v", final="SEALED")
    guard = {}
    for name in ("train", "validation", "final", "replay"):
        try:
            guard[name] = {"resolved": repr(pipeline.split_of(SimpleNamespace(splits=splits),
                                                              name))}
        except Exception as exc:  # noqa: BLE001  (the refusal is the evidence)
            guard[name] = {"raises": f"{type(exc).__name__}: {exc}"}
    result["split_guard"] = guard
    result["sealed_window_reachable"] = "resolved" in guard["final"]
    # replay is the cache role of the evaluation split: same window as validation,
    # never the sealed final window.
    result["replay_maps_to_validation"] = (guard["replay"].get("resolved")
                                          == guard["validation"].get("resolved"))

    result["max_abs_deviation"] = worst
    result["verdict"] = "pass" if (worst <= THRESHOLD
                                   and not result["sealed_window_reachable"]
                                   and result["replay_maps_to_validation"]) else "fail"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "context_table_block"},
                     indent=1)[:2000])
    print(f"[pipeline] verdict={result['verdict']} max_abs={worst:g} -> {args.out}")
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
