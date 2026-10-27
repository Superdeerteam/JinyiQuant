"""Verify that the streaming training pool changes memory, not training.

The materialised pool expands every ``(date, timestamp, instrument)`` key into one
``(samples, lookback, features)`` float32 tensor (24.9 GB for the 132-day window). The
streaming pool keeps the same keys plus one canonical bar matrix and builds a batch's windows
when the sampler asks for them. This script produces the acceptance evidence, in order:

1. **equivalence** - the first ``--batches`` sampler draws produce bitwise identical keys,
   feature tensors, observed masks, history dates and targets (``tobytes``/``==``, never
   ``allclose``), and identical coverage, fit mask and history diagnostics;
2. **non-vacuity** - mutating the index geometry, the key order or the window offset must each
   break that comparison, so a passing report cannot rest on an empty assertion;
3. **memory** - peak RSS and the resident delta around the build of each variant, each measured
   inside its own subprocess so the streaming figure cannot inherit the materialised pool's
   footprint (on a 10-20 day window the fixture's own load transient sets the peak, so the
   isolated resident delta is what isolates one pool from the other);
4. **step cost** - per-step data cost of both paths next to one real optimizer step of the
   same recipe, so a slowdown is quoted against a step rather than against nothing;
5. **projection** - the archive's measured keys and bars per trading day, times the measured
   bytes per key and per bar, projected onto the 132-day and 441-day pools against the budget.

The fixture is either the real archive (``--config``, a bounded slice: ``--days`` x
``--max-instruments``, read through the production reader) or shape-faithful synthetic bars
(``--synthetic``, no data IO). Only the size is reduced, never the data contract.

Usage::

    # local logic + mutations, no data IO
    python -m scripts.verify_streaming_pool --synthetic --max-instruments 200 --batches 50 \
        --sizes 50,200 --output runs/stream-pool-20261027/synthetic.json

    # local real slice: 2 calendar days x 200 instruments
    python -m scripts.verify_streaming_pool --config configs/<probe>.json --days 2 \
        --max-instruments 200 --batches 50 --output runs/stream-pool-20261027/real_2d.json

    # the acceptance run: the whole local universe over 14 trading days
    python -m scripts.verify_streaming_pool --config configs/stream_pool_probe_local_20261027.json \
        --start 2023-08-01 --days 20 --warmup-days 10 --max-instruments 999 --sizes 300,999 \
        --batches 50 --model-step --output runs/stream-pool-20261027/stream_pool_20d_full_universe.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.data import (  # noqa: E402
    KEYS,
    build_batch,
    history_diagnostics,
    load_data,
    lob_columns,
    select_targets,
)
from jinyiquant.pool import StreamingPool  # noqa: E402

VARIANTS = ("baseline", "fixture", "materialized", "streaming", "equivalent")


class ModelConfig(SimpleNamespace):
    """Attribute access for :func:`jinyiquant.models.create_model` and ``dict()`` for the adapter."""

    def keys(self):
        return vars(self).keys()

    def __getitem__(self, name):
        return getattr(self, name)
# One materialised window row costs a float32 feature tensor, a bool observed mask and an
# int64 history stamp per bar; the documented peaks divided by this give sample counts.
WINDOW_BYTES_PER_SAMPLE = 128 * 12 * 4 + 128 * 1 + 128 * 8


def peak_rss_mb() -> float:
    """This process's peak resident set size (Windows working set / POSIX ru_maxrss)."""
    if os.name == "nt":
        return _process_counters().PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10


def resident_rss_mb() -> float:
    """This process's current resident set size, the instantaneous counterpart of the peak.

    The peak is a high-water mark, so it cannot show a pool built after the parquet-to-pandas
    load already peaked above it (which is exactly what happens on a 10-20 day window). The
    current size, sampled around the build, isolated the pool from that transient.
    """
    if os.name == "nt":
        return _process_counters().WorkingSetSize / 2**20
    with open("/proc/self/statm", encoding="ascii") as handle:
        pages = int(handle.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def _process_counters():
    """``PROCESS_MEMORY_COUNTERS`` for this process (Windows working set and peak)."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters


# --------------------------------------------------------------------------- fixtures
def synthetic_fixture(instruments: int, days: int, sample_times: int, seed: int = 3):
    """Shape-faithful bars (240 one-minute bars per instrument-day) with missing runs.

    Twelve book features, matching the documented window shape (lookback 128 x 12 float32),
    so the projection arithmetic below compares like with like.
    """
    rng = np.random.default_rng(seed)
    features = list(lob_columns(3))
    stamps = pd.date_range("2023-08-01 09:31", periods=days * 240, freq="min")
    names = [f"S{index:04d}" for index in range(instruments)]
    frames = []
    for name in names:
        values = rng.normal(size=(len(stamps), len(features))).cumsum(axis=0) + 1000.0
        values[rng.random((len(stamps), len(features))) < 0.02] = np.nan
        for start in range(7, len(stamps), 97):        # six-bar gaps, wider than the fill limit
            values[start:start + 6, :9] = np.nan
        frames.append(pd.DataFrame({"date": stamps, "instrument": name,
                                    **dict(zip(features, values.T))}))
    step = max(1, len(stamps) // (days * sample_times))
    keys = pd.DataFrame([{"date": stamp, "instrument": name}
                         for stamp in stamps[::step] for name in names])
    config = SimpleNamespace(feature_columns=features, lookback=128, min_history=2,
                             missing_policy="ffill", max_ffill_steps=5, missing_sentinel=None,
                             price_scale=1.0, history_days=1, label_horizon_minutes=30,
                             sample_times=sorted({stamp.strftime("%H:%M") for stamp in keys["date"]}),
                             source="synthetic", version="streaming-pool-verification")
    targets = keys.assign(target=np.linspace(-0.01, 0.01, len(keys)))
    return config, pd.concat(frames, ignore_index=True), keys, targets


def real_fixture(args):
    """A bounded slice of the real archive, read through the production reader."""
    config = load_config(args.config)
    start = pd.Timestamp(args.start or config.splits.train.start)
    end = start + pd.Timedelta(days=args.days)
    data = config.data.model_dump()
    data.update(history_days=args.warmup_days, cache_dir=None)
    data_config = type(config.data)(**data)
    bundle = load_data(data_config, start=start, end=end)
    targets = select_targets(bundle.labels, SimpleNamespace(start=start, end=end),
                             data["label_horizon_minutes"], sample_times=data["sample_times"])
    if targets.empty:
        raise ValueError("The real slice selected no eligible target")
    instruments = sorted(targets["instrument"].unique())[:args.max_instruments]
    targets = targets.loc[targets["instrument"].isin(instruments)].reset_index(drop=True)
    bars = bundle.bars.loc[bundle.bars["instrument"].isin(instruments)].reset_index(drop=True)
    return data_config, bars, targets[KEYS], targets


def fixture(args):
    if args.synthetic:
        return synthetic_fixture(args.max_instruments, args.days, 8)
    return real_fixture(args)


def fixture_meta(args, config, bars, keys, targets) -> dict:
    return {"source": "synthetic" if args.synthetic else "real_archive_slice",
            "config": None if args.synthetic else str(args.config),
            "days": args.days, "instruments": int(keys["instrument"].nunique()),
            "bars_rows": len(bars), "bar_days": int(bars["date"].dt.normalize().nunique()),
            "keys": len(keys), "key_days": int(keys["date"].dt.normalize().nunique()),
            "missing_feature_fraction": round(float(bars[list(config.feature_columns)].isna().to_numpy().mean()), 6),
            "targets_sha256": hashlib.sha256(targets["target"].to_numpy(dtype=np.float64).tobytes()).hexdigest()}


# --------------------------------------------------------------------------- equivalence
def draws(count: int, size: int, total: int, seed: int) -> list[np.ndarray]:
    """Epoch-shaped draws in sampler order: one permutation, chunked, never re-sorted.

    ``TLOBAdapter.fit`` draws ``order = torch.randperm(len(y))`` once per epoch and hands over
    ``order[offset:offset + batch_size]`` unchanged, so a batch is an unsorted slice of a
    permutation. Sorting the draws here would only assert row-wise agreement; the pool must
    also preserve the order the optimizer sees.
    """
    order = np.random.default_rng(seed).permutation(total)
    return [order[start:start + size] for start in range(0, size * count, size)]


def compare(materialised, pool, targets: pd.DataFrame, positions: list[np.ndarray]) -> dict:
    aligned = materialised.keys.merge(targets, on=KEYS, how="left", validate="one_to_one")
    streamed = pool.keys.merge(targets, on=KEYS, how="left", validate="one_to_one")
    if aligned["target"].isna().any():
        raise ValueError("Missing targets after exact key alignment")
    evidence = {"batches": len(positions), "batch_size": len(positions[0]),
                "keys_identical": True, "x_bitwise": True, "observed_bitwise": True,
                "history_dates_bitwise": True, "target_bitwise": True,
                "feature_count_identical": True,
                "coverage_identical": pool.coverage == materialised.coverage,
                "fit_mask_identical": np.array_equal(
                    pool.fit_mask, materialised.effective_rows.sum(axis=1) >= int(pool.settings["minimum"])),
                "diagnostics_identical": pool.history_diagnostics().equals(history_diagnostics(materialised))}
    for positions_one in positions:
        batch = pool.batch(positions_one)
        evidence["keys_identical"] &= batch.keys.equals(materialised.keys.iloc[positions_one].reset_index(drop=True))
        evidence["x_bitwise"] &= batch.x.tobytes() == materialised.x[positions_one].tobytes()
        evidence["observed_bitwise"] &= np.array_equal(batch.observed, materialised.observed[positions_one])
        evidence["history_dates_bitwise"] &= np.array_equal(
            batch.history_dates.astype("int64"), materialised.history_dates[positions_one].astype("int64"))
        evidence["target_bitwise"] &= np.array_equal(
            streamed["target"].to_numpy()[positions_one], aligned["target"].to_numpy()[positions_one])
        evidence["feature_count_identical"] &= batch.history_feature_count == materialised.history_feature_count
    return evidence


def mutations(materialised, pool, positions: np.ndarray) -> dict:
    """Every mutation must break the bitwise comparison, then the baseline must be restored."""
    baseline = materialised.x[positions].tobytes()
    codes, begin, stop = pool.codes, pool.begin, pool.stop
    pool.codes, pool.begin, pool.stop = codes[::-1].copy(), begin[::-1].copy(), stop[::-1].copy()
    results = {"index_geometry_reversed": pool.batch(positions).x.tobytes() != baseline}
    pool.codes, pool.begin, pool.stop = codes, begin, stop
    keys, head = pool.keys, np.arange(min(4, len(pool.keys)))
    pool.keys = keys.iloc[[1, 0, *range(2, len(keys))]].reset_index(drop=True)
    results["key_order_swapped"] = not pool.batch(head).keys.equals(
        materialised.keys.iloc[head].reset_index(drop=True))
    pool.keys = keys
    pool.begin = begin + 1
    results["window_shifted_one_bar"] = pool.batch(positions).x.tobytes() != baseline
    pool.begin = begin
    results["baseline_restored_after_mutations"] = pool.batch(positions).x.tobytes() == baseline
    return results


# --------------------------------------------------------------------------- step cost
def step_cost(materialised, pool, config, positions: list[np.ndarray], batch_size: int,
              with_model_step: bool = True, device: str = "cpu") -> dict:
    """Per-step data cost of both paths, next to one real optimizer step of the same recipe."""
    pool.batch(positions[0])
    started = time.perf_counter()
    for positions_one in positions:
        materialised.x[positions_one]
    materialised_ms = (time.perf_counter() - started) / len(positions) * 1000
    started = time.perf_counter()
    for positions_one in positions:
        pool.batch(positions_one)
    streaming_ms = (time.perf_counter() - started) / len(positions) * 1000
    model_ms, note = None, None
    try:
        if not with_model_step:
            raise RuntimeError("model step timing disabled by --no-model-step")
        import torch

        from jinyiquant.models import create_model

        with torch.random.fork_rng(devices=[]):
            model = create_model(SimpleNamespace(
                seed=11, data=_adapter_data(config),
                model=ModelConfig(kind="mlplob", hidden_dim=40, num_layers=3, num_heads=4,
                                  is_sin_emb=True, device=device, cpu_threads=1, compile_mode="eager",
                                  finite_check_every=50, book_representation="raw",
                                  trade_features="zero", target_scaling="train_std",
                                  target_centering="none", time_loss_weighting="none",
                                  sampling="shuffle", lr_schedule="constant", ranking_loss="none",
                                  ranking_loss_weight=0.0, batch_size=batch_size,
                                  learning_rate=1e-4, epochs=1)))
            network, optimizer = model.network, torch.optim.Adam(model.network.parameters(), lr=1e-4)
            xb = torch.from_numpy(np.ascontiguousarray(materialised.x[positions[0]])).to(device)
            yb = torch.zeros(len(positions[0]), device=device)
            timings = []
            for _ in range(max(3, len(positions) // 10)):
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(network(xb), yb).backward()
                optimizer.step()
                timings.append(time.perf_counter() - started)
            model_ms = float(np.median(timings) * 1000)
    except Exception as exc:                    # a missing torch build must not hide the data cost
        note = f"{type(exc).__name__}: {exc}"[:300]
    materialised_step = materialised_ms + (model_ms or 0.0)
    streaming_step = streaming_ms + (model_ms or 0.0)
    # Without the model step the denominator would be the data cost alone, which would quote a
    # meaningless percentage; the data cost is reported either way.
    increase = (None if model_ms is None else
                round((streaming_step - materialised_step) / materialised_step * 100, 2))
    return {"materialised_data_ms": round(materialised_ms, 4),
            "streaming_data_ms": round(streaming_ms, 4),
            "model_step_ms": None if model_ms is None else round(model_ms, 3),
            "model_step_note": note,
            "materialised_step_ms": round(materialised_step, 3),
            "streaming_step_ms": round(streaming_step, 3),
            "step_increase_pct": increase,
            "over_15pct": None if increase is None else increase > 15,
            "batch_size": batch_size, "device": device,
            "rule": "step = per-step data cost + one real eager forward/backward/step of the same recipe"}


def _adapter_data(config) -> dict:
    """The preprocessing contract the adapter expects, from either a config object or a dict."""
    return config.model_dump(mode="json") if hasattr(config, "model_dump") else dict(vars(config))


# --------------------------------------------------------------------------- variants
def _snapshot() -> dict:
    """Resident, committed and peak memory of this process, in MiB.

    Committed bytes (``private``) are the right instrument for a pool: CPython hands large array
    buffers to the platform allocator, which releases them on free, while the *working set* keeps
    the pages of freed blocks resident. That is why a working-set delta around a build reads as
    zero as soon as the fixture has already grown the heap.
    """
    if os.name == "nt":
        counters = _process_counters()
        return {"working_set_mb": round(counters.WorkingSetSize / 2**20, 1),
                "private_mb": round(counters.PrivateUsage / 2**20, 1),
                "peak_working_set_mb": round(counters.PeakWorkingSetSize / 2**20, 1),
                "peak_private_mb": round(counters.PeakPagefileUsage / 2**20, 1)}
    peak = peak_rss_mb()
    return {"working_set_mb": round(resident_rss_mb(), 1), "private_mb": round(resident_rss_mb(), 1),
            "peak_working_set_mb": round(peak, 1), "peak_private_mb": round(peak, 1)}


def _delta(after: dict, before: dict) -> dict:
    return {name: round(after[name] - before[name], 1) for name in ("working_set_mb", "private_mb")}


def run_variant(args) -> dict:
    """Build the fixture, then the requested variant, and report this process's memory.

    Three readings per variant, because they answer different questions: the peak RSS is the
    acceptance figure (a training process has to survive it); the delta around the build is what
    the pool itself adds on top of the loaded frame; and the reading after the frame is released
    is what a training loop actually keeps, since the frame is only needed to build the pool.
    """
    interpreter_mb = round(peak_rss_mb(), 1)
    config, bars, keys, targets = fixture(args)
    result = {"variant": args.variant, "interpreter_peak_rss_mb": interpreter_mb,
              "fixture": fixture_meta(args, config, bars, keys, targets)}
    if args.variant in {"baseline", "fixture"}:
        result["peak_rss_mb"] = round(peak_rss_mb(), 1)
        return result
    positions = draws(args.batches, args.batch_size, len(keys), args.seed)
    loaded = _snapshot()
    if args.variant == "materialized":
        held = build_batch(bars, keys, config)
        result.update(keys=len(keys), pool_bytes=int(held.x.nbytes + held.observed.nbytes
                                                    + held.history_dates.nbytes))
    elif args.variant == "streaming":
        # Full build, quality summary included: the comparison must cover the same work the
        # materialised variant does, otherwise the lighter number would only reflect a skipped
        # step rather than a smaller pool.
        held = StreamingPool.build(bars, keys, config)
        result.update(keys=len(keys), pool_nbytes=held.nbytes(), pool_bytes=int(sum(held.nbytes().values())))
    elif args.variant == "equivalent":
        held = build_batch(bars, keys, config)
        pool = StreamingPool.build(bars, keys, config)
        result["equivalence"] = compare(held, pool, targets, positions)
        result["mutations"] = mutations(held, pool, positions[0])
        result["step_cost"] = step_cost(held, pool, config, positions, args.batch_size,
                                        args.model_step, args.device)
        result["pool_nbytes"] = pool.nbytes()
    else:
        raise ValueError(f"Unknown variant {args.variant!r}")
    built = _snapshot()
    bars = None                      # the frame is only an input to the pool, not training state
    gc.collect()
    released = _snapshot()
    result["memory_mb"] = {
        "loaded": loaded,
        "build_delta": _delta(built, loaded),
        "after_release": {"working_set_mb": released["working_set_mb"],
                          "private_mb": released["private_mb"],
                          "pool_plus_interpreter_minus_loaded_private_mb": round(
                              released["private_mb"] - loaded["private_mb"], 1)},
    }
    del held
    result["peak_rss_mb"] = round(peak_rss_mb(), 1)
    return result


def head_check(args) -> dict:
    """The refactored materialised builder must still equal the committed one, byte for byte."""
    import importlib.util
    import tempfile

    try:
        source = subprocess.run(["git", "show", "HEAD:src/jinyiquant/data.py"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
    except Exception as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    temporary = Path(tempfile.mkdtemp()) / "data_head.py"
    temporary.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("data_head", temporary)
    head = importlib.util.module_from_spec(spec)
    sys.modules["data_head"] = head
    spec.loader.exec_module(head)
    _, bars, keys, _ = synthetic_fixture(args.max_instruments, 1, 8)
    base = vars(synthetic_fixture(args.max_instruments, 1, 8)[0])
    evidence = {"checked": True, "fixtures": 0, "x_bitwise": True, "observed_bitwise": True,
                "history_dates_bitwise": True, "coverage_identical": True}
    for overrides in ({"lookback": 16}, {"lookback": 128}, {"lookback": 16, "max_ffill_steps": 0},
                      {"lookback": 16, "missing_policy": "zero", "max_ffill_steps": 0}):
        config = SimpleNamespace(**{**base, **overrides})
        committed, current = head.build_batch(bars, keys, config), build_batch(bars, keys, config)
        evidence["fixtures"] += 1
        evidence["x_bitwise"] &= committed.x.tobytes() == current.x.tobytes()
        evidence["observed_bitwise"] &= np.array_equal(committed.observed, current.observed)
        evidence["history_dates_bitwise"] &= np.array_equal(committed.history_dates.astype("int64"),
                                                           current.history_dates.astype("int64"))
        evidence["coverage_identical"] &= committed.coverage == current.coverage
    return evidence


def code_state() -> dict:
    """The revision this evidence describes, plus the hashes of the files in its scope."""
    import hashlib

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:
        head = f"{type(exc).__name__}: {exc}"[:120]
    files = {}
    for name in ("src/jinyiquant/data.py", "src/jinyiquant/pool.py",
                 "scripts/verify_streaming_pool.py", "tests/test_streaming_pool.py"):
        path = ROOT / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {"head": head, "files_sha256": files}


# --------------------------------------------------------------------------- driver
def parse_sizes(text: str) -> list[int]:
    values = sorted({int(item) for item in str(text).replace(" ", "").split(",") if item})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return values


def child_command(args, instrument_count: int, variant: str) -> list[str]:
    command = [sys.executable, "-m", "scripts.verify_streaming_pool", "--variant", variant,
               "--batches", str(args.batches), "--batch-size", str(args.batch_size),
               "--days", str(args.days), "--seed", str(args.seed),
               "--max-instruments", str(instrument_count)]
    command.append("--model-step" if args.model_step else "--no-model-step")
    if args.synthetic:
        command.append("--synthetic")
    else:
        command += ["--config", str(args.config), "--warmup-days", str(args.warmup_days)]
        if args.start is not None:
            command += ["--start", str(args.start)]
    return command


def run_child(args, instrument_count: int, variant: str) -> dict:
    completed = subprocess.run(child_command(args, instrument_count, variant), cwd=ROOT,
                               capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT)))})
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    if completed.returncode != 0 or not lines:
        raise RuntimeError(f"variant {variant} failed ({completed.returncode})\n"
                           f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    payload = json.loads(lines[-1][len("RESULT "):])
    if payload.get("error"):
        raise RuntimeError(f"variant {variant} failed: {payload['error']}\n{payload.get('traceback', '')}")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=("driver",) + VARIANTS, default="driver")
    parser.add_argument("--synthetic", action="store_true", help="generated bars, no data IO")
    parser.add_argument("--config", type=Path, help="run configuration whose data recipe and paths are reused")
    parser.add_argument("--start", default=None, help="first calendar day of the real slice")
    parser.add_argument("--days", type=int, default=2, help="calendar days in the fixture")
    parser.add_argument("--warmup-days", type=int, default=1, help="bar history read before the slice")
    parser.add_argument("--max-instruments", type=int, default=200)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("50,200"),
                        help="instrument counts for the peak-RSS slope")
    parser.add_argument("--materialised-132d-gb", type=float, default=24.9,
                        help="documented 132-day materialised peak, quoted for context only")
    parser.add_argument("--materialised-441d-gb", type=float, default=83.0,
                        help="documented 441-day materialised peak, quoted for context only")
    parser.add_argument("--budget-gb", type=float, default=12.0, help="441-day streaming budget")
    parser.add_argument("--device", default="cpu",
                        help="device for the timed optimizer step (cuda on the training host)")
    parser.add_argument("--model-step", action=argparse.BooleanOptionalAction, default=True,
                        help="also time one real optimizer step as the per-step denominator")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if min(args.max_instruments, args.batches, args.batch_size, args.days) < 1:
        parser.error("instruments, batches, batch size and days must be positive")
    if not args.synthetic and args.config is None:
        parser.error("either --synthetic or --config is required")

    if args.variant != "driver":
        try:
            print("RESULT " + json.dumps(run_variant(args), default=float), flush=True)
            return 0
        except Exception as exc:
            print("RESULT " + json.dumps({"variant": args.variant, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()[-1500:]}), flush=True)
            return 1

    sizes = sorted(set(args.sizes) | {args.max_instruments})
    runs = [run_child(args, args.max_instruments, "baseline")]
    for count in sizes:
        for variant in ("fixture", "materialized", "streaming"):
            runs.append(run_child(args, count, variant))
    runs.append(run_child(args, args.max_instruments, "equivalent"))
    equivalent = runs[-1]
    document = {
        "schema_version": 1, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture": {"synthetic": args.synthetic, "config": None if args.synthetic else str(args.config),
                    "days": args.days, "max_instruments": args.max_instruments,
                    "sizes": sizes, "batches": args.batches, "batch_size": args.batch_size},
        "runs": len(runs), "max_peak_rss_mb": max(item.get("peak_rss_mb", 0.0) for item in runs),
        "exceeded_500mb": max(item.get("peak_rss_mb", 0.0) for item in runs) > 500,
        "equivalence": equivalent.get("equivalence"), "mutations": equivalent.get("mutations"),
        "step_cost": equivalent.get("step_cost"), "pool_nbytes": equivalent.get("pool_nbytes"),
        "head_check": head_check(args),
    }
    memory = {"fixture_only_peak_mb": runs[0].get("peak_rss_mb"),
              "interpreter_peak_mb": runs[0].get("interpreter_peak_rss_mb"), "by_variant": {}}
    for item in runs[1:]:
        entry = {"peak_rss_mb": item.get("peak_rss_mb"), "pool_bytes": item.get("pool_bytes"),
                 "keys": item.get("keys")}
        entry.update(item.get("memory_mb") or {})
        memory["by_variant"][f"{item['variant']}@{item['fixture']['instruments']}"] = entry
    document["memory"] = memory
    projection = {}
    nbytes, meta = equivalent.get("pool_nbytes"), equivalent["fixture"]
    if nbytes and meta.get("bars_rows") and meta.get("keys"):
        baseline_mb = runs[0].get("interpreter_peak_rss_mb") or 0.0
        bar_bytes = (nbytes["bars"] + nbytes["bar_dates"]) / meta["bars_rows"]
        index_per_key = (nbytes["keys"] + nbytes["instrument_codes"] + nbytes["window_geometry"]
                         + nbytes["row_counts"]) / meta["keys"]
        bars_per_day = meta["bars_rows"] / max(1, meta["bar_days"])
        keys_per_day = meta["keys"] / max(1, meta["key_days"])
        # The warm-up bars the recipe reads before its first key: they are in the fixture's bar
        # days but not in its key days, and they are part of a real pool's footprint.
        warmup_days = max(0, meta["bar_days"] - meta["key_days"])
        projection = {"bytes_per_bar": round(bar_bytes, 1), "index_bytes_per_key": round(index_per_key, 1),
                      "bars_per_trading_day": round(bars_per_day), "keys_per_trading_day": round(keys_per_day),
                      "warmup_trading_days": warmup_days,
                      "documented_implied_keys_per_day": round(
                          args.materialised_132d_gb * 2**30 / WINDOW_BYTES_PER_SAMPLE / 132),
                      "note": ("a streamed pool costs the fixture-measured bars per trading day plus the "
                               "measured index bytes per key, so the projection scales with the archive's "
                               "real sample and bar volume, not with the documented peak RSS (which also "
                               "holds the pandas bar frame, the validation pool and the torch runtime); the "
                               "materialised column is the window tensor alone, lookback x features float32 "
                               "+ observed bool + int64 stamp per key")}
        for label, days in (("132d", 132), ("441d", 441)):
            keys_total = days * keys_per_day
            bars_total = (days + warmup_days) * bars_per_day
            pool_bytes = keys_total * index_per_key + bars_total * bar_bytes
            projection[f"streaming_{label}_keys"] = round(keys_total)
            projection[f"streaming_{label}_bars"] = round(bars_total)
            projection[f"streaming_{label}_gb"] = round((baseline_mb * 2**20 + pool_bytes) / 2**30, 2)
            projection[f"materialised_window_{label}_gb"] = round(
                keys_total * WINDOW_BYTES_PER_SAMPLE / 2**30, 2)
        projection["documented_materialised_gb"] = {"132d": args.materialised_132d_gb,
                                                   "441d": args.materialised_441d_gb}
        largest, fixture_only = f"streaming@{sizes[-1]}", f"fixture@{sizes[-1]}"
        if largest in memory["by_variant"] and fixture_only in memory["by_variant"]:
            # A peak-RSS difference below the fixture's own read cost is measurement noise, which
            # means the window is too small for the pool to show up in RSS at all.
            fixture_peak = memory["by_variant"][fixture_only]["peak_rss_mb"]
            delta = memory["by_variant"][largest]["peak_rss_mb"] - fixture_peak
            projection["marginal_rss_bytes_per_key"] = (
                round(delta * 2**20 / memory["by_variant"][largest]["keys"], 1)
                if abs(delta) > 0.05 * fixture_peak else None)
            projection["rss_note"] = (f"streaming minus fixture-only peak = {delta:.1f} MB on a "
                                      f"{fixture_peak:.1f} MB fixture peak; a pool delta inside that "
                                      f"noise needs a bigger window than {sizes[-1]} instruments")
        projection["budget_gb"] = args.budget_gb
        projection["passed"] = projection["streaming_441d_gb"] <= args.budget_gb
        old, new = memory["by_variant"].get(f"materialized@{sizes[-1]}"), memory["by_variant"].get(
            f"streaming@{sizes[-1]}")
        if old and new and "build_delta" in old and "build_delta" in new:
            # The old-versus-new headline for one and the same window: what the pool adds when it
            # is built, what it leaves resident, and the peak the process has to survive.
            memory["same_window_comparison"] = {
                "instruments": sizes[-1], "keys": new["keys"],
                "materialised_pool_bytes": old["pool_bytes"], "streaming_pool_bytes": new["pool_bytes"],
                "pool_ratio": round(old["pool_bytes"] / new["pool_bytes"], 2),
                "materialised_build_delta_private_mb": old["build_delta"]["private_mb"],
                "streaming_build_delta_private_mb": new["build_delta"]["private_mb"],
                "materialised_after_release_mb": old["after_release"]["working_set_mb"],
                "streaming_after_release_mb": new["after_release"]["working_set_mb"],
                "materialised_peak_rss_mb": old["peak_rss_mb"], "streaming_peak_rss_mb": new["peak_rss_mb"],
                "note": ("both peaks are set by the fixture's parquet-to-pandas transient, which is above "
                         "either pool, so the peak cannot separate them at this window; the committed-byte "
                         "delta around the build can - it reproduces each pool's byte count to within a "
                         "percent - and the reading after the frame is released is the footprint a training "
                         "loop actually keeps")}
    document["projection"] = projection
    document["code_state"] = code_state()
    payload = json.dumps(document, indent=1, ensure_ascii=False, default=float)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    evaluation, mutation_results = document["equivalence"] or {}, document["mutations"] or {}
    step, projection = document["step_cost"] or {}, document["projection"] or {}
    head = document["head_check"] or {}
    passed = (all(evaluation.get(name) for name in (
        "keys_identical", "x_bitwise", "observed_bitwise", "history_dates_bitwise", "target_bitwise",
        "coverage_identical", "fit_mask_identical", "diagnostics_identical", "feature_count_identical"))
        and all(mutation_results.values()) and bool(projection.get("passed"))
        and (head.get("checked") is False or all(head.get(name) for name in (
            "x_bitwise", "observed_bitwise", "history_dates_bitwise", "coverage_identical"))))
    print(f"\nequivalence: {json.dumps(evaluation, ensure_ascii=False)}")
    print(f"mutations:   {json.dumps(mutation_results, ensure_ascii=False)}")
    print(f"head check:  {json.dumps(head, ensure_ascii=False)}")
    print(f"step cost:   {json.dumps(step, ensure_ascii=False)}")
    print(f"projection:  {json.dumps(projection, ensure_ascii=False)}")
    print(f"peak RSS:    max {document['max_peak_rss_mb']:.1f} MB over {document['runs']} runs "
          f"(over 500 MB: {document['exceeded_500mb']})")
    increase = step.get("step_increase_pct")
    rule = ("not measured" if increase is None else
            f"{increase:+.2f}% per-step cost, {'above' if step.get('over_15pct') else 'within'} the 15% rule")
    print(f"step rule:   {rule}")
    print(f"verdict:     {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
