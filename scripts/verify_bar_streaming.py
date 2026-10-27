"""Acceptance evidence for bounded chunked bar loading (``load_data(stream_bars=True)``).

The 927-day window does not fit the host because of the *bar load*, not the window tensor:
132 days cost 5 GB resident and a 19.3 GB transient (2.6x) as one pandas frame, which
extrapolates to ~175 GB at 927 days. The streamed reader assembles the same canonical float32
arrays in bounded chunks, so its peak is ``target arrays + one chunk``. This script produces
the evidence, in order:

1. **equivalence** - for every measured window size, the streamed read and the default frame
   read must hash to the same canonical arrays (``values``/``dates``/``offsets``/``names``) and
   the same per-file provenance records. The arrays are compared as buffers, so this is
   ``tobytes`` equality without holding both sides at once;
2. **equivalence at the pool level** - on a small window, the windows ``build_batch`` assembles
   from the frame and the windows ``StreamingPool`` assembles from the streamed arrays are
   compared bitwise (feature tensor, observed mask, history dates, coverage), and with
   ``--device`` the same batch is round-tripped through the device;
3. **memory** - every window size is loaded inside its own subprocess, once per path, and the
   peak RSS is reported next to the canonical array bytes. The two paths' bytes-per-row are
   fitted from the measured sizes and extrapolated to ``--target-days``, which is the number
   the 172 GB host has to survive;
4. **conclusion** - one verdict per acceptance criterion, and a non-zero exit code when any of
   them fails, so a green run is reproducible evidence rather than a judgement call.

Usage::

    # local: 3 trading days, bitwise + memory, no device
    python -m scripts.verify_bar_streaming --config configs/stream_pool_probe_local_20261027.json \
        --days 3 --sizes 2,3 --max-instruments 200 --device cpu \
        --output runs/stream-pool-20261027/bar_stream_3d.json

    # container acceptance: the real window, GPU round-trip
    python -m scripts.verify_bar_streaming --config configs/mlplob_v3_132d_csz_19800.json \
        --days 132 --sizes 60,132 --max-instruments 999 --device cuda:0 \
        --output /mnt/work/bar_stream_132d.json
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
    DEFAULT_BAR_CHUNK_ROWS,
    BarArrays,
    bar_arrays,
    build_batch,
    load_data,
    select_targets,
)
from jinyiquant.pool import StreamingPool  # noqa: E402


# --------------------------------------------------------------------------- memory probes
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


def peak_rss_mb() -> float:
    """This process's peak resident set size (Windows working set / POSIX ru_maxrss)."""
    if os.name == "nt":
        return _process_counters().PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10


def resident_rss_mb() -> float:
    """This process's current resident set size, the instantaneous counterpart of the peak.

    The peak is a high-water mark, so a float64 load transient hides the resident cost that
    the projection needs. The resident delta around the load is nearly linear in rows, which
    is what turns two small windows into a marginal bytes-per-row figure.
    """
    if os.name == "nt":
        return _process_counters().WorkingSetSize / 2**20
    with open("/proc/self/statm", encoding="ascii") as handle:
        pages = int(handle.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def canonical_digest(arrays: BarArrays) -> dict:
    """Hash the canonical arrays as buffers: equality here is ``tobytes`` equality.

    Hashing avoids holding the other path's copy, which is what makes an equivalence check
    affordable on the full window instead of only on a slice.
    """
    values = np.ascontiguousarray(arrays.values)
    dates = np.ascontiguousarray(arrays.dates).view(np.int64)   # datetime64 has no buffer format
    offsets = np.ascontiguousarray(arrays.offsets)
    return {"values": hashlib.sha256(memoryview(values)).hexdigest(),
            "dates": hashlib.sha256(memoryview(dates)).hexdigest(),
            "offsets": hashlib.sha256(memoryview(offsets)).hexdigest(),
            "names": hashlib.sha256("\n".join(str(name) for name in arrays.names).encode()).hexdigest()}


# --------------------------------------------------------------------------- window helpers
def distinct_days(stamps: np.ndarray, block: int = 1 << 24) -> int:
    """Exact number of distinct calendar days, unioned block by block (bounded temporaries)."""
    days = set()
    for begin in range(0, len(stamps), block):
        days.update(np.unique(stamps[begin:begin + block].astype("datetime64[D]").view(np.int64)).tolist())
    return len(days)


def window(args, days: float, *, stream_bars: bool) -> dict:
    """Load one window through one path in this process and report what it cost.

    The peak is read immediately after the load returns: hashing the arrays afterwards is
    evidence gathering, not part of the load, and must not inflate the figure.
    """
    config = load_config(args.config)
    start = pd.Timestamp(args.start or config.splits.train.start)
    end = start + pd.Timedelta(days=days)
    data = SimpleNamespace(**{**config.data.model_dump(), "history_days": args.warmup_days})
    gc.collect()
    resident_before = resident_rss_mb()
    began = time.monotonic()
    bundle = load_data(data, start=start, end=end, stream_bars=stream_bars,
                       chunk_rows=args.chunk_rows)
    if stream_bars:
        arrays = bundle.arrays
    else:
        arrays = bar_arrays(bundle.bars, list(data.feature_columns))
    peak = peak_rss_mb()
    resident_after = resident_rss_mb()
    digest = canonical_digest(arrays)
    return {"path": "streamed_chunks" if stream_bars else "whole_frame",
            "days": days, "warmup_days": args.warmup_days, "chunk_rows": args.chunk_rows,
            "start": str(start), "end": str(end),
            "rows": arrays.rows, "bar_days": distinct_days(arrays.dates),
            "instruments": int(len(arrays.names)),
            "width": arrays.width, "target_mb": round(sum(arrays.nbytes().values()) / 2**20, 1),
            "canonical": digest, "provenance": hashlib.sha256(
                json.dumps(bundle.metadata["files"]["bars"], sort_keys=True).encode()).hexdigest(),
            "bar_rows_metadata": bundle.metadata["bar_rows"],
            "streamed_audit": bundle.metadata.get("bars_streamed"),
            "peak_rss_mb": round(peak, 1),
            "resident_before_mb": round(resident_before, 1),
            "resident_after_mb": round(resident_after, 1),
            "resident_delta_mb": round(resident_after - resident_before, 1),
            "seconds": round(time.monotonic() - began, 1)}


def child(args, days: float, stream_bars: bool) -> dict:
    command = [sys.executable, "-m", "scripts.verify_bar_streaming", "--worker", "window",
               "--config", str(args.config), "--days", str(days),
               "--warmup-days", str(args.warmup_days), "--chunk-rows", str(args.chunk_rows),
               "--stream-bars" if stream_bars else "--whole-frame"]
    if args.start:
        command += ["--start", args.start]
    finished = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if finished.returncode != 0:
        raise SystemExit(f"worker failed ({'streamed' if stream_bars else 'frame'} "
                         f"{days}d):\n{finished.stdout}\n{finished.stderr}")
    return json.loads(finished.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- pool equivalence
def pool_check(args) -> dict:
    """Compare the windows both readings assemble, batch for batch, and on the device."""
    config = load_config(args.config)
    start = pd.Timestamp(args.start or config.splits.train.start)
    end = start + pd.Timedelta(days=args.pool_days)
    data = SimpleNamespace(**{**config.data.model_dump(), "history_days": args.warmup_days})
    streamed = load_data(data, start=start, end=end, stream_bars=True, chunk_rows=args.chunk_rows)
    framed = load_data(data, start=start, end=end)
    frames = streamed.metadata["files"]["bars"]
    evidence = {"days": args.pool_days, "bars": streamed.arrays.rows,
                "arrays_bitwise": canonical_digest(streamed.arrays) == canonical_digest(
                    bar_arrays(framed.bars, list(data.feature_columns))),
                "provenance_identical": frames == framed.metadata["files"]["bars"]}
    targets = select_targets(framed.labels, SimpleNamespace(start=start, end=end),
                             data.label_horizon_minutes, sample_times=data.sample_times)
    if targets.empty:
        evidence["skipped"] = "the slice selected no eligible target"
        return evidence
    keep = sorted(targets["instrument"].unique())[:args.max_instruments]
    keys = targets.loc[targets["instrument"].isin(keep), ["date", "instrument"]].reset_index(drop=True)
    bars = framed.bars.loc[framed.bars["instrument"].isin(keep)].reset_index(drop=True)
    reference = build_batch(bars, keys, data)
    pool = StreamingPool.from_arrays(streamed.arrays, keys, data)
    order = np.random.default_rng(args.seed).permutation(len(keys))
    positions = [order[begin:begin + args.batch_size]
                 for begin in range(0, args.batches * args.batch_size, args.batch_size)]
    evidence.update({"instruments": len(keep), "keys": len(keys), "batches": len(positions),
                     "x_bitwise": True, "observed_bitwise": True, "history_dates_bitwise": True,
                     "coverage_identical": pool.coverage == reference.coverage,
                     "fit_mask_identical": np.array_equal(
                         pool.fit_mask,
                         reference.effective_rows.sum(axis=1) >= int(pool.settings["minimum"]))})
    device_verified = False
    for positions_one in positions:
        batch = pool.batch(positions_one)
        evidence["x_bitwise"] &= batch.x.tobytes() == reference.x[positions_one].tobytes()
        evidence["observed_bitwise"] &= np.array_equal(batch.observed, reference.observed[positions_one])
        evidence["history_dates_bitwise"] &= batch.history_dates.tobytes() == \
            reference.history_dates[positions_one].tobytes()
        if args.device and args.device != "cpu":
            import torch

            moved = torch.from_numpy(batch.x).to(args.device)
            device_verified = moved.cpu().numpy().tobytes() == batch.x.tobytes()
            evidence["x_bitwise"] &= device_verified
    evidence["device"] = args.device
    evidence["device_roundtrip_verified"] = device_verified
    return evidence


# --------------------------------------------------------------------------- extrapolation
def marginal(sizes: list[dict], field: str) -> dict:
    """Least-squares ``field = a + b * rows``; needs two sizes with distinct row counts."""
    distinct = {entry["rows"]: entry[field] for entry in sizes}
    if len(distinct) < 2:
        return {"defined": False, "reason": "window sizes must differ in row count",
                "points": len(distinct)}
    rows = np.array(sorted(distinct), dtype=float)
    values = np.array([distinct[int(row)] for row in rows])
    slope, intercept = np.polyfit(rows, values, 1)
    return {"defined": True, "points": len(rows),
            "bytes_per_row": round(float(slope) * 2**20, 1),
            "intercept_mb": round(float(intercept), 1)}


def anchor(args, target_rows: int) -> dict | None:
    """Linear projection from an externally measured frame-path peak (``--old-anchor``).

    The frame path's transient multiplier is only visible on a window that is large enough to
    dominate the interpreter, which a small local window is not. When the host's own
    measurement is known (132 days: 19.3 GB), scaling it by rows is the honest upper bound.
    """
    if not args.old_anchor:
        return None
    days, peak_gb = args.old_anchor
    rows_per_day = target_rows / args.target_days
    return {"anchor_days": days, "anchor_peak_gb": peak_gb,
            "model": "peak * target_rows / anchor_rows (linear in the bar count)",
            "projected_peak_gb": round(peak_gb * target_rows / (rows_per_day * days), 2)}


def projection(args, old: list[dict], new: list[dict]) -> dict:
    """Per-row measured cost, then the peak each path implies for ``--target-days``.

    Streamed: the target arrays are unavoidable data, so the bound is their size at the
    projected row count plus the largest measured overhead (peak minus array bytes) - a
    constant the window length must not grow. Frame: the marginal resident bytes per row
    extrapolated linearly, which is the extrapolation the current path cannot survive.
    """
    rows_per_day = max(entry["rows"] / max(entry["bar_days"], 1) for entry in new)
    target_rows = rows_per_day * args.target_days
    bytes_per_row = new[0]["width"] * 4 + 8
    target_gb = target_rows * bytes_per_row / 2**30
    overhead = max(entry["peak_rss_mb"] - entry["target_mb"] for entry in new)
    streamed_gb = target_gb + overhead / 1024
    old_marginal = marginal(old, "resident_delta_mb")
    old_gb = None
    if old_marginal["defined"]:
        old_gb = (old_marginal["intercept_mb"]
                  + old_marginal["bytes_per_row"] * target_rows / 2**20) / 1024
    return {"target_days": args.target_days,
            "measured": {"rows_per_trading_day": round(rows_per_day),
                         "target_rows": round(target_rows),
                         "arrays_per_row_bytes": bytes_per_row,
                         "target_array_gb": round(target_gb, 2),
                         "trading_days": {entry["days"]: entry["bar_days"] for entry in new}},
            "new_path": {"model": "target arrays + measured block overhead (independent of days)",
                         "array_gb": round(target_gb, 2),
                         "overhead_mb_upper_bound": round(overhead, 1),
                         "projected_peak_gb": round(streamed_gb, 2),
                         "marginal": marginal(new, "resident_delta_mb"),
                         "peak_slope": marginal(new, "peak_rss_mb")},
            "old_path": {"model": "marginal resident bytes per row, fitted and extrapolated",
                         "marginal": old_marginal,
                         "peak_slope": marginal(old, "peak_rss_mb"),
                         "projected_peak_gb": None if old_gb is None else round(old_gb, 2),
                         "anchored": anchor(args, target_rows)},
            "host_budget_gb": 172.0, "goal_gb": 60.0}


# --------------------------------------------------------------------------- driver
def parse_days(text: str) -> list[float]:
    values = sorted({float(item) for item in str(text).replace(" ", "").split(",") if item})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("sizes must be positive day counts")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/stream_pool_probe_local_20261027.json"))
    parser.add_argument("--start", default=None, help="window start (default: split train start)")
    parser.add_argument("--days", type=float, default=3.0,
                        help="window length (calendar days after --start) of the pool check")
    parser.add_argument("--sizes", default=None,
                        help="comma separated window lengths (calendar days) for the memory and "
                             "equivalence sweep; they must differ in bar rows")
    parser.add_argument("--target-days", type=float, default=927.0,
                        help="window length the projection is reported for")
    parser.add_argument("--warmup-days", type=int, default=None, help="bar warm-up before the window")
    parser.add_argument("--max-instruments", type=int, default=200)
    parser.add_argument("--pool-days", type=float, default=None,
                        help="window length of the pool-level comparison (default: min(--days, 10), "
                             "because that check holds both readings at once)")
    parser.add_argument("--streamed-only", action="store_true",
                        help="skip the frame path (a window the default reader cannot hold at all)")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_BAR_CHUNK_ROWS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--old-anchor", default=None, metavar="DAYS:PEAK_GB",
                        help="measured frame-path peak to extrapolate the old path from, "
                             "e.g. 132:19.3 (the 132-day probe)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker", choices=["window"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stream-bars", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--whole-frame", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        args.stream_bars = bool(args.stream_bars)
    if args.warmup_days is None:
        args.warmup_days = int(load_config(args.config).data.history_days)
    if args.pool_days is None:
        args.pool_days = min(args.days, 10.0)
    if args.old_anchor:
        days, peak = str(args.old_anchor).split(":")
        args.old_anchor = (float(days), float(peak))
    if args.sizes is None:
        args.sizes = f"{args.days}"
    args.sizes = parse_days(args.sizes)
    return args


def run(args) -> dict:
    sizes = args.sizes if args.sizes[-1] >= args.days else [*args.sizes, args.days]
    per_size = []
    for days in sizes:
        new = child(args, days, True)
        old = None if args.streamed_only else child(args, days, False)
        per_size.append({"days": days, "old": old, "new": new,
                         "canonical_identical": None if old is None else old["canonical"] == new["canonical"],
                         "provenance_identical": None if old is None else old["provenance"] == new["provenance"],
                         "rows_identical": None if old is None else old["rows"] == new["rows"]})
    compared = [entry for entry in per_size if entry["old"] is not None]
    equivalence = None if not compared else all(
        entry["canonical_identical"] and entry["provenance_identical"] and entry["rows_identical"]
        for entry in compared)
    pools = pool_check(args)
    frames = [entry["old"] for entry in per_size if entry["old"] is not None] or [per_size[0]["new"]]
    projection_result = projection(args, frames, [entry["new"] for entry in per_size])
    projected = projection_result["new_path"]["projected_peak_gb"]
    goal = projected <= projection_result["goal_gb"]
    under_host = projected <= projection_result["host_budget_gb"]
    # Decoupling is measured, not assumed: the streamed peak must grow at most half as fast
    # per row as the frame path's, i.e. it tracks the float32 arrays rather than the float64
    # pipeline. On a small window both slopes carry the interpreter's fixed arena noise, which
    # is why the comparison is between the two paths and not against a constant.
    new_slope = projection_result["new_path"]["peak_slope"]
    old_slope = projection_result["old_path"]["peak_slope"]
    decoupled = None if args.streamed_only else bool(
        new_slope["defined"] and old_slope["defined"]
        and new_slope["bytes_per_row"] <= 0.5 * old_slope["bytes_per_row"])
    pool_ok = True if "skipped" in pools else all(pools.get(name, False) for name in
                  ("arrays_bitwise", "x_bitwise", "observed_bitwise", "history_dates_bitwise",
                   "coverage_identical", "fit_mask_identical"))
    verdicts = {"canonical_arrays_bitwise_identical": None if equivalence is None else bool(equivalence),
                "provenance_identical": None if not compared else
                all(entry["provenance_identical"] for entry in compared),
                "pool_windows_bitwise_identical": bool(pool_ok),
                "peak_grows_with_target_arrays_only": decoupled,
                "projection_under_goal": bool(goal), "projection_under_host": bool(under_host)}
    context = {"device": args.device, "chunk_rows": args.chunk_rows,
               "config": str(args.config), "start": args.start}
    return {"kind": "bar-streaming-acceptance", "created_at_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "context": context, "sizes": per_size,
        "pool": pools, "projection": projection_result, "verdicts": verdicts,
        "passed": all(value for value in verdicts.values() if value is not None)}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker:
        result = window(args, args.days, stream_bars=args.stream_bars)
        print(json.dumps(result, sort_keys=True))
        return 0
    result = run(args)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(f"{'window':>7} {'rows':>11} {'days':>4} {'arrays':>8} | {'frame peak':>10} {'dRes':>8} "
          f"| {'streamed peak':>13} {'dRes':>8} | bitwise")
    for entry in result["sizes"]:
        old, new = entry["old"], entry["new"]
        frame_cells = (f"| {'-':>10} {'-':>8} " if old is None else
                       f"| {old['peak_rss_mb']:>7.1f} MB {old['resident_delta_mb']:>7.1f}M ")
        print(f"{entry['days']:>5.0f}d {new['rows']:>11,} {new['bar_days']:>4} {new['target_mb']:>7.1f}M "
              f"{frame_cells}| {new['peak_rss_mb']:>10.1f} MB {new['resident_delta_mb']:>7.1f}M "
              f"| {entry['canonical_identical']}")
    pools = result["pool"]
    print(f"pool: x={pools.get('x_bitwise')} observed={pools.get('observed_bitwise')} "
          f"dates={pools.get('history_dates_bitwise')} coverage={pools.get('coverage_identical')} "
          f"device={pools.get('device')}")
    projected = result["projection"]
    print(f"per row: arrays {projected['measured']['arrays_per_row_bytes']} B | "
          f"peak slope frame {projected['old_path']['peak_slope'].get('bytes_per_row')} B, "
          f"streamed {projected['new_path']['peak_slope'].get('bytes_per_row')} B | "
          f"resident slope frame {projected['old_path']['marginal'].get('bytes_per_row')} B, "
          f"streamed {projected['new_path']['marginal'].get('bytes_per_row')} B")
    anchored = projected["old_path"]["anchored"]
    if not projected["old_path"]["marginal"]["defined"]:
        print("frame path: no marginal fit (one window size or equal row counts)")
    if anchored:
        print(f"frame path from the {anchored['anchor_days']:.0f}-day anchor "
              f"{anchored['anchor_peak_gb']} GB -> {anchored['projected_peak_gb']} GB")
    print(f"projected at {projected['target_days']:.0f} days "
          f"({projected['measured']['target_rows']:,} rows): frame "
          f"{projected['old_path']['projected_peak_gb']} GB, streamed "
          f"{projected['new_path']['projected_peak_gb']} GB "
          f"(arrays {projected['measured']['target_array_gb']} GB + "
          f"{projected['new_path']['overhead_mb_upper_bound']} MB overhead)")
    print(json.dumps(result["verdicts"], sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
