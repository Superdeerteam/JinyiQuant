"""Isolated synthetic benchmark of submission data paths, without model inference.

Run with the project's Python environment and PYTHONPATH=src;.  This measures
query materialization, preprocessing, and one final timestamp's model inputs.
It does not measure DAI server memory, model inference, scoring, or platform SLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def worker(args):
    import numpy as np
    import pandas as pd

    from jinyiquant.data import build_batch, preprocess_bars
    from modified_tlob.schema import lob_columns
    from modified_tlob.submission import iter_bar_days, query_frame, section_history

    days = pd.bdate_range("2022-01-03", periods=args.days)
    start, end = days[0], days[-1] + pd.Timedelta(hours=15)
    features = list(lob_columns(3))
    config = {
        "feature_columns": features, "lookback": 128, "min_history": 128,
        "missing_policy": "ffill", "max_ffill_steps": 3, "missing_sentinel": -999,
        "price_scale": 1.0, "sample_times": ["14:30"],
    }
    instruments = np.array([f"{i:06d}.SZ" for i in range(args.securities)], dtype=object)
    stats = {"calls": 0, "total_rows": 0, "max_rows": 0, "max_span_seconds": 0.0}

    class QueryResult:
        def __init__(self, lower, upper):
            self.lower, self.upper = pd.Timestamp(lower), pd.Timestamp(upper)

        def df(self):
            # Only timestamps are listed before materialization. No full-market
            # DataFrame is retained by the query double between requests.
            minutes = []
            for day in days:
                if day > self.upper or day + pd.Timedelta(days=1) <= self.lower:
                    continue
                values = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min")
                values = values.append(pd.date_range(
                    day + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min"))
                minutes.extend(values[(values >= self.lower) & (values <= self.upper)].to_numpy())
            minute_array = np.asarray(minutes, dtype="datetime64[ns]")
            dates = np.repeat(minute_array, args.securities)
            security = np.tile(np.arange(args.securities, dtype=np.int32), len(minutes))
            elapsed_minutes = ((dates - np.datetime64(start)) / np.timedelta64(1, "m")).astype(np.int64)
            row_id = elapsed_minutes * args.securities + security
            frame = pd.DataFrame({"date": dates, "instrument": instruments[security]})
            for index, name in enumerate(features):
                values = ((row_id % 10007) * 0.001 + index + 1).astype(np.float32)
                values[(row_id + index * 17) % 997 == 0] = np.nan
                frame[name] = values
            stats["total_rows"] += len(frame)
            stats["max_rows"] = max(stats["max_rows"], len(frame))
            return frame

    def query(sql, *, filters):
        if "synthetic_bars" not in sql or set(filters) != {"date"}:
            raise ValueError("Unexpected query in data-path benchmark")
        lower, upper = map(pd.Timestamp, filters["date"])
        stats["calls"] += 1
        stats["max_span_seconds"] = max(stats["max_span_seconds"], (upper - lower).total_seconds())
        return QueryResult(lower, upper)

    keys = pd.DataFrame({
        "date": np.repeat(days[-1] + pd.Timedelta(hours=14, minutes=30), args.securities),
        "instrument": instruments,
    })
    started = time.perf_counter()
    if args.worker == "full_range":
        bars = query_frame(query, "synthetic_bars", ["date", "instrument", *features], start, end)
        bars["date"] = pd.to_datetime(bars["date"])
        bars = bars.loc[(bars.date >= start) & (bars.date <= end)]
        bars = preprocess_bars(bars, config)
        # Same full-table fingerprint allocation made by the previous submission.
        hashlib.sha256(pd.util.hash_pandas_object(bars, index=False).to_numpy().tobytes()).hexdigest()
        batch = build_batch(bars, keys, config)
    else:
        batch = None
        for bars, metadata in iter_bar_days(query, "synthetic_bars", start, end, config):
            if pd.Timestamp(metadata["start"]).normalize() == days[-1]:
                batch = build_batch(section_history(bars, keys, config["lookback"]), keys, config)
            del bars
        if batch is None:
            raise RuntimeError("Final timestamp was not processed")
    hashes = {}
    for name in ("x", "observed", "history_dates"):
        array = np.ascontiguousarray(getattr(batch, name))
        hashes[name] = hashlib.sha256(
            str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()
    hashes["keys"] = hashlib.sha256(
        pd.util.hash_pandas_object(batch.keys, index=False).to_numpy().tobytes()).hexdigest()
    print(json.dumps({
        "mode": args.worker, "status": "completed",
        "data_pipeline_seconds": time.perf_counter() - started,
        "query": stats, "input_hashes": hashes, "coverage": batch.coverage,
        "input_shape": list(batch.x.shape), "preprocessing": config,
    }), flush=True)


def run_child(args, mode):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
           "--securities", str(args.securities), "--days", str(args.days)]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    process = psutil.Process(proc.pid)
    peak, os_peak_sum, samples, max_processes, failure = 0, 0, 0, 0, None
    while proc.poll() is None:
        try:
            # A Windows venv executable can be a small redirector. Include its
            # actual Python child, and enforce the same budget on the full tree.
            processes = [process, *process.children(recursive=True)]
            rss, lifetime_peak = 0, 0
            for child in processes:
                try:
                    info = child.memory_info()
                    rss += info.rss
                    lifetime_peak += getattr(info, "peak_wset", info.rss)
                except psutil.NoSuchProcess:
                    continue
            peak = max(peak, rss)
            os_peak_sum = max(os_peak_sum, lifetime_peak)
            max_processes = max(max_processes, len(processes))
            samples += 1
            if max(peak, os_peak_sum) > args.rss_cap_mib * 2**20:
                failure = "rss_cap_exceeded"
            elif psutil.virtual_memory().available < args.reserve_mib * 2**20:
                failure = "system_available_memory_reserve_reached"
            elif time.perf_counter() - started > args.wall_seconds:
                failure = "wall_timeout"
            if failure:
                for child in reversed(processes):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.02)
    stdout, stderr = proc.communicate()
    result = {
        "mode": mode, "status": "failed", "returncode": proc.returncode,
        "peak_rss_bytes": peak, "peak_rss_mib": peak / 2**20, "memory_samples": samples,
        "os_process_lifetime_peak_sum_bytes": os_peak_sum,
        "max_monitored_processes": max_processes,
        "memory_scope": "sampled process_tree RSS sum; includes Windows venv redirector and Python child",
        "process_seconds": time.perf_counter() - started,
    }
    if failure:
        result["failure_reason"] = failure
    elif proc.returncode == 0:
        try:
            result.update(json.loads(stdout))
        except json.JSONDecodeError:
            result["failure_reason"] = "invalid_worker_output"
    else:
        result["failure_reason"] = "worker_failed"
    if stderr:
        result["stderr"] = stderr[-8000:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--securities", type=int, default=64)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--rss-cap-mib", type=int, default=768)
    parser.add_argument("--reserve-mib", type=int, default=384)
    parser.add_argument("--wall-seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("runs/submission-memory-20261020/benchmark.json"))
    parser.add_argument("--worker", choices=["full_range", "streamed"])
    args = parser.parse_args()
    if min(args.securities, args.days, args.rss_cap_mib, args.reserve_mib, args.wall_seconds) <= 0:
        parser.error("Sizes and safety budgets must be positive")
    if args.worker:
        worker(args)
        return 0
    if args.output.exists():
        parser.error("Output exists; use a new --output path to preserve prior results")
    root = Path(__file__).resolve().parents[1]
    available = psutil.virtual_memory().available
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_data_pipeline_only_single_final_timestamp",
        "limitations": [
            "No model inference, training, DAI server execution, or evaluator measured.",
            "Synthetic query construction overhead is included in both paths.",
            "Only the final timestamp's model inputs are compared; this is not a platform SLA test.",
        ],
        "configuration": {
            "securities": args.securities, "trading_days": args.days, "bars_per_day": 240,
            "features": 12, "lookback": 128, "rss_cap_mib": args.rss_cap_mib,
            "system_reserve_mib": args.reserve_mib, "wall_seconds_per_child": args.wall_seconds,
            "available_memory_before_bytes": available, "python": sys.version,
        },
        "sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [root / "modified_tlob/submission.py", root / "src/jinyiquant/data.py",
                                 Path(__file__).resolve()]},
        "runs": [], "passed": False,
    }
    if available < (args.rss_cap_mib + args.reserve_mib) * 2**20:
        report["status"] = "blocked_resource_budget"
    else:
        for mode in ("full_range", "streamed"):
            report["runs"].append(run_child(args, mode))
            if report["runs"][-1]["status"] != "completed":
                break
        completed = len(report["runs"]) == 2 and all(r["status"] == "completed" for r in report["runs"])
        equal = completed and report["runs"][0]["input_hashes"] == report["runs"][1]["input_hashes"]
        rows_equal = completed and report["runs"][0]["query"]["total_rows"] == report["runs"][1]["query"]["total_rows"]
        report.update(status="completed" if completed else "failed",
                      exact_inputs_equal=equal, source_row_counts_equal=rows_equal, passed=equal and rows_equal)
        if completed:
            baseline, streamed = report["runs"]
            report["peak_rss_ratio_streamed_over_full"] = streamed["peak_rss_bytes"] / baseline["peak_rss_bytes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
