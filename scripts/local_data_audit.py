"""Read-only inventory/schema audit and a bounded first-file sample; no raw output."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROCESS_STARTED = time.perf_counter()
PROCESS_CPU_STARTED = time.process_time()

# Include third-party import costs in the process budget.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

FORMAT_VERSION = 1
OHLC = ("open", "high", "low", "close")
BOOK_PRICES = tuple(f"{side}_price{level}" for side in ("ask", "bid") for level in (1, 2, 3))
LIMITATION = "仅检查全部文件清单与 IPC 结构，并检查最早文件的有限样本；不能证明全年数据合格，也不能据月份确定官方可训练区间。"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path, check_budget=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            if check_budget:
                check_budget()
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes() -> int | None:
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return None


class RunLog:
    def __init__(self, directory: Path, run_id: str, experiment_id: str):
        self.path = directory / "events.jsonl"
        self.run_id, self.experiment_id = run_id, experiment_id

    def event(self, phase: str, event_type: str, context: dict, level: str = "INFO") -> None:
        payload = {"format_version": FORMAT_VERSION, "utc_time": utc_now(), "level": level,
                   "phase": phase, "event_type": event_type, "run_id": self.run_id,
                   "experiment_id": self.experiment_id, "context": context}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def month_range(first: str, last: str) -> list[str]:
    return list(pd.period_range(first[:4] + "-" + first[4:], last[:4] + "-" + last[4:], freq="M").strftime("%Y%m"))


def inventory(source: Path) -> list[dict]:
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append({"relative_path": path.relative_to(source).as_posix(), "size_bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    return files


def sample_metrics(table: pa.Table, expected_month: str) -> dict:
    """Return aggregates only. Arrow nulls and floating NaN/Inf are distinct."""
    frame = table.to_pandas()
    columns = {}
    for field in table.schema:
        column = table[field.name]
        null_mask = column.is_null().to_numpy()
        numeric = pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        finite = np.isfinite(column.to_numpy()) if numeric else None
        columns[field.name] = {"null_count": int(null_mask.sum()),
                               "nonfinite_nonnull_count": int((~finite & ~null_mask).sum()) if numeric else None}

    dates = frame["date"]
    good_dates = dates.notna()
    minutes = dates.dt.hour * 60 + dates.dt.minute
    regular = minutes.between(9 * 60 + 31, 11 * 60 + 30) | minutes.between(13 * 60 + 1, 15 * 60)
    null_keys = dates.isna() | frame["instrument_id"].isna()
    minute_aligned = dates.eq(dates.dt.floor("min"))
    time_info = {"min_timestamp": dates.min().isoformat() if good_dates.any() else None,
                 "max_timestamp": dates.max().isoformat() if good_dates.any() else None,
                 "distinct_dates": int(dates.dt.date.nunique()), "source_timezone_declared": False,
                 "nonminute_aligned_count": int((good_dates & ~minute_aligned).sum()),
                 "outside_filename_month_count": int((good_dates & dates.dt.strftime("%Y%m").ne(expected_month)).sum()),
                 "weekend_count": int((good_dates & dates.dt.dayofweek.ge(5)).sum()),
                 "outside_regular_session_count": int((good_dates & ~regular).sum()),
                 "regular_session_check_is_diagnostic": True,
                 "global_time_monotonic": bool(dates[good_dates].is_monotonic_increasing)}
    missing = {}
    for name in OHLC:
        values = frame[name]
        missing[name] = {"documented_minus_one_count": int(values.eq(-1).sum()), "zero_count": int(values.eq(0).sum()),
                         "other_negative_count": int((values.lt(0) & values.ne(-1)).sum()),
                         "nonpositive_count": int(values.le(0).sum())}
    book = {name: {"documented_zero_missing_count": int(frame[name].eq(0).sum()),
                   "negative_price_count": int(frame[name].lt(0).sum())} for name in BOOK_PRICES}
    valid = np.ones(len(frame), dtype=bool)
    for name in OHLC:
        valid &= frame[name].gt(0).to_numpy() & np.isfinite(frame[name].to_numpy())
    high, low, opening, close = (frame[name].to_numpy() for name in ("high", "low", "open", "close"))
    bad_range = valid & ((high < low) | (high < opening) | (high < close) | (low > opening) | (low > close))
    result = {"rows": len(frame), "distinct_instrument_ids": int(frame["instrument_id"].nunique()),
              "null_key_rows": int(null_keys.sum()),
              "duplicate_key_excess_rows": int(frame.duplicated(["date", "instrument_id"], keep="first").sum()),
              "duplicate_key_involved_rows": int(frame.duplicated(["date", "instrument_id"], keep=False).sum()),
              "time": time_info, "columns": columns, "ohlc_missing": missing, "book_price_missing": book,
              "ohlc_range": {"eligible_rows": int(valid.sum()), "excluded_missing_or_nonfinite_rows": int((~valid).sum()),
                             "invalid_range_rows": int(bad_range.sum())}}
    result["checks"] = {
        "sample_nonempty": len(frame) > 0,
        "keys_nonnull": result["null_key_rows"] == 0,
        "keys_unique_in_sample": result["duplicate_key_excess_rows"] == 0,
        "timestamps_nonnull": columns["date"]["null_count"] == 0,
        "timestamps_minute_aligned": time_info["nonminute_aligned_count"] == 0,
        "timestamps_match_local_file_month": time_info["outside_filename_month_count"] == 0,
        "numeric_nonnull_values_finite": all(item["nonfinite_nonnull_count"] in (None, 0) for item in columns.values()),
        "ohlc_documented_negative_sentinel_only": all(item["other_negative_count"] == 0 for item in missing.values()),
        "book_prices_nonnegative": all(item["negative_price_count"] == 0 for item in book.values()),
        "ohlc_range_consistent_when_present": int(bad_range.sum()) == 0,
    }
    return result


def self_checks() -> dict:
    values = {name: [100, 100, 100, 100] for name in BOOK_PRICES + OHLC}
    values.update({"high": [110, 110, 110, 95], "low": [90, 90, 90, 90], "open": [100, 100, -1, 100],
                   "close": [100, 100, 100, 105], "ask_price1": [100, 100, 0, -1],
                   "date": pa.array([datetime(2020, 1, 2, 9, 31), datetime(2020, 1, 2, 9, 31), datetime(2020, 1, 2, 9, 32), None], type=pa.timestamp("ns")),
                   "instrument_id": [1, 1, 2, 3], "adjust_factor": pa.array([1.0, 2.0, np.inf, np.nan], type=pa.float32(), from_pandas=False)})
    result = sample_metrics(pa.table(values), "202001")
    assertions = {"duplicate_detected": result["duplicate_key_excess_rows"] == 1,
                  "missing_ohlc_excluded": result["ohlc_range"]["excluded_missing_or_nonfinite_rows"] == 1,
                  "bad_range_detected": result["ohlc_range"]["invalid_range_rows"] == 1,
                  "book_zero_separate_from_negative": result["book_price_missing"]["ask_price1"] == {"documented_zero_missing_count": 1, "negative_price_count": 1},
                  "nan_inf_separate_from_null": result["columns"]["adjust_factor"] == {"null_count": 0, "nonfinite_nonnull_count": 2},
                  "null_time_detected": result["columns"]["date"]["null_count"] == 1}
    if not all(assertions.values()):
        raise AssertionError("Aggregate validation self-check failed: " + ", ".join(name for name, passed in assertions.items() if not passed))
    return assertions


def build_report(manifest: dict, metrics: dict) -> str:
    sample = metrics.get("sample", {})
    missing_ohlc = sum(item["nonpositive_count"] for item in sample.get("ohlc_missing", {}).values())
    missing_book = sum(item["documented_zero_missing_count"] for item in sample.get("book_price_missing", {}).values())
    lines = ["# 本地数据盘点与有限样本检查", "", f"- run_id：{manifest['run_id']}", f"- 状态：{manifest['status']}",
             f"- 实际耗时：{metrics['cost']['elapsed_seconds']:.3f} 秒；GPU 使用：0。",
             f"- 全部源文件：{len(manifest.get('source_files', []))}；Feather 文件：{metrics.get('inventory', {}).get('feather_count', 0)}。",
             f"- 总字节数：{metrics.get('inventory', {}).get('total_size_bytes', 0)}。",
             f"- 全文件 IPC 结构检查通过数：{metrics.get('schema', {}).get('passed_files', 0)}。",
             f"- 有限样本行数：{sample.get('rows', 0)}；重复键多余行：{sample.get('duplicate_key_excess_rows', '未检查')}。",
             f"- OHLC 非正值单元格：{missing_ohlc}；盘口价格 0 单元格：{missing_book}；二者分别按缺失统计。",
             f"- 剔除 OHLC 缺失后的价格范围异常行：{sample.get('ohlc_range', {}).get('invalid_range_rows', '未检查')}。",
             f"- 列空值总数：{sum(item['null_count'] for item in sample.get('columns', {}).values())}；非空数值中的 NaN/Inf 总数：{sum(item['nonfinite_nonnull_count'] or 0 for item in sample.get('columns', {}).values())}。",
             f"- 样本时间范围：{sample.get('time', {}).get('min_timestamp', '未检查')} 至 {sample.get('time', {}).get('max_timestamp', '未检查')}。",
             f"- 常规连续交易时段外：{sample.get('time', {}).get('outside_regular_session_count', '未检查')} 行；仅诊断，不直接判定数据损坏。",
             "", "## 自动验收", ""]
    lines.extend(f"- {name}：{'通过' if passed else '失败'}。" for name, passed in metrics.get("checks", {}).items())
    if manifest.get("failure"):
        lines.extend(["", "失败原因：" + manifest["failure"]["message"]])
    lines.extend(["", "## 结论边界与下一步", "", LIMITATION,
                  "本次没有标签、评估股票池或股票 ID 映射核验，也没有监督训练、标签重建、归一化拟合或 bmeta.bin 内容解析。",
                  "本比赛不提供 SDK。后续在平台 AIStudio 中核对官方训练标签、评估股票池与股票 ID 的关系，并确认配套文件是否允许在本地使用；监督训练条件目前未齐。",
                  "源时间戳未声明时区；本次保留原值，常规交易时段检查仅为诊断，不代替官方采样规则。",
                  "文档中的月份仅表示本地文件分区，官方允许开发日期仍为 unconfirmed。",
                  "所有输出均为聚合、文件元信息或结构信息；原始样本未输出、未保存。", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "local_data_audit.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"]).resolve()
    source = (root / config["source_directory"]).resolve()
    output = (root / config["output_directory"]).resolve()
    if output == source or source in output.parents:
        raise ValueError("Audit artifacts cannot be written into the source dataset")
    budget = config["budget"]
    if budget["gpu_count"] != 0 or not 0 < budget["max_sample_rows"] <= 65536 or not 0 < budget["max_seconds"] <= 120:
        raise ValueError("This audit requires 0 GPUs, at most 65,536 sample rows and at most 120 seconds")
    started = PROCESS_STARTED
    cpu_started = PROCESS_CPU_STARTED
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z_" + uuid4().hex[:8]
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    log = RunLog(directory, run_id, config["experiment_id"])
    manifest = {"format_version": FORMAT_VERSION, "run_id": run_id, "experiment_id": config["experiment_id"],
                "started_utc": utc_now(), "status": "running", "hypothesis": "本地文件结构可统一读取，最早文件有限样本可开展基础数据校验。",
                "acceptance": "清单和结构符合配置；样本关键检查通过；源文件元信息不变；资源不超预算。",
                "budget": budget, "official_training_dates": config["official_training_dates"], "time_split": "not_defined_no_training",
                "seed": None, "determinism": "deterministic file ordering and first-file prefix; no GPU or random sampling",
                "environment": {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                                "processor": platform.processor(), "cpu_count": os.cpu_count(), "gpu_usage": 0,
                                "libraries": {name: importlib.metadata.version(name) for name in ("pyarrow", "pandas", "numpy")}},
                "source_directory": str(source), "source_read_mode": "read-only memory map and file rb", "source_files": [],
                "scope": LIMITATION, "raw_data_exported": False, "label_reconstruction": False, "bmeta_content_read": False,
                "retention": "retain every audit run and its source/config snapshots; no dataset cache or model artifacts created"}
    metrics = {"format_version": FORMAT_VERSION, "run_id": run_id, "checks": {}, "cost": {}}
    exit_code = 1

    def check_budget() -> None:
        if time.perf_counter() - started > budget["max_seconds"]:
            raise TimeoutError("Audit exceeded the declared wall-clock budget")

    log.event("startup", "run_started", {"budget": budget, "official_training_dates": "unconfirmed", "raw_data_exported": False})
    write_json(directory / "manifest.json", manifest)
    try:
        snapshots = directory / "snapshots"
        snapshots.mkdir()
        manifest["source_snapshots"] = []
        for path in (Path(__file__).resolve(), config_path):
            saved = snapshots / path.name
            shutil.copyfile(path, saved)
            manifest["source_snapshots"].append({"original_path": str(path), "snapshot_path": saved.relative_to(directory).as_posix(), "sha256": sha256(saved, check_budget)})
        log.event("archive", "source_config_snapshotted", {"snapshot_count": 2})
        metrics["self_checks"] = self_checks()
        metrics["checks"]["behavior_self_checks"] = all(metrics["self_checks"].values())
        log.event("validation", "behavior_self_checks_completed", metrics["self_checks"])
        if not source.is_dir():
            raise FileNotFoundError("Configured source directory does not exist")
        manifest["source_files"] = inventory(source)
        feather_files = [item for item in manifest["source_files"] if item["relative_path"].endswith(".feather")]
        expected = config["inventory_expectation"]
        expected_names = [month + ".0.feather" for month in month_range(expected["first_month"], expected["last_month"])]
        actual_names = [item["relative_path"] for item in feather_files]
        allowed_names = set(expected_names + [expected["metadata_filename"]])
        metrics["inventory"] = {"file_count": len(manifest["source_files"]), "feather_count": len(feather_files),
                                "total_size_bytes": sum(item["size_bytes"] for item in manifest["source_files"]),
                                "observed_filename_months": [name[:6] for name in actual_names],
                                "not_an_official_training_interval": True,
                                "metadata_only_files": [item["relative_path"] for item in manifest["source_files"] if not item["relative_path"].endswith(".feather")]}
        metrics["checks"].update({"expected_monthly_files_present": actual_names == expected_names,
                                 "expected_feather_count": len(feather_files) == expected["feather_count"],
                                 "metadata_file_present": (source / expected["metadata_filename"]).is_file(),
                                 "no_unexpected_source_files": all(item["relative_path"] in allowed_names for item in manifest["source_files"])})
        log.event("inventory", "source_inventory_completed", metrics["inventory"])
        write_json(directory / "manifest.json", manifest)
        schema_results = []
        for item in feather_files:
            check_budget()
            path = source / item["relative_path"]
            result = {"relative_path": item["relative_path"]}
            try:
                with pa.memory_map(str(path), "r") as mapped:
                    reader = pa.ipc.open_file(mapped)
                    fields = {field.name: str(field.type) for field in reader.schema}
                    result.update({"record_batch_count": reader.num_record_batches, "column_count": len(reader.schema),
                                   "fields": fields, "schema_matches": list(fields.items()) == list(config["expected_schema"].items()),
                                   "format": "Arrow IPC file / Feather v2", "rows_decoded": 0})
            except Exception:
                result.update({"schema_matches": False, "exception_type": sys.exc_info()[0].__name__})
                log.event("schema", "schema_read_failed", {"relative_path": item["relative_path"], "exception_stack": traceback.format_exc(), "recovery_state": "continuing_metadata_checks_only"}, "ERROR")
            schema_results.append(result)
        metrics["schema"] = {"passed_files": sum(item["schema_matches"] for item in schema_results), "files": schema_results,
                             "all_source_rows_decoded_for_schema": 0, "metadata_equality_required": False}
        metrics["checks"]["all_ipc_schemas_match"] = bool(schema_results) and all(item["schema_matches"] for item in schema_results)
        log.event("schema", "all_schema_checks_completed", {"file_count": len(schema_results), "passed_files": metrics["schema"]["passed_files"], "rows_decoded": 0})
        if not feather_files:
            raise ValueError("No Feather files available for the bounded sample")
        first = source / feather_files[0]["relative_path"]
        first_result = schema_results[0]
        if not first_result["schema_matches"]:
            raise ValueError("Earliest file schema mismatch; sample validation not attempted")
        if not re.fullmatch(r"\d{6}\.0\.feather", first.name):
            raise ValueError("Earliest filename does not provide a valid local month partition")
        manifest["sample_source"] = {"relative_path": first.relative_to(source).as_posix(), "sha256": sha256(first, check_budget),
                                     "hash_scope": "entire earliest file byte stream; other source files not hashed", "selection": "first record-batch prefix", "max_rows": budget["max_sample_rows"]}
        with pa.memory_map(str(first), "r") as mapped:
            reader = pa.ipc.open_file(mapped)
            if reader.num_record_batches == 0:
                raise ValueError("Earliest Feather file has no record batches")
            batches = []
            decoded_rows = 0
            for batch_index in range(reader.num_record_batches):
                check_budget()
                if decoded_rows == budget["max_sample_rows"]:
                    break
                batch = reader.get_batch(batch_index)
                if batch.num_rows > budget["max_sample_rows"] - decoded_rows:
                    raise ValueError("IPC record batch exceeds the remaining bounded sample; row validation not attempted")
                batches.append(batch)
                decoded_rows += batch.num_rows
            sample = pa.Table.from_batches(batches)
            manifest["sample_source"]["decoded_rows"] = decoded_rows
            manifest["sample_source"]["record_batches_read"] = len(batches)
            metrics["sample"] = sample_metrics(sample, first.name[:6])
        metrics["checks"].update(metrics["sample"]["checks"])
        metrics["checks"]["sample_row_budget"] = metrics["sample"]["rows"] <= budget["max_sample_rows"]
        log.event("sample", "bounded_sample_validated", {"rows": metrics["sample"]["rows"], "checks": metrics["sample"]["checks"], "raw_data_exported": False})
        check_budget()
        metrics["checks"]["source_files_metadata_unchanged"] = inventory(source) == manifest["source_files"]
        metrics["checks"]["wall_clock_budget"] = time.perf_counter() - started <= budget["max_seconds"]
        failed = [name for name, passed in metrics["checks"].items() if not passed]
        if failed:
            manifest["status"] = "failed"
            manifest["failure"] = {"type": "AcceptanceFailure", "message": "Automatic checks failed: " + ", ".join(failed), "failed_checks": failed}
            log.event("validation", "acceptance_failed", {**manifest["failure"], "recovery_state": "artifacts_retained_no_training"}, "ERROR")
            exit_code = 2
        else:
            manifest["status"] = "completed"
            exit_code = 0
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error), "exception_stack": traceback.format_exc()}
        log.event("failure", "run_interrupted" if isinstance(error, KeyboardInterrupt) else "run_failed", {**manifest["failure"], "recovery_state": "artifacts_retained_no_training"}, "ERROR")
    finally:
        metrics["cost"] = {"elapsed_seconds": round(time.perf_counter() - started, 6), "cpu_seconds": round(time.process_time() - cpu_started, 6),
                           "gpu_count": 0, "gpu_hours": 0, "peak_process_rss_bytes": peak_rss_bytes(),
                           "validated_sample_rows": metrics.get("sample", {}).get("rows", 0)}
        manifest["ended_utc"] = utc_now()
        manifest["actual_cost"] = metrics["cost"]
        manifest["artifact_directory"] = str(directory)
        write_json(directory / "metrics.json", metrics)
        (directory / "report.txt").write_text(build_report(manifest, metrics), encoding="utf-8")
        log.event("archive", "aggregate_artifacts_saved", {"artifacts": ["metrics.json", "report.txt"], "raw_data_exported": False})
        log.event("end", "run_ended", {"status": manifest["status"], "exit_code": exit_code, "cost": metrics["cost"]}, "INFO" if exit_code == 0 else "ERROR")
        manifest["artifacts"] = [{"path": name, "sha256": sha256(directory / name)} for name in ("events.jsonl", "metrics.json", "report.txt")]
        write_json(directory / "manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": manifest["status"], "elapsed_seconds": metrics["cost"]["elapsed_seconds"], "artifact_directory": str(directory),
                      "feather_files": metrics.get("inventory", {}).get("feather_count", 0), "sample_rows": metrics.get("sample", {}).get("rows", 0),
                      "failed_checks": [name for name, passed in metrics["checks"].items() if not passed]}, ensure_ascii=False, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
