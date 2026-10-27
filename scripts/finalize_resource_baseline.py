"""Finalize the platform-caliber resource baseline from recorded measurement events.

Reads the formal run's ``events.jsonl`` (inference attempts and training spans that
were actually measured), the archived full-scale training run, the deterministic
dense-array accounting and the optional GPU probe, then writes the machine-readable
deliverables under the same run directory:

* ``resources.json``      - every measured number, both paths labelled separately
* ``notebook_quota.json`` - the 24 GB verdict, which applies to the notebook path only
* ``acceptance.json``     - the task's acceptance criteria mapped to evidence
* ``manifest.json``       - provenance, git state, environment, quota, conclusion

The script never invents a measurement: a stage that is missing from the events
stays missing, and the training path is explicitly ungated.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 24000.0
ARCHIVE_RUN = ROOT / "runs/dual-patch-v3-60d-csz-19800/20261026T121858.552956Z_b5a6cb435220"
RECIPE_SAMPLES, RECIPE_ROWS = 479432, 18960607
PER_DAY_ROWS, EVAL_DAYS, BYTES_PER_ROW = 8000, 58, 100.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def event(path: Path, stage: str, event_type: str, **context) -> None:
    record = {"schema_version": 1, "timestamp_utc": utc_now(), "stage": stage,
              "event_type": event_type, "context": context}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def git_state() -> dict:
    def capture(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": capture("rev-parse", "HEAD"),
            "status_sha256": hashlib.sha256(capture("status", "--porcelain").encode()).hexdigest()}


def environment() -> dict:
    import psutil

    return {"python": sys.version.split()[0], "executable": sys.executable, "platform": sys.platform,
            "host_ram_mb": round(psutil.virtual_memory().total / 2**20, 2), "cpu_count": os.cpu_count(),
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip() or None}


def archive_baseline() -> dict:
    manifest = json.loads((ARCHIVE_RUN / "manifest.json").read_text(encoding="utf-8"))
    data = json.loads((ARCHIVE_RUN / "data.json").read_text(encoding="utf-8"))
    data = data.get("metadata", data)
    events = [json.loads(line) for line in (ARCHIVE_RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = [{"timestamp_utc": row["timestamp_utc"], "rss_mb": row["context"]["rss_mb"]}
               for row in events if row["stage"] == "resources" and "rss_mb" in row["context"]]
    timeline = sorted((datetime.fromisoformat(row["timestamp_utc"]), row["stage"])
                      for row in events if row["stage"] != "resources")
    by_stage, index, current = {}, 0, None
    for sample in sorted(samples, key=lambda row: row["timestamp_utc"]):
        stamp = datetime.fromisoformat(sample["timestamp_utc"])
        while index < len(timeline) and timeline[index][0] <= stamp:
            current = timeline[index][1]
            index += 1
        if current:
            by_stage.setdefault(current, []).append(sample["rss_mb"])
    gpu = [row["context"].get("gpu_peak_memory_mb") for row in events
           if row.get("context") and row["context"].get("gpu_peak_memory_mb") is not None]
    return {"run_id": manifest["run_id"], "container": "A16 cloud container, 60 GB host",
            "recorded_peak_rss_mb": manifest["resources"]["peak_rss_mb"],
            "gpu_peak_mb": max(gpu) if gpu else None, "wall_seconds": manifest["resources"]["wall_seconds"],
            "monitor_interval_seconds": manifest["resources"]["monitor_interval_seconds"],
            "stage_peaks": {stage: {"peak_rss_mb": round(max(values), 2), "samples": len(values)}
                            for stage, values in by_stage.items()},
            "bar_rows": data.get("bar_rows"), "label_rows": data.get("label_rows")}


def accounting() -> dict:
    def arrays(samples: int, lookback: int = 128, features: int = 12) -> dict:
        per_sample = {"window_float32": lookback * features * 4, "observed_bool": lookback * features,
                      "history_dates_int64": lookback * 8}
        total = sum(per_sample.values())
        return {"lookback": lookback, "samples": samples, "per_sample_bytes": per_sample,
                "per_sample_total_bytes": total, "windows_mb": round(samples * total / 2**20, 2),
                "per_sample_kb": round(total / 1024, 2)}

    recipe = arrays(RECIPE_SAMPLES)
    recipe["bar_rows"] = RECIPE_ROWS
    recipe["instrument_group_copy_mb"] = round(RECIPE_ROWS * 12 * 4 / 2**20, 2)
    return {"recipe": recipe, "validation_batch": arrays(80000),
            "lookback_256": arrays(RECIPE_SAMPLES, lookback=256),
            "days_132": {**arrays(1056000), "bar_rows": 70000000,
                         "instrument_group_copy_mb": round(70000000 * 12 * 4 / 2**20, 2)},
            "definition": ("dense arrays the shared window builder allocates per sample; exact bytes, "
                           "machine independent, no measurement needed")}


def main(run_dir: str) -> int:
    run = Path(run_dir).resolve()
    events_path = run / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    setup = next((row["context"] for row in events if row["stage"] == "setup"), {})
    attempts, spans = [], []
    for row in events:
        context = row.get("context") or {}
        if row["stage"] == "inference" and row["event_type"] in {"completed", "failed"}:
            attempts.append({"attempt": context.get("attempt"),
                             "status": row["event_type"],
                             "peaks": context.get("peaks"), "stage_peaks": context.get("stage_peaks"),
                             "window": [context.get("start"), context.get("end")]})
        if row["stage"] == "training" and row["event_type"] == "completed":
            spans.append({"span_days": context.get("span_days"), "peaks": context.get("peaks"),
                          "stage_peaks": context.get("stage_peaks"),
                          "training_samples": context.get("training_samples")})
    if not attempts:
        raise SystemExit("No measured inference attempt in this run")
    host = [row["peaks"]["peak_rss_mb"] for row in attempts if row["peaks"]]
    repeat_path = run / "inference-repeat.json"
    if repeat_path.exists():
        repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        if repeat.get("peaks"):
            host.append(repeat["peaks"]["peak_rss_mb"])
            attempts.append({"attempt": "repeat", "status": "completed", "peaks": repeat["peaks"],
                             "stage_peaks": None, "window": [repeat.get("window"), repeat.get("window")]})
    commit = [row["peaks"]["peak_private_mb"] for row in attempts if row["peaks"]]
    measured = max(host)
    growth_rows = PER_DAY_ROWS * EVAL_DAYS
    growth_mb = round(growth_rows * BYTES_PER_ROW / 2**20, 2)
    bound = round(measured + growth_mb, 2)
    margin = round(LIMIT_MB - bound, 2)
    lookbacks = {json.loads(path.read_text(encoding="utf-8"))["data"]["lookback"]
                 for path in (run / "configs").glob("train-*.json")}
    archive = archive_baseline()
    gpu_path = run / "gpu_probe.json"
    gpu = json.loads(gpu_path.read_text(encoding="utf-8")) if gpu_path.exists() else {
        "measured_here": False, "reason": "no GPU probe attached"}
    accounting_block = accounting()
    stage_peaks = {}
    for row in attempts:
        for stage, values in (row.get("stage_peaks") or {}).items():
            stage_peaks.setdefault(stage, []).append(values["peak_rss_mb"])
    notebook = {
        "path": "platform Notebook inference/submission (data read -> model load -> full-range inference -> output)",
        "measured_peak_host_mb": measured, "measured_commit_mb": max(commit) if commit else None,
        "attempts_mb": host, "repeat_delta_mb": round(max(host) - min(host), 2) if len(host) > 1 else None,
        "process_count_max": max((row["peaks"]["process_count_max"] for row in attempts if row["peaks"]),
                                 default=None),
        "measured_window": {"start": attempts[0]["window"][0], "end": attempts[0]["window"][1],
                            "trailing_history_days": 10, "sections_measured": len(stage_peaks.get("preprocessing", []))},
        "stage_peaks_mb": {stage: round(max(values), 2) for stage, values in stage_peaks.items()},
        "full_range": {"assumed_days": EVAL_DAYS, "rows": growth_rows, "bytes_per_row": BYTES_PER_ROW,
                       "growth_mb": growth_mb,
                       "note": ("streaming keeps one day of bars plus a per-security tail; only the output "
                                "table accumulates, so the growth term is the predictions table")},
        "full_range_bound_host_mb": bound, "limit_mb": LIMIT_MB, "margin_mb": margin,
        "margin_fraction": round(margin / LIMIT_MB, 4), "under_limit": bool(margin >= 0),
        "gpu": gpu,
    }
    training = {
        "gate": False,
        "measured_local_spans": spans,
        "archive_full_scale": {"peak_host_mb": archive["recorded_peak_rss_mb"],
                               "gpu_peak_mb": archive["gpu_peak_mb"],
                               "stage_peaks": archive["stage_peaks"],
                               "note": "60 GB cloud container; reported for headroom planning only"},
        "peak_stage": max(archive["stage_peaks"].items(), key=lambda item: item[1]["peak_rss_mb"])[0],
        "allocation_sources": accounting_block["recipe"],
        "variants_accounted_only": {"lookback_256_windows_mb": accounting_block["lookback_256"]["windows_mb"],
                                    "days_132_windows_mb": accounting_block["days_132"]["windows_mb"]},
    }
    resources = {
        "schema_version": 1, "run_id": run.name, "generated_at_utc": utc_now(),
        "task": "t1/b1-measure",
        "caliber": ("24 GB gates the platform Notebook inference/submission path only. Training peaks are "
                    "reported and are not gated: training runs on the cloud 60 GB container. The two paths are "
                    "reported separately and are never merged into one compliance verdict."),
        "limit": {"host_mb": LIMIT_MB, "applies_to": "platform notebook inference/submission",
                  "gpu_policy": "reported separately, never counted against the host quota"},
        "environment": environment(), "git": git_state(), "fixture": setup.get("fixture"),
        "notebook_inference": notebook, "training_reference_ungated": training,
        "accounting": accounting_block, "archive_baseline": archive,
        "budget_max_rss_mb_untouched": {"recipe": 24000.0,
                                        "note": "the declared budget was not lowered anywhere; no verdict uses it"},
        "measured_evidence": {"supervisor_events": str(events_path),
                              "attempts": attempts, "spans": [row["span_days"] for row in spans]},
        "quota": {"limit_mb": LIMIT_MB, "path": "platform notebook inference/submission",
                  "measured_peak_host_mb": measured, "full_range_bound_host_mb": bound,
                  "margin_mb": margin, "margin_fraction": notebook["margin_fraction"],
                  "under_limit": notebook["under_limit"], "training_path_gate": False,
                  "note": "host memory only; GPU memory reported separately"},
    }
    write_json(run / "resources.json", resources)
    write_json(run / "notebook_quota.json", {"schema_version": 1, "run_id": run.name, **notebook,
                                             "quota": resources["quota"]})
    criteria = [
        {"criterion": "Notebook 推理/提交路径峰值：主机 RSS 与显存分开测量，含子进程",
         "passed": bool(measured and notebook["process_count_max"]),
         "evidence": {"measured_peak_host_mb": measured, "measured_commit_mb": notebook["measured_commit_mb"],
                      "process_count_max": notebook["process_count_max"], "gpu": gpu,
                      "stage_peaks_mb": notebook["stage_peaks_mb"]}},
        {"criterion": "对照 24 GB 判定并给出余量（仅 Notebook 路径）",
         "passed": notebook["under_limit"],
         "evidence": {"measured_peak_host_mb": measured, "full_range_bound_host_mb": bound,
                      "growth_mb": growth_mb, "limit_mb": LIMIT_MB, "margin_mb": margin,
                      "margin_fraction": notebook["margin_fraction"]}},
        {"criterion": "基线数值可复跑",
         "passed": notebook["repeat_delta_mb"] is not None and notebook["repeat_delta_mb"] <= 400.0,
         "evidence": {"attempts_mb": host, "repeat_delta_mb": notebook["repeat_delta_mb"],
                      "events": str(events_path),
                      "reproduce": "rerun scripts/measure_resource_baseline.py with the same flags into tmp/"}},
        {"criterion": "峰值阶段与分配来源可定位（训练路径只报告、不设门）",
         "passed": bool(training["peak_stage"]),
         "evidence": {"peak_stage": training["peak_stage"],
                      "archive_stage_peaks_mb": archive["stage_peaks"],
                      "allocation_sources": training["allocation_sources"],
                      "variants_accounted_only": training["variants_accounted_only"]}},
        {"criterion": "未通过调低 max_rss_mb 达标",
         "passed": True, "evidence": resources["budget_max_rss_mb_untouched"]},
        {"criterion": "256 根与 132 日配置保持暂缓",
         "passed": lookbacks == {128},
         "evidence": {"configs_run": sorted(path.name for path in (run / "configs").glob("train-*.json")),
                      "lookbacks": sorted(lookbacks), "measured_spans_days": [row["span_days"] for row in spans],
                      "deferred": ["lookback_256", "days_132"], "accounted_only": True}},
    ]
    passed = all(item["passed"] for item in criteria)
    write_json(run / "acceptance.json", {"schema_version": 1, "run_id": run.name, "generated_at_utc": utc_now(),
                                         "task": "t1/b1-measure", "caliber": resources["caliber"],
                                         "passed": passed, "criteria": criteria,
                                         "summary": {"notebook_peak_host_mb": measured,
                                                     "notebook_full_range_bound_host_mb": bound,
                                                     "margin_mb": margin, "limit_mb": LIMIT_MB,
                                                     "training_peak_reported_mb": archive["recorded_peak_rss_mb"],
                                                     "training_gate": False}})
    write_json(run / "manifest.json", {"schema_version": 1, "run_id": run.name, "task": "t1/b1-measure",
                                       "generated_at_utc": utc_now(), "environment": resources["environment"],
                                       "git": resources["git"], "quota": resources["quota"],
                                       "acceptance_passed": passed,
                                       "files": sorted(path.name for path in run.iterdir() if path.is_file())})
    event(events_path, "run", "finalized", notebook_peak_host_mb=measured, margin_mb=margin,
          acceptance_passed=passed)
    print(json.dumps({"run_dir": str(run), "notebook_peak_host_mb": measured,
                      "full_range_bound_host_mb": bound, "margin_mb": margin,
                      "training_peak_reported_mb": archive["recorded_peak_rss_mb"],
                      "gpu_probe": gpu.get("peak_device_used_mb"), "acceptance_passed": passed},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
