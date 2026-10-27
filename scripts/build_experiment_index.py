"""Build a deterministic, queryable experiment index from ``runs/**/manifest.json``.

The manifest is the only fact source. Every manifest under ``runs/`` is indexed
exactly once, including failed, interrupted, budget-exceeded and running attempts
and every attempt of the same experiment. Two directories are excluded:

* ``runs/frozen/`` - the stage-A frozen snapshot copies manifests; they are
  history duplicates and are referenced as a separate ``snapshots`` section;
* ``runs/index/`` - this index's own output.

Missing required fields are reported as *gaps* and never filled in by hand.
Symlinked directories are not followed, so aliased copies (for example pytest
``*current`` junctions) do not duplicate entries.

Outputs (``runs/index`` by default): ``manifest_index.jsonl`` (canonical,
byte-identical across rebuilds), ``manifest_index.sqlite3`` (queryable mirror),
``summary.json`` (coverage, status distribution, attempts, gaps, snapshots).

Query examples::

    python scripts/build_experiment_index.py --query --experiment-id <id>
    python scripts/build_experiment_index.py --query --status failed \\
        --since 2026-10-01 --until 2026-11-01
    sqlite3 runs/index/manifest_index.sqlite3 "select status, count(*) from runs group by 1"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
INDEX_FILENAME = "manifest_index.jsonl"
SQLITE_FILENAME = "manifest_index.sqlite3"
SUMMARY_FILENAME = "summary.json"
VIEWED_WINDOWS_FILENAME = "viewed_windows.json"
EXCLUDED_TOP_LEVEL = ("frozen", "index")
SEALED_INTERVAL = ("2024-10-08", "2025-01-01")
DATE_FORMAT = "%Y-%m-%d"
FROZEN_SNAPSHOT_INDEX = "runs/frozen/a-20261026/sha256_index.json"
FAILURE_STATUSES = ("failed", "interrupted", "budget_exceeded")
STATUS_GROUPS = {
    "completed": "success",
    "succeeded": "success",
    "failed": "failed",
    "budget_exceeded": "failed",
    "interrupted": "interrupted",
    "running": "running",
}
REQUIRED_FIELDS = (
    "run_id",
    "experiment_id",
    "status",
    "started_at",
    "ended_at",
    "code.commit",
    "code.dirty_fingerprint",
    "config_sha256",
    "plan_sha256",
    "seed",
    "data.source",
    "data.version",
    "data.sha256",
    "splits.evaluation_split",
    "splits.train.start",
    "splits.train.end",
    "splits.validation.start",
    "splits.validation.end",
    "splits.final.start",
    "splits.final.end",
    "metrics.sha256",
    "artifacts",
)
MISSING_ARCHIVES = (
    {
        "path": "runs/training-duration-20261015",
        "kind": "missing_archive",
        "reason": "档案缺失：运行目录及其 decision.json 在工作区不存在",
        "evidence": ["modified_tlob/training_duration_20261015.json",
                     "runs/legacy_caliber.json#unfrozen"],
        "counts_as_index_entry": False,
    },
    {
        "path": "runs/training-progression",
        "kind": "missing_archive",
        "reason": "档案缺失：运行目录及其 decision.json 在工作区不存在",
        "evidence": ["modified_tlob/training_progression_20261015.json",
                     "runs/legacy_caliber.json#unfrozen"],
        "counts_as_index_entry": False,
    },
    {
        "path": "runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b/windows/jan2024/tlob_s29",
        "kind": "derived_artifact",
        "reason": "子窗口重放产物没有独立 manifest（父 run 有 manifest 并已作为一条运行登记）；"
                  "不生成索引条目，避免把派生窗口当成一次独立实验",
        "evidence": ["runs/multiscale-replay-20261026/20261026T133118.869042Z_bcdc779dc66b/manifest.json",
                     "runs/frozen/a-20261026/sha256_index.json"],
        "counts_as_index_entry": False,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relative_posix(path: Path, root: Path) -> str:
    """Lexical relative POSIX path (does not touch the filesystem)."""
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:
        return path.absolute().as_posix()


def display_path(path: Path) -> str:
    """Repository-relative POSIX path when possible, else the absolute path."""
    try:
        return path.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return path.absolute().as_posix()


def canonical_config_sha256(document) -> str:
    """Hash convention used by the run manifests for their config snapshot."""
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_date(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()[:10]
    try:
        datetime.strptime(text, DATE_FORMAT)
    except ValueError:
        return None
    return text


def interval_overlaps(start, end, interval: tuple[str, str]) -> bool | None:
    start_date, end_date = parse_date(start), parse_date(end)
    if start_date is None or end_date is None:
        return None
    return start_date < interval[1] and end_date > interval[0]


def parse_timestamp(value) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def dig(document, dotted: str):
    node = document
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
        if node is None:
            return None
    return node


def discover_manifests(root: Path) -> list[Path]:
    """Return every manifest under ``root`` once, without following symlinks."""
    found: list[Path] = []
    root_key = os.path.normcase(os.path.normpath(str(root)))
    for directory, subdirs, files in os.walk(root, followlinks=False):
        at_root = os.path.normcase(os.path.normpath(directory)) == root_key
        subdirs[:] = sorted(
            name for name in subdirs
            if not (at_root and name in EXCLUDED_TOP_LEVEL))
        if "manifest.json" in files:
            found.append(Path(directory) / "manifest.json")
    return sorted(found, key=lambda path: relative_posix(path, root))


def normalize_status(status) -> str:
    if not isinstance(status, str) or not status:
        return "unknown"
    return STATUS_GROUPS.get(status, "other")


def normalize_artifacts(document, run_dir: Path, root: Path) -> list[dict]:
    raw = document.get("artifacts")
    if not isinstance(raw, list):
        return []
    rows = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        target = run_dir / str(item["path"])
        rows.append({
            "path": display_path(target),
            "declared_path": str(item["path"]),
            "sha256": item.get("sha256"),
            "bytes": item.get("bytes"),
            "exists": target.is_file(),
        })
    return sorted(rows, key=lambda row: row["path"])


def extract_metrics(document, run_dir: Path, root: Path, artifacts: list[dict]) -> dict | None:
    listed = next((row for row in artifacts if row["declared_path"] == "metrics.json"), None)
    path = run_dir / "metrics.json"
    if path.is_file():
        row = {
            "path": display_path(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "source": "run_directory",
        }
        if listed:
            row["manifest_sha256"] = listed["sha256"]
            row["matches_manifest"] = listed["sha256"] == row["sha256"]
        return row
    if listed:
        return {"path": listed["path"], "sha256": listed["sha256"], "bytes": listed["bytes"],
                "source": "manifest_artifacts", "exists": False}
    return None


def extract_failure(document, run_dir: Path, root: Path) -> dict | None:
    error = document.get("error")
    reason = None
    if isinstance(error, dict):
        reason = error.get("reason") or error.get("message") or canonical_json(error)
    elif isinstance(error, str):
        reason = error
    record = None
    path = run_dir / "failure.json"
    if path.is_file():
        record = {"path": display_path(path), "sha256": sha256_file(path),
                  "bytes": path.stat().st_size}
        if reason is None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                reason = payload.get("reason") or payload.get("message")
    if reason is None and record is None:
        return None
    return {"reason": reason, "error": error, "record": record}


def extract_code(document, run_dir: Path, root: Path) -> dict:
    code = document.get("code") if isinstance(document.get("code"), dict) else {}
    status_text = code.get("working_tree_status") or ""
    snapshot = code.get("snapshot")
    snapshot_path = None
    if snapshot:
        target = run_dir / str(snapshot)
        snapshot_path = display_path(target) if target.is_file() else None
    return {
        "commit": code.get("commit"),
        "dirty": bool(str(status_text).strip()),
        "dirty_fingerprint": code.get("snapshot_sha256"),
        "snapshot_path": snapshot_path,
        "declared_snapshot": snapshot,
        "content_sha256": code.get("content_sha256"),
        "working_tree_status_sha256": (hashlib.sha256(status_text.encode("utf-8")).hexdigest()
                                       if status_text else None),
        "working_tree_status_lines": len(status_text.splitlines()) if status_text else 0,
    }


def extract_splits(document) -> dict:
    data = document.get("data_version") if isinstance(document.get("data_version"), dict) else {}
    validation = {key: document.get(key) for key in ("validation_start", "validation_end")}
    final = {key: document.get(key) for key in ("final_start", "final_end")}
    target = {key: document.get(key) for key in ("start_date", "end_date")}
    return {
        "evaluation_split": document.get("evaluation_split"),
        "train": {"start": data.get("requested_start"), "end": data.get("requested_end")},
        "validation": validation if any(value is not None for value in validation.values()) else None,
        "final": final if any(value is not None for value in final.values()) else None,
        "target": target if any(value is not None for value in target.values()) else None,
        "window": document.get("window"),
        "period": document.get("period"),
        "month": document.get("month"),
    }


def read_metrics_summary(path: Path) -> dict:
    summary = {"day_count": None, "valid_days": None, "readable": False}
    if not path.is_file():
        return summary
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return summary
    if not isinstance(document, dict):
        return summary
    summary["readable"] = True
    summary["day_count"] = document.get("day_count")
    summary["valid_days"] = document.get("valid_days")
    return summary


def extract_declared(document, run_dir: Path, root: Path) -> dict:
    """Read the run's own config snapshot: declared boundaries are not access evidence."""
    configs = sorted(path for path in run_dir.glob("config*.json") if path.is_file())
    empty = {
        "present": False,
        "config_snapshots": [],
        "config_snapshot": None,
        "splits": {"train": None, "validation": None, "final": None},
        "evaluation_split": None,
        "final_access": None,
        "data": {"version": None, "source": None, "allowed_start": None, "allowed_end": None},
        "final_overlaps_sealed": None,
        "declaration_note": "a declared boundary is not evidence that the data was read or evaluated",
    }
    if not configs:
        return empty
    chosen = next((path for path in configs if path.name == "config.json"), configs[0])
    payload = json.loads(chosen.read_text(encoding="utf-8"))
    raw_sha = sha256_file(chosen)
    canonical_sha = canonical_config_sha256(payload)
    manifest_config_sha256 = document.get("config_sha256")
    splits = payload.get("splits") if isinstance(payload.get("splits"), dict) else {}

    def bound(node):
        if not isinstance(node, dict):
            return None
        start, end = node.get("start"), node.get("end")
        if start is None and end is None:
            return None
        return {"start": start, "end": end}

    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    final = bound(splits.get("final"))
    return {
        "present": True,
        "config_snapshots": [
            {"path": display_path(path), "sha256": sha256_file(path), "selected": path == chosen}
            for path in configs],
        "config_snapshot": {
            "path": display_path(chosen),
            "sha256": raw_sha,
            "canonical_sha256": canonical_sha,
            "bytes": chosen.stat().st_size,
            "manifest_config_sha256": manifest_config_sha256,
            "matches_manifest_config_sha256": (
                manifest_config_sha256 == canonical_sha if manifest_config_sha256 else None),
        },
        "splits": {"train": bound(splits.get("train")),
                   "validation": bound(splits.get("validation")),
                   "final": final},
        "evaluation_split": payload.get("evaluation_split"),
        "final_access": payload.get("final_access"),
        "data": {"version": data.get("version"), "source": data.get("source"),
                 "allowed_start": data.get("allowed_start"), "allowed_end": data.get("allowed_end")},
        "final_overlaps_sealed": (
            interval_overlaps(final["start"], final["end"], SEALED_INTERVAL) if final else None),
        "declaration_note": "a declared boundary is not evidence that the data was read or evaluated",
    }


def extract_evaluated(run_dir: Path, root: Path, metrics_summary: dict) -> dict:
    """Read the dates actually present in the run's own daily artifacts."""
    files = sorted(run_dir.glob("daily*.csv"))
    base = {
        "determination": "unknown",
        "from": None,
        "to": None,
        "days": None,
        "dates_sha256": None,
        "source_files": [],
        "cross_check": {},
        "reason": "no_daily_artifact",
        "sealed": {"interval": list(SEALED_INTERVAL), "intersects": None,
                   "intersecting_dates": [], "after_sealed_end": [],
                   "reason": "evaluated_dates_unknown"},
    }
    if not files:
        return base
    dates: set[str] = set()
    source_files: list[dict] = []
    unparseable = 0
    for path in files:
        parsed: set[str] = set()
        rows = 0
        error = None
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "date" not in fields:
                error = "missing_date_column"
            else:
                for row in reader:
                    rows += 1
                    date = parse_date(row.get("date"))
                    if date is None:
                        unparseable += 1
                    else:
                        parsed.add(date)
        dates |= parsed
        source_files.append({"path": display_path(path), "sha256": sha256_file(path),
                             "bytes": path.stat().st_size, "rows": rows, "dates": len(parsed),
                             "date_min": min(parsed) if parsed else None,
                             "date_max": max(parsed) if parsed else None, "error": error})
    cross = {"metrics_day_count": metrics_summary.get("day_count"),
             "metrics_valid_days": metrics_summary.get("valid_days"),
             "metrics_readable": metrics_summary.get("readable", False),
             "matches_day_count": None}
    if not dates:
        return {**base, "source_files": source_files, "cross_check": cross,
                "unparseable_rows": unparseable,
                "reason": "daily_artifact_without_parseable_dates"}
    ordered = sorted(dates)
    determination = "partial" if unparseable else "exact"
    reason = "unparseable_daily_rows" if unparseable else None
    day_count = cross["metrics_day_count"]
    if isinstance(day_count, int):
        cross["matches_day_count"] = day_count == len(ordered)
        if day_count > len(ordered):
            determination = "partial"
            reason = "metrics_day_count_exceeds_daily_dates"
    inside = [date for date in ordered if SEALED_INTERVAL[0] <= date < SEALED_INTERVAL[1]]
    after = [date for date in ordered if date >= SEALED_INTERVAL[1]]
    return {
        "determination": determination,
        "from": ordered[0],
        "to": ordered[-1],
        "days": len(ordered),
        "dates_sha256": hashlib.sha256(chr(10).join(ordered).encode("utf-8")).hexdigest(),
        "source_files": source_files,
        "unparseable_rows": unparseable,
        "cross_check": cross,
        "reason": reason,
        "sealed": {"interval": list(SEALED_INTERVAL), "intersects": bool(inside),
                   "intersecting_dates": inside, "after_sealed_end": after,
                   "reason": "evaluated_dates_known"},
    }


def duration_seconds(started_at, ended_at) -> float | None:
    start, end = parse_timestamp(started_at), parse_timestamp(ended_at)
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def extract_entry(manifest_path: Path, root: Path) -> dict:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    manifest_rel = display_path(manifest_path)
    artifacts = normalize_artifacts(document, run_dir, root)
    metrics = extract_metrics(document, run_dir, root, artifacts)
    metrics_summary = read_metrics_summary(run_dir / "metrics.json")
    declared = extract_declared(document, run_dir, root)
    evaluated = extract_evaluated(run_dir, root, metrics_summary)
    data_version = document.get("data_version") if isinstance(document.get("data_version"),
                                                               dict) else {}
    data = {
        "source": data_version.get("source"),
        "version": data_version.get("version"),
        "sha256": data_version.get("data_sha256"),
        "requested_start": data_version.get("requested_start"),
        "requested_end": data_version.get("requested_end"),
        "bars_start": data_version.get("bars_start"),
        "declared_version": declared["data"]["version"],
        "declared_source": declared["data"]["source"],
        "backfilled_from": None,
    }
    backfilled = []
    for name in ("version", "source"):
        if data[name] is None and declared["data"][name] is not None:
            data[name] = declared["data"][name]
            backfilled.append(name)
    if backfilled:
        data["backfilled_from"] = "run_config_snapshot"
    splits = extract_splits(document)
    if declared["present"]:
        for name in ("train", "validation", "final"):
            if declared["splits"][name] is not None:
                splits[name] = declared["splits"][name]
        if splits["evaluation_split"] is None:
            splits["evaluation_split"] = declared["evaluation_split"]
        splits["declared_source"] = declared["config_snapshot"]
        splits["declaration_note"] = declared["declaration_note"]
    entry = {
        "index_key": manifest_rel,
        "manifest_path": manifest_rel,
        "run_dir": display_path(run_dir),
        "run_id": document.get("run_id"),
        "experiment_id": document.get("experiment_id"),
        "attempt_index": None,
        "experiment_attempts": None,
        "status": document.get("status"),
        "status_group": normalize_status(document.get("status")),
        "started_at": document.get("started_at"),
        "ended_at": document.get("ended_at"),
        "duration_seconds": duration_seconds(document.get("started_at"),
                                             document.get("ended_at")),
        "exit_code": document.get("exit_code"),
        "code": extract_code(document, run_dir, root),
        "config_sha256": document.get("config_sha256"),
        "plan_sha256": document.get("plan_sha256"),
        "seed": document.get("seed"),
        "data": data,
        "splits": splits,
        "declared": declared,
        "evaluated": evaluated,
        "metrics": metrics,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "failure": extract_failure(document, run_dir, root),
        "manifest_schema_version": document.get("schema_version"),
        "manifest_kind": document.get("type") or document.get("kind"),
        "is_test_fixture": "pytest-base-temp" in manifest_rel.split("/"),
        "companions": {
            "metrics_json": (run_dir / "metrics.json").is_file(),
            "failure_json": (run_dir / "failure.json").is_file(),
            "decision_json": (run_dir / "decision.json").is_file(),
            "events_jsonl": (run_dir / "events.jsonl").is_file(),
            "config_json": declared["present"],
            "daily_csv_count": len(list(run_dir.glob("daily*.csv"))),
        },
        "missing_fields": [],
    }
    entry["missing_fields"] = missing_fields(entry)
    return entry


def field_value(entry: dict, dotted: str):
    node = entry
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def missing_fields(entry: dict) -> list[str]:
    missing = [name for name in REQUIRED_FIELDS
               if field_value(entry, name) in (None, "", [])]
    failure = entry.get("failure")
    if entry.get("status") in FAILURE_STATUSES and not (failure and failure.get("reason")):
        missing.append("failure.reason")
    return sorted(set(missing))


def assign_attempts(entries: list[dict]) -> None:
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        experiment_id = entry.get("experiment_id")
        if experiment_id:
            groups.setdefault(experiment_id, []).append(entry)
    for rows in groups.values():
        rows.sort(key=lambda row: (row.get("started_at") or "", row["index_key"]))
        for position, entry in enumerate(rows, start=1):
            entry["attempt_index"] = position
            entry["experiment_attempts"] = len(rows)


def status_counts(entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry.get("status") if entry.get("status") is not None else "null"
        counts[str(key)] = counts.get(str(key), 0) + 1
    return dict(sorted(counts.items()))


def group_counts(entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status_group"]] = counts.get(entry["status_group"], 0) + 1
    return dict(sorted(counts.items()))


def experiment_summary(entries: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for entry in entries:
        experiment_id = entry.get("experiment_id")
        if not experiment_id or entry["is_test_fixture"]:
            continue
        row = rows.setdefault(experiment_id, {
            "experiment_id": experiment_id, "attempts": 0, "statuses": {},
            "first_started_at": None, "last_ended_at": None,
            "run_ids": [],
        })
        row["attempts"] += 1
        status = entry.get("status") if entry.get("status") is not None else "null"
        row["statuses"][str(status)] = row["statuses"].get(str(status), 0) + 1
        started = entry.get("started_at")
        if started and (row["first_started_at"] is None or started < row["first_started_at"]):
            row["first_started_at"] = started
        ended = entry.get("ended_at")
        if ended and (row["last_ended_at"] is None or ended > row["last_ended_at"]):
            row["last_ended_at"] = ended
        if entry.get("run_id"):
            row["run_ids"].append(entry["run_id"])
    for row in rows.values():
        row["statuses"] = dict(sorted(row["statuses"].items()))
        row["run_ids"] = sorted(set(row["run_ids"]))
    return [rows[key] for key in sorted(rows)]


def viewed_window_record(entry: dict) -> dict:
    evaluated = entry["evaluated"]
    declared = entry["declared"]
    sealed = evaluated["sealed"]
    return {
        "run_id": entry["run_id"],
        "experiment_id": entry["experiment_id"],
        "manifest_path": entry["manifest_path"],
        "is_test_fixture": entry["is_test_fixture"],
        "determination": evaluated["determination"],
        "evaluated_from": evaluated["from"],
        "evaluated_to": evaluated["to"],
        "evaluated_days": evaluated["days"],
        "dates_sha256": evaluated["dates_sha256"],
        "source_files": evaluated["source_files"],
        "cross_check": evaluated["cross_check"],
        "reason": evaluated["reason"],
        "sealed_interval": sealed["interval"],
        "intersects_sealed": sealed["intersects"],
        "intersecting_dates": sealed["intersecting_dates"],
        "after_sealed_end": sealed["after_sealed_end"],
        "declared_final": declared["splits"]["final"],
        "declares_sealed_final": declared["final_overlaps_sealed"],
        "config_snapshot": declared["config_snapshot"],
    }


def build_viewed_windows(entries: list[dict]) -> dict:
    """Research-area registry: what each run actually evaluated, versus what it declared."""
    records = [viewed_window_record(entry) for entry in entries]
    exact = [row for row in records if row["determination"] == "exact"]
    partial = [row for row in records if row["determination"] == "partial"]
    unknown = [row for row in records if row["determination"] == "unknown"]
    hits = [row for row in records if row["intersects_sealed"] is True]
    declared_only = [row for row in records
                     if row["declares_sealed_final"] is True and row["intersects_sealed"] is not True]
    beyond = [row for row in records if row["after_sealed_end"]]
    declared_true = [row for row in records if row["declares_sealed_final"] is True]
    counts = {
        "total": len(records),
        "exact": len(exact),
        "partial": len(partial),
        "unknown": len(unknown),
        "intersects_sealed_true": len(hits),
        "intersects_sealed_false": sum(1 for row in records if row["intersects_sealed"] is False),
        "intersects_sealed_unknown": sum(1 for row in records
                                         if row["intersects_sealed"] is None),
        "declared_sealed_final": len(declared_true),
        "declared_sealed_final_without_actual_intersection": len(declared_only),
        "declared_sealed_final_observed_false": sum(
            1 for row in declared_true if row["intersects_sealed"] is False),
        "declared_sealed_final_unknown": sum(
            1 for row in declared_true if row["intersects_sealed"] is None),
        "after_sealed_end": len(beyond),
    }
    configs = [row["config_snapshot"] for row in records if row["config_snapshot"]]
    coverage = {
        "config_snapshots": len(configs),
        "config_snapshot_matched_manifest": sum(
            1 for row in configs if row["matches_manifest_config_sha256"] is True),
        "config_snapshot_mismatched_manifest": sum(
            1 for row in configs if row["matches_manifest_config_sha256"] is False),
        "config_snapshot_unlinked_manifest": sum(
            1 for row in configs if row["matches_manifest_config_sha256"] is None),
        "daily_artifact_runs": len(exact) + len(partial),
        "daily_artifact_files": sum(len(row["source_files"]) for row in records),
    }
    if hits:
        wording = (f"实际评估日期与封存区间相交的运行 {len(hits)} 条（见 intersects_sealed）；"
                   f"另有 {counts['unknown']} 条 unknown 无逐日证据、未验证。")
    else:
        wording = ("在可确定子集内未发现读取封存区间：exact "
                   f"{counts['exact']} / partial {counts['partial']} 条均无相交日期；"
                   f"{counts['unknown']} 条 unknown 无逐日证据，未验证，不能据此断言全体未读取。"
                   "配置中声明的 final 区间（declared_final）不是读取证据。")
    return {
        "schema_version": 1,
        "kind": "research_area_registry",
        "sealed_interval": list(SEALED_INTERVAL),
        "sealed_interval_semantics": "half-open [start, end); determined from daily artifacts, never from config declarations",
        "scope": "primary evidence = the run directory's own config*.json and daily*.csv; "
                 "derived reports nested under report/ or windows/ and fixture archives are not "
                 "treated as the run's primary evaluation artifact",
        "wording": wording,
        "counts": counts,
        "coverage": coverage,
        "runs": records,
        "intersects_sealed": hits,
        "declared_sealed_final_but_not_observed": declared_only,
        "unknown": [{"run_id": row["run_id"], "experiment_id": row["experiment_id"],
                     "manifest_path": row["manifest_path"], "reason": row["reason"]}
                    for row in unknown],
        "after_sealed_end": [{"run_id": row["run_id"], "manifest_path": row["manifest_path"],
                              "is_test_fixture": row["is_test_fixture"],
                              "dates": row["after_sealed_end"]} for row in beyond],
    }


def write_json(document, path: Path) -> str:
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1) + chr(10)).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def gaps_summary(entries: list[dict], root: Path) -> dict:
    per_field: dict[str, int] = {}
    affected = 0
    for entry in entries:
        if entry["missing_fields"]:
            affected += 1
        for name in entry["missing_fields"]:
            per_field[name] = per_field.get(name, 0) + 1
    missing_archives: list[dict] = []
    manifestless_artifacts: list[dict] = []
    if is_repository_runs(root):
        for row in MISSING_ARCHIVES:
            target = ROOT / row["path"]
            present = target.exists()
            if row.get("kind") == "derived_artifact":
                if not present or not any(target.rglob("manifest.json")):
                    manifestless_artifacts.append({**row, "exists": present})
            elif not present:
                missing_archives.append(dict(row))
    return {
        "entries_with_gaps": affected,
        "entries_without_gaps": len(entries) - affected,
        "total_gap_instances": sum(per_field.values()),
        "per_field": dict(sorted(per_field.items())),
        "fields_missing_for_all_entries": sorted(
            name for name, count in per_field.items() if count == len(entries)),
        "missing_archives": missing_archives,
        "missing_archives_count": len(missing_archives),
        "manifestless_artifacts": manifestless_artifacts,
        "manifestless_artifacts_count": len(manifestless_artifacts),
        "unindexed_total": len(missing_archives) + len(manifestless_artifacts),
        "note": "gaps record what the manifest does not state; nothing is backfilled or "
                "inferred, and missing archives or manifest-less derived artifacts yield no "
                "synthetic index entries",
    }


def is_repository_runs(root: Path) -> bool:
    return (ROOT / "runs").resolve() == root.resolve()


def snapshot_summary(root: Path) -> dict | None:
    path = ROOT / FROZEN_SNAPSHOT_INDEX
    if not path.is_file() or not is_repository_runs(root):
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    frozen_root = root / "frozen"
    frozen_manifests = discover_manifests(frozen_root) if frozen_root.is_dir() else []
    frozen_status = {}
    for manifest in frozen_manifests:
        try:
            status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = "unreadable"
        key = status if status is not None else "null"
        frozen_status[str(key)] = frozen_status.get(str(key), 0) + 1
    return {
        "path": FROZEN_SNAPSHOT_INDEX,
        "sha256": sha256_file(path),
        "totals": index.get("totals"),
        "frozen_manifest_count": len(frozen_manifests),
        "frozen_manifest_statuses": dict(sorted(frozen_status.items())),
        "note": "frozen copies are not experiments and stay out of the index and status counts; "
                "their running/failed statuses are frozen history as-is",
    }


def build_summary(entries: list[dict], root: Path, viewed: dict | None = None) -> dict:
    fixtures = [entry for entry in entries if entry["is_test_fixture"]]
    experiments = experiment_summary(entries)
    viewed = viewed or build_viewed_windows(entries)
    determinations = {key: viewed["counts"][key] for key in ("exact", "partial", "unknown")}
    return {
        "schema_version": SCHEMA_VERSION,
        "caliber": "manifest.json is the fact source; run-internal config snapshots and daily "
                   "artifacts backfill declared boundaries and actually evaluated dates; "
                   "deterministic rebuild; no manual backfill",
        "root": display_path(root),
        "entries": {
            "total": len(entries),
            "test_fixtures": len(fixtures),
            "historical_runs": len(entries) - len(fixtures),
            "top_level_directories": len([p for p in root.iterdir() if p.is_dir()])
            if root.is_dir() else 0,
            "declared_splits_backfilled": sum(1 for entry in entries
                                              if entry["declared"]["present"]),
            "evaluated_dates_determined": determinations["exact"] + determinations["partial"],
            "evaluated_determination": determinations,
        },
        "status_counts": status_counts(entries),
        "status_group_counts": group_counts(entries),
        "experiments": experiments,
        "experiment_count": len(experiments),
        "viewed_windows": {
            "path": VIEWED_WINDOWS_FILENAME,
            "counts": viewed["counts"],
            "coverage": viewed["coverage"],
            "wording": viewed["wording"],
        },
        "gaps": gaps_summary(entries, root),
        "snapshots": snapshot_summary(root),
        "excluded": {
            "paths": [f"{display_path(root)}/{name}" for name in EXCLUDED_TOP_LEVEL],
            "reason": "frozen/ = stage-A snapshot duplicates; index/ = this index's output",
        },
    }


def write_jsonl(entries: list[dict], path: Path) -> str:
    payload = "".join(canonical_json(entry) + "\n" for entry in entries).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


SCHEMA_SQL = """
CREATE TABLE runs (
    index_key TEXT PRIMARY KEY,
    manifest_path TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    run_id TEXT,
    experiment_id TEXT,
    attempt_index INTEGER,
    experiment_attempts INTEGER,
    status TEXT,
    status_group TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    duration_seconds REAL,
    exit_code INTEGER,
    code_commit TEXT,
    code_dirty INTEGER,
    code_dirty_fingerprint TEXT,
    code_snapshot_path TEXT,
    config_sha256 TEXT,
    plan_sha256 TEXT,
    seed INTEGER,
    data_source TEXT,
    data_version TEXT,
    data_sha256 TEXT,
    split_evaluation TEXT,
    split_train_start TEXT,
    split_train_end TEXT,
    split_validation_start TEXT,
    split_validation_end TEXT,
    split_final_start TEXT,
    split_final_end TEXT,
    evaluated_from TEXT,
    evaluated_to TEXT,
    evaluated_days INTEGER,
    dates_sha256 TEXT,
    determination TEXT NOT NULL,
    intersects_sealed INTEGER,
    declared_final_start TEXT,
    declared_final_end TEXT,
    declares_sealed_final INTEGER,
    config_snapshot_path TEXT,
    config_snapshot_sha256 TEXT,
    config_snapshot_matches INTEGER,
    metrics_path TEXT,
    metrics_sha256 TEXT,
    artifact_count INTEGER NOT NULL,
    failure_reason TEXT,
    is_test_fixture INTEGER NOT NULL,
    missing_fields TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE gaps (
    index_key TEXT NOT NULL,
    field TEXT NOT NULL,
    PRIMARY KEY (index_key, field)
);
CREATE INDEX runs_experiment_idx ON runs (experiment_id);
CREATE INDEX runs_status_idx ON runs (status);
CREATE INDEX runs_started_idx ON runs (started_at);
"""


def sqlite_row(entry: dict) -> tuple:
    splits = entry["splits"]
    train, validation, final = splits["train"] or {}, splits["validation"] or {}, splits["final"] or {}
    evaluated = entry["evaluated"]
    sealed = evaluated["sealed"]
    declared = entry["declared"]
    config = declared["config_snapshot"] or {}
    metrics = entry["metrics"] or {}
    failure = entry["failure"] or {}
    code = entry["code"]
    data = entry["data"]
    intersects = sealed["intersects"]
    return (
        entry["index_key"], entry["manifest_path"], entry["run_dir"], entry["run_id"],
        entry["experiment_id"], entry["attempt_index"], entry["experiment_attempts"],
        entry["status"], entry["status_group"], entry["started_at"], entry["ended_at"],
        entry["duration_seconds"], entry["exit_code"], code["commit"], int(code["dirty"]),
        code["dirty_fingerprint"], code["snapshot_path"], entry["config_sha256"],
        entry["plan_sha256"], entry["seed"], data["source"], data["version"], data["sha256"],
        splits["evaluation_split"], train.get("start"), train.get("end"),
        validation.get("start"), validation.get("end"), final.get("start"), final.get("end"),
        evaluated["from"], evaluated["to"], evaluated["days"], evaluated["dates_sha256"],
        evaluated["determination"], None if intersects is None else int(intersects),
        (declared["splits"]["final"] or {}).get("start"),
        (declared["splits"]["final"] or {}).get("end"),
        None if declared["final_overlaps_sealed"] is None
        else int(declared["final_overlaps_sealed"]),
        config.get("path"), config.get("sha256"),
        None if config.get("matches_manifest_config_sha256") is None
        else int(config["matches_manifest_config_sha256"]),
        metrics.get("path"), metrics.get("sha256"), entry["artifact_count"], failure.get("reason"),
        int(entry["is_test_fixture"]), canonical_json(entry["missing_fields"]),
        canonical_json(entry),
    )


def build_sqlite(entries: list[dict], path: Path) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA_SQL)
        rows = [sqlite_row(entry) for entry in entries]
        if rows:
            connection.executemany(
                "INSERT INTO runs VALUES (" + ",".join("?" * len(rows[0])) + ")", rows)
        connection.executemany(
            "INSERT INTO gaps VALUES (?,?)",
            [(entry["index_key"], name) for entry in entries for name in entry["missing_fields"]])
        connection.commit()
    finally:
        connection.close()


def build_index(root: Path, output_dir: Path) -> dict:
    if not root.is_dir():
        raise SystemExit(f"runs root missing: {root}")
    entries = [extract_entry(path, root) for path in discover_manifests(root)]
    assign_attempts(entries)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_sha = write_jsonl(entries, output_dir / INDEX_FILENAME)
    build_sqlite(entries, output_dir / SQLITE_FILENAME)
    viewed = build_viewed_windows(entries)
    viewed_sha = write_json(viewed, output_dir / VIEWED_WINDOWS_FILENAME)
    summary = build_summary(entries, root, viewed)
    summary["artifacts"] = {
        "jsonl": {"path": INDEX_FILENAME, "entries": len(entries),
                  "bytes": (output_dir / INDEX_FILENAME).stat().st_size, "sha256": index_sha},
        "sqlite": {"path": SQLITE_FILENAME,
                   "bytes": (output_dir / SQLITE_FILENAME).stat().st_size,
                   "sha256": sha256_file(output_dir / SQLITE_FILENAME)},
        "viewed_windows": {"path": VIEWED_WINDOWS_FILENAME,
                          "bytes": (output_dir / VIEWED_WINDOWS_FILENAME).stat().st_size,
                          "sha256": viewed_sha},
    }
    (output_dir / SUMMARY_FILENAME).write_bytes(
        (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8"))
    return summary


def read_entries(output_dir: Path) -> list[dict]:
    path = output_dir / INDEX_FILENAME
    if not path.is_file():
        raise SystemExit(f"index not built: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _coerce_status(status: list[str] | None) -> set[str] | None:
    if not status:
        return None
    values: set[str] = set()
    for item in status:
        values.update(part.strip() for part in item.split(",") if part.strip())
    return values or None


def query_records(records: list[dict], experiment_id: str | None = None,
                  status: list[str] | None = None, run_id: str | None = None,
                  since: str | datetime | None = None, until: str | datetime | None = None,
                  determination: list[str] | None = None,
                  intersects_sealed: str | None = None) -> list[dict]:
    """Filter entries by experiment/run/status/started_at window, determination and sealed flag."""
    statuses = _coerce_status(status)
    determinations = _coerce_status(determination)
    sealed_filter = None
    if intersects_sealed is not None:
        sealed_filter = {"true": True, "false": False, "unknown": None}.get(
            str(intersects_sealed).strip().lower())
        if sealed_filter is False and str(intersects_sealed).strip().lower() != "false":
            raise SystemExit("--intersects-sealed must be true, false or unknown")
    since_at = parse_timestamp(since) if isinstance(since, str) else since
    until_at = parse_timestamp(until) if isinstance(until, str) else until
    matched = []
    for record in records:
        if experiment_id and record.get("experiment_id") != experiment_id:
            continue
        if run_id and record.get("run_id") != run_id:
            continue
        record_status = record.get("status") if record.get("status") is not None else "null"
        if statuses and str(record_status) not in statuses:
            continue
        record_determination = record.get("evaluated", {}).get("determination")
        if determinations and record_determination not in determinations:
            continue
        if intersects_sealed is not None:
            observed = record.get("evaluated", {}).get("sealed", {}).get("intersects")
            if observed is not sealed_filter:
                continue
        if since_at or until_at:
            started = parse_timestamp(record.get("started_at"))
            if started is None:
                continue
            if since_at and started < since_at:
                continue
            if until_at and started >= until_at:
                continue
        matched.append(record)
    return matched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "runs",
                        help="manifest root (default: runs/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/index")
    parser.add_argument("--query", action="store_true", help="query the built index instead of rebuilding")
    parser.add_argument("--experiments", action="store_true", help="print the experiment summary")
    parser.add_argument("--experiment-id")
    parser.add_argument("--run-id")
    parser.add_argument("--status", action="append", help="status filter; repeat or comma-separate")
    parser.add_argument("--determination", action="append",
                        help="evaluated-date determination filter: exact/partial/unknown")
    parser.add_argument("--intersects-sealed", choices=("true", "false", "unknown"),
                        help="filter by observed sealed-interval intersection")
    parser.add_argument("--since", help="inclusive started_at lower bound (UTC ISO)")
    parser.add_argument("--until", help="exclusive started_at upper bound (UTC ISO)")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.query or args.experiments:
        entries = read_entries(output_dir)
        if args.experiments:
            summary = json.loads((output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
            print(json.dumps({"experiment_count": summary["experiment_count"],
                              "experiments": summary["experiments"]},
                             ensure_ascii=False, indent=1))
            return 0
        matched = query_records(entries, experiment_id=args.experiment_id, status=args.status,
                                run_id=args.run_id, since=args.since, until=args.until,
                                determination=args.determination,
                                intersects_sealed=args.intersects_sealed)
        if args.limit:
            matched = matched[:args.limit]
        for entry in matched:
            print(canonical_json(entry))
        return 0
    root = args.root if args.root.is_absolute() else ROOT / args.root
    summary = build_index(root, output_dir)
    print(f"indexed {summary['entries']['total']} manifests -> {output_dir}")
    print(f"status_counts={summary['status_counts']}")
    print(f"gaps: entries={summary['gaps']['entries_with_gaps']} "
          f"instances={summary['gaps']['total_gap_instances']} "
          f"missing_archives={summary['gaps']['missing_archives_count']}")
    print(f"evaluated_determination={summary['entries']['evaluated_determination']} "
          f"declared_splits_backfilled={summary['entries']['declared_splits_backfilled']}")
    print(f"viewed_windows: {summary['viewed_windows']['wording']}")
    print(f"jsonl_sha256={summary['artifacts']['jsonl']['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
