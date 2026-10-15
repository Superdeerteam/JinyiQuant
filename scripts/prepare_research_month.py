"""Stream one monthly selection from a TAR archive with bounded output and time."""

import argparse
import hashlib
import json
import re
import tarfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prepare_month(archive_path, output_root, month, max_bytes=8 * 1024**3, wall_seconds=1200):
    if not re.fullmatch(r"\d{4}(0[1-9]|1[0-2])", month):
        raise ValueError("month must be YYYYMM")
    if max_bytes <= 0 or wall_seconds <= 0:
        raise ValueError("Output and time budgets must be positive")
    archive_path = Path(archive_path).resolve(strict=True)
    source_stat = archive_path.stat()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    bars = output_root / "bars"
    bars.mkdir()
    started = time.monotonic()
    manifest = {
        "schema_version": 1, "run_id": uuid.uuid4().hex,
        "experiment_id": f"prepare-research-{month}", "started_at_utc": utc_now(),
        "status": "running", "month": month, "full_archive_scan": False,
        "source": {"path": str(archive_path), "bytes": source_stat.st_size,
                   "mtime_ns": source_stat.st_mtime_ns, "verification": "stat_and_member_sha256"},
        "budget": {"max_retained_bytes": max_bytes, "wall_seconds": wall_seconds},
        "compressed_bytes_read": 0, "members_seen": 0, "retained_bytes": 0, "artifacts": [],
    }
    manifest_path = output_root / "manifest.json"
    progress_at = started

    def save_manifest():
        temporary = output_root / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    with (output_root / "events.jsonl").open("x", encoding="utf-8") as events:
        def event(kind, **context):
            record = {"schema_version": 1, "timestamp_utc": utc_now(),
                      "level": "ERROR" if kind == "failed" else "INFO", "stage": "preparation",
                      "event_type": kind, "run_id": manifest["run_id"],
                      "experiment_id": manifest["experiment_id"],
                      "elapsed_seconds": round(time.monotonic() - started, 3), **context}
            line = json.dumps(record, ensure_ascii=False)
            events.write(line + "\n")
            events.flush()
            print(line, flush=True)

        def check_budget():
            nonlocal progress_at
            now = time.monotonic()
            if now - started > wall_seconds:
                raise TimeoutError(f"Preparation exceeded {wall_seconds} seconds")
            if now - progress_at >= 15:
                event("progress", compressed_bytes_read=manifest["compressed_bytes_read"],
                      archive_bytes=source_stat.st_size, members_seen=manifest["members_seen"],
                      retained_bytes=manifest["retained_bytes"])
                progress_at = now

        class BudgetReader:
            def __init__(self, raw):
                self.raw = raw

            def read(self, size):
                check_budget()
                chunk = self.raw.read(size)
                manifest["compressed_bytes_read"] += len(chunk)
                return chunk

        save_manifest()
        event("started", source=manifest["source"], month=month, budget=manifest["budget"])
        try:
            pattern = re.compile(rf"{month}(?:\.\d+)?\.feather")
            with archive_path.open("rb") as raw:
                with tarfile.open(fileobj=BudgetReader(raw), mode="r|gz", bufsize=1024 * 1024) as archive:
                    for member in archive:
                        check_budget()
                        manifest["members_seen"] += 1
                        name = PurePosixPath(member.name)
                        if not pattern.fullmatch(name.name):
                            continue
                        if (not member.isfile() or name.is_absolute() or ".." in name.parts
                                or "\\" in member.name or ":" in member.name):
                            raise ValueError(f"Unsafe selected member: {member.name}")
                        if manifest["retained_bytes"] + member.size > max_bytes:
                            raise ValueError("Selected month exceeds retained-byte budget")
                        target = bars / name.name
                        if target.exists():
                            raise ValueError(f"Duplicate member basename: {name.name}")
                        partial = target.with_suffix(target.suffix + ".partial")
                        manifest["active_member"] = {"member": member.name, "partial_path": str(partial)}
                        save_manifest()
                        event("member_started", member=member.name, bytes=member.size)
                        source = archive.extractfile(member)
                        if source is None:
                            raise ValueError(f"Member cannot be read: {member.name}")
                        digest = hashlib.sha256()
                        written = 0
                        with source, partial.open("xb") as destination:
                            while chunk := source.read(4 * 1024 * 1024):
                                check_budget()
                                written += len(chunk)
                                if written > member.size or manifest["retained_bytes"] + written > max_bytes:
                                    raise ValueError("Member exceeded its declared size or retained-byte budget")
                                destination.write(chunk)
                                digest.update(chunk)
                        if written != member.size:
                            raise ValueError(f"Truncated member: {member.name}")
                        partial.rename(target)
                        artifact = {"member": member.name, "path": str(target), "bytes": written,
                                    "sha256": digest.hexdigest()}
                        manifest["artifacts"].append(artifact)
                        manifest["retained_bytes"] += written
                        manifest.pop("active_member", None)
                        save_manifest()
                        event("member_saved", **artifact)
            manifest["full_archive_scan"] = True
            after = archive_path.stat()
            if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source archive changed during preparation")
            if not manifest["artifacts"]:
                raise ValueError(f"No feather members found for month {month}")
            manifest["status"] = "succeeded"
            event("succeeded", files=len(manifest["artifacts"]), retained_bytes=manifest["retained_bytes"])
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc),
                                 "traceback": traceback.format_exc(),
                                 "recovery": "Preserve this failed archive; rerun into a new output directory"}
            event("failed", **manifest["error"])
            raise
        finally:
            manifest["finished_at_utc"] = utc_now()
            manifest["elapsed_seconds"] = time.monotonic() - started
            save_manifest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--month", required=True)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--wall-seconds", type=float, default=1200)
    args = parser.parse_args()
    prepare_month(args.archive, args.output, args.month, args.max_bytes, args.wall_seconds)


if __name__ == "__main__":
    main()
