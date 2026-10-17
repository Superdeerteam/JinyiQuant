"""Prepare only December 2021 and the twelve research months of 2022."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

MONTHS = ("202112", *(f"2022{month:02d}" for month in range(1, 13)))
BAR_MEMBERS = tuple(f"cpt_jyc_2026_e2e_bar1m/{month}.0.feather" for month in MONTHS)
LABEL_MEMBERS = tuple(
    f"cpt_jyc_2026_vwap/cpt_jyc_2026_vwap_{month[:4]}-{month[4:]}.parquet" for month in MONTHS
)
BAR_ARCHIVE = "cpt_jyc_2026_e2e_bar1m.tar.gz"
LABEL_ARCHIVE = "cpt_jyc_2026_vwap.rar"
STATIC_FILES = ("cpt_jyc_2026_instruments.parquet", "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _copy_artifact(source, target, max_bytes, expected_bytes=None):
    """Write an exclusive regular file; incomplete files retain their .partial suffix."""
    partial = target.with_name(target.name + ".partial")
    if target.exists():
        raise FileExistsError(target)
    digest, written = hashlib.sha256(), 0
    with partial.open("xb") as destination:
        while chunk := source.read(4 * 1024**2):
            written += len(chunk)
            if written > max_bytes or (expected_bytes is not None and written > expected_bytes):
                raise ValueError(f"Member exceeds byte budget: {target.name}")
            destination.write(chunk)
            digest.update(chunk)
    if expected_bytes is not None and written != expected_bytes:
        raise ValueError(f"Truncated member: {target.name}")
    return {"path": str(target), "bytes": written, "sha256": digest.hexdigest()}, partial


def _extract_label(executable, archive, member, target):
    # stdout extraction never lets the archive choose a destination path.
    command = ([executable, "-xOf", str(archive), member] if Path(executable).name.startswith("bsdtar")
               else [executable, "p", "-inul", str(archive), member])
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            with process.stdout:
                artifact, partial = _copy_artifact(process.stdout, target, 64 * 1024**2)
            code = process.wait(timeout=120)
            if code:
                errors.seek(0)
                message = errors.read(8192).decode("utf-8", errors="replace")
                raise RuntimeError(f"Label extraction failed ({code}): {member}: {message}")
            if artifact["bytes"] == 0:
                raise ValueError(f"Empty label member: {member}")
            partial.rename(target)
            return artifact
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def prepare(archives_dir, output_dir):
    archives_dir = Path(archives_dir).resolve(strict=True)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "bars").mkdir()
    (output / "labels").mkdir()
    run_id = uuid.uuid4().hex
    manifest = {"schema_version": 1, "run_id": run_id, "experiment_id": "prepare-annual-baseline-2022",
                "status": "running", "started_at_utc": _now(), "months": list(MONTHS),
                "source_files": [], "artifacts": [], "full_bar_archive_scan": False,
                "reserved_label_members_extracted": 0}

    def save():
        temporary = output / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "manifest.json")

    with (output / "events.jsonl").open("x", encoding="utf-8") as events:
        def event(kind, **context):
            record = {"schema_version": 1, "timestamp_utc": _now(), "level": "ERROR" if kind == "failed" else "INFO",
                      "stage": "preparation", "event_type": kind, "run_id": run_id,
                      "experiment_id": manifest["experiment_id"], "context": context}
            line = json.dumps(record, ensure_ascii=False)
            events.write(line + "\n")
            events.flush()
            print(line, flush=True)

        save()
        event("started", archives_dir=str(archives_dir), output_dir=str(output))
        try:
            for name in (BAR_ARCHIVE, LABEL_ARCHIVE, *STATIC_FILES):
                source = archives_dir / name
                before = source.stat()
                record = {"path": str(source), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                          "sha256": _digest(source)}
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Source changed while hashing: {name}")
                manifest["source_files"].append(record)
                save()
                event("source_hashed", **record)
            seen = set()
            with tarfile.open(archives_dir / BAR_ARCHIVE, mode="r|gz", bufsize=1024**2) as archive:
                for member in archive:
                    if member.name not in BAR_MEMBERS:
                        continue
                    if not member.isfile():
                        raise ValueError(f"Selected member is not a regular file: {member.name}")
                    if member.name in seen:
                        raise ValueError(f"Duplicate selected member: {member.name}")
                    if member.size > 512 * 1024**2:
                        raise ValueError(f"Member exceeds byte budget: {member.name}")
                    source = archive.extractfile(member)
                    with source:
                        artifact, partial = _copy_artifact(
                            source, output / "bars" / Path(member.name).name, 512 * 1024**2, member.size)
                    partial.rename(artifact["path"])
                    artifact["member"] = member.name
                    manifest["artifacts"].append(artifact)
                    seen.add(member.name)
                    save()
                    event("member_saved", **artifact)
            manifest["full_bar_archive_scan"] = True
            if seen != set(BAR_MEMBERS):
                raise ValueError(f"Missing selected bars: {sorted(set(BAR_MEMBERS) - seen)}")
            executable = shutil.which("bsdtar") or shutil.which("unrar")
            if executable is None:
                raise RuntimeError("Install bsdtar (libarchive-tools) or unrar to extract the selected labels")
            command = ([executable, "-tf", str(archives_dir / LABEL_ARCHIVE)]
                       if Path(executable).name.startswith("bsdtar")
                       else [executable, "lb", "-c-", str(archives_dir / LABEL_ARCHIVE)])
            listing = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
            names = listing.stdout.splitlines()
            for member in LABEL_MEMBERS:
                if names.count(member) != 1:
                    raise ValueError(f"Label member must occur exactly once: {member}")
                artifact = _extract_label(executable, archives_dir / LABEL_ARCHIVE, member,
                                          output / "labels" / Path(member).name)
                artifact["member"] = member
                manifest["artifacts"].append(artifact)
                save()
                event("member_saved", **artifact)
            for name in STATIC_FILES:
                record = next(record for record in manifest["source_files"] if Path(record["path"]).name == name)
                with (archives_dir / name).open("rb") as source:
                    artifact, partial = _copy_artifact(source, output / name, record["bytes"], record["bytes"])
                if artifact["sha256"] != record["sha256"]:
                    raise ValueError(f"Source changed while copying: {name}")
                partial.rename(artifact["path"])
                manifest["artifacts"].append(artifact)
                save()
                event("file_saved", **artifact)
            for record in manifest["source_files"]:
                stat = Path(record["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                    raise ValueError(f"Source changed during extraction: {record['path']}")
            manifest["status"] = "succeeded"
            event("succeeded", artifacts=len(manifest["artifacts"]))
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            event("failed", **manifest["error"])
            raise
        finally:
            manifest["finished_at_utc"] = _now()
            save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.archives_dir, args.output_dir)


if __name__ == "__main__":
    main()
