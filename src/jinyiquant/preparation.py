"""Bounded, opt-in extraction of one source member from the large bar archive."""

from __future__ import annotations

import hashlib
import json
import tarfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def _safe_member_name(member: str) -> PurePosixPath:
    if not member or "\\" in member or ":" in member:
        raise ValueError("Archive member must be a relative POSIX path")
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError("Absolute paths and parent traversal are forbidden")
    if path.suffix.lower() not in {".feather", ".arrow", ".parquet", ".csv"}:
        raise ValueError("Select one supported tabular data member")
    return path


def prepare_data(archive_path: str | Path, member: str, output_root: str | Path,
                 max_member_bytes: int = 1_073_741_824) -> Path:
    """Extract one named regular file into a new directory without overwriting.

    TAR/GZIP must be traversed until the named member. Calling this for a late
    member can take time, but no unrelated file is extracted or retained. The
    original archive is identified by stat and exact member name; the extracted
    content receives SHA256. The manifest explicitly records this distinction.
    """
    name = _safe_member_name(member)
    if max_member_bytes <= 0:
        raise ValueError("max_member_bytes must be positive")
    archive_path = Path(archive_path).resolve(strict=True)
    source_stat = archive_path.stat()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    preparation_id = datetime.now(timezone.utc).strftime("prepare_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
    destination = root / preparation_id
    destination.mkdir(exist_ok=False)
    manifest = {"schema_version": 1, "preparation_id": preparation_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "source": {"path": str(archive_path), "bytes": source_stat.st_size,
                           "mtime_ns": source_stat.st_mtime_ns, "member": member,
                           "sha256": None, "verification": "stat_and_selected_member_only"},
                "max_member_bytes": max_member_bytes, "artifacts": []}
    manifest_path = destination / "preparation.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    save_manifest()
    try:
        with tarfile.open(archive_path, mode="r|gz" if archive_path.suffix == ".gz" else "r|*") as archive:
            selected = None
            for entry in archive:
                if entry.name == member:
                    selected = entry
                    break
            if selected is None:
                raise ValueError(f"Archive member not found: {member}")
            if not selected.isfile():
                raise ValueError("Only regular file members can be prepared; links and directories are forbidden")
            if selected.size > max_member_bytes:
                raise ValueError(f"Member is {selected.size} bytes, exceeding the declared preparation budget")
            # Use basename only inside a newly created directory, and open with
            # exclusive creation; never trust the member's directory hierarchy.
            target = destination / name.name
            if target.resolve().parent != destination.resolve():
                raise ValueError("Unsafe resolved extraction target")
            temporary = target.with_name(target.name + ".partial")
            digest = hashlib.sha256()
            written = 0
            source = archive.extractfile(selected)
            if source is None:
                raise ValueError("Selected member cannot be read")
            with source, temporary.open("xb") as output:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    written += len(chunk)
                    if written > max_member_bytes or written > selected.size:
                        raise ValueError("Member exceeded its declared size or preparation budget")
                    output.write(chunk)
                    digest.update(chunk)
            if written != selected.size:
                raise ValueError("Truncated archive member")
            after = archive_path.stat()
            if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source archive changed during extraction")
            temporary.rename(target)
            manifest["artifacts"] = [{"path": str(target), "bytes": written, "sha256": digest.hexdigest()}]
        manifest["status"] = "succeeded"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        return destination
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                             "recovery": "Any .partial file is incomplete; repeat into a new preparation directory"}
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        raise
