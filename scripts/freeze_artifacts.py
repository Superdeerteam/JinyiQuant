"""Copy the runnable reference artifacts into a frozen snapshot with a sha256 index.

The snapshot is append-only and read-only with respect to history: sources are
only read, every copied file is re-hashed after the copy, and an existing frozen
file is never replaced by different bytes. ``--plan`` supplies the declared
source list; the resulting ``sha256_index.json`` records per-file digests, a tree
digest per source, and the totals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", ".DS_Store"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise SystemExit(f"freeze source missing: {source}")
    return sorted(path for path in source.rglob("*")
                  if path.is_file() and not SKIP_PARTS.intersection(path.parts))


def tree_digest(rows: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative, sha in sorted(rows):
        digest.update(relative.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n")
    return digest.hexdigest()


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def freeze(sources: list[str], dest: Path, created_at: str | None = None) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    source_rows: list[dict] = []
    for raw in sources:
        source = (ROOT / raw).resolve()
        try:
            source.relative_to(ROOT)
        except ValueError as exc:  # pragma: no cover - declared plans stay inside the repo
            raise SystemExit(f"freeze source outside the repository: {raw}") from exc
        rows: list[tuple[str, str]] = []
        total_bytes = 0
        reused = 0
        for path in source_files(source):
            relative = path.relative_to(ROOT).as_posix()
            target = dest / "artifacts" / relative
            source_sha = sha256_file(path)
            if target.exists():
                if sha256_file(target) != source_sha:
                    raise SystemExit(f"frozen file differs, refusing to overwrite: {target}")
                reused += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                if sha256_file(target) != source_sha:
                    raise SystemExit(f"copy verification failed: {target}")
            stat = path.stat()
            rows.append((relative, source_sha))
            total_bytes += stat.st_size
            files.append({
                "source": relative,
                "frozen": target.relative_to(dest).as_posix(),
                "bytes": stat.st_size,
                "sha256": source_sha,
                "mtime_utc": utc_iso(stat.st_mtime),
            })
        source_rows.append({
            "source": raw,
            "kind": "dir" if source.is_dir() else "file",
            "files": len(rows),
            "bytes": total_bytes,
            "tree_sha256": tree_digest(rows),
            "reused_identical": reused,
        })
        print(f"froze {raw}: {len(rows)} files, {total_bytes / 1e6:.1f} MB", flush=True)
    index = {
        "schema_version": 1,
        "snapshot_id": dest.name,
        "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
        "policy": "copy-only; sha256 recomputed on every frozen copy; sources are never modified",
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "sources": source_rows,
        "files": files,
        "totals": {
            "sources": len(source_rows),
            "files": len(files),
            "bytes": sum(row["bytes"] for row in files),
        },
    }
    output = dest / "sha256_index.json"
    output.write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True, help="freeze_plan.json with a sources list")
    parser.add_argument("--dest", type=Path, default=ROOT / "runs/frozen/a-20261026")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    dest = args.dest if args.dest.is_absolute() else ROOT / args.dest
    created_at = datetime.now(timezone.utc).isoformat()
    index = freeze(list(plan["sources"]), dest, created_at)
    index["plan"] = {
        "path": args.plan.resolve().relative_to(ROOT).as_posix(),
        "sha256": sha256_file(args.plan),
        "snapshot_id": plan.get("snapshot_id"),
    }
    (dest / "sha256_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"totals: {index['totals']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
