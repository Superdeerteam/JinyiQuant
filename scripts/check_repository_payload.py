"""Reject local datasets and generated artifacts from the Git upload payload.

The default checks added/modified staged blobs, not working-tree copies. CI uses
``--mode tracked`` to check every file in the index. Neither mode reads datasets
from disk. This is a packaging guard, not a general-purpose secret scanner.
"""

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 2 * 1024 * 1024
FORBIDDEN_DIRECTORIES = {
    "数据", "data", "runs", "scratch", ".tmp", "cache", ".cache", ".venv", "dist", "build",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".uv_cache", "node_modules",
}
FORBIDDEN_SUFFIXES = {
    ".csv", ".tsv", ".parquet", ".pq", ".feather", ".arrow", ".orc", ".npz", ".npy",
    ".pkl", ".pickle", ".h5", ".hdf", ".hdf5", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".joblib", ".bin", ".pyc", ".pyo",
    ".exe", ".dll", ".so", ".dylib", ".whl", ".jsonl", ".ndjson", ".log",
    ".tar", ".gz", ".tgz", ".bz", ".bz2", ".xz", ".zip", ".rar", ".7z", ".zst", ".lz", ".lzo", ".br",
    ".pem", ".key", ".p12", ".pfx",
}


def git(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *args], capture_output=True, check=True,
    ).stdout


def path_reasons(path: str) -> list[str]:
    normalized = PurePosixPath(path.replace("\\", "/").lower())
    reasons = []
    if any(part in FORBIDDEN_DIRECTORIES or part.startswith((".test-tmp-", ".pytest-eval-"))
           or part.endswith(".egg-info") for part in normalized.parts[:-1]):
        reasons.append("local_data_or_runtime_directory")
    if normalized.suffix in FORBIDDEN_SUFFIXES:
        reasons.append("dataset_archive_or_artifact_extension")
    if normalized.name == ".env" or normalized.name.startswith(".env."):
        reasons.append("local_credentials_filename")
    return reasons


def check_repository(repository: Path, mode: str = "staged") -> dict:
    if mode not in {"staged", "tracked"}:
        raise ValueError("mode must be staged or tracked")
    selected = None
    if mode == "staged":
        selected = set(git(repository, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z").split(b"\0"))
    entries = []
    for entry in git(repository, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        file_mode, oid, stage = metadata.decode("ascii").split()
        if selected is not None and raw_path not in selected and stage == "0":
            continue
        path = raw_path.decode("utf-8")
        entries.append((path, file_mode, oid, stage))
    violations = []
    total_bytes = 0
    for path, file_mode, oid, stage in sorted(entries):
        reasons = path_reasons(path)
        if stage != "0":
            reasons.append("unresolved_merge")
        if file_mode not in {"100644", "100755"}:
            reasons.append("non_regular_file")
        size = int(git(repository, "cat-file", "-s", oid))
        total_bytes += size
        if size > MAX_FILE_BYTES:
            reasons.append("file_exceeds_2_mib")
        # Never load oversized or already forbidden data blobs into memory.
        if not reasons:
            content = git(repository, "cat-file", "blob", oid)
            try:
                content.decode("utf-8-sig")
            except UnicodeDecodeError:
                reasons.append("non_utf8_binary_content")
            if b"\0" in content:
                reasons.append("binary_content")
        if reasons:
            violations.append({"path": path, "bytes": size, "reasons": reasons})
    return {
        "schema_version": 1, "mode": mode, "source": "git_index_blobs", "passed": not violations,
        "files_checked": len(entries), "total_bytes": total_bytes, "max_file_bytes": MAX_FILE_BYTES,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("staged", "tracked"), default="staged")
    args = parser.parse_args()
    try:
        result = check_repository(args.repository, args.mode)
    except (ValueError, subprocess.CalledProcessError) as exc:
        result = {"schema_version": 1, "mode": args.mode, "passed": False,
                  "error": type(exc).__name__, "message": "Unable to inspect Git index; upload must stop."}
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
