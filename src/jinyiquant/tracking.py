"""Append-only event archives, source capture and transactional audit index."""

import importlib.metadata
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import traceback
import uuid
import zipfile
from pathlib import Path

import psutil

from .io import canonical_json, digest, file_sha256, read_json, redact, utc_now, write_json

_MODULE = Path(__file__).resolve()
PROJECT_ROOT = _MODULE.parents[2] if _MODULE.parent.parent.name == "src" else _MODULE.parents[1]


def new_run_dir(output_root: Path) -> Path:
    name = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z") + "_" + uuid.uuid4().hex[:12]
    path = output_root / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def resource_usage() -> dict:
    # A lazy lookup keeps Ridge/Mean independent of the optional torch runtime.
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and getattr(cuda, "is_initialized", lambda: False)():
        from modified_tlob.device import device_resources
        return device_resources(torch.device("cuda"))
    p = psutil.Process()
    return {"rss_mb": p.memory_info().rss / 2**20, "cpu_seconds": sum(p.cpu_times()[:2]),
            "gpu_memory_mb": None, "gpu_note": "CPU reference implementation; GPU not used"}


class Recorder:
    def __init__(self, run_dir: Path, experiment_id: str):
        self.run_dir = Path(run_dir)
        self.experiment_id = experiment_id
        self.run_id = self.run_dir.name
        self._lock = threading.Lock()

    def event(self, stage: str, event_type: str, *, level: str = "INFO", **context):
        value = {"schema_version": 1, "timestamp_utc": utc_now(), "level": level, "stage": stage,
                 "event_type": event_type, "run_id": self.run_id, "experiment_id": self.experiment_id,
                 "context": redact(context)}
        with self._lock, (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(canonical_json(value) + "\n")
            f.flush()

    def failure(self, stage: str, exc: BaseException):
        self.event(stage, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", level="ERROR",
                   exception_type=type(exc).__name__, reason=str(exc),
                   traceback="".join(traceback.format_exception(exc)), recovery_state="restart_as_new_run")


def capture_source(run_dir: Path) -> dict:
    # Explicit allowlist: no raw datasets, environment, credentials or unrelated user files.
    roots = [PROJECT_ROOT / x for x in
             ("src", "jinyiquant", "tests", "configs", "docs", "scripts", "modified_tlob", ".github")]
    paths = [p for root in roots if root.exists() for p in root.rglob("*")
             if p.is_file() and (p.suffix in {".py", ".json", ".md", ".txt", ".toml", ".ipynb", ".lock", ".yml", ".yaml"}
                                or p.name == "LICENSE")
             and "__pycache__" not in p.parts
             and (p.suffix != ".md" or p.name in {"README.md", "AGENTS.md"})]
    paths += [PROJECT_ROOT / x for x in ("pyproject.toml", "uv.lock", ".python-version", ".gitattributes", "README.md", "AGENTS.md",
              "inference.ipynb", "training_config.json", "submission_config.json",
              "requirements-cpu.lock.txt", "requirements-cuda.lock.txt")
              if (PROJECT_ROOT / x).is_file()]
    inventory = {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in sorted(paths)}
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            z.write(p, p.relative_to(PROJECT_ROOT))
    def git(*args):
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short"),
            "snapshot": "source.zip", "snapshot_sha256": file_sha256(run_dir / "source.zip"),
            "content_sha256": digest(inventory), "files": inventory}


def environment() -> dict:
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "total_memory_mb": psutil.virtual_memory().total / 2**20,
            "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "determinism": {"device": "cpu", "blas_threads": 1,
                            "note": "Tolerance-based reproducibility; BLAS/platform rounding may vary"}}


def artifacts(run_dir: Path) -> list[dict]:
    return [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size,
             "sha256": file_sha256(p)} for p in sorted(run_dir.rglob("*"))
            if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")]


def audit_db(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output_root / "audit.sqlite3", timeout=30)
    connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS final_access "
                       "(scope TEXT, run_id TEXT PRIMARY KEY, purpose TEXT, timestamp_utc TEXT)")
    connection.commit()
    return connection


def claim_final_access(config, run_id: str):
    # Key excludes model/seed/experiment to prevent repeated views by renaming experiments.
    scope = digest({"source": config.data.source, "version": config.data.version,
                    "final": config.splits.final.model_dump(mode="json")})
    with audit_db(config.output_root) as db:
        db.execute("BEGIN IMMEDIATE")
        count = db.execute("SELECT COUNT(*) FROM final_access WHERE scope=?", (scope,)).fetchone()[0]
        if count >= config.final_access.max_views:
            raise ValueError(f"Final split view limit reached ({count}); access denied")
        db.execute("INSERT INTO final_access VALUES (?,?,?,?)",
                   (scope, run_id, config.final_access.purpose, utc_now()))


def finish(run_dir: Path, status: str, exit_code: int, *, resources=None, error=None):
    manifest = read_json(run_dir / "manifest.json")
    rec = Recorder(run_dir, manifest["experiment_id"])
    rec.event("run", "ended", status=status, exit_code=exit_code, resources=resources)
    manifest.update(status=status, exit_code=exit_code, ended_at=utc_now())
    if resources is not None:
        manifest["resources"] = resources
    if error:
        manifest["error"] = redact(error)
    manifest["artifacts"] = artifacts(run_dir)
    write_json(run_dir / "manifest.json", manifest)
    with audit_db(run_dir.parent) as db:
        db.execute("INSERT INTO runs VALUES (?,?)", (manifest["run_id"], canonical_json(manifest)))
    # Convenience export; SQLite is the canonical concurrent-safe index.
    with (run_dir.parent / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(canonical_json({k: manifest.get(k) for k in
                               ("run_id", "experiment_id", "status", "exit_code", "started_at", "ended_at")}) + "\n")
    return manifest
