"""Stable, strict JSON and content fingerprints; no application dependencies."""

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def redact(value):
    """Exclude credential-like fields and redact common credentials inside errors."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"password|secret|token|credential|api.?key", k, re.I)
                else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'''(?i)(["']?(?:password|secret|token|api[_-]?key)["']?\s*:\s*)(["'])(.*?)\2''',
                       r'\1"[REDACTED]"', value)
        value = re.sub(r"(?i)(password|secret|token|api[_-]?key)([\s:=]+)[^\s,;]+", r"\1\2[REDACTED]", value)
        return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
    return value
