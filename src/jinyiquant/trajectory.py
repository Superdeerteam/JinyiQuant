"""Continuous training trajectory: pre-registered budgets, fixed LR rule and exposure accounting.

A trajectory is one uninterrupted optimization run cut into pre-registered budget points
by checkpoints. Nothing here trains a model. The module converts effective-epoch budgets
into optimizer steps against the pool that is actually used, states the learning-rate rule
once (so no budget point can silently use a different schedule), and turns per-row exposure
counters into the per-day and per-epoch accounting that must reconcile with the declared
budget. Accounting is what keeps a configuration from claiming more epochs than the run
actually covered.

An effective epoch is one pass over the usable training pool measured in samples, not in
batch boundaries: ``steps * batch_size / usable_samples``. A short last batch therefore
cannot inflate the declared exposure.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LR_SCHEDULES = ("constant",)
CHECKPOINT_FILENAME = "training_state.pt"
CHECKPOINT_METADATA_FILENAME = "training_state.json"
EXPOSURE_FILENAME = "exposure.json"
TRAJECTORY_FILENAME = "trajectory.json"
SEALED_INTERVAL = ("2024-10-08", "2025-01-01")


def batches_per_epoch(samples: int, batch_size: int) -> int:
    """Optimizer steps in one pass over ``samples`` rows (the last batch may be short)."""
    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")
    return -(-samples // batch_size)


def steps_for_effective_epochs(effective_epochs: float, samples: int, batch_size: int) -> int:
    """Optimizer steps whose sample exposure equals the declared effective epochs."""
    if not np.isfinite(effective_epochs) or effective_epochs <= 0:
        raise ValueError("effective_epochs must be positive and finite")
    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")
    steps = int(round(float(effective_epochs) * samples / batch_size))
    if steps < 1:
        raise ValueError("effective_epochs is smaller than one optimizer step")
    return steps


def effective_epochs_for_steps(steps: int, samples: int, batch_size: int) -> float:
    if steps < 0 or samples < 1 or batch_size < 1:
        raise ValueError("steps must be nonnegative and samples/batch_size positive")
    return steps * batch_size / samples


def learning_rate_at(schedule: str, base_learning_rate: float, step: int) -> float:
    """The frozen learning-rate rule; it never depends on the total budget."""
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"Unsupported learning-rate schedule: {schedule!r}")
    if not np.isfinite(base_learning_rate) or base_learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    if step < 0:
        raise ValueError("learning-rate step must be nonnegative")
    return float(base_learning_rate)


def validate_budget_points(points) -> list[dict]:
    """Check pre-registered budget points: unique names, increasing effective epochs."""
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("A trajectory requires at least two budget points")
    names, previous = set(), 0.0
    validated = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Each budget point must be an object")
        name = point.get("name")
        epochs = point.get("effective_epochs")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Budget point names must be nonempty and unique")
        if not isinstance(epochs, (int, float)) or not np.isfinite(epochs) or epochs <= previous:
            raise ValueError(f"Budget point {name!r} must increase the effective-epoch budget")
        names.add(name)
        previous = float(epochs)
        validated.append({"name": name, "effective_epochs": float(epochs)})
    return validated


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_digest(plan: dict) -> str:
    """Stable digest of a pre-registered trajectory plan."""
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_plan(path: str | Path) -> dict:
    """Read a pre-registered trajectory plan and refuse an under-specified one."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("schema_version", "experiment_id", "recipe_config", "recipe_sha256",
                "budget_points", "lr_schedule", "exposure"):
        if key not in document:
            raise ValueError(f"Trajectory plan is missing {key!r}")
    if document["schema_version"] != 1:
        raise ValueError("Unsupported trajectory plan schema_version")
    if not re.fullmatch(r"[0-9a-f]{64}", str(document["recipe_sha256"])):
        raise ValueError("recipe_sha256 must pin the fixed recipe file")
    if document["lr_schedule"] not in LR_SCHEDULES:
        raise ValueError(f"Unsupported learning-rate schedule: {document['lr_schedule']!r}")
    document["budget_points"] = validate_budget_points(document["budget_points"])
    exposure = document["exposure"]
    if not 0 < float(exposure.get("min_coverage", 0)) <= 1:
        raise ValueError("exposure.min_coverage must be in (0, 1]")
    if float(exposure.get("effective_epochs_atol", -1)) < 0:
        raise ValueError("exposure.effective_epochs_atol must be nonnegative")
    return document


def derive_budget_steps(points: list[dict], samples: int, batch_size: int) -> list[dict]:
    """Attach the optimizer steps and realised exposure to each pre-registered point."""
    rows, previous = [], 0
    for point in points:
        steps = steps_for_effective_epochs(point["effective_epochs"], samples, batch_size)
        if steps <= previous:
            raise ValueError(f"Budget point {point['name']!r} does not add optimizer steps")
        previous = steps
        rows.append({**point, "steps": steps, "segment_steps": steps - (rows[-1]["steps"] if rows else 0),
                     "realised_effective_epochs": effective_epochs_for_steps(steps, samples, batch_size),
                     "batches_per_epoch": batches_per_epoch(samples, batch_size)})
    return rows


def exposure_report(keys: pd.DataFrame, sample_mask, exposure_counts, *,
                    batch_size: int, steps: int, per_epoch_steps: dict[str, int] | None = None,
                    declared: dict | None = None) -> dict:
    """Turn per-row exposure counters into per-day, per-epoch and overall accounting."""
    counts = np.asarray(exposure_counts, dtype=np.int64)
    mask = np.ones(len(keys), dtype=bool) if sample_mask is None else np.asarray(sample_mask, dtype=bool)
    if mask.shape != (len(keys),) or not mask.any():
        raise ValueError("Exposure accounting requires an aligned nonempty training mask")
    if counts.shape != (int(mask.sum()),):
        raise ValueError("Exposure counters must have one entry per usable training row")
    if counts.min() < 0:
        raise ValueError("Exposure counters must be nonnegative")
    usable = keys.loc[mask]
    dates = pd.to_datetime(usable["date"]).dt.normalize()
    per_day = []
    for date, rows in usable.groupby(dates, sort=True).groups.items():
        positions = np.asarray(rows)
        day_counts = counts[positions]
        per_day.append({
            "date": str(pd.Timestamp(date).date()),
            "usable_samples": int(len(positions)),
            "samples_seen": int(day_counts.sum()),
            "unique_samples": int((day_counts > 0).sum()),
            "coverage": float((day_counts > 0).mean()),
            "min_exposure": int(day_counts.min()),
            "max_exposure": int(day_counts.max()),
        })
    total = int(len(counts))
    seen = int(counts.sum())
    unique = int((counts > 0).sum())
    report = {
        "schema_version": 1,
        "steps": int(steps),
        "batch_size": int(batch_size),
        "usable_samples": total,
        "samples_seen": seen,
        "expected_samples_seen": int(steps) * int(batch_size),
        "unique_samples": unique,
        "coverage": unique / total,
        "effective_epochs": seen / total,
        "min_exposure": int(counts.min()),
        "max_exposure": int(counts.max()),
        "mean_exposure": float(counts.mean(dtype=np.float64)),
        "exposure_counts_sha256": hashlib.sha256(counts.astype("<i8").tobytes()).hexdigest(),
        "per_day": per_day,
        "per_epoch": [{"epoch_index": int(index), "steps": int(value)}
                      for index, value in sorted((per_epoch_steps or {}).items())],
        "declared": declared,
        "note": "samples_seen counts every sample fed to the optimizer; unique_samples/counts are "
                "per usable training row; effective_epochs = samples_seen / usable_samples",
    }
    if report["samples_seen"] != report["expected_samples_seen"]:
        raise ValueError("Exposure counters disagree with steps * batch_size")
    return report


def reconcile_exposure(report: dict, declared: dict) -> dict:
    """Declared budget versus measured exposure, failing rather than trusting config."""
    issues = []
    tolerance = float(declared["effective_epochs_atol"])
    if report["steps"] != int(declared["steps"]):
        issues.append(f"steps {report['steps']} != declared {declared['steps']}")
    if abs(report["effective_epochs"] - float(declared["effective_epochs"])) > tolerance:
        issues.append(f"effective epochs {report['effective_epochs']:.9f} outside "
                      f"{declared['effective_epochs']} +/- {tolerance}")
    if report["coverage"] < float(declared["min_coverage"]):
        issues.append(f"coverage {report['coverage']:.6f} below {declared['min_coverage']}")
    short = [row["date"] for row in report["per_day"] if row["unique_samples"] < row["usable_samples"]]
    if report["coverage"] >= 1.0 and short:
        issues.append(f"{len(short)} training days were not fully covered")
    return {"passed": not issues, "issues": issues,
            "steps": report["steps"], "effective_epochs": report["effective_epochs"],
            "coverage": report["coverage"], "declared": declared}


def write_json(document, path: str | Path) -> str:
    """Write stable, human-readable JSON and return its sha256."""
    payload = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
