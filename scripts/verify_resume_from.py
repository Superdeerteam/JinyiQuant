"""Prove ``--resume-from`` continues a trajectory bit for bit.

Continuing a finished run from its last budget-point checkpoint must produce exactly the
trajectory the original run would have produced: the same per-step loss, the same parameters, the
same validation scores and the same artifacts. This script compares one continuous trajectory (A)
with two continuations of it, on one pool and one plan:

* **B1** resumes from the first checkpoint (mid-epoch) and trains the remaining two points, so the
  epoch boundary is crossed inside the resumed run;
* **B2** resumes from the second checkpoint (exactly at an epoch end) and trains the last point, so
  the ``batch_index == 0`` restore path is exercised.

The plan's budget points are chosen from the pool that was actually built: the first is a few
steps in, the second lands exactly on an epoch end and the third one step later. A and B share the
recipe, the plan and the prepared pool, so any difference comes from the restore path.

Usage::

    python -m scripts.verify_resume_from --config configs/mlplob_v3_60d_csz_19800.json \
        --output runs/stream-pool-20261027/resume_from_60d.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.io import write_json  # noqa: E402
from jinyiquant.trajectory import (  # noqa: E402
    CHECKPOINT_FILENAME,
    batches_per_epoch,
    derive_budget_steps,
)
from scripts import run_training_trajectory as driver  # noqa: E402


def sha256_bytes(*parts: bytes) -> str:
    value = hashlib.sha256()
    for part in parts:
        value.update(len(part).to_bytes(8, "little"))
        value.update(part)
    return value.hexdigest()


def resume_plan(config, usable: int, targets: list[int] | None = None) -> dict:
    """Three budget points at absolute step counts; the first is a few steps in.

    The default targets put the second point exactly on an epoch end and the third one step past
    it: ``batch_index == 0`` takes the other restore branch (``generator`` is not rewound to
    ``generator_before_epoch``), so both branches get covered. ``targets`` overrides them, which
    is how a cheaper probe keeps the same shape.
    """
    batch = int(config.model.batch_size)
    epoch = batches_per_epoch(usable, batch)
    steps = targets or [max(1, round(0.0004 * usable / batch)), epoch, epoch + 1]
    points = [{"name": f"r{index + 1}", "effective_epochs": step * batch / usable}
              for index, step in enumerate(steps)]
    return {"schema_version": 1, "experiment_id": config.experiment_id,
            "budget_points": points, "lr_schedule": "constant",
            "exposure": {"min_coverage": 1e-6, "effective_epochs_atol": 0.01},
            "probe": {"steps": steps, "batches_per_epoch": epoch,
                      "note": "probe plan for --resume-from equivalence; not registered"}}


def points_of(run_dir: Path) -> dict[str, Path]:
    """Checkpoint directory of every budget point the run executed, by point name.

    The directory keeps the point's index in the plan, not in the executed subset, so the
    trajectory record is the authority (a resumed run skips points and keeps the numbering).
    """
    document = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    return {row["name"]: run_dir / Path(row["checkpoint"]).parent
            for row in document["completed_points"]}


def checkpoint_state(directory: Path) -> dict[str, bytes]:
    """Every byte a resumed update depends on, flat and comparable."""
    payload = torch.load(directory / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    flat = {f"network.{name}": value.contiguous().numpy().tobytes()
            for name, value in payload["network_state_dict"].items()}
    for index, entry in enumerate(payload["optimizer_state_dict"]["state"].values()):
        for key, value in entry.items():
            flat[f"optimizer.{index}.{key}"] = (value.contiguous().numpy().tobytes()
                                                if torch.is_tensor(value) else repr(value).encode())
    flat["generator_before_epoch"] = payload["generator_before_epoch"].numpy().tobytes()
    for name in ("position", "fingerprint", "rng"):
        flat[name] = json.dumps(payload[name], sort_keys=True, default=str).encode()
    if payload.get("exposure_counts") is not None:
        flat["exposure_counts"] = payload["exposure_counts"].numpy().tobytes()
    return flat


def loss_trace(plan: dict, config, prepared: dict, validation, output: Path, *,
               resume_from: Path | None = None) -> tuple[dict, list[float]]:
    """Run one driver trajectory, capturing the training loss of every optimizer step."""
    losses = []
    original_mse, original_step = torch.nn.functional.mse_loss, torch.optim.Adam.step
    marks = []

    def recorded_mse(*args, **kwargs):
        value = original_mse(*args, **kwargs)
        losses.append(float(value.detach()))
        return value

    def marked_step(optimizer, *args, **kwargs):
        marks.append(time.perf_counter())
        return original_step(optimizer, *args, **kwargs)

    torch.nn.functional.mse_loss, torch.optim.Adam.step = recorded_mse, marked_step
    started = time.perf_counter()
    try:
        result = driver.run_trajectory(plan, config, prepared, validation, output,
                                       streaming=True, resume_from=resume_from)
    finally:
        torch.nn.functional.mse_loss, torch.optim.Adam.step = original_mse, original_step
    result["seconds"] = round(time.perf_counter() - started, 2)
    intervals = np.diff(np.asarray(marks, dtype=np.float64))
    result["step_ms"] = None if not len(intervals) else round(float(np.median(intervals)) * 1000, 3)
    return result, losses


def artifacts(run_dir: Path, name: str) -> dict[str, str]:
    """The files one budget point leaves behind, by content digest."""
    state = checkpoint_state(points_of(run_dir)[name])
    return {"checkpoint": sha256_bytes(*[state[key] for key in sorted(state)]),
            "model": hashlib.sha256((run_dir / f"model-{name}.pt").read_bytes()).hexdigest(),
            "weights": hashlib.sha256(
                (run_dir / f"modified_tlob_weights-{name}.json").read_bytes()).hexdigest(),
            "predictions": hashlib.sha256(
                (run_dir / "budget_points" / f"{name}_predictions.parquet").read_bytes()).hexdigest()}


def run(args) -> dict:
    config = load_config(args.config)
    root = Path(args.output_root)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    driver.capture_source = lambda run_dir: {"commit": "0" * 40, "content_sha256": "0" * 64,
                                             "snapshot": None, "files": {}}

    print("building the streamed pool", flush=True)
    started = time.perf_counter()
    prepared = driver.prepare_training(config, streaming=True)
    validation = driver.prepare_validation(config, prepared["bundle"])
    usable = int(prepared["fit_mask"].sum())
    targets = [int(value) for value in args.steps.split(",")] if args.steps else None
    if targets is not None:
        if len(targets) < 3 or sorted(targets) != targets or targets[0] < 1:
            raise SystemExit("--steps needs three or more strictly increasing step counts")
    plan = resume_plan(config, usable, targets)
    steps = [row["steps"] for row in
             derive_budget_steps(plan["budget_points"], usable, config.model.batch_size)]
    print(f"pool built in {time.perf_counter() - started:.1f}s: usable={usable} steps={steps}",
          flush=True)

    print("A: one continuous trajectory", flush=True)
    a_dir = root / "A"
    a_dir.mkdir()
    a_result, a_losses = loss_trace(plan, config, prepared, validation, a_dir)
    if a_result["status"] != "completed":
        raise SystemExit(f"A failed: {a_result['error']}")
    a_points = points_of(a_dir)

    print("B1: resume from the first checkpoint (mid-epoch)", flush=True)
    b1_dir = root / "B1"
    b1_dir.mkdir()
    b1_result, b1_losses = loss_trace(plan, config, prepared, validation, b1_dir,
                                      resume_from=a_dir / a_points["r1"] / CHECKPOINT_FILENAME)
    print("B2: resume from the second checkpoint (epoch end)", flush=True)
    b2_dir = root / "B2"
    b2_dir.mkdir()
    b2_result, b2_losses = loss_trace(plan, config, prepared, validation, b2_dir,
                                      resume_from=a_dir / a_points["r2"] / CHECKPOINT_FILENAME)

    report = {"config": str(Path(args.config).resolve()), "usable_samples": usable, "steps": steps,
              "seconds": {"A": a_result["seconds"], "B1": b1_result["seconds"],
                          "B2": b2_result["seconds"]},
              "step_ms": {"A": a_result["step_ms"], "B1": b1_result["step_ms"]},
              "losses": {"A": len(a_losses), "B1": len(b1_losses), "B2": len(b2_losses)},
              "A_vs_B1": {}, "A_vs_B2": {}}
    # B1 trains from step ``steps[0] + 1`` and B2 from ``steps[1] + 1``: the tail of A's trace
    # has to match the continuation's trace element by element.
    tail1, tail2 = a_losses[steps[0]:], a_losses[steps[1]:]
    report["A_vs_B1"]["losses_identical"] = bool(tail1 == b1_losses)
    report["A_vs_B1"]["losses_compared"] = len(tail1)
    report["A_vs_B1"]["max_abs_loss_difference"] = max(
        (abs(a - b) for a, b in zip(tail1, b1_losses)), default=None)
    report["A_vs_B2"]["losses_identical"] = bool(tail2 == b2_losses)
    report["A_vs_B2"]["losses_compared"] = len(tail2)
    report["A_vs_B2"]["max_abs_loss_difference"] = max(
        (abs(a - b) for a, b in zip(tail2, b2_losses)), default=None)
    for label, run_dir, names in (("A_vs_B1", b1_dir, [p for p in a_points if p != "r1"]),
                                  ("A_vs_B2", b2_dir, ["r3"])):
        b_points = points_of(run_dir)
        report[label]["points"] = {}
        for name in names:
            left, right = artifacts(a_dir, name), artifacts(run_dir, name)
            differing = [key for key in left if left[key] != right[key]]
            report[label]["points"][name] = {"differing": differing, "digests": left}
            if left["checkpoint"] != right["checkpoint"]:
                a_state, b_state = checkpoint_state(a_points[name]), checkpoint_state(b_points[name])
                report[label]["points"][name]["checkpoint_fields"] = [
                    key for key in set(a_state) | set(b_state)
                    if a_state.get(key) != b_state.get(key)][:12]
    report["executed"] = {"A": [row["name"] for row in a_result["trajectory"]],
                          "B1": [row["name"] for row in b1_result["trajectory"]],
                          "B2": [row["name"] for row in b2_result["trajectory"]]}
    report["passed"] = bool(
        report["A_vs_B1"]["losses_identical"] and report["A_vs_B2"]["losses_identical"]
        and all(not value["differing"] for label in ("A_vs_B1", "A_vs_B2")
                for value in report[label]["points"].values())
        and report["executed"]["B1"] == ["r2", "r3"] and report["executed"]["B2"] == ["r3"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/mlplob_v3_60d_csz_19800.json",
                        help="registered recipe to replay (a 60-day window keeps it cheap)")
    parser.add_argument("--steps", default=None,
                        help="comma-separated absolute step targets for the three budget points; "
                             "default lands the second point exactly on an epoch end")
    parser.add_argument("--output-root", default="runs/resume-from-probe")
    parser.add_argument("--output", default=None, help="report path (default: stdout only)")
    args = parser.parse_args(argv)
    report = run(args)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, report)
        print(f"wrote {destination}")
    print(json.dumps({key: report[key] for key in
                      ("passed", "usable_samples", "steps", "losses", "A_vs_B1", "A_vs_B2",
                       "executed", "seconds")}, indent=1, ensure_ascii=False)[:4000])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
