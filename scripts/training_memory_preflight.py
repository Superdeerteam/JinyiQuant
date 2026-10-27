"""Pre-flight the enlarged training pool on the host before any formal trajectory.

Runs the registered configuration end to end - pool build, validation tensorisation, model
construction and a short 200-500 step optimizer window - while sampling host RSS and, on CUDA,
allocated/reserved GPU memory at a fixed interval.  Sampling runs from a ``SIGALRM`` handler
instead of a sampler thread because the training container's sandbox denies ``pthread_create``
(``CLONE_THREAD`` -> EPERM); a periodic signal is delivered to the same thread between
bytecodes and needs no new thread.

The measurement is persisted as ``preflight.json`` plus ``samples.jsonl`` next to the run, and
the gate is read from that file: ``processes`` x the measured per-process peak must fit the
host with margin, otherwise the pre-flight aborts the round.  No formal trajectory is started
here and nothing beyond the pre-flight step budget is trained.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from jinyiquant.io import write_json
from jinyiquant.models import create_model
from jinyiquant.trajectory import derive_budget_steps, file_sha256, plan_digest

from scripts.run_training_trajectory import (
    prepare_training,
    prepare_validation,
    trajectory_config,
)

STEPS_RANGE = (200, 500)


class IntervalSampler:
    """Sample RSS and GPU counters every ``interval`` seconds without creating a thread."""

    def __init__(self, interval: float, cuda: bool) -> None:
        self.interval = float(interval)
        self.cuda = cuda
        self.started = time.monotonic()
        self.samples: list[dict] = []
        self._previous = None
        self._process = psutil.Process()

    def _handle(self, signum, frame) -> None:  # noqa: ARG002 - signal handler signature
        sample = self.sample()
        if sample is not None:
            self.samples.append(sample)
        signal.setitimer(signal.ITIMER_REAL, self.interval)

    def sample(self) -> dict | None:
        now = time.monotonic()
        row = {
            "t_seconds": round(now - self.started, 3),
            "rss_mb": round(self._process.memory_info().rss / 2**20, 2),
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
        }
        if self.cuda:
            try:
                row["gpu_allocated_mb"] = round(torch.cuda.memory_allocated(0) / 2**20, 2)
                row["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 2**20, 2)
            except RuntimeError:
                row["gpu_allocated_mb"] = None
                row["gpu_reserved_mb"] = None
        return row

    def start(self) -> None:
        signal.signal(signal.SIGALRM, self._handle)
        signal.setitimer(signal.ITIMER_REAL, self.interval)

    def stop(self) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def peak_of(samples: list[dict], key: str) -> float | None:
    values = [row[key] for row in samples if row.get(key) is not None]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--processes", type=int, default=3,
                        help="parallel training processes the measured peak is projected onto")
    parser.add_argument("--host-margin", type=float, default=0.9,
                        help="fraction of host RAM the projected peak may occupy")
    parser.add_argument("--gpu-margin", type=float, default=0.9)
    parser.add_argument("--gpu-share", type=int, default=2,
                        help="training processes that share one GPU at run time")
    args = parser.parse_args()

    if not STEPS_RANGE[0] <= args.steps <= STEPS_RANGE[1]:
        parser.error(f"--steps must be within {STEPS_RANGE}")
    if not 0.5 <= args.host_margin < 1.0 or not 0.5 <= args.gpu_margin < 1.0:
        parser.error("margins must be in [0.5, 1.0)")
    if args.gpu_share < 1:
        parser.error("--gpu-share must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": {"path": args.plan.as_posix(), "sha256": file_sha256(args.plan)},
        "plan_digest": plan_digest(json.loads(args.plan.read_text(encoding="utf-8"))),
        "steps": args.steps,
        "interval_seconds": args.interval_seconds,
        "concurrent_processes": args.processes,
        "processes_per_gpu": args.gpu_share,
        "status": "failed",
        "failure": None,
        "measurement": None,
        "gate": None,
    }
    sampler = IntervalSampler(args.interval_seconds, cuda=False)
    sampler.start()
    model = None
    prepared = None
    started = time.monotonic()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        config, recipe_guard = trajectory_config(plan, args.repo.resolve(), output_dir / "runs")
        document["experiment_id"] = config.experiment_id
        document["seed"] = config.seed
        document["device"] = config.model.device
        document["recipe_guard"] = recipe_guard
        if config.model.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this host; a CPU peak would not measure "
                               "the registered configuration")
        sampler.cuda = config.model.device != "cpu"
        prepared = prepare_training(config)
        usable = int(prepared["fit_mask"].sum())
        budget = derive_budget_steps(plan["budget_points"], usable, config.model.batch_size)
        document["usable_samples"] = usable
        document["budget_points"] = budget
        prepare_validation(config, prepared["bundle"])
        model = create_model(config)
        progress = model.fit(prepared["train"], prepared["targets"],
                             sample_mask=prepared["fit_mask"], step_budget=args.steps,
                             track_exposure=True)
        if progress["steps"] != args.steps:
            raise ValueError(f"pre-flight stopped at step {progress['steps']}, expected {args.steps}")
        document["training"] = {"steps": progress["steps"], "samples_seen": progress["samples_seen"]}
        document["status"] = "completed"
    except BaseException as exc:  # noqa: BLE001 - the measurement must record any failure
        document["failure"] = {"type": type(exc).__name__, "reason": str(exc)[:500]}
    finally:
        document["sample_count"] = len(sampler.samples)
        final_sample = sampler.sample()
        if final_sample is not None:
            sampler.samples.append(final_sample)
        sampler.stop()

    peak_rss = peak_of(sampler.samples, "rss_mb")
    peak_gpu = peak_of(sampler.samples, "gpu_allocated_mb")
    peak_gpu_reserved = peak_of(sampler.samples, "gpu_reserved_mb")
    if torch.cuda.is_available():
        try:
            peak_gpu = max(peak_gpu or 0.0, round(torch.cuda.max_memory_allocated(0) / 2**20, 2))
            peak_gpu_reserved = max(peak_gpu_reserved or 0.0,
                                    round(torch.cuda.max_memory_reserved(0) / 2**20, 2))
        except RuntimeError:
            pass
    host_mb = psutil.virtual_memory().total / 2**20
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in sampler.samples:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    measurement = {
        "peak_rss_mb": peak_rss,
        "peak_gpu_allocated_mb": peak_gpu,
        "peak_gpu_reserved_mb": peak_gpu_reserved,
        "host_memory_mb": round(host_mb, 1),
        "host_available_mb": round(psutil.virtual_memory().available / 2**20, 1),
        "wall_seconds": round(time.monotonic() - started, 1),
        "samples_path": samples_path.name,
    }
    document["measurement"] = measurement
    if peak_rss is not None:
        projected = peak_rss * args.processes
        host_ok = projected <= args.host_margin * host_mb
        gpu_total = None
        if peak_gpu is not None:
            gpu_total = round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        projected_gpu = (None if peak_gpu is None else round(peak_gpu * args.gpu_share, 1))
        gpu_ok = (peak_gpu is None or gpu_total is None
                  or projected_gpu <= args.gpu_margin * gpu_total)
        document["gate"] = {
            "projected_rss_mb": round(projected, 1),
            "host_budget_mb": round(args.host_margin * host_mb, 1),
            "host_passed": bool(host_ok),
            "projected_gpu_mb": projected_gpu,
            "gpu_total_mb": gpu_total,
            "gpu_budget_mb": (None if gpu_total is None else round(args.gpu_margin * gpu_total, 1)),
            "gpu_passed": bool(gpu_ok),
            "passed": bool(host_ok and gpu_ok and document["status"] == "completed"),
            "rule": (f"{args.processes} x measured per-process peak must stay under "
                     f"{args.host_margin:.0%} of host RAM, and {args.gpu_share} processes sharing "
                     f"one GPU must stay under {args.gpu_margin:.0%} of that GPU"),
        }
    write_json(output_dir / "preflight.json", document)
    gate = document["gate"] or {}
    print(json.dumps({"status": document["status"], "measurement": measurement,
                      "gate": gate, "failure": document["failure"]}, ensure_ascii=False))
    return 0 if gate.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
