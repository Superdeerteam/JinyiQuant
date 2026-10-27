"""Prove the streaming pool trains exactly like the materialised pool on a real recipe.

``StreamingPool`` replaced the per-process window tensor with a key index plus one canonical bar
matrix, and ``TLOBAdapter.fit`` accepts a streaming pool by building each sampler draw's windows
instead of indexing a prefetched tensor. That wiring is only acceptable if the optimizer cannot
tell the two apart, so this script runs the same recipe on both pools in one process and reports,
in order:

1. **pool identity** - the same keys and the same usable-sample mask, so the sampler, the fitted
   transforms and the budget accounting see one pool, not two;
2. **trajectory equivalence** - the same initial parameters, the same per-step training loss
   (``float`` equality, never ``allclose``) and a bitwise identical final ``state_dict``;
3. **step cost** - median wall time of one optimizer step (batch build + forward + backward +
   update) for each pool, quoted as the streaming increase over the materialised path, with the
   acceptance bound applied;
4. **footprint** - peak RSS and GPU peak of each pass, measured around the pool build and the fit.

The recipe comes from a registered config; only the step budget is reduced (``--steps``). Each
variant is fitted twice: once with the training loss recorded per step (the equivalence pass,
which drains the device stream every step) and once without it (the timing pass, so the measured
step is the shipped one).

Usage::

    python -m scripts.verify_streaming_training --config configs/mlplob_v3_132d_csz_19800.json \
        --steps 200 --output runs/stream-pool-20261027/training_equiv_132d_gpu.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.data import build_batch, load_data, select_targets  # noqa: E402
from jinyiquant.io import file_sha256, write_json  # noqa: E402
from jinyiquant.models import create_model  # noqa: E402
from jinyiquant.pool import StreamingPool  # noqa: E402
from jinyiquant.runner import _aligned_targets  # noqa: E402

# One step may cost at most this much more on the streaming pool than on the materialised one.
STEP_INCREASE_LIMIT_PCT = 15.0
TIMING_WARMUP_STEPS = 10


def peak_rss_mb() -> float | None:
    """This process's peak resident set size (Linux ``VmHWM``), in MiB."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def digest(tensors: dict[str, bytes]) -> str:
    """Order-independent digest of parameter names and raw bytes."""
    value = hashlib.sha256()
    for name in sorted(tensors):
        value.update(name.encode("utf-8"))
        value.update(tensors[name])
    return value.hexdigest()


def bytes_of(state) -> dict[str, bytes]:
    return {name: value.detach().cpu().contiguous().numpy().tobytes()
            for name, value in state.items()}


def train_once(config, pool, target, mask, *, steps: int, record_losses: bool) -> dict:
    """One ``fit`` on ``pool``: loss trace, per-step seconds and the fitted parameters.

    ``torch.optim.Adam.step`` marks the completion of one iteration, so the intervals between
    marks are whole steps. The equivalence pass additionally records the training loss of every
    step, which drains the device stream once per step; the timing pass does not.
    """
    import torch

    model = create_model(config)
    initial = bytes_of(model.network.state_dict())
    losses, marks = [], []
    original_mse, original_step = torch.nn.functional.mse_loss, torch.optim.Adam.step

    def recorded_mse(*args, **kwargs):
        value = original_mse(*args, **kwargs)
        losses.append(float(value.detach()))
        return value

    def marked_step(optimizer, *args, **kwargs):
        marks.append(time.perf_counter())
        return original_step(optimizer, *args, **kwargs)

    if record_losses:
        torch.nn.functional.mse_loss = recorded_mse
    torch.optim.Adam.step = marked_step
    started = time.perf_counter()
    try:
        progress = model.fit(pool, target, sample_mask=mask, step_budget=steps)
    finally:
        torch.nn.functional.mse_loss, torch.optim.Adam.step = original_mse, original_step
    elapsed = time.perf_counter() - started
    intervals = np.diff(np.asarray(marks, dtype=np.float64))
    measured = intervals[TIMING_WARMUP_STEPS:] if len(intervals) > TIMING_WARMUP_STEPS else intervals
    return {"progress": progress, "initial": initial, "state": bytes_of(model.network.state_dict()),
            "device": str(model.device), "losses": losses, "seconds": round(elapsed, 2),
            "steps_marked": len(marks),
            "step_ms": None if not len(measured) else {
                "median": round(float(np.median(measured) * 1000), 3),
                "mean": round(float(measured.mean() * 1000), 3),
                "p10": round(float(np.percentile(measured, 10) * 1000), 3),
                "p90": round(float(np.percentile(measured, 90) * 1000), 3)},
            "peak_rss_mb": peak_rss_mb(),
            "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available() else None}


def pass_report(run: dict) -> dict:
    return {"seconds": run["seconds"], "peak_rss_mb": run["peak_rss_mb"],
            "gpu_peak_mb": run["gpu_peak_mb"], "steps_marked": run["steps_marked"],
            "steps": run["progress"]["steps"], "step_ms": run["step_ms"]}


def step_time_report(streaming: list[dict], materialised: list[dict]) -> dict:
    """Step cost per pool: ``min`` of the per-round medians, so contention spikes cannot pick a side."""
    def summary(rounds):
        medians = [run["median"] for run in rounds]
        return {"rounds": rounds, "min_median_ms": min(medians),
                "medians_ms": medians,
                "p10_p90_ms": [min(run["p10"] for run in rounds), max(run["p90"] for run in rounds)]}

    left, right = summary(streaming), summary(materialised)
    base, candidate = right["min_median_ms"], left["min_median_ms"]
    increase = (candidate - base) / base * 100 if base else None
    return {"streaming": left, "materialised": right,
            "materialised_step_ms": base, "streaming_step_ms": candidate,
            "materialised_steps_per_second": round(1000 / base, 2) if base else None,
            "streaming_steps_per_second": round(1000 / candidate, 2) if candidate else None,
            "step_increase_pct": None if increase is None else round(increase, 2),
            "limit_pct": STEP_INCREASE_LIMIT_PCT,
            "within_limit": None if increase is None else bool(increase <= STEP_INCREASE_LIMIT_PCT),
            "warmup_steps_skipped": TIMING_WARMUP_STEPS,
            "rule": "interval between successive optimizer steps (batch build + forward + backward + "
                    "update) without a per-step device sync; rounds alternate the pool order and the "
                    "acceptance uses the min of the per-round medians, after dropping the first "
                    f"{TIMING_WARMUP_STEPS} intervals"}


def equivalence_report(streaming: dict, materialised: dict, steps: int) -> dict:
    """Equivalence evidence; ``steps`` is what was requested, a run may stop earlier."""
    shared = [name for name in streaming["state"] if name in materialised["state"]]
    differing = [name for name in shared if streaming["state"][name] != materialised["state"][name]]
    ran = streaming["progress"]["steps"]
    steps_identical = ran == materialised["progress"]["steps"]
    losses_identical = (steps_identical and len(streaming["losses"]) == len(materialised["losses"]) == ran
                        and streaming["losses"] == materialised["losses"])
    return {"steps_requested": steps, "steps": ran, "steps_identical": bool(steps_identical),
            "stop_reason": [streaming["progress"]["stop_reason"],
                            materialised["progress"]["stop_reason"]],
            "initial_state_identical": streaming["initial"] == materialised["initial"],
            "losses_recorded": [len(streaming["losses"]), len(materialised["losses"])],
            "losses_identical": losses_identical,
            "max_abs_loss_difference": max((abs(a - b) for a, b in
                                            zip(streaming["losses"], materialised["losses"])), default=None),
            "losses": streaming["losses"],
            "state_tensors_compared": len(shared),
            "state_tensors_differing": differing,
            "state_identical": bool(shared) and not differing
            and streaming["state"].keys() == materialised["state"].keys(),
            "state_sha256": {"streaming": digest(streaming["state"]),
                             "materialised": digest(materialised["state"])}}



def prepare(config_path):
    """The recipe's training pool inputs: bars, targets and the keys the pool indexes."""
    config = load_config(config_path)
    training_times = config.data.training_sample_times or config.data.sample_times
    bundle = load_data(config.data, start=config.splits.train.start, end=config.splits.train.end)
    targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                             sample_times=training_times)
    if targets.empty:
        raise ValueError("The recipe's training split selects no targets")
    return config, bundle, targets


def run(args) -> dict:
    import torch

    started = time.perf_counter()
    print("loading data", flush=True)
    config, bundle, targets = prepare(args.config)
    print(f"prepared {len(bundle.bars)} bars, {len(targets)} targets", flush=True)
    keys = targets[["date", "instrument"]]
    prepared_seconds = round(time.perf_counter() - started, 2)

    print("streaming pool build", flush=True)
    pool_started = time.perf_counter()
    pool = StreamingPool.build(bundle.bars, keys, config.data)
    aligned = _aligned_targets(pool, targets)
    target = aligned.target.to_numpy(dtype=np.float64)
    mask = pool.fit_mask
    if not mask.any():
        raise ValueError("The recipe's usable-sample mask selects nothing")
    streaming_keys = pool.keys.copy()
    streaming_build = round(time.perf_counter() - pool_started, 2)
    report = {"config": str(Path(args.config).resolve()), "seed": config.seed,
              "requested_device": config.model.device, "steps": args.steps,
              "prepared_seconds": prepared_seconds, "bars_rows": int(len(bundle.bars)),
              "targets": int(len(targets)),
              "pool": {"keys": int(len(pool)), "usable": int(mask.sum()),
                       "streaming_build_seconds": streaming_build,
                       "streaming_nbytes": int(sum(pool.nbytes().values()))},
              "code": {"adapter_sha256": file_sha256(ROOT / "modified_tlob" / "adapter.py"),
                       "pool_sha256": file_sha256(ROOT / "src" / "jinyiquant" / "pool.py")}}

    # Streaming first: the materialised window tensor must not be resident during this pass.
    print("streaming pass", flush=True)
    stream_equivalence = train_once(config, pool, target, mask, steps=args.steps, record_losses=True)
    streaming_only_rss = peak_rss_mb()
    # Queried only now: the adapter's device preflight must own CUDA initialization.
    report["environment"] = {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": stream_equivalence["device"],
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    report["kernel_path"] = stream_equivalence["progress"]["compile"]

    print("materialised pool build", flush=True)
    build_started = time.perf_counter()
    materialised = build_batch(bundle.bars, keys, config.data)
    report["pool"]["materialised_build_seconds"] = round(time.perf_counter() - build_started, 2)
    report["pool"]["keys_identical"] = bool(materialised.keys.equals(streaming_keys))
    report["pool"]["fit_mask_identical"] = bool(np.array_equal(
        materialised.effective_rows.sum(axis=1) >= int(config.data.min_history), mask))
    del aligned
    gc.collect()
    print("materialised pass", flush=True)
    materialised_equivalence = train_once(config, materialised, target, mask, steps=args.steps,
                                          record_losses=True)

    # Timing rounds alternate the order, so a shift in ambient load cannot favour one pool.
    pools = {"streaming": pool, "materialised": materialised}
    timing = {variant: [] for variant in pools}
    for round_index in range(args.timing_rounds):
        print(f"timing round {round_index + 1}", flush=True)
        for variant in (("streaming", "materialised") if round_index % 2 == 0
                        else ("materialised", "streaming")):
            timing[variant].append(train_once(config, pools[variant], target, mask, steps=args.steps,
                                              record_losses=False))
    report["equivalence"] = equivalence_report(stream_equivalence, materialised_equivalence, args.steps)
    report["streaming"] = {"equivalence_pass": pass_report(stream_equivalence),
                           "timing_pass": [pass_report(run) for run in timing["streaming"]]}
    report["materialised"] = {"equivalence_pass": pass_report(materialised_equivalence),
                              "timing_pass": [pass_report(run) for run in timing["materialised"]]}
    report["step_time"] = step_time_report([run["step_ms"] for run in timing["streaming"]],
                                           [run["step_ms"] for run in timing["materialised"]])
    report["memory"] = {"peak_rss_after_streaming_pass_mb": streaming_only_rss,
                        "peak_rss_end_mb": peak_rss_mb(),
                        "gpu_peak_mb": max(run["gpu_peak_mb"] or 0 for run in
                                           timing["streaming"] + timing["materialised"]) or None,
                        "note": "the first figure is the standalone streaming footprint before the "
                                "materialised pool exists; the end figure includes both pools"}
    report["seconds"] = round(time.perf_counter() - started, 2)
    report["passed"] = bool(report["equivalence"]["state_identical"]
                            and report["equivalence"]["losses_identical"]
                            and report["equivalence"]["steps_identical"]
                            and report["equivalence"]["initial_state_identical"]
                            and report["pool"]["keys_identical"]
                            and report["pool"]["fit_mask_identical"]
                            and report["step_time"]["within_limit"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="registered recipe (run config) to replay")
    parser.add_argument("--steps", type=int, default=200, help="optimizer steps per pass")
    parser.add_argument("--timing-rounds", type=int, default=2,
                        help="interleaved timing rounds; the acceptance uses the min per-round median")
    parser.add_argument("--output", default=None, help="report path (default: stdout only)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.timing_rounds < 1:
        parser.error("--timing-rounds must be positive")
    report = run(args)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, report)
        print(f"wrote {destination}")
    summary = {key: report[key] for key in ("passed", "pool", "equivalence", "step_time",
                                            "memory", "seconds")}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
