"""Prove the trajectory driver trains identically on the streaming pool and the materialised one.

``scripts/run_training_trajectory.py`` takes ``--stream-bars``: the bar window is read in bounded
chunks into the canonical bar matrix and the pool indexes it, instead of materialising one
``(samples, lookback, features)`` window tensor. That switch is only acceptable if the driver
cannot tell the two pools apart, so this script replays the driver's own preparation functions -
``prepare_training`` and ``prepare_validation`` - once per pool and reports:

1. **pool identity** - the same key table, usable-sample mask and aligned targets, therefore the
   same ``usable_samples`` and the same derived budget points;
2. **trajectory equivalence** - the same initial parameters, the same per-step training loss
   (``float`` equality, never ``allclose``) and a bitwise identical final ``state_dict``;
3. **scoring equivalence** - the validation batch of the bounded path equals the materialised one,
   the validation scores equal, and ``TLOBAdapter._predict`` on a pool equals ``_predict`` on the
   materialised batch;
4. **footprint and step cost** - peak RSS per pool and the median step interval.

The streaming pass runs first, so its footprint is measured before a window tensor exists; each
pass resets the kernel's peak-RSS counter (``/proc/self/clear_refs``) so the figures are per pool.

Usage::

    python -m scripts.verify_trajectory_streaming --config configs/mlplob_v3_132d_csz_19800.json \
        --steps 120 --output runs/stream-pool-20261027/trajectory_equiv_132d.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.io import file_sha256, write_json  # noqa: E402
from jinyiquant.models import create_model  # noqa: E402
from jinyiquant.pool import StreamingPool  # noqa: E402
from jinyiquant.runner import inference_keys  # noqa: E402
from scripts.run_training_trajectory import prepare_training, prepare_validation  # noqa: E402
from scripts.verify_streaming_training import (  # noqa: E402
    bytes_of,
    equivalence_report,
    peak_rss_mb,
    step_time_report,
)

WARMUP_STEPS = 10


def sha256(*parts: bytes) -> str:
    """Digest of named byte fields, each length-prefixed so concatenation is unambiguous."""
    value = hashlib.sha256()
    for part in parts:
        value.update(len(part).to_bytes(8, "little"))
        value.update(part)
    return value.hexdigest()


def reset_peak_rss() -> None:
    """Zero this process's peak-RSS high-water mark so the next pass reports its own."""
    try:
        Path("/proc/self/clear_refs").write_text("5", encoding="ascii")
    except OSError:
        pass


def host_memory() -> dict:
    """``/proc/meminfo`` figures in MiB; the pre-flight gate reads the same numbers."""
    figures = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, _, rest = line.partition(":")
            if name in {"MemTotal", "MemAvailable"}:
                figures[name] = int(rest.split()[0]) // 1024
    except OSError:
        pass
    return figures


def pool_identity(prepared: dict) -> dict:
    """What the sampler, the fitted transforms and the budget accounting see: keys, mask, targets."""
    keys = prepared["train"].keys
    dates = keys["date"].to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes()
    instruments = "\x00".join(keys["instrument"].astype(str)).encode("utf-8")
    targets = np.ascontiguousarray(prepared["targets"], dtype="<f8").tobytes()
    mask = np.ascontiguousarray(prepared["fit_mask"], dtype="?").tobytes()
    return {"keys": int(len(keys)), "usable_samples": int(prepared["fit_mask"].sum()),
            "keys_sha256": sha256(dates, instruments), "targets_sha256": sha256(targets),
            "fit_mask_sha256": sha256(mask)}


def batch_identity(batch) -> dict:
    """Keys plus the exact window bytes of one already-built batch."""
    keys = batch.keys
    dates = keys["date"].to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes()
    instruments = "\x00".join(keys["instrument"].astype(str)).encode("utf-8")
    return {"keys": int(len(keys)), "keys_sha256": sha256(dates, instruments),
            "x_sha256": sha256(np.ascontiguousarray(batch.x).tobytes()),
            "observed_sha256": sha256(np.ascontiguousarray(batch.observed).tobytes()),
            "history_dates_sha256": sha256(np.ascontiguousarray(batch.history_dates).tobytes()),
            "shape": list(batch.x.shape)}


def fit_pass(config, prepared: dict, validation, steps: int) -> dict:
    """One ``fit`` on the driver's pool: loss trace, step intervals and the fitted parameters."""
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

    torch.nn.functional.mse_loss, torch.optim.Adam.step = recorded_mse, marked_step
    started = time.perf_counter()
    try:
        progress = model.fit(prepared["train"], prepared["targets"],
                             sample_mask=prepared["fit_mask"], step_budget=steps)
    finally:
        torch.nn.functional.mse_loss, torch.optim.Adam.step = original_mse, original_step
    seconds = time.perf_counter() - started
    intervals = np.diff(np.asarray(marks, dtype=np.float64))
    measured = intervals[WARMUP_STEPS:] if len(intervals) > WARMUP_STEPS else intervals
    return {"model": model, "initial": initial, "losses": losses, "progress": progress,
            "state": bytes_of(model.network.state_dict()), "device": str(model.device),
            "scores": model.predict(validation), "seconds": round(seconds, 2),
            "step_ms": None if not len(measured) else {
                "median": round(float(np.median(measured) * 1000), 3),
                "p10": round(float(np.percentile(measured, 10) * 1000), 3),
                "p90": round(float(np.percentile(measured, 90) * 1000), 3)}}


def scoring_evidence(config, model, bundle, validation_batch) -> dict:
    """A pool over the validation window must score exactly like the materialised batch.

    The pool is built from the materialised pass's own bar frame restricted to the window plus its
    warm-up, so the windows are the ones the frame path uses and only the pool/``_predict``
    chunking differs. The draws are recorded to show the chunking actually happened.
    """
    split = getattr(config.splits, config.evaluation_split)
    start = np.datetime64(split.start) - np.timedelta64(int(config.data.history_days), "D")
    window = bundle.bars.loc[bundle.bars["date"] >= start].reset_index(drop=True)
    keys = inference_keys(bundle, split, config.data)
    pool = StreamingPool.build(window, keys, config.data)
    draws, draw = [], pool.batch

    def counting(indices):
        draws.append(len(np.asarray(indices)))
        return draw(indices)

    pool.batch = counting
    pool_scores = model._predict(pool)
    batch_scores = model._predict(validation_batch)
    return {"rows": int(len(pool)), "bar_rows": int(len(window)), "predict_chunks": draws,
            "scores_identical": bool(pool_scores.tobytes() == batch_scores.tobytes()),
            "max_abs_score_difference": float(np.abs(pool_scores - batch_scores).max()),
            "scores_sha256": {"pool": sha256(pool_scores.tobytes()),
                              "materialised": sha256(batch_scores.tobytes())}}


def run(args) -> dict:
    import torch

    config = load_config(args.config)
    report = {"config": str(Path(args.config).resolve()), "steps": args.steps, "seed": config.seed,
              "code": {"driver_sha256": file_sha256(ROOT / "scripts" / "run_training_trajectory.py"),
                       "adapter_sha256": file_sha256(ROOT / "modified_tlob" / "adapter.py"),
                       "pool_sha256": file_sha256(ROOT / "src" / "jinyiquant" / "pool.py")}}
    started = time.perf_counter()
    report["host"] = {"memory_before_pass": host_memory()}

    print("streaming pass", flush=True)
    reset_peak_rss()
    began = time.perf_counter()
    streaming = prepare_training(config, streaming=True)
    streaming_validation = prepare_validation(config, streaming["bundle"])
    stream_build = round(time.perf_counter() - began, 2)
    stream_identity = pool_identity(streaming)
    stream_validation_identity = batch_identity(streaming_validation)
    stream_footprint = {key: int(value) for key, value in streaming["train"].nbytes().items()}
    stream_pass = fit_pass(config, streaming, streaming_validation, args.steps)
    stream_rss = peak_rss_mb()
    del streaming, streaming_validation
    gc.collect()

    print("materialised pass", flush=True)
    reset_peak_rss()
    began = time.perf_counter()
    materialised = prepare_training(config)
    materialised_validation = prepare_validation(config, materialised["bundle"])
    materialised_build = round(time.perf_counter() - began, 2)
    materialised_identity = pool_identity(materialised)
    materialised_validation_identity = batch_identity(materialised_validation)
    materialised_pass = fit_pass(config, materialised, materialised_validation, args.steps)
    materialised_rss = peak_rss_mb()

    print("scoring pass", flush=True)
    scoring = scoring_evidence(config, materialised_pass["model"], materialised["bundle"],
                               materialised_validation)
    report["host"]["memory_after_materialised_build"] = host_memory()

    def pass_report(run: dict, build: float, rss: float, footprint: dict | None) -> dict:
        return {"build_seconds": build, "peak_rss_mb": rss, "step_ms": run["step_ms"],
                "fit_seconds": run["seconds"], "steps": run["progress"]["steps"],
                "samples_seen": run["progress"]["samples_seen"],
                "stop_reason": run["progress"]["stop_reason"],
                "footprint_bytes": footprint}

    validation_scores = {"streaming": sha256(stream_pass["scores"].tobytes()),
                         "materialised": sha256(materialised_pass["scores"].tobytes())}
    report["streaming"] = {"pool": stream_identity, "validation": stream_validation_identity,
                           **pass_report(stream_pass, stream_build, stream_rss, stream_footprint)}
    report["materialised"] = {"pool": materialised_identity,
                              "validation": materialised_validation_identity,
                              **pass_report(materialised_pass, materialised_build, materialised_rss,
                                            None)}
    report["scoring"] = scoring
    report["equivalence"] = equivalence_report(stream_pass, materialised_pass, args.steps)
    report["equivalence"]["validation_scores_sha256"] = validation_scores
    report["equivalence"]["validation_scores_identical"] = bool(
        stream_pass["scores"].tobytes() == materialised_pass["scores"].tobytes())
    report["equivalence"]["validation_batch_identical"] = bool(
        stream_validation_identity == materialised_validation_identity)
    report["equivalence"]["pool_identical"] = bool(stream_identity == materialised_identity)
    report["step_time"] = step_time_report([stream_pass["step_ms"]],
                                          [materialised_pass["step_ms"]])
    report["step_time"]["note"] = ("one round per pool, reported not gated: the streaming step "
                                   "cost was accepted at +5.38% before this wiring")
    report["environment"] = {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": materialised_pass["device"],
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compile": materialised_pass["progress"]["compile"]}
    report["seconds"] = round(time.perf_counter() - started, 2)
    report["passed"] = bool(
        report["equivalence"]["pool_identical"]
        and report["equivalence"]["validation_batch_identical"]
        and report["equivalence"]["validation_scores_identical"]
        and report["equivalence"]["steps_identical"]
        and report["equivalence"]["initial_state_identical"]
        and report["equivalence"]["losses_identical"]
        and report["equivalence"]["state_identical"]
        and scoring["scores_identical"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="registered recipe (run config) to replay")
    parser.add_argument("--steps", type=int, default=120, help="optimizer steps per pass")
    parser.add_argument("--output", default=None, help="report path (default: stdout only)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    report = run(args)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, report)
        print(f"wrote {destination}")
    summary = {"passed": report["passed"],
               "equivalence": {key: report["equivalence"][key] for key in
                               ("steps", "steps_identical", "initial_state_identical",
                                "losses_identical", "max_abs_loss_difference",
                                "state_tensors_compared", "state_tensors_differing",
                                "state_identical", "pool_identical",
                                "validation_batch_identical", "validation_scores_identical")},
               "scoring": {key: report["scoring"][key] for key in
                           ("rows", "predict_chunks", "scores_identical")},
               "step_time": {key: report["step_time"][key] for key in
                             ("materialised_step_ms", "streaming_step_ms", "step_increase_pct")},
               "memory": {"streaming_peak_rss_mb": report["streaming"]["peak_rss_mb"],
                          "materialised_peak_rss_mb": report["materialised"]["peak_rss_mb"]},
               "seconds": report["seconds"]}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
