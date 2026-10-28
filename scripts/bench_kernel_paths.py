"""Steady-state kernel-path benchmark for the training loop.

Measurement harness for two trainer changes: replacing the per-step host-visible
finiteness/loss synchronisation with sampled guards plus device-side accumulation, and
optionally compiling the trunk math with ``torch.compile`` (CUDA Graphs come in through
``mode="reduce-overhead"``). It runs the production ``TLOBAdapter.fit`` path on one
prepared training pool and derives the per-step cost from the trainer's own ``progress``
markers, so data preparation, first-call autotuning and the compile/equivalence probe
never enter the number::

    step_ms = 1000 * (elapsed[last] - elapsed[first]) / (step[last] - step[first])

``--warmup-steps`` is also the marker interval, and the step budget is exactly
``warmup + measure``, so the measured window holds only steady-state steps and no epoch
boundary, checkpoint, validation pass or log-cadence change.

Variants are presets over the model config. ``legacy`` applies no override, so running
this script from a checkout of the previous code measures the before/after reference.

Usage::

    python -m scripts.bench_kernel_paths --config scratch/bench_kernel_10d_local.json \\
        --variants legacy,eager-strict,eager-sampled,aot-eager \\
        --warmup-steps 200 --measure-steps 200 --output scratch/bench_kernel_2060.json
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from jinyiquant.config import RunConfig
from jinyiquant.data import KEYS, Batch, build_batch, load_data, select_targets
from jinyiquant.runner import _aligned_targets, inference_keys
from modified_tlob.adapter import TLOBAdapter

# Variant label -> model-config overrides. The labels are stable so results from
# different hosts/checkouts can be compared directly.
VARIANTS = {
    # Previous behaviour (the reference run is started from a HEAD checkout).
    "legacy": {},
    # Only the per-step loss readout moved to the device; guards stay per step.
    "eager-strict": {"compile_mode": "eager", "finite_check_every": 1},
    # The shipped configuration: sampled guards and device-side loss accumulation.
    "eager-sampled": {"compile_mode": "eager", "finite_check_every": 50},
    "compile-default": {"compile_mode": "default", "finite_check_every": 50},
    "compile-reduce-overhead": {"compile_mode": "reduce-overhead", "finite_check_every": 50},
    # Backend that needs no Triton: the diagnostic option on non-Linux hosts.
    "aot-eager": {"compile_mode": "aot-eager", "finite_check_every": 50},
}
KEPT_EVENTS = {"progress", "compile_enabled", "compile_fallback", "kernel_path"}


def _apply_overrides(raw, items):
    for item in items or ():
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override must be key=value, got {item!r}")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return raw


def _collector():
    events = []

    def event(stage, name, **context):
        events.append({"stage": stage, "event": name, **context})

    return events, SimpleNamespace(event=event)


def _prepare_pool(config, events, *, with_validation=False):
    """Read the training split exactly like the production runner does."""
    recorder = SimpleNamespace(event=lambda stage, name, **context: events.append(
        {"stage": stage, "event": name, **context}))
    start, end = config.splits.train.start, config.splits.train.end
    if with_validation:
        end = getattr(config.splits, config.evaluation_split).end
    bundle = load_data(config.data, start=start, end=end)
    for entry in bundle.cache_events + bundle.data_events:
        recorder.event("data", entry["event"], **{k: v for k, v in entry.items() if k != "event"})
    training_times = config.data.training_sample_times or config.data.sample_times
    targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                             sample_times=training_times)
    train = build_batch(bundle.bars, targets[KEYS], config.data)
    targets = _aligned_targets(train, targets)
    mask = train.effective_rows.sum(axis=1) >= config.data.min_history
    validation = None
    if with_validation:
        split = getattr(config.splits, config.evaluation_split)
        validation = build_batch(bundle.bars, inference_keys(bundle, split, config.data), config.data)
    return train, targets.target.to_numpy(), mask, validation


def _probe_batch(batch, size):
    size = min(size, len(batch.keys))
    return Batch(keys=batch.keys.iloc[:size].reset_index(drop=True),
                 x=np.ascontiguousarray(batch.x[:size]),
                 observed=np.ascontiguousarray(batch.observed[:size]),
                 history_dates=batch.history_dates[:size],
                 coverage=batch.coverage,
                 history_feature_count=batch.history_feature_count)


def _synthetic_pool(config, samples):
    """Shape-faithful synthetic training pool: same lookback/features/sample times.

    Removes the data-IO prerequisite so the same pooled benchmark can run on any host
    (and on a previous-code checkout) with identical bytes for both runs. Values are
    random, so only timing and exact-equivalence evidence come from these runs.
    """
    data = config.data.model_dump(mode="json")
    times = tuple(data.get("training_sample_times") or data["sample_times"])
    lookback, features = int(data["lookback"]), len(data["feature_columns"])
    per_timestamp = 20
    days = max(1, -(-samples // (per_timestamp * len(times))))
    dates = pd.bdate_range("2023-04-14", periods=days)
    stamps = np.repeat(np.arange(days * len(times)), per_timestamp)[:samples]
    keys = pd.DataFrame({
        "date": pd.to_datetime([f"{dates[index // len(times)].date()} {times[index % len(times)]}"
                                for index in stamps]),
        "instrument": np.resize([f"S{index:03d}" for index in range(per_timestamp)], samples)})
    rng = np.random.default_rng(config.seed)
    x = rng.normal(size=(samples, lookback, features)).astype(np.float32)
    batch = Batch(keys=keys, x=x, observed=np.ones(x.shape, dtype=bool),
                  history_dates=np.full((samples, lookback), np.datetime64("NaT")),
                  coverage={}, history_feature_count=features)
    targets = rng.normal(scale=0.01, size=samples).astype(np.float32)
    return batch, targets, np.ones(samples, dtype=bool), None


def _state_digest(network):
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def _environment():
    def version(distribution):
        try:
            import importlib.metadata as metadata
            return metadata.version(distribution)
        except Exception:
            return None

    try:  # documentation only: reduce-overhead enables CUDA-graph trees in inductor
        from torch._inductor import config as inductor_config
        graphs = {"cudagraphs": bool(inductor_config.triton.cudagraphs),
                  "cudagraph_trees": bool(inductor_config.triton.cudagraph_trees)}
    except Exception:
        graphs = None
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": version("triton"),
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "device_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "inductor_graphs": graphs,
    }


def _markers(events):
    return [(int(entry["step"]), float(entry["elapsed_seconds"]))
            for entry in events if entry["event"] == "progress"]


def _run_variant(config, pool, label, warmup_steps, measure_steps):
    model_config = config.model.model_dump(mode="json")
    model_config["log_every_steps"] = warmup_steps
    model_config.update(VARIANTS[label])
    events, recorder = _collector()
    adapter = TLOBAdapter(model_config, config.data, seed=config.seed, recorder=recorder)
    batch, targets, mask, validation = pool
    started = time.perf_counter()
    progress = adapter.fit(batch, targets, sample_mask=mask, step_budget=warmup_steps + measure_steps)
    wall_seconds = time.perf_counter() - started
    markers = _markers(events)
    steps = [step for step, _ in markers]
    first, last = markers[0], markers[-1]
    measured_steps = last[0] - first[0]
    if measured_steps < measure_steps:
        raise RuntimeError(f"{label}: measured window is {measured_steps} steps, expected >= {measure_steps}; "
                           f"markers={steps}")
    measured_seconds = last[1] - first[1]
    step_ms = 1000.0 * measured_seconds / measured_steps
    probe = _probe_batch(batch, config.model.batch_size)
    predictions = adapter.predict(probe)
    validation_seconds = None
    if validation is not None:
        # Same call path as the trajectory driver's per-budget-point scoring
        # (``_predict``: eval mode, inference_mode, batch_size chunking) so a graph
        # rebuild or first-call cost is inside the measured round trip.
        validation_started = time.perf_counter()
        adapter._predict(validation)
        validation_seconds = time.perf_counter() - validation_started
    report = progress.get("compile") if isinstance(progress, dict) else None
    result = {
        "label": label,
        "overrides": VARIANTS[label],
        "kernel_status": (report or {}).get("status", "legacy"),
        "compile": report,
        "finite_check_every": model_config.get("finite_check_every"),
        "fit_wall_seconds": wall_seconds,
        "steps": int(progress["steps"]) if isinstance(progress, dict) else None,
        "marker_steps": steps,
        "measured_window_steps": measured_steps,
        "measured_window_seconds": measured_seconds,
        "step_ms": step_ms,
        "steps_per_second": 1000.0 / step_ms,
        "samples_per_second": 1000.0 * config.model.batch_size / step_ms,
        "validation_seconds": validation_seconds,
        "validation_samples": (None if validation is None else int(len(validation.keys))),
        "state_digest": _state_digest(adapter.network),
        "prediction_digest": hashlib.sha256(np.ascontiguousarray(predictions).tobytes()).hexdigest(),
        "events": [entry for entry in events if entry["event"] in KEPT_EVENTS],
    }
    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, predictions


def _equivalence(results, predictions):
    reference = next((label for label in results if results[label]["kernel_status"] == "eager"), None)
    if reference is None:
        return {"reference": None, "note": "no eager variant in this run; nothing to compare against"}
    base = predictions[reference]
    comparisons = {}
    for label, result in results.items():
        candidate = predictions[label]
        difference = float(np.max(np.abs(candidate - base))) if len(candidate) == len(base) else None
        comparisons[label] = {
            "reference": reference,
            "state_identical": result["state_digest"] == results[reference]["state_digest"],
            "prediction_digest_identical": result["prediction_digest"] == results[reference]["prediction_digest"],
            "max_abs_prediction_difference": difference,
            "forward_equivalence_atol": (result["compile"] or {}).get("equivalence_atol"),
            "forward_max_abs_difference": (result["compile"] or {}).get("max_abs_forward_difference"),
        }
    return {"reference": reference, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="RunConfig JSON (same schema as production configs)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted override into the config, e.g. --set model.device=cuda")
    parser.add_argument("--variants", default="eager-strict,eager-sampled",
                        help=f"comma-separated presets from {sorted(VARIANTS)}")
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="excluded warmup steps and marker interval (also the first timed step)")
    parser.add_argument("--measure-steps", type=int, default=200, help="minimum measured steps")
    parser.add_argument("--probe-batch", type=int, default=32, help="rows used for the state prediction digest")
    parser.add_argument("--validation", action="store_true",
                        help="also build the evaluation split and time one full scoring round trip")
    parser.add_argument("--synthetic-samples", type=int, default=0,
                        help="build a shape-faithful synthetic pool instead of reading data (0 = real data)")
    parser.add_argument("--output", required=True, help="JSON result path")
    args = parser.parse_args(argv)
    labels = [label.strip() for label in args.variants.split(",") if label.strip()]
    unknown = sorted(set(labels) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("warmup and measure steps must be positive")
    if args.measure_steps % args.warmup_steps:
        parser.error("measure-steps must be a multiple of warmup-steps so the markers bracket exactly")

    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = RunConfig.model_validate(_apply_overrides(raw, args.set))
    pool_events = []
    prepared = (_synthetic_pool(config, args.synthetic_samples) if args.synthetic_samples
                else _prepare_pool(config, pool_events, with_validation=args.validation))
    batch, targets, mask = prepared[:3]
    usable = int(mask.sum())
    steps_per_epoch = int(np.ceil(usable / config.model.batch_size))
    protocol = {
        "config": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "train_split": [str(config.splits.train.start), str(config.splits.train.end)],
        "pool_samples": int(len(targets)),
        "usable_samples": usable,
        "batch_size": config.model.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "dataset": (f"synthetic_shape_faithful:{args.synthetic_samples}" if args.synthetic_samples
                    else "real_training_split"),
        "validation_split": ([str(config.splits.validation.start), str(config.splits.validation.end)]
                             if args.validation else None),
        "timing": "trainer progress markers: (elapsed[last]-elapsed[first])/(step[last]-step[first]); "
                  "no epoch boundary, validation, checkpoint or data preparation inside the window; "
                  "the compile/equivalence probe runs before the timed window",
    }
    if steps_per_epoch <= args.warmup_steps + args.measure_steps:
        raise RuntimeError(f"pool yields {steps_per_epoch} steps/epoch, need more than "
                           f"{args.warmup_steps + args.measure_steps} to stay inside one epoch")

    results, predictions = {}, {}
    for label in labels:
        result, prediction = _run_variant(config, prepared, label, args.warmup_steps, args.measure_steps)
        results[label] = result
        predictions[label] = prediction
        validation_note = ("" if result["validation_seconds"] is None else
                           f"  validation={result['validation_seconds'] * 1000:.0f}ms"
                           f"/{result['validation_samples']} rows")
        print(f"{label:<24} status={result['kernel_status']:<9} "
              f"step_ms={result['step_ms']:.3f} steps/s={result['steps_per_second']:.1f} "
              f"window={result['measured_window_steps']} steps  fit_wall={result['fit_wall_seconds']:.1f}s"
              f"{validation_note}")
        if result["compile"] and result["compile"].get("fallback"):
            print(f"{'':<24} fallback: {result['compile']['fallback']}")

    payload = {
        "protocol": protocol,
        "environment": _environment(),
        "variants": results,
        "equivalence": _equivalence(results, predictions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                 default=lambda value: value.item() if hasattr(value, "item") else str(value)),
                      encoding="utf-8")
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
