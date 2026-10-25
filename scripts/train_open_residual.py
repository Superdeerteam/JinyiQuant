"""Fit the frozen-V2 09:30 opening-residual head on the frozen cross-sectional target space.

The trunk and the preprocessing contract are loaded read-only from the frozen V2
checkpoint; only the 320->32->1 head is optimised, and only on history-valid 09:30
samples of the declared training window. Labels are mapped into the checkpoint's
saved per-timestamp statistics, so no target statistic is re-estimated here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Support both `python -m scripts.train_open_residual` and direct execution of this file.
if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if (root / "src").is_dir():
        sys.path.insert(0, str(root / "src"))

import numpy as np
import pandas as pd
import torch

from jinyiquant.checks import check
from jinyiquant.config import RunConfig, load_config
from jinyiquant.data import build_batch, load_data, select_targets
from jinyiquant.io import file_sha256, redact, utc_now, write_json
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from modified_tlob.adapter import (
    OPEN_SAMPLE_TIME,
    TLOBAdapter,
    apply_frozen_timestamp_zscore,
    frozen_stats_sha256,
    open_market_mask,
    state_dict_sha256,
)
from modified_tlob.device import device_resources

KEYS = ["date", "instrument"]
STEPS, LEARNING_RATE, BATCH_SIZE = 1500, 1e-4, 32
HEAD_STATE = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def _with_seed(config, seed):
    if seed is None or seed == config.seed:
        return config
    return RunConfig.model_validate({**config.snapshot(), "seed": seed})


def _opening_probe(bars, targets, config, sample_time):
    """Mixed-section batch from the newest training date: real bars, no labels.

    The training pool only holds 09:30 keys, so invariance of every other section is
    checked on this probe against the frozen checkpoint's own predictions.
    """
    date = targets.date.dt.normalize().max()
    instruments = targets.loc[targets.date.dt.normalize() == date, "instrument"].drop_duplicates().to_numpy()
    keys = pd.concat([pd.DataFrame({
        "date": date + pd.Timedelta(hours=int(stamp[:2]), minutes=int(stamp[3:])),
        "instrument": instruments}) for stamp in config.data.sample_times], ignore_index=True)
    probe = build_batch(bars, keys, config.data)
    missing = (probe.observed[..., :probe.history_feature_count].any(axis=2).sum(axis=1)
               < config.data.min_history)
    return probe, open_market_mask(probe.keys, sample_time) & ~missing


def open_residual_training(config_path, base_weights, output_root=None, *, device="auto", steps=STEPS,
                           learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, sample_time=OPEN_SAMPLE_TIME,
                           seed=None, expected_base_sha256=None):
    """Train the opening head on frozen trunk outputs and write a self-contained archive."""
    config = _with_seed(load_config(config_path), seed)
    if config.model.kind != "modified_tlob":
        raise ValueError("Open residual training requires modified_tlob")
    if config.evaluation_split != "validation":
        raise ValueError("Open residual training must stay on the development validation split")
    if config.model.target_centering != "timestamp_zscore_train_rms_matched":
        raise ValueError("Open residual training requires the checkpoint's frozen CSZ target space")
    if sample_time not in (config.data.training_sample_times or config.data.sample_times):
        raise ValueError("The opening sample time must be a declared training sample time")
    base_weights = Path(base_weights).resolve()
    base_sha256 = file_sha256(base_weights)
    if expected_base_sha256 is not None and base_sha256 != expected_base_sha256:
        raise ValueError("Frozen trunk weights do not match the declared sha256")
    run_dir = new_run_dir(Path(output_root or config.output_root).resolve())
    recorder = Recorder(run_dir, config.experiment_id)
    started = time.monotonic()
    manifest = {"schema_version": 1, "run_id": run_dir.name, "experiment_id": config.experiment_id,
                "hypothesis": config.hypothesis, "status": "running", "started_at": utc_now(),
                "scope": "frozen_trunk_0930_residual_head_only",
                "training": {"sample_time": sample_time, "seed": config.seed, "steps": steps,
                             "learning_rate": learning_rate, "batch_size": batch_size,
                             "base_weights": str(base_weights), "base_weights_sha256": base_sha256}}
    write_json(run_dir / "manifest.json", manifest)
    recorder.event("run", "started", base_weights=str(base_weights), base_weights_sha256=base_sha256,
                   seed=config.seed, steps=steps, learning_rate=learning_rate, batch_size=batch_size)
    try:
        manifest.update(code=capture_source(run_dir), environment=environment())
        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "config.json", config.snapshot())
        (run_dir / "config.sha256").write_text(config.checksum + "\n", encoding="utf-8")
        recorder.event("loading", "started", path=str(base_weights))
        model = TLOBAdapter.load(base_weights, device=device, recorder=recorder)
        declared = config.data.model_dump(mode="json")
        if any(declared[key] != value for key, value in model.preprocessing.items()):
            raise ValueError("Declared data configuration does not match the frozen checkpoint preprocessing")
        if model.config["target_centering"] != config.model.target_centering:
            raise ValueError("Frozen checkpoint is not in the declared target-centering space")
        write_json(run_dir / "device.json", {"selected": str(model.device), "requested": device})
        base_state = {name: tensor.detach().cpu().clone()
                      for name, tensor in model.network.state_dict().items()}
        recorder.event("loading", "model_loaded", device=str(model.device), frozen=True,
                       base_state_sha256=state_dict_sha256(base_state), target_scale=model.target_scale,
                       intercept=model.intercept, target_centering=model.config["target_centering"])

        recorder.event("data", "loading", split="train", sample_time=sample_time)
        bundle = load_data(config.data, start=config.splits.train.start, end=config.splits.train.end)
        # 128-bar windows are float32 either way, so halving the resident bar table halves the
        # peak of the 60-day window build without changing a single training input value.
        for column in config.data.feature_columns:
            bundle.bars[column] = bundle.bars[column].astype(np.float32)
        for event in bundle.cache_events:
            recorder.event("data_cache", event["event"], **{k: v for k, v in event.items() if k != "event"})
        for event in bundle.data_events:
            recorder.event("data", event["event"], level="WARNING", **{k: v for k, v in event.items() if k != "event"})
        write_json(run_dir / "data.json", bundle.metadata)
        targets = select_targets(bundle.labels, config.splits.train, config.data.label_horizon_minutes,
                                 sample_times=(sample_time,))
        if targets.empty:
            raise ValueError("The opening training window contains no eligible labels")
        if len(targets) > config.budget.max_train_samples:
            raise ValueError("Opening training sample budget exceeded; declare a shorter window")
        batch = build_batch(bundle.bars, targets[KEYS], config.data)
        aligned = batch.keys.merge(targets[KEYS + ["target"]], on=KEYS, how="left", validate="one_to_one")
        if aligned.target.isna().any():
            raise ValueError("Missing labels after exact key alignment")
        write_json(run_dir / "coverage.json", batch.coverage)
        recorder.event("preprocessing", "completed", samples=len(targets), coverage=batch.coverage)

        raw_target = aligned.target.to_numpy(dtype=np.float32)
        transformed, transform = apply_frozen_timestamp_zscore(batch.keys, raw_target,
                                                               model.target_centering_stats)
        optimization_targets = (transformed.astype(np.float64) / model.target_scale).astype(np.float32)
        if not np.isfinite(optimization_targets).all():
            raise FloatingPointError("Nonfinite optimization targets")
        recorder.event("preprocessing", "target_transform_used", **transform)

        baseline = model.predict(batch)
        probe, probe_opening = _opening_probe(bundle.bars, targets, config, sample_time)
        del bundle, aligned  # the raw bars are no longer needed; free them before training
        frozen = TLOBAdapter.load(base_weights, device=device)
        probe_baseline = frozen.predict(probe)
        model.attach_open_residual(seed=config.seed)
        attached = model.predict(batch)
        progress = model.fit_open_residual(batch, optimization_targets, max_steps=steps,
                                           learning_rate=learning_rate, batch_size=batch_size)
        after = model.predict(batch)
        probe_after = model.predict(probe)
        missing = (batch.observed[..., :model.lob_feature_count].any(axis=2).sum(axis=1)
                   < model.preprocessing["min_history"])
        applicable = open_market_mask(batch.keys, sample_time) & ~missing
        base_mse = float(np.square(baseline[applicable] - transformed[applicable]).mean())
        fitted_mse = float(np.square(after[applicable] - transformed[applicable]).mean())
        metrics = {
            "scope": "in-sample residual fit only; out-of-sample paired comparison is a separate run",
            "training": progress, "target_transform": transform,
            "target_scale": model.target_scale, "base_weights_sha256": base_sha256,
            "in_sample": {"samples": int(applicable.sum()), "base_mse": base_mse, "fitted_mse": fitted_mse,
                          "mse_reduction": None if base_mse <= 0 else 1 - fitted_mse / base_mse,
                          "target_std": float(transformed[applicable].std()),
                          "base_prediction_std": float(baseline[applicable].std()),
                          "fitted_prediction_std": float(after[applicable].std()),
                          "correction_mean": float((after - baseline)[applicable].mean()),
                          "correction_std": float((after - baseline)[applicable].std())},
            "coverage": {"samples": len(targets), "history_valid": int(applicable.sum()),
                         "insufficient_history": int(missing.sum())},
        }
        write_json(run_dir / "metrics.json", metrics)
        pd.DataFrame({"date": batch.keys.date, "instrument": batch.keys.instrument,
                      "target": raw_target, "transformed_target": transformed,
                      "optimization_target": optimization_targets,
                      "base_prediction": baseline, "head_prediction": after}).to_parquet(
            run_dir / "train_targets.parquet", index=False)

        recorder.event("saving", "started", path="model.pt")
        model.save(run_dir / "model.pt")
        model.export_text(run_dir / "modified_tlob_weights.json")
        saved = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
        rerun = TLOBAdapter.load(run_dir / "model.pt", device=device)
        text_model = TLOBAdapter.load(run_dir / "modified_tlob_weights.json", device=device)
        base_unchanged = set(saved["state_dict"]) == set(base_state) and all(
            torch.equal(saved["state_dict"][name], base_state[name]) for name in base_state)
        checks = [
            check("base_weights_read_only", file_sha256(base_weights) == base_sha256,
                  path=str(base_weights), sha256=base_sha256),
            check("base_state_unchanged", base_unchanged,
                  frozen_parameters=int(sum(p.numel() for p in model.network.parameters())),
                  trunk_grad_parameters=int(sum(p.grad is not None for p in model.network.parameters())),
                  base_state_sha256=state_dict_sha256(base_state)),
            check("head_checkpoint_schema", saved["schema_version"] == 4
                  and set(saved["open_residual"]["state_dict"]) == HEAD_STATE,
                  schema_version=saved["schema_version"],
                  base_state_sha256=saved["open_residual"]["base_state_sha256"]),
            check("zero_initialization_identity", np.array_equal(baseline, attached), samples=len(baseline)),
            check("opening_only_application",
                  np.array_equal(probe_baseline[~probe_opening], probe_after[~probe_opening])
                  and not np.array_equal(probe_baseline[probe_opening], probe_after[probe_opening]),
                  opening_samples=int(probe_opening.sum()),
                  non_opening_samples=int((~probe_opening).sum()),
                  probe_date=str(probe.keys.date.dt.normalize().max().date())),
            check("frozen_target_space", transform["stats_sha256"] ==
                  frozen_stats_sha256(model.target_centering_stats),
                  stats_sha256=transform["stats_sha256"],
                  checkpoint_per_timestamp_sha256=transform["checkpoint_per_timestamp_sha256"],
                  timestamps=transform["timestamps"], rows=transform["rows"]),
            check("checkpoint_roundtrip",
                  np.array_equal(rerun.predict(batch), after) and np.array_equal(text_model.predict(batch), after)
                  and rerun.target_scale == model.target_scale and rerun.open_residual_head is not None
                  and rerun.open_residual_metadata["base_state_sha256"] == saved["open_residual"]["base_state_sha256"],
                  target_scale=rerun.target_scale,
                  text_weights_sha256=file_sha256(run_dir / "modified_tlob_weights.json")),
            check("training_sample_scope", set(targets.date.dt.strftime("%H:%M")) == {sample_time}
                  and bool(((targets.date >= config.splits.train.start)
                            & (targets.date < config.splits.train.end)).all()),
                  sample_time=sample_time, samples=len(targets),
                  start=str(targets.date.min()), end=str(targets.date.max())),
        ]
        passed = all(item["passed"] for item in checks)
        write_json(run_dir / "checks.json", {"schema_version": 1, "passed": passed, "checks": checks})
        open_residual = {"base_weights": str(base_weights), "base_weights_sha256": base_sha256,
                         "base_state_sha256": state_dict_sha256(base_state), "seed": config.seed,
                         "checkpoint_seed": model.seed,
                         "sample_time": sample_time, "steps": steps, "learning_rate": learning_rate,
                         "batch_size": batch_size, "target_scale": model.target_scale,
                         "training": progress, "target_transform": transform,
                         "weights_sha256": file_sha256(run_dir / "modified_tlob_weights.json")}
        write_json(run_dir / "open_residual.json", open_residual)
        (run_dir / "report.txt").write_text(
            f"Run: {run_dir.name}\nFrozen trunk: {base_weights} ({base_sha256})\n"
            f"Head: 320->32->1, zero-initialised last layer, {progress['samples']} history-valid 09:30 samples, "
            f"{progress['steps']} steps\nIn-sample MSE {base_mse:.6g} -> {fitted_mse:.6g}\n"
            f"Checks: {sum(item['passed'] for item in checks)}/{len(checks)} passed. "
            "Evidence: in-sample fit only; promotion needs the predeclared paired out-of-sample comparison.\n",
            encoding="utf-8")
        if not passed:
            raise ValueError("Open residual acceptance checks failed; see checks.json")
        recorder.event("open_residual", "completed", samples=progress["samples"], steps=progress["steps"],
                       in_sample=metrics["in_sample"], resources=device_resources(model.device))
        finish(run_dir, "completed", 0, resources={"wall_seconds": time.monotonic() - started,
                                                  **device_resources(model.device)})
        return run_dir
    except BaseException as exc:
        recorder.failure("open_residual", exc)
        finish(run_dir, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
               130 if isinstance(exc, KeyboardInterrupt) else 1,
               resources={"wall_seconds": time.monotonic() - started},
               error={"type": type(exc).__name__, "reason": redact(str(exc))})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-weights", required=True, type=Path, help="Frozen V2 checkpoint (read-only)")
    parser.add_argument("--output-root", type=Path, help="Parent for unique run archives")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sample-time", default=OPEN_SAMPLE_TIME, help="Opening section HH:MM")
    parser.add_argument("--seed", type=int, help="Override the trunk seed recorded in the config")
    parser.add_argument("--expected-base-sha256", help="Fail unless the frozen weights match this digest")
    args = parser.parse_args(argv)
    options = vars(args)
    try:
        path = open_residual_training(options.pop("config"), **options)
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"exit_code": 1, "error_type": type(exc).__name__, "reason": redact(str(exc))}))
        return 1
    print(json.dumps({"exit_code": 0, "output_dir": str(path),
                      "weights": str(path / "modified_tlob_weights.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
