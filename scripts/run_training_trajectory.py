"""Run one continuous training trajectory to its pre-registered budget points.

The recipe stays fixed; only the training amount changes. The budget points are checkpoints of
one uninterrupted optimizer trajectory, so every segment after the first resumes the exact
optimizer state, RNG streams and sampler position of the previous one, and each segment records
the sample exposure it actually produced before the next budget point starts. Evaluation stays
on the validation split; nothing here reads the sealed final interval.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import torch

from jinyiquant.config import RunConfig, load_config
from jinyiquant.data import build_batch, load_data, select_targets
from jinyiquant.io import read_json, write_json
from jinyiquant.models import create_model, model_filename
from jinyiquant.runner import inference_keys
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir
from jinyiquant.trajectory import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_METADATA_FILENAME,
    derive_budget_steps,
    file_sha256,
    load_plan,
    plan_digest,
    reconcile_exposure,
)
from modified_tlob.adapter import TLOBAdapter

BUDGET_ONLY_OVERRIDES = {"model.max_steps", "output_root"}


def trajectory_config(plan: dict, repo: Path, output_root: Path) -> tuple[RunConfig, dict]:
    """The fixed recipe with only the budget control relaxed, plus the recorded deviation."""
    recipe_path = (repo / plan["recipe_config"]).resolve()
    if file_sha256(recipe_path) != plan["recipe_sha256"]:
        raise ValueError("Recipe configuration changed since pre-registration; "
                         "a trajectory may only change the training amount")
    recipe = load_config(recipe_path)
    last_point = plan["budget_points"][-1]["effective_epochs"]
    model = {**recipe.model.model_dump(),
             "max_steps": None,
             "epochs": max(recipe.model.epochs, int(math.ceil(last_point)) + 1)}
    configured = RunConfig.model_validate({**recipe.model_dump(), "model": model,
                                           "output_root": output_root})
    deviations = sorted(_flatten_changes(recipe.snapshot(), configured.snapshot()))
    unexpected = [path for path in deviations if path not in BUDGET_ONLY_OVERRIDES]
    if unexpected:
        raise ValueError(f"Trajectory run changed more than the training amount: {unexpected}")
    return configured, {"recipe_config": plan["recipe_config"],
                        "recipe_sha256": plan["recipe_sha256"],
                        "deviations": deviations}


def _flatten_changes(left, right, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            paths += _flatten_changes(left.get(key), right.get(key), f"{prefix}{key}.")
        return paths
    return [] if left == right else [prefix[:-1]]


def _aligned_targets(batch, targets):
    merged = batch.keys.merge(targets, on=["date", "instrument"], how="left", validate="one_to_one")
    if merged.target.isna().any():
        raise ValueError("Missing targets after exact key alignment")
    return merged


def prepare_training(config: RunConfig) -> dict:
    """Build the fixed-recipe training pool exactly as the standard pipeline does."""
    split = getattr(config.splits, config.evaluation_split)
    bundle = load_data(config.data, start=config.splits.train.start, end=split.end)
    times = config.data.training_sample_times or config.data.sample_times
    train_targets = select_targets(bundle.labels, config.splits.train,
                                   config.data.label_horizon_minutes, sample_times=times)
    if train_targets.empty:
        raise ValueError("Training must contain eligible targets")
    train = build_batch(bundle.bars, train_targets[["date", "instrument"]], config.data)
    train_targets = _aligned_targets(train, train_targets)
    fit_mask = train.effective_rows.sum(axis=1) >= config.data.min_history
    if not fit_mask.any():
        raise ValueError("Training pool has no sample with enough history")
    return {"bundle": bundle, "train": train, "targets": train_targets.target.to_numpy(),
            "fit_mask": fit_mask}


def prepare_validation(config: RunConfig, bundle):
    """The validation split the budget points are scored on; the final split is never read."""
    split = getattr(config.splits, config.evaluation_split)
    return build_batch(bundle.bars, inference_keys(bundle, split, config.data), config.data)


def run_trajectory(plan: dict, config: RunConfig, prepared: dict, validation, run_dir: Path) -> dict:
    """One continuous checkpointed trajectory, scoring every pre-registered budget point."""
    bundle, train = prepared["bundle"], prepared["train"]
    targets, fit_mask = prepared["targets"], prepared["fit_mask"]
    usable = int(fit_mask.sum())
    budget = derive_budget_steps(plan["budget_points"], usable, config.model.batch_size)
    derivation = {"usable_samples": usable, "batch_size": config.model.batch_size,
                  "rule": "steps = round(effective_epochs * usable_samples / batch_size)",
                  "budget_points": budget}
    run_dir = Path(run_dir)
    rec = Recorder(run_dir, config.experiment_id)
    write_json(run_dir / "config.json", config.snapshot())
    (run_dir / "config.sha256").write_text(config.checksum + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "run_id": run_dir.name, "experiment_id": config.experiment_id,
                "hypothesis": config.hypothesis, "status": "running",
                "config_sha256": config.checksum, "seed": config.seed,
                "evaluation_split": config.evaluation_split,
                "plan": {"path": plan.get("_path", ""), "sha256": plan_digest(plan)},
                "budget": derivation}
    write_json(run_dir / "manifest.json", manifest)
    manifest.update(code=capture_source(run_dir), environment=environment())
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "data.json", bundle.metadata)
    write_json(run_dir / "budget.json", derivation)
    rec.event("run", "started", config_sha256=config.checksum, plan_sha256=plan_digest(plan),
              usable_samples=usable, budget_points=[row["name"] for row in budget])
    point_dir = run_dir / "budget_points"
    point_dir.mkdir(exist_ok=True)
    checkpoint_dir = run_dir / "trajectory"
    points_dir = checkpoint_dir / "points"
    trajectory, status, exit_code, error = [], "completed", 0, None
    started = time.monotonic()
    try:
        payload = None
        for index, row in enumerate(budget, start=1):
            point_checkpoint = points_dir / f"{index:02d}-{row['name']}"
            declared = {"steps": row["steps"], "effective_epochs": row["effective_epochs"],
                        "effective_epochs_atol": float(plan["exposure"]["effective_epochs_atol"]),
                        "min_coverage": float(plan["exposure"]["min_coverage"])}
            model = create_model(config, recorder=rec)
            progress = model.fit(train, targets, sample_mask=fit_mask, step_budget=row["steps"],
                                 resume=payload, checkpoint_at_steps=[row["steps"]],
                                 checkpoint_dir=point_checkpoint, track_exposure=True)
            report = model.exposure_report(declared=declared)
            verdict = reconcile_exposure(report, declared)
            if not verdict["passed"]:
                raise ValueError(f"budget point {row['name']!r} exposure is inconsistent: "
                                 f"{verdict['issues']}")
            if progress["steps"] != row["steps"]:
                raise ValueError(f"budget point {row['name']!r} stopped at step {progress['steps']}")
            write_json(run_dir / "exposure.json", report)
            write_json(point_dir / f"{row['name']}.json",
                       {"budget_point": row, "progress": progress, "exposure": report,
                        "reconciliation": verdict})
            suffix = Path(model_filename(config.model.kind)).suffix
            model.save(run_dir / f"model-{row['name']}{suffix}")
            weights = run_dir / f"modified_tlob_weights-{row['name']}.json"
            model.export_text(weights)
            reloaded = TLOBAdapter.load(weights, device="cpu", recorder=rec)
            for key, value in model.network.state_dict().items():
                if not torch.equal(value.cpu(), reloaded.network.state_dict()[key]):
                    raise ValueError(f"exported weights do not reload at budget point {row['name']!r}")
            predictions = validation.keys.copy()
            predictions["score"] = model.predict(validation)
            predictions.to_parquet(point_dir / f"{row['name']}_predictions.parquet", index=False)
            # One self-contained training state per budget point; the rolling copy is a
            # convenience, never the only source.
            metadata = read_json(point_checkpoint / CHECKPOINT_METADATA_FILENAME)
            metadata["budget_point"] = {**row, "coverage": report["coverage"],
                                        "reconciliation_passed": verdict["passed"]}
            write_json(point_checkpoint / CHECKPOINT_METADATA_FILENAME, metadata)
            shutil.copyfile(point_checkpoint / CHECKPOINT_FILENAME,
                            checkpoint_dir / CHECKPOINT_FILENAME)
            write_json(checkpoint_dir / CHECKPOINT_METADATA_FILENAME, metadata)
            payload = torch.load(point_checkpoint / CHECKPOINT_FILENAME, map_location="cpu",
                                 weights_only=True)
            trajectory.append({"name": row["name"], "steps": progress["steps"],
                               "samples_seen": progress["samples_seen"],
                               "resumed": progress["resumed"],
                               "effective_epochs": report["effective_epochs"],
                               "coverage": report["coverage"],
                               "checkpoint": f"trajectory/points/{index:02d}-{row['name']}/"
                                             f"{CHECKPOINT_FILENAME}",
                               "model": f"model-{row['name']}{suffix}",
                               "weights": weights.name})
            write_json(run_dir / "trajectory.json",
                       {"schema_version": 1, "plan": plan_digest(plan),
                        "budget": seed_budget(budget), "completed_points": trajectory})
            rec.event("trajectory", "budget_point_reached", name=row["name"],
                      steps=progress["steps"], samples_seen=progress["samples_seen"],
                      effective_epochs=report["effective_epochs"], coverage=report["coverage"])
    except BaseException as exc:  # noqa: BLE001 - the run archive must record any failure
        rec.failure("trajectory", exc)
        status, exit_code = "failed", 1
        error = {"type": type(exc).__name__, "reason": str(exc)[:500]}
    finish(run_dir, status, exit_code,
           resources={"wall_seconds": time.monotonic() - started,
                      "gpu_used": config.model.device != "cpu"},
           error=error)
    return {"status": status, "exit_code": exit_code, "budget": derivation,
            "trajectory": trajectory, "error": error}


def seed_budget(budget: list[dict]) -> list[dict]:
    """The executed budget without the private bookkeeping keys of the derivation."""
    keys = ("name", "effective_epochs", "steps", "realised_effective_epochs")
    return [{key: row[key] for key in keys} for row in budget]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path,
                        default=Path("configs/training_trajectory_20261027_plan.json"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--budget-file", type=Path, default=None,
                        help="write the derived budget points and stop, without training")
    args = parser.parse_args()
    plan = {**load_plan(args.plan), "_path": args.plan.as_posix()}
    output_root = args.output_root or (args.repo / "runs" / plan["experiment_id"])
    config, recipe_guard = trajectory_config(plan, args.repo.resolve(), output_root.resolve())
    prepared = prepare_training(config)
    usable = int(prepared["fit_mask"].sum())
    budget = derive_budget_steps(plan["budget_points"], usable, config.model.batch_size)
    derivation = {"usable_samples": usable, "batch_size": config.model.batch_size,
                  "rule": "steps = round(effective_epochs * usable_samples / batch_size)",
                  "budget_points": budget}
    if args.budget_file is not None:
        write_json(args.budget_file, derivation)
        print(f"derived budgets -> {args.budget_file}: "
              f"{[(row['name'], row['steps']) for row in budget]}")
        return 0
    validation = prepare_validation(config, prepared["bundle"])
    run_dir = new_run_dir(config.output_root)
    result = run_trajectory(plan, config, prepared, validation, run_dir)
    print(f"{result['status']} {run_dir} points={len(result['trajectory'])} "
          f"guard={recipe_guard['deviations']}")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
