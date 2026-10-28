"""R14: pretrain on non-label minutes, then fine-tune on the labels at a matched budget.

One pre-registered plan drives one seed. The pretraining task is fixed (masked-segment
reconstruction on the minute columns the supervised recipe never samples), the pretraining
steps are a declared share of the control run's scored budget, and the fine-tune then trains
for exactly ``control steps - pretraining steps`` steps. Total optimizer steps, total samples
seen and the recipe/data/split identity are therefore the same as the pure-supervised control,
which is what makes the two arms comparable.

Guards, all fail-closed:

* the pretraining read stops at the training split end and ``bar_window_guard`` re-checks the
  bar extrema; the final (sealed) interval is never read;
* ``assert_train_window`` re-checks every pretraining key against the training split;
* the pretraining minute grid is recomputed from the recipe's own sample times and must equal
  the pre-registered grid, so a pretraining pool cannot silently reuse label columns;
* the control run's config must match this run's recipe except identity/output fields.

Usage (on hz-4, after ``source /mnt/work/run_env.sh``)::

    python scripts/run_pretrain_finetune.py --plan configs/r14_pretrain_441d_csz_seed11_plan.json \
        --output-root /mnt/work/runs/r14-pretrain-441d-seed11 --stream-bars
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_training_trajectory as driver  # noqa: E402

from jinyiquant.data import load_data  # noqa: E402
from jinyiquant.io import read_json, write_json  # noqa: E402
from jinyiquant.pool import training_pool  # noqa: E402
from jinyiquant.pretrain import (  # noqa: E402
    PretrainSpec,
    apply_checkpoint,
    assert_train_window,
    bar_window_guard,
    month_inventory,
    non_label_minutes,
    pretrain_keys,
    run_pretraining,
    save_checkpoint,
)
from jinyiquant.tracking import (Recorder, capture_source, environment, finish,
                                new_run_dir)  # noqa: E402
from jinyiquant.trajectory import SEALED_INTERVAL, derive_budget_steps, plan_digest  # noqa: E402

PLAN_SCHEMA = "r14-pretrain-plan/1"
IDENTITY_FIELDS = {"experiment_id", "hypothesis", "output_root"}


def pretrain_plan(path: Path) -> dict:
    """Read the R14 plan and refuse an under-specified one.

    The plan is deliberately stricter than a trajectory plan: one scored budget point, the
    pretraining task, the control run it is matched against, and the arithmetic that has to
    balance. ``load_plan`` is not used because a trajectory there requires two points.
    """
    document = read_json(path)
    for key in ("schema_version", "experiment_id", "recipe_config", "recipe_sha256",
                "budget_points", "lr_schedule", "exposure", "pretrain", "control",
                "budget_match"):
        if key not in document:
            raise ValueError(f"R14 plan is missing {key!r}")
    if document["schema_version"] != PLAN_SCHEMA:
        raise ValueError(f"Unsupported R14 plan schema_version {document['schema_version']!r}")
    points = document["budget_points"]
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError("An R14 plan declares exactly one scored budget point")
    point = points[0]
    if not isinstance(point.get("name"), str) or not point["name"]:
        raise ValueError("The scored budget point needs a name")
    if not float(point.get("effective_epochs", 0)) > 0:
        raise ValueError("The scored budget point needs a positive effective-epoch budget")
    for key in ("experiment_id", "path", "budget_point"):
        if key not in document["control"]:
            raise ValueError(f"R14 plan control block is missing {key!r}")
    for key in ("pretrain_steps", "finetune_steps", "control_steps", "rule"):
        if key not in document["budget_match"]:
            raise ValueError(f"R14 plan budget_match block is missing {key!r}")
    document["_path"] = Path(path).as_posix()
    return document


def control_record(plan: dict) -> dict:
    """The control run's scored budget point: its step count is the budget to match."""
    control = plan["control"]
    root = Path(control["path"])
    matches = []
    for path in sorted(root.glob("*/trajectory.json")):
        trajectory = read_json(path)
        row = next((value for value in trajectory["completed_points"]
                    if value["name"] == control["budget_point"]), None)
        if row is None:
            continue
        budget = read_json(path.parent / "budget.json")
        matches.append({"run_dir": path.parent.as_posix(), "steps": int(row["steps"]),
                        "effective_epochs": float(row["effective_epochs"]),
                        "usable_samples": int(budget["usable_samples"]),
                        "experiment_id": trajectory.get("experiment_id")
                        or read_json(path.parent / "config.json").get("experiment_id"),
                        "budget_point": control["budget_point"],
                        "config": read_json(path.parent / "config.json")})
    if len(matches) != 1:
        raise ValueError(f"Control {control['experiment_id']!r} needs exactly one run holding "
                         f"budget point {control['budget_point']!r}; found {len(matches)}")
    return matches[0]


def budget_basis(budget: dict) -> str:
    """The sentence the report has to quote: steps are matched, supervised samples are not."""
    supervised = budget["finetune_steps"] * budget["batch_size"]
    control = budget["control_steps"] * budget["batch_size"]
    return (f"steps only: pretraining and fine-tune together spend the control's "
            f"{budget['control_steps']} optimiser steps, so the arm sees {supervised} supervised "
            f"samples against the control's {control}")


def budget_record(budget: dict, *, plan_sha256: str, metrics: dict) -> dict:
    """The archived budget arithmetic, with the sample asymmetry spelled out in words."""
    supervised = budget["finetune_steps"] * budget["batch_size"]
    control_samples = budget["control_steps"] * budget["batch_size"]
    return {"pretrain_steps": budget["pretrain_steps"],
            "finetune_steps": budget["finetune_steps"],
            "total_steps": budget["total_steps"], "control_steps": budget["control_steps"],
            "matched": budget["matched"], "plan_sha256": plan_sha256,
            "pretrain_wall_seconds": metrics["wall_seconds"], "control": budget["control"],
            "scored_point": budget["scored_point"],
            "budget_basis": budget["budget_basis"],
            "interpretation": (
                "Optimiser steps are matched, supervised samples are not: the arm spends "
                f"{budget['finetune_steps']} steps x {budget['batch_size']} = {supervised} "
                f"supervised samples against the control's {budget['control_steps']} steps x "
                f"{budget['batch_size']} = {control_samples} (10% fewer). A small loss therefore "
                "cannot separate 'pretraining did not help' from 'the arm saw 10% less supervised "
                "data'; only a paired gain well above the seed noise, or the sample-matched "
                "follow-up arm, can tell them apart.")}


def matched_budget(plan: dict, config, control: dict, spec: PretrainSpec) -> dict:
    """The pre-registered arithmetic: pretrain + fine-tune steps == the control's steps."""
    usable = control["usable_samples"]
    steps = derive_budget_steps(plan["budget_points"], usable, config.model.batch_size)[0]
    finetune = control["steps"] - spec.steps
    if finetune < 1:
        raise ValueError("Pretraining consumes the whole control budget")
    if steps["steps"] != finetune:
        raise ValueError(f"The scored budget point derives {steps['steps']} fine-tune steps but "
                         f"the matched budget needs {finetune}")
    declared = plan["budget_match"]
    if int(declared["pretrain_steps"]) != spec.steps:
        raise ValueError("Plan pretrain_steps does not match the pretraining spec")
    if int(declared["finetune_steps"]) != finetune:
        raise ValueError("Plan finetune_steps does not match the derived fine-tune steps")
    if int(declared["control_steps"]) != control["steps"]:
        raise ValueError("Plan control_steps does not match the control run")
    budget = {"rule": declared["rule"], "control_steps": control["steps"],
              "pretrain_steps": spec.steps, "finetune_steps": finetune,
              "total_steps": spec.steps + finetune,
              "matched": spec.steps + finetune == control["steps"],
              "usable_samples": usable, "batch_size": config.model.batch_size,
              "control": {key: control[key] for key in ("experiment_id", "run_dir", "budget_point",
                                                       "steps", "effective_epochs")},
              "scored_point": steps}
    budget["budget_basis"] = budget_basis(budget)
    return budget


PATH_FIELDS = ("bars_path", "labels_path", "instruments_path", "instrument_map_path",
               "cache_dir")


def _canonical_snapshot(snapshot: dict) -> dict:
    """A snapshot whose filesystem paths compare as POSIX strings, whatever type they came in.

    ``load_config`` re-resolves the data paths against the recipe directory, so one side carries
    ``Path`` objects and the other plain strings straight out of the control run's config.json.
    """
    canonical = {key: value for key, value in snapshot.items() if key != "output_root"}
    data = dict(canonical.get("data") or {})
    for key in PATH_FIELDS:
        value = data.get(key)
        if isinstance(value, str | Path) and str(value):
            data[key] = Path(value).as_posix()
    if data:
        canonical["data"] = data
    return canonical


def recipe_identity(plan: dict, config, control: dict) -> dict:
    """The control run must be this recipe: same data, splits, model and evaluation."""
    changed = sorted(set(driver._flatten_changes(  # noqa: SLF001 - the driver's own comparison
        _canonical_snapshot(control["config"]), _canonical_snapshot(config.snapshot()))))
    unexpected = [path for path in changed if path.split(".")[0] not in IDENTITY_FIELDS
                  and path not in IDENTITY_FIELDS]
    if unexpected:
        raise ValueError(f"The control run is not this recipe: {unexpected}")
    return {"deviating_fields": changed, "identical_fields": "recipe_snapshot"}


def pretraining_pool(config, spec: PretrainSpec, *, minutes: list[str]) -> tuple:
    """Read the training-split bars only and index the non-label windows."""
    bundle = load_data(config.data, start=config.splits.train.start,
                       end=config.splits.train.end, stream_bars=True)
    if bundle.arrays is None:
        raise ValueError("Pretraining needs the streamed bar arrays, not a materialised frame")
    bars = bar_window_guard(bundle.arrays.dates, train_end=config.splits.train.end,
                            validation_start=getattr(config.splits, config.evaluation_split).start,
                            sealed_start=SEALED_INTERVAL[0])
    keys = pretrain_keys(bundle.universe, train_start=config.splits.train.start,
                         train_end=config.splits.train.end, minutes=minutes,
                         per_slot=spec.instruments_per_slot, seed=spec.seed)
    window = assert_train_window(keys, train_start=config.splits.train.start,
                                 train_end=config.splits.train.end)
    months = month_inventory(bundle.arrays.dates)
    pool_config = {**config.data.model_dump(), "sample_times": list(minutes),
                   "training_sample_times": list(minutes)}
    pool = training_pool(None, keys, pool_config, streaming=True, arrays=bundle.arrays)
    return bundle, pool, {"bars": bars, "keys": window, "minutes": list(minutes),
                          "months": months, "pool_coverage": pool.coverage}


NEWLINE = chr(10)


def write_archive_start(run_dir: Path, config, plan: dict, budget: dict) -> None:
    """Write the archive header before any training.

    ``run_trajectory`` writes the same files once the fine-tune starts; writing them first means
    a failure inside the pretraining phase is archived by ``finish`` instead of raising a second
    error about a missing manifest that hides the real one.
    """
    write_json(run_dir / "config.json", config.snapshot())
    (run_dir / "config.sha256").write_text(config.checksum + NEWLINE, encoding="utf-8")
    manifest = {"schema_version": 1, "run_id": run_dir.name,
                "experiment_id": config.experiment_id, "hypothesis": config.hypothesis,
                "status": "pretraining", "config_sha256": config.checksum, "seed": config.seed,
                "evaluation_split": config.evaluation_split,
                "plan": {"path": plan.get("_path", ""), "sha256": plan_digest(plan)},
                "budget": budget}
    write_json(run_dir / "manifest.json", manifest)
    manifest.update(code=capture_source(run_dir), environment=environment())
    write_json(run_dir / "manifest.json", manifest)


def verify_forward_compatibility(legacy_repo: Path, current_repo: Path, weights: Path) -> dict:
    """The arm has to reproduce the control's forward pass bit for bit, or it is not comparable.

    ``modified_tlob/model.py`` in this repository adds the representation interface the arm needs
    to pretrain the trunk. This check runs the control's own weights through both implementations
    and refuses the run when the outputs differ at all.
    """
    verifier = importlib.import_module("verify_mlplob_equivalence")
    record = verifier.verify(str(legacy_repo), str(current_repo), str(weights))
    if not record["bitwise_equal"]:
        raise ValueError(f"MLPLOB forward pass differs from {legacy_repo} by "
                         f"{record['max_abs_diff']}; the arm would not be comparable to the control")
    return record



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--stream-bars", action="store_true",
                        help="stream the supervised pool from the bar matrix (the R14 default)")
    parser.add_argument("--log-every-steps", type=int, default=200)
    parser.add_argument("--check-only", action="store_true",
                        help="validate the plan, the control match and the budget arithmetic")
    parser.add_argument("--compat-repo", type=Path, default=Path("/mnt/work/repo_stream2"),
                        help="legacy repository whose MLPLOB the control was trained on")
    parser.add_argument("--compat-weights", type=Path, default=None,
                        help="the control's exported weights, to prove the forward pass is unchanged")
    args = parser.parse_args()
    plan = pretrain_plan(args.plan)
    output_root = (args.output_root or (args.repo / "runs" / plan["experiment_id"])).resolve()
    config, recipe_guard = driver.trajectory_config(plan, args.repo.resolve(), output_root)
    spec = PretrainSpec.from_mapping(plan["pretrain"])
    if spec.seed != config.seed:
        raise ValueError(f"Pretraining seed {spec.seed} must equal the recipe seed {config.seed}")
    minutes = non_label_minutes(config.data.sample_times)
    if list(spec.minutes) != minutes:
        raise ValueError("Pre-registered pretraining minutes do not match the recipe's "
                         f"non-label grid: {len(spec.minutes)} vs {len(minutes)}")
    control = control_record(plan)
    budget = matched_budget(plan, config, control, spec)
    identity = recipe_identity(plan, config, control)
    summary = {"plan": plan["_path"], "plan_sha256": plan_digest(plan),
               "experiment_id": config.experiment_id, "seed": config.seed,
               "output_root": output_root.as_posix(), "recipe": recipe_guard,
               "pretrain": spec.as_mapping(), "budget": budget, "control_identity": identity,
               "non_label_minutes": len(minutes),
               "budget_basis": budget["budget_basis"]}
    if args.compat_weights is not None:
        summary["forward_compatibility"] = verify_forward_compatibility(
            args.compat_repo, args.repo, args.compat_weights)
    if args.check_only:
        print(_json(summary))
        return 0
    run_dir = new_run_dir(output_root)
    rec = Recorder(run_dir, config.experiment_id)
    rec.event("pretrain", "planned", plan=plan["_path"], plan_sha256=plan_digest(plan),
              spec=spec.as_mapping(), minutes=minutes, budget=budget,
              control_identity=identity, recipe=recipe_guard)
    write_archive_start(run_dir, config, plan, budget)
    started = time.monotonic()
    try:
        if args.compat_weights is not None:
            record = verify_forward_compatibility(args.compat_repo, args.repo, args.compat_weights)
            write_json(run_dir / "pretrain" / "compatibility.json", record)
            rec.event("pretrain", "forward_compatibility", **{
                key: record[key] for key in ("legacy_repo", "current_repo", "weights",
                                             "max_abs_diff", "bitwise_equal", "model_sha256")})
        else:
            rec.event("pretrain", "forward_compatibility",
                      status="not_checked: no --compat-weights was supplied")
        bundle, pool, guards = pretraining_pool(config, spec, minutes=minutes)
        adapter = driver.create_model(config, recorder=rec)
        metrics = run_pretraining(adapter.network, pool, spec, device=adapter.device,
                                  recorder=rec, log_every_steps=args.log_every_steps)
        checkpoint = run_dir / "pretrain" / "checkpoint.pt"
        record = save_checkpoint(checkpoint, adapter.network, metrics.pop("decoder_state"),
                                 spec, metadata={"guards": guards, "budget": budget,
                                                 "control": budget["control"],
                                                 "metrics": {key: metrics[key] for key in
                                                             ("steps", "samples_seen", "epochs",
                                                              "first_loss", "final_loss",
                                                              "wall_seconds")}})
        write_json(run_dir / "pretrain" / "spec.json",
                   {"schema_version": 1, "plan": plan["_path"], "plan_sha256": plan_digest(plan),
                    "experiment_id": config.experiment_id, "seed": config.seed,
                    "spec": spec.as_mapping(), "minutes": minutes, "guards": guards,
                    "budget": budget, "checkpoint": record})
        write_json(run_dir / "pretrain" / "metrics.json", {key: value for key, value in metrics.items()
                                                          if key != "decoder_state"})
        write_json(run_dir / "pretrain" / "months.json", guards["months"])
        rec.event("pretrain", "completed", steps=metrics["steps"], samples_seen=metrics["samples_seen"],
                  first_loss=metrics["first_loss"], final_loss=metrics["final_loss"],
                  wall_seconds=metrics["wall_seconds"], encoder_sha256=record["encoder_sha256"],
                  months_sha256=guards["months"]["sha256"], months_last=guards["months"]["last"])
        matched = budget_record(budget, plan_sha256=plan_digest(plan),
                              metrics=metrics)
        write_json(run_dir / "pretrain" / "budget_match.json", matched)
        del pool, bundle, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        prepared = driver.prepare_training(config, streaming=args.stream_bars)
        validation = driver.prepare_validation(config, prepared["bundle"])

        # The shared driver stays byte-identical for the runs already using it, so the trunk
        # transfer hooks the one name run_trajectory resolves at call time instead of editing it.
        original = driver.create_model

        def factory(configuration, recorder):
            model = original(configuration, recorder=recorder)
            apply_checkpoint(model, checkpoint, recorder=recorder)
            return model

        driver.create_model = factory
        try:
            result = driver.run_trajectory(plan, config, prepared, validation, run_dir,
                                           streaming=args.stream_bars)
        finally:
            driver.create_model = original
    except BaseException as exc:  # noqa: BLE001 - the archive must record any failure
        rec.failure("pretrain_finetune", exc)
        finish(run_dir, "failed", 1, resources={"wall_seconds": time.monotonic() - started,
                                               "gpu_used": config.model.device != "cpu"},
               error={"type": type(exc).__name__, "reason": str(exc)[:500]})
        raise
    print(_json({"status": result["status"], "run_dir": run_dir.as_posix(),
                 "points": [row["name"] for row in result["trajectory"]],
                 "budget_match": matched, "plan_sha256": summary["plan_sha256"]}))
    return result["exit_code"]


def _json(document) -> str:
    import json

    return json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
