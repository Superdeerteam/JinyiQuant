"""Pre-register one R14 arm: the recipe, the plan, and the budget arithmetic.

The recipe is the R1/441d supervised recipe with a new identity, so the fine-tune sees exactly
the data, splits, model and evaluation the control saw; the plan freezes the pretraining task
(masked-segment reconstruction on the non-label minute grid), the control run it is matched
against, and the step arithmetic that has to balance. ``run_pretrain_finetune.py --check-only``
re-derives all of it from the control run's own artifacts before anything is trained.

Usage::

    python scripts/make_r14_plan.py --control-config .tmp_r14/ctl_seed11_config.json \
        --seed 11 --control-path /mnt/work/runs/scaling-441d-seed11

writes ``configs/r14-pretrain-441d-csz-seed11.json`` and
``configs/r14-pretrain-441d-csz-seed11_plan.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.config import RunConfig, load_config  # noqa: E402
from jinyiquant.io import file_sha256, read_json, write_json  # noqa: E402
from jinyiquant.pretrain import PretrainSpec, non_label_minutes  # noqa: E402
from jinyiquant.trajectory import derive_budget_steps  # noqa: E402

PLAN_SCHEMA = "r14-pretrain-plan/1"
SHARE_RULE = ("pretrain_steps = round(share * control_steps); "
              "finetune_steps = control_steps - pretrain_steps, so the arm trains exactly the "
              "control's total optimizer steps and samples")
HYPOTHESIS = ("R14: the 31 five-minute columns a day that the supervised recipe never samples "
              "carry order-book shape information; reconstructing a masked contiguous span of a "
              "128-bar window that ends at one of those columns transfers a better initialisation "
              "for the labelled fine-tune than the same fine-tune from a fresh trunk, at an "
              "identical total training budget.")


def recipe_from_control(control: dict, *, seed: int, output_root: str, experiment_id: str) -> dict:
    """The control recipe under this arm's identity: nothing else may differ."""
    document = {**control, "experiment_id": experiment_id, "hypothesis": HYPOTHESIS,
                "output_root": output_root, "seed": seed}
    RunConfig.model_validate(document)  # refuse a recipe this pipeline cannot run
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control-config", type=Path, required=True,
                        help="the control run's config.json (scaling-441d-seedNN)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control-path", default="/mnt/work/runs/scaling-441d-seed11",
                        help="the control run root on the training host")
    parser.add_argument("--control-steps", type=int, default=145269,
                        help="the control run's scored budget point, in optimizer steps")
    parser.add_argument("--usable-samples", type=int, default=3521680)
    parser.add_argument("--share", type=float, default=0.10,
                        help="pretraining share of the control budget")
    parser.add_argument("--point-name", default="r14-eq132",
                        help="the scored budget point, named after the control point it matches")
    parser.add_argument("--mask-length", type=int, default=32)
    parser.add_argument("--instruments-per-slot", type=int, default=10)
    parser.add_argument("--pretrain-steps", type=int, default=None,
                        help="override the derived pretraining step count")
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing")
    args = parser.parse_args()
    control = read_json(args.control_config)
    experiment_id = f"r14-pretrain-441d-csz-seed{args.seed}"
    output_root = f"/mnt/work/runs/{experiment_id}"
    recipe = recipe_from_control(control, seed=args.seed, output_root=output_root,
                                 experiment_id=experiment_id)
    recipe_name = f"{experiment_id}.json"
    recipe_path = args.repo / "configs" / recipe_name
    if not args.dry_run:
        write_json(recipe_path, recipe)
    config = RunConfig.model_validate(recipe)
    batch_size = config.model.batch_size
    pretrain_steps = (args.pretrain_steps if args.pretrain_steps is not None
                      else int(round(args.share * args.control_steps)))
    finetune_steps = args.control_steps - pretrain_steps
    if finetune_steps < 1:
        raise SystemExit("The pretraining share consumes the whole control budget")
    effective_epochs = round(finetune_steps * batch_size / args.usable_samples, 6)
    batches_per_epoch = -(-args.usable_samples // batch_size)
    if finetune_steps < batches_per_epoch:
        raise SystemExit(f"Fine-tune budget {finetune_steps} steps cannot cover one epoch "
                         f"({batches_per_epoch} batches), so the declared coverage of 1.0 is "
                         f"unreachable and the driver would reject the point after training")
    points = [{"name": args.point_name, "effective_epochs": effective_epochs}]
    derived = derive_budget_steps(points, args.usable_samples, batch_size)[0]
    if derived["steps"] != finetune_steps:
        raise SystemExit(f"Derived fine-tune steps {derived['steps']} do not equal the matched "
                         f"budget {finetune_steps}; change --control-steps or the point name")
    minutes = non_label_minutes(config.data.sample_times)
    pretrain = {"steps": pretrain_steps, "batch_size": batch_size,
                "learning_rate": args.pretrain_learning_rate,
                "mask_length": args.mask_length,
                "instruments_per_slot": args.instruments_per_slot,
                "seed": args.seed, "minutes": minutes}
    PretrainSpec.from_mapping(pretrain)  # refuse a task this pipeline cannot run
    plan = {"schema_version": PLAN_SCHEMA, "experiment_id": experiment_id,
            "hypothesis": HYPOTHESIS,
            "recipe_config": f"configs/{recipe_name}",
            "recipe_sha256": file_sha256(recipe_path) if not args.dry_run else None,
            "budget_points": points, "lr_schedule": config.model.lr_schedule,
            "exposure": {"min_coverage": 1.0, "effective_epochs_atol": 0.005},
            "seeds": [args.seed],
            "pretrain": pretrain,
            "control": {"experiment_id": control["experiment_id"], "path": args.control_path,
                        "budget_point": "scaling-1.32" if args.control_steps == 145269
                        else f"control-{args.control_steps}"},
            "budget_match": {"rule": SHARE_RULE, "share": args.share,
                             "pretrain_steps": pretrain_steps,
                             "finetune_steps": finetune_steps,
                             "control_steps": args.control_steps,
                             "total_steps": args.control_steps,
                             "note": f"supervised exposure is {effective_epochs} epoch; the point "
                                     "name marks the control budget point it matches"},
            "acceptance": {"paired_rank_ic_min_improvement": 0.003,
                           "paired_rank_ic_positive_windows": "7/8",
                           "screen_windows": ["nov2023", "jan2024", "feb2024", "mar2024",
                                              "apr2024", "may2024", "jul2024", "aug2024"],
                           "comparison": "candidate - control, paired by (window, trading day)"},
            "compute": {"pretrain_wall_seconds_estimate": 900,
                        "finetune_wall_seconds_estimate": 2800}}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    plan_path = args.repo / "configs" / f"{experiment_id}_plan.json"
    write_json(plan_path, plan)
    print(json.dumps({"recipe": recipe_path.as_posix(), "plan": plan_path.as_posix(),
                      "plan_sha256": file_sha256(plan_path),
                      "recipe_sha256": plan["recipe_sha256"],
                      "budget": plan["budget_match"], "minutes": len(minutes)},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
