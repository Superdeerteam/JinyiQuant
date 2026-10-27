"""Generate and register the level-1 data ladder supplement (overnight fill).

The registered level-1 probe (`b4_data_probe_prereg_20261027`) stays the sole
authority for the escalation verdict. This supplement only adds measurements
*inside* the same 132-day window: a finer training-amount curve on independent
seeds, a same-environment 60-day curve, and extra support seeds. It never
escalates, never promotes and never reads a sealed interval.

Usage:
    python scripts/data_ladder_prereg.py --generate
    python scripts/data_ladder_prereg.py --register
    python scripts/data_ladder_prereg.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

PREREG_NAME = "b4_data_ladder_prereg_20261027.json"
REGISTRATION_NAME = "b4_data_ladder_prereg_20261027.registration.json"

TEMPLATE_PLAN = "data_probe_132d_20261027_seed11_plan.json"
BASE_132D = "mlplob_v3_132d_csz_19800.json"
BASE_60D = "mlplob_v3_60d_csz_19800.json"

HOST_CACHE_132D = "/root/cache/v2-132d-csz-20261027"
HOST_CACHE_60D = "/root/cache/v2-60d-csz-20261027"
HOST_DATA = "/root/data"
RUN_ROOT = "/mnt/work/runs"

# measured, not estimated: seed-11 precheck on the clean tree
MEASURED = {
    132: {"usable_samples": 1054256, "batches_per_epoch": 32946,
          "steps": {"0.66": 21744, "1.32": 43488, "2.64": 86976,
                    "3.96": 130464, "5.28": 173952}},
    60: {"usable_samples": 479432, "batches_per_epoch": 14983,
         "steps": {"0.66": 9888, "1.32": 19777, "2.64": 39553}},
}

COMPONENTS = [
    {
        "id": "A",
        "rung_trading_days": 132,
        "train_window": ["2023-04-14", "2023-10-31"],
        "seeds": [53, 59, 61],
        "points": ["0.66", "1.32", "2.64", "3.96", "5.28"],
        "min_coverage": 0.65,
        "role": "training-amount curve at the registered 132-day window, on seeds that the registered probe does not use",
    },
    {
        "id": "B",
        "rung_trading_days": 60,
        "train_window": ["2023-08-01", "2023-10-31"],
        "seeds": [11, 29, 47],
        "points": ["0.66", "1.32", "2.64"],
        "min_coverage": 0.65,
        "role": "same-environment 60-day control curve; the registered control is not re-run and is not replaced",
    },
    {
        "id": "C",
        "rung_trading_days": 132,
        "train_window": ["2023-04-14", "2023-10-31"],
        "seeds": [67, 71, 73],
        "points": ["1.32", "2.64"],
        "role": "extra support seeds for the C1 interval; descriptive, never a replacement for the registered three seeds",
    },
]

POINT_ROLE = {
    "0.66": "low-amount curve point below the registered anchor",
    "1.32": "registered anchor magnitude (C1 left side)",
    "2.64": "registered second point magnitude (C2 left / C3 right)",
    "3.96": "right-shift test point",
    "5.28": "registered refusal point magnitude, re-tested on more data",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def _write(name: str, payload: dict) -> Path:
    path = CONFIGS / name
    # LF on purpose: git normalises on add, so a byte-exact copy and a `git archive`
    # checkout must hash identically or the driver refuses the pinned recipe sha256.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def _gpu_map(seeds: list[int]) -> dict[str, int]:
    return {str(seed): index % 2 for index, seed in enumerate(seeds)}


def build_recipe(component: dict, seed: int) -> str:
    """The frozen recipe with host paths, seed and run root changed - nothing else."""
    days = component["rung_trading_days"]
    base = _read(BASE_132D if days == 132 else BASE_60D)
    base["seed"] = seed
    base["experiment_id"] = f"ladder-{days}d-seed{seed}"
    base["output_root"] = f"{RUN_ROOT}/{base['experiment_id']}"
    base["data"]["bars_path"] = f"{HOST_DATA}/cpt_jyc_2026_e2e_bar1m"
    base["data"]["labels_path"] = f"{HOST_DATA}/cpt_jyc_2026_vwap"
    base["data"]["instruments_path"] = f"{HOST_DATA}/cpt_jyc_2026_instruments(1).parquet"
    base["data"]["instrument_map_path"] = (
        f"{HOST_DATA}/cpt_jyc_2026_stock_bar1m_instrument_mapping.csv")
    base["data"]["cache_dir"] = HOST_CACHE_132D if days == 132 else HOST_CACHE_60D
    base["hypothesis"] = (
        f"Training-amount ladder supplement (component {component['id']}): the frozen MLPLOB v3 CSZ "
        f"recipe on the {days}-trading-day window {component['train_window'][0]}..{component['train_window'][1]}, "
        "scored at several effective-epoch points of one continuous trajectory; only the training amount "
        "changes, and no escalation or promotion is claimed from this run.")
    name = f"ladder_{days}d_csz_seed{seed}.json"
    _write(name, base)
    return f"configs/{name}"


def build_plan(component: dict, seed: int, recipe_rel: str) -> str:
    template = _read(TEMPLATE_PLAN)
    days = component["rung_trading_days"]
    seeds = component["seeds"]
    points = component["points"]
    experiment_id = f"ladder-{days}d-seed{seed}"
    plan = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at_utc": _now(),
        "status": "execution_plan_of_registered_ladder",
        "registration": {
            "path": f"configs/{PREREG_NAME}",
            "role": f"component {component['id']}",
            "note": ("this supplement adds measurements inside the registered level-1 window; the escalation "
                     "verdict stays with b4_data_probe_prereg_20261027 and is not re-read here"),
        },
        "question_zh": (f"在 {days} 个交易日窗口（{component['train_window'][0]}..{component['train_window'][1]}）上，"
                        f"同配方单条连续轨迹在 {'/'.join(points)} 个有效 epoch 点的样本外 RankIC 各是多少，"
                        f"峰值落在哪个训练量上。"),
        "hypothesis": ("if the out-of-sample RankIC peak moves to a larger training amount when the window holds "
                       "more days, the training-amount optimum is data-limited; if the peak stays at the smallest "
                       "amount, more data does not support more training."),
        "factor": {
            "changed": "none beyond the training amount of one continuous trajectory",
            "train_window": component["train_window"],
            "train_trading_days": days,
            "note": "no data, label, split or model setting differs from the frozen recipe of this rung",
        },
        "environment_only_changes": {
            "data.bars_path": "host data root",
            "data.labels_path": "host data root",
            "data.instruments_path": "host data root",
            "data.instrument_map_path": "host data root",
            "data.cache_dir": "host cache root (shared across the seeds of this rung)",
            "output_root": "host run root",
        },
        "recipe_config": recipe_rel,
        "recipe_sha256": _sha(REPO / recipe_rel),
        "recipe_frozen_baseline": template["recipe_frozen_baseline"],
        "recipe_frozen": template["recipe_frozen"],
        "budget_only_overrides": template["budget_only_overrides"],
        "budget_points": [
            {"name": f"ladder-{point}", "effective_epochs": float(point), "role": POINT_ROLE[point]}
            for point in points
        ],
        "one_continuous_trajectory": True,
        "lr_schedule": "constant",
        "exposure": {**template["exposure"],
                     "min_coverage": component.get("min_coverage", 1.0)},
        "resume": template["resume"],
        "derived_reference": {
            "estimate_not_budget": True,
            "trading_days": days,
            "usable_samples_measured": MEASURED[days]["usable_samples"],
            "batch_size": 32,
            "steps_expected": {point: MEASURED[days]["steps"][point] for point in points},
            "calibration": ("measured on the clean HEAD tree for the seed-11 registered precheck; the driver "
                            "re-derives usable samples and steps from the pool actually built"),
            "deviation_tolerance": {"relative": 0.05, "applies_to": ["usable_samples", "steps"]},
        },
        "cloud": {
            "host": "hz-4.matpool.com:26197 (24 cores / 172 GB / 2x RTX 3090 24GB)",
            "concurrency": {
                "processes": len(seeds),
                "gpus": 2,
                "one_process_per_gpu": False,
                "sharing": "seeds are packed 3 per GPU when the memory pre-flight allows it",
                "threads_per_process": 4,
                "seed_gpu_map": _gpu_map(seeds),
            },
            "checkpoint_dir": "trajectory/",
            "artifacts_to_retrieve": template["cloud"]["artifacts_to_retrieve"],
        },
        "evaluation_scope": template["evaluation_scope"],
    }
    name = f"ladder_{days}d_seed{seed}_{len(points)}point_plan.json"
    _write(name, plan)
    return f"configs/{name}"


def build_prereg(artifacts: dict[str, dict]) -> dict:
    return {
        "schema_version": 1,
        "registration_id": "b4-data-ladder-20261027",
        "owner_role": "captain",
        "phase": "B4 data-window research, level-1 supplemental measurements",
        "status": "registered",
        "one_shot": False,
        "registered_at_source": "written before any ladder process started",
        "purpose": (
            "Fill an idle cloud window with level-1 measurements that sharpen the registered data-amount "
            "question, without escalating to level 2, without promoting anything and without reading the "
            "sealed interval."
        ),
        "question": (
            "Within the registered 132-day window, where does the out-of-sample RankIC peak sit on the "
            "training-amount axis, and does the peak move when the window holds 132 instead of 60 days?"
        ),
        "relation_to_registered_probe": {
            "escalation_authority": "configs/b4_data_probe_prereg_20261027.json",
            "note": ("the C1/C2/C3 verdict and the escalation decision belong to that document; this supplement "
                     "neither re-reads it nor may overturn it"),
            "no_level_2": True,
            "level_2_status": "still requires its own pre-registration and is still not authorized by anything here",
        },
        "components": [
            {
                "id": component["id"],
                "train_trading_days": component["rung_trading_days"],
                "train_window": component["train_window"],
                "seeds": component["seeds"],
                "effective_epoch_points": [float(p) for p in component["points"]],
                "role": component["role"],
                "plans": artifacts[component["id"]]["plans"],
            }
            for component in COMPONENTS
        ],
        "measured_budget_reference": {
            str(days): value for days, value in MEASURED.items()
        },
        "declared_reading": {
            "primary_curve_component": "A",
            "peak_rule": (
                "for each seed, the peak point is the argmax of out-of-sample RankIC over the declared points; "
                "the component-level peak is the mode of the three seed peaks and is reported together with all "
                "three values, never as a single number"
            ),
            "right_shift_rule": (
                "declare 'the optimum moved right with more data' only when the 132-day component peak sits at a "
                "point larger than the 60-day component peak for at least two of three seeds, and the seed-mean "
                "RankIC at that larger point exceeds the seed-mean RankIC at 1.32 by at least 0.0017 - the same "
                "magnitude as the registered minimum meaningful improvement, borrowed and not re-derived"
            ),
            "cross_environment_rule": (
                "component B is a machine and torch-build robustness check against the recorded 60-day values; "
                "if any point of seed 11 shifts by more than 0.0020 the cross-environment comparability of the "
                "registered control is reported as impaired and the C1 reading is flagged, not silently kept"
            ),
            "support_seed_rule": (
                "component C widens the C1 interval for reporting only; the registered three seeds remain the "
                "registered evidence and a six-seed interval may not be presented as the registered result"
            ),
        },
        "prohibitions": [
            "no escalation, no level 2, no promotion, no replacement of the incumbent",
            "no reading of the sealed interval",
            "no change to any promotion threshold, secondary line, net5 limit, cost scenario or period count",
            "no reporting of a single seed as a candidate",
            "no re-selection of a component, seed or point after its value has been read",
            "no C1 verdict from this document, in either direction",
            "no relaxation of the exposure reconciliation, the coverage floor or the resume fail-closed rule",
        ],
        "execution_queue": [
            {"priority": 1, "component": "A", "processes": 3},
            {"priority": 2, "component": "B", "processes": 3},
            {"priority": 3, "component": "C", "processes": 3},
        ],
        "queue_rules": [
            "a component may start only when a free process slot exists and the memory pre-flight allows it",
            "each plan is one process and one uninterrupted trajectory; a plan is never split",
            "a failed process is retried at most once, then skipped and reported; the queue continues",
            "the whole queue is drained in priority order and stops when the window closes",
            "partial completion is reported as partial: a component with fewer than three finished seeds is described as incomplete",
        ],
        "open_items_after_registration": [
            "the real peak location is read from the replay of exported weights, not from the cloud validation score",
        ],
        "generated_artifacts": artifacts,
    }


def generate(only: list[str] | None = None) -> dict:
    """Regenerate selected components only; untouched rungs keep their exact bytes."""
    artifacts: dict[str, dict] = {}
    previous = None
    if only:
        prereg_path = CONFIGS / PREREG_NAME
        previous = json.loads(prereg_path.read_text(encoding="utf-8")) if prereg_path.exists() else {}
    for component in COMPONENTS:
        if only and component["id"] not in only:
            kept = next((c for c in (previous or {}).get("components", [])
                         if c["id"] == component["id"]), None)
            if kept is None:
                raise SystemExit(f"component {component['id']} is not in the previous prereg")
            artifacts[component["id"]] = {"recipes": kept.get("recipes", []), "plans": kept["plans"]}
            continue
        recipes, plans = [], []
        for seed in component["seeds"]:
            recipe_rel = build_recipe(component, seed)
            recipes.append(recipe_rel)
            plans.append(build_plan(component, seed, recipe_rel))
        artifacts[component["id"]] = {"recipes": recipes, "plans": plans}
    prereg = build_prereg(artifacts)
    path = _write(PREREG_NAME, prereg)
    return {"prereg": str(path.relative_to(REPO)), "sha256": _sha(path), "artifacts": artifacts}


def register() -> dict:
    prereg_path = CONFIGS / PREREG_NAME
    if not prereg_path.exists():
        raise SystemExit("generate the ladder first")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    plan_hashes = {}
    for component in prereg["components"]:
        for plan_rel in component["plans"]:
            plan = json.loads((REPO / plan_rel).read_text(encoding="utf-8"))
            recipe_rel = plan["recipe_config"]
            if plan["recipe_sha256"] != _sha(REPO / recipe_rel):
                raise SystemExit(f"recipe changed after the plan was generated: {recipe_rel}")
            plan_hashes[plan_rel] = _sha(REPO / plan_rel)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    record = {
        "schema_version": 1,
        "prereg_path": f"configs/{PREREG_NAME}",
        "prereg_sha256": _sha(prereg_path),
        "registered_at": _now(),
        "git_head": head,
        "plan_sha256": plan_hashes,
        "recipe_sha256": {
            plan_rel: json.loads((REPO / plan_rel).read_text(encoding="utf-8"))["recipe_sha256"]
            for plan_rel in plan_hashes
        },
        "ordering": {
            "prereg_written_before_any_ladder_process": True,
            "registered_against": "no ladder run had started at registration time",
        },
    }
    out = _write(REGISTRATION_NAME, record)
    return {"registration": str(out.relative_to(REPO)), "sha256": _sha(out), "record": record}


def check() -> dict:
    """Fail if any pin broke after registration."""
    reg_path = CONFIGS / REGISTRATION_NAME
    if not reg_path.exists():
        raise SystemExit("ladder is not registered")
    record = json.loads(reg_path.read_text(encoding="utf-8"))
    problems = []
    prereg_path = REPO / record["prereg_path"]
    if _sha(prereg_path) != record["prereg_sha256"]:
        problems.append("prereg sha256 changed")
    for plan_rel, digest in record["plan_sha256"].items():
        if _sha(REPO / plan_rel) != digest:
            problems.append(f"plan changed: {plan_rel}")
        plan = json.loads((REPO / plan_rel).read_text(encoding="utf-8"))
        if _sha(REPO / plan["recipe_config"]) != plan["recipe_sha256"]:
            problems.append(f"recipe changed: {plan['recipe_config']}")
    if problems:
        raise SystemExit("; ".join(problems))
    return {"status": "ok", "plans": len(record["plan_sha256"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--register", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--only", nargs="+", default=None,
                        help="regenerate only these component ids; other rungs keep their bytes")
    args = parser.parse_args(argv)
    if args.generate:
        report = generate(args.only)
    elif args.register:
        report = register()
    else:
        report = check()
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
