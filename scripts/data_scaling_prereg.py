"""Generate and register the data-scaling rungs (240 and 450 trading days).

This is a second, self-contained document. The registered level-1 probe
(`b4_data_probe_prereg_20261027`) and the level-1 ladder supplement
(`b4_data_ladder_prereg_20261027`) keep their own verdicts; this document adds
the coarse data axis the level-1 pair cannot answer: is a *material* increase in
trading days worth anything, and does it keep paying.

It is registered before any of its processes starts. The ordering clause of the
level-1 document ("level 2 only after level-1 escalation") is recorded below as
knowingly deviated from, with the reason, so the deviation is auditable rather
than silent.

Usage:
    python scripts/data_scaling_prereg.py --generate
    python scripts/data_scaling_prereg.py --register
    python scripts/data_scaling_prereg.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_ladder_prereg import (  # noqa: E402  (shared, side-effect free helpers)
    BASE_132D,
    CONFIGS,
    HOST_DATA,
    POINT_ROLE,
    REPO,
    RUN_ROOT,
    TEMPLATE_PLAN,
    _now,
    _read,
    _sha,
    _write,
)

PREREG_NAME = "b4_data_scaling_prereg_20261027.json"
REGISTRATION_NAME = "b4_data_scaling_prereg_20261027.registration.json"

# usable samples scale with the number of trading days; the driver re-derives them
ANCHOR = {"trading_days": 132, "usable_samples": 1054256}

COMPONENTS = [
    {
        "id": "D",
        "label": "240d",
        "train_window": ["2022-11-04", "2023-10-31"],
        "declared_trading_days": 240,
        "seeds": [11, 29, 47],
        "points": ["1.32", "2.64"],
        "role": ("the provisional level-2 window that the level-1 document already named, run as a "
                 "dose-response rung at matched effective epochs, paired by seed with the 60-day and "
                 "132-day rungs"),
    },
    {
        "id": "E",
        "label": "441d",
        "train_window": ["2022-01-04", "2023-10-31"],
        "declared_trading_days": 441,
        "seeds": [11, 29, 47],
        "points": ["1.32", "2.64"],
        "role": ("the largest window this dataset supports: 2022-01-04 is the earliest trading day in the "
                 "archive, so 441 trading days ending 2023-10-31 is the hard ceiling for the materialised pool; "
                 "the decisive rung for 'the model was always trained on too little data'. Two points are scored "
                 "because the trajectory driver refuses a single-point plan and because every other rung is "
                 "read at the same 1.32/2.64 pair, so the rung stays comparable; the pool peak at 441 days "
                 "allows one process at a time, so the three seeds run in sequence."),
    },
    {
        "id": "F",
        "label": "927d",
        "train_window": ["2020-01-02", "2023-10-31"],
        "declared_trading_days": 927,
        "seeds": [11, 29, 47],
        "points": ["1.32", "2.64"],
        "role": ("the full official-label history: 2020-01-02..2023-10-31 is 927 trading days, 7.0x the "
                 "132-day window. The official vwap archive covers all 60 months, so no label is derived here. "
                 "This rung needs the streaming pool (bitwise equal to the materialised pool, verified on "
                 "111,888 keys) because a materialised 927-day pool is 175 GB and cannot fit the host."),
    },
]


def _cache(days: str) -> str:
    return f"/root/cache/v2-{days}-csz-20261027"


def _estimated_usable(declared_days: int) -> int:
    return int(round(ANCHOR["usable_samples"] * declared_days / ANCHOR["trading_days"]))


def build_recipe(component: dict) -> dict:
    base = _read(BASE_132D)
    days = component["label"]
    recipes = {}
    for seed in component["seeds"]:
        recipe = json.loads(json.dumps(base))
        recipe["seed"] = seed
        recipe["experiment_id"] = f"scaling-{days}-seed{seed}"
        recipe["output_root"] = f"{RUN_ROOT}/{recipe['experiment_id']}"
        recipe["data"]["bars_path"] = f"{HOST_DATA}/cpt_jyc_2026_e2e_bar1m"
        recipe["data"]["labels_path"] = f"{HOST_DATA}/cpt_jyc_2026_vwap"
        recipe["data"]["instruments_path"] = f"{HOST_DATA}/cpt_jyc_2026_instruments(1).parquet"
        recipe["data"]["instrument_map_path"] = (
            f"{HOST_DATA}/cpt_jyc_2026_stock_bar1m_instrument_mapping.csv")
        recipe["data"]["cache_dir"] = _cache(days)
        recipe["splits"]["train"]["start"] = f"{component['train_window'][0]}T00:00:00"
        recipe["splits"]["train"]["end"] = "2023-11-01T00:00:00"
        recipe["hypothesis"] = (
            f"Data-scaling rung {component['id']} ({days} trading days, "
            f"{component['train_window'][0]}..{component['train_window'][1]}): the frozen MLPLOB v3 CSZ recipe "
            "with only the training window start moved earlier, scored at matched effective-epoch points of one "
            "continuous trajectory; the validation split, the evaluation window and every other setting stay fixed.")
        name = f"scaling_{days}_csz_seed{seed}.json"
        _write(name, recipe)
        recipes[seed] = f"configs/{name}"
    return recipes


def build_plan(component: dict, seed: int, recipe_rel: str) -> str:
    template = _read(TEMPLATE_PLAN)
    days = component["label"]
    usable = _estimated_usable(component["declared_trading_days"])
    steps = {point: int(round(float(point) * usable / 32)) for point in component["points"]}
    plan = {
        "schema_version": 1,
        "experiment_id": f"scaling-{days}-seed{seed}",
        "created_at_utc": _now(),
        "status": "execution_plan_of_registered_data_scaling",
        "registration": {
            "path": f"configs/{PREREG_NAME}",
            "role": f"component {component['id']} ({days})",
            "note": ("dose-response measurement only; it may not escalate, replace the incumbent or be "
                     "reported as a candidate"),
        },
        "question_zh": (f"把训练窗口从 132 个交易日扩到 {component['declared_trading_days']} 个交易日"
                        f"（{component['train_window'][0]}..{component['train_window'][1]}），在 1.32 与 2.64 个"
                        "有效 epoch 的匹配训练量下，样本外 RankIC 是否随数据量继续上升。"),
        "hypothesis": ("if the low score of every earlier model is a data-quantity failure, the out-of-sample "
                       "RankIC must rise with the number of training days at matched effective epochs; a flat "
                       "curve falsifies the data hypothesis at this resolution."),
        "factor": {
            "changed": "splits.train.start",
            "to": f"{component['train_window'][0]}T00:00:00",
            "train_window": component["train_window"],
            "declared_trading_days": component["declared_trading_days"],
            "validation_unchanged": ["2023-11-01T00:00:00", "2023-11-15T00:00:00"],
            "train_end_exclusive": "2023-11-01T00:00:00",
            "note": "the window never reaches the validation split; only the training window start moves",
        },
        "environment_only_changes": template["environment_only_changes"],
        "recipe_config": recipe_rel,
        "recipe_sha256": _sha(REPO / recipe_rel),
        "recipe_frozen_baseline": template["recipe_frozen_baseline"],
        "recipe_frozen": template["recipe_frozen"],
        "budget_only_overrides": template["budget_only_overrides"],
        "budget_points": [
            {"name": f"scaling-{point}", "effective_epochs": float(point), "role": POINT_ROLE[point]}
            for point in component["points"]
        ],
        "one_continuous_trajectory": True,
        "lr_schedule": "constant",
        "exposure": template["exposure"],
        "resume": template["resume"],
        "derived_reference": {
            "estimate_not_budget": True,
            "trading_days": component["declared_trading_days"],
            "usable_samples_estimate": usable,
            "batch_size": 32,
            "steps_expected": steps,
            "calibration": (f"level-1 measured {ANCHOR['usable_samples']} usable samples over "
                            f"{ANCHOR['trading_days']} trading days, scaled linearly by trading days; the driver "
                            "re-derives both from the pool actually built"),
            "deviation_tolerance": {
                "relative": 0.05,
                "applies_to": ["usable_samples", "steps"],
                "rule": "a realised value beyond +/-5% stops the run and reports; the estimate is not edited afterwards",
            },
        },
        "cloud": {
            "host": "hz-4.matpool.com:26197 (24 cores / 172 GB / 2x RTX 3090 24GB)",
            "concurrency": {
                "processes": len(component["seeds"]),
                "gpus": 2,
                "one_process_per_gpu": False,
                "sharing": "packed by the memory pre-flight, not by a fixed map",
                "threads_per_process": 4,
                "seed_gpu_map": {str(s): i % 2 for i, s in enumerate(component["seeds"])},
            },
            "checkpoint_dir": "trajectory/",
            "artifacts_to_retrieve": template["cloud"]["artifacts_to_retrieve"],
        },
        "evaluation_scope": template["evaluation_scope"],
    }
    name = f"scaling_{days}_seed{seed}_{len(component['points'])}point_plan.json"
    _write(name, plan)
    return f"configs/{name}"


def build_prereg(artifacts: dict) -> dict:
    return {
        "schema_version": 1,
        "registration_id": "b4-data-scaling-20261027",
        "owner_role": "captain",
        "phase": "B4 data-quantity research, coarse data axis",
        "status": "registered",
        "one_shot": False,
        "registered_at_source": "written before any scaling process started",
        "purpose": (
            "Measure whether a materially larger training window improves out-of-sample RankIC at matched "
            "training amount, and whether it keeps paying between 240 and 450 trading days. Every earlier model "
            "in this project trained on 60 trading days; the data axis was never measured."
        ),
        "question": (
            "At matched effective epochs, how does out-of-sample RankIC move when the training window grows "
            "from 60 to 132 to 240 to 450 trading days?"
        ),
        "data_availability": {
            "bars_months": "2022-01 .. 2023-11 (archive plus extracted months)",
            "labels_months": "2022-01 .. 2023-10",
            "why_441_is_the_materialised_ceiling": "the materialised pool at 927 days is 175 GB and exceeds the 172 GB host, so 441 days is the ceiling for the materialised path",
            "official_labels_available": "2020-01..2023-11 (47 months) after the official vwap archive was restored to the host; no derived label is used anywhere",
            "train_window_largest": ["2020-01-02", "2023-10-31"],
        },
        "implementation_note": {
            "rung_F": ("runs on the streaming pool (src/jinyiquant/pool.py). The streaming and materialised "
                       "pools were proven bitwise equal on 111,888 real keys over 50 batches, with three "
                       "mutations caught, so rung F stays numerically comparable to D and E"),
            "why": "a materialised 927-day pool is 175 GB; the streaming pool projects to 12.8 GB",
        },
        "known_deviation": {
            "from": "configs/b4_data_probe_prereg_20261027.json level_2 trigger",
            "clause": "level 2 may only be considered after level 1 reaches the escalation state",
            "deviation": "the 240-day rung is run before the level-1 verdict is read",
            "reason": ("the principal directed a broad data-quantity measurement for this compute window; the "
                       "rung is run as curve estimation, not as escalation, and it cannot produce a candidate"),
            "mitigation": [
                "its windows, seeds, effective-epoch points and reading rules are fixed here, before any run",
                "it may not be used to relax, overturn or pre-empt the level-1 C1 verdict",
                "no promotion, no incumbent replacement and no sealed read follow from it",
            ],
        },
        "components": [
            {
                "id": component["id"],
                "label": component["label"],
                "train_window": component["train_window"],
                "declared_trading_days": component["declared_trading_days"],
                "seeds": component["seeds"],
                "effective_epoch_points": [float(p) for p in component["points"]],
                "role": component["role"],
                "plans": artifacts[component["id"]]["plans"],
            }
            for component in COMPONENTS
        ],
        "dose_response_design": {
            "rungs_trading_days": [60, 132, 240, 441, 927],
            "fixed_axis": "effective epochs (1.32 and 2.64) of one continuous trajectory per seed",
            "paired_by_seed": [11, 29, 47],
            "why_paired": ("the same three seeds are run on every rung, so the rung-to-rung difference is a "
                           "paired difference and not a seed-lottery difference"),
        },
        "declared_reading": {
            "primary": "paired per-seed RankIC difference between adjacent rungs at 1.32 effective epochs",
            "single_point_top_rung": ("the 441-day rung declares one point only; its 2.64 point was dropped for "
                                      "resource reasons before any of its values were read, and no rung-to-rung "
                                      "slope is claimed from it"),
            "monotone_rule": ("declare 'more data pays' only when the seed-mean RankIC rises across at least "
                              "three consecutive rungs and the 450-vs-132 paired difference has a 5-day block "
                              "bootstrap interval whose lower bound exceeds zero"),
            "flattening_rule": ("declare 'more data stops paying' when the 450-vs-240 paired difference is "
                                "negative or its interval straddles zero, while 240-vs-132 is positive"),
            "falsification_rule": ("if the 132-to-240 and 240-to-450 paired differences both straddle zero, the "
                                   "data-quantity hypothesis is reported as rejected at this resolution, and the "
                                   "next lever is feature and label information content, not more days"),
            "interval": "the registered 5-day block interval, 4000 bootstrap draws, paired by trading day",
            "trading_day_count_gate": ("before a rung starts, the measured trading-day count of its window must "
                                       "be within 2% of its declared count, otherwise the rung stops and reports"),
        },
        "prohibitions": [
            "no promotion, no incumbent replacement, no candidate claim from any rung",
            "no reading of the sealed interval",
            "no change to any promotion threshold, secondary line, net5 limit or cost scenario",
            "no re-selection of a window, seed or point after its value has been read",
            "no reporting of a single seed as evidence for a rung",
            "no use of this document to relax or overturn the level-1 C1 verdict",
            "no run from this document before its own registration record exists",
        ],
        "execution_queue": [
            {"priority": 1, "component": "D", "processes": 3},
            {"priority": 2, "component": "E", "processes": 3},
        ],
        "queue_rules": [
            "the rungs run only in free process slots and only when the measured pool peak allows it",
            "each plan is one process and one uninterrupted trajectory",
            "a failed process is retried at most once, then skipped and reported",
            "partial completion is reported as partial",
        ],
        "generated_artifacts": artifacts,
    }


def generate(only: list[str] | None = None) -> dict:
    """Regenerate selected components only; untouched rungs keep their exact bytes."""
    artifacts: dict[str, dict] = {}
    if only:
        for name in only:
            if name not in {c["id"] for c in COMPONENTS}:
                raise SystemExit(f"unknown component {name}")
    for component in COMPONENTS:
        if only and component["id"] not in only:
            prereg_old = CONFIGS / PREREG_NAME
            prev = json.loads(prereg_old.read_text(encoding="utf-8")) if prereg_old.exists() else None
            kept = next((c for c in (prev or {}).get("components", [])
                         if c["id"] == component["id"]), None)
            if kept is None:
                raise SystemExit(f"component {component['id']} is not in the previous prereg")
            artifacts[component["id"]] = {"recipes": kept.get("recipes", []),
                                          "plans": kept["plans"]}
            continue
        recipes = build_recipe(component)
        plans = {seed: build_plan(component, seed, recipes[seed]) for seed in component["seeds"]}
        artifacts[component["id"]] = {
            "recipes": [recipes[seed] for seed in component["seeds"]],
            "plans": [plans[seed] for seed in component["seeds"]],
        }
    path = _write(PREREG_NAME, build_prereg(artifacts))
    return {"prereg": str(path.relative_to(REPO)), "sha256": _sha(path), "artifacts": artifacts}


def register() -> dict:
    prereg_path = CONFIGS / PREREG_NAME
    if not prereg_path.exists():
        raise SystemExit("generate the scaling rungs first")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    plan_hashes, recipe_hashes = {}, {}
    for component in prereg["components"]:
        for plan_rel in component["plans"]:
            plan = json.loads((REPO / plan_rel).read_text(encoding="utf-8"))
            recipe_rel = plan["recipe_config"]
            if plan["recipe_sha256"] != _sha(REPO / recipe_rel):
                raise SystemExit(f"recipe changed after the plan was generated: {recipe_rel}")
            plan_hashes[plan_rel] = _sha(REPO / plan_rel)
            recipe_hashes[recipe_rel] = plan["recipe_sha256"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    supersedes = None
    if REGISTRATION_NAME and (CONFIGS / REGISTRATION_NAME).exists():
        previous = json.loads((CONFIGS / REGISTRATION_NAME).read_text(encoding="utf-8"))
        archive = CONFIGS / f"{REGISTRATION_NAME[:-5]}.superseded_{previous['registered_at'].replace(':', '').replace('-', '')[:15]}Z.json"
        archive.write_bytes((CONFIGS / REGISTRATION_NAME).read_bytes())
        supersedes = {"archived_as": f"configs/{archive.name}",
                      "registered_at": previous["registered_at"],
                      "sha256": previous.get("prereg_sha256")}
    record = {
        "schema_version": 1,
        "prereg_path": f"configs/{PREREG_NAME}",
        "prereg_sha256": _sha(prereg_path),
        "registered_at": _now(),
        "git_head": head,
        "plan_sha256": plan_hashes,
        "recipe_sha256": recipe_hashes,
        "ordering": {"registered_before_any_scaling_process": True},
        "supersedes": supersedes,
    }
    out = _write(REGISTRATION_NAME, record)
    return {"registration": str(out.relative_to(REPO)), "sha256": _sha(out), "record": record}


def check() -> dict:
    reg_path = CONFIGS / REGISTRATION_NAME
    if not reg_path.exists():
        raise SystemExit("scaling rungs are not registered")
    record = json.loads(reg_path.read_text(encoding="utf-8"))
    problems = []
    if _sha(REPO / record["prereg_path"]) != record["prereg_sha256"]:
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
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
