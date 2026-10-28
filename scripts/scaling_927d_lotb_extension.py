"""Register and generate the 927-day seed-lot-B replication (seeds 53/59/61).

The registered data-scaling component F declares exactly three seeds (11/29/47).
This document adds a *declared* replication lot on the same rung so that the
newest rung can be judged on six seeds instead of three. It is written and
registered before any score of these three seeds exists; it changes no recipe
field other than `seed`, no budget point, no evaluation window and no threshold,
and it does not alter the registered component-F verdict.

Usage:
    python scripts/scaling_927d_lotb_extension.py --generate
    python scripts/scaling_927d_lotb_extension.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_scaling_prereg as ds  # noqa: E402  (registered generator, reused verbatim)
from data_ladder_prereg import CONFIGS, _now, _read, _sha, _write  # noqa: E402

RUNG = "927d"
PARENT_PREREG = "b4_data_scaling_prereg_20261027.json"
REGISTRATION = "b4_scaling_927d_lotb_extension_20261028.registration.json"
LOT_A_SEEDS = [11, 29, 47]
LOT_B_SEEDS = [53, 59, 61]
POINTS = ["1.32", "2.64"]
TRAIN_WINDOW = ["2020-01-02", "2023-10-31"]
DECLARED_TRADING_DAYS = 927
STEPS_PER_POINT = {"1.32": 305227, "2.64": 610453}


def _component(seeds: list[int]) -> dict:
    return {
        "id": "F",
        "label": RUNG,
        "train_window": list(TRAIN_WINDOW),
        "declared_trading_days": DECLARED_TRADING_DAYS,
        "seeds": list(seeds),
        "points": list(POINTS),
        "raise_max_train_samples": True,
        "role": "927-day seed-lot-B replication of component F",
    }


def _recipe_rel(seed: int) -> str:
    return f"configs/scaling_{RUNG}_csz_seed{seed}.json"


def _plan_rel(seed: int) -> str:
    return f"configs/scaling_{RUNG}_seed{seed}_{len(POINTS)}point_plan.json"


def _capture(seeds: list[int]) -> dict:
    """Run the registered generators without touching configs/ on disk."""
    captured: dict[str, dict] = {}
    real_write = ds._write

    def capture(name: str, payload: dict) -> Path:
        captured[name] = payload
        return CONFIGS / name

    ds._write = capture
    try:
        ds.build_recipe(_component(seeds))
    finally:
        ds._write = real_write
    return captured


def _diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}{key}"
        if key not in a or key not in b:
            out.append(f"{path}: present in only one side")
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            out.extend(_diff(a[key], b[key], f"{path}."))
        elif a[key] != b[key]:
            out.append(f"{path}: {a[key]!r} != {b[key]!r}")
    return out


def verify_generator() -> list[str]:
    """The generator must reproduce the registered component-F recipe for seed 11."""
    captured = _capture([11])
    name = f"scaling_{RUNG}_csz_seed11.json"
    if name not in captured:
        return [f"generator did not emit {name}"]
    return _diff(captured[name], _read(name))


def generate() -> dict:
    problems = verify_generator()
    if problems:
        raise SystemExit("generator no longer reproduces the registered recipe:\n  "
                         + "\n  ".join(problems))

    ds.build_recipe(_component(LOT_B_SEEDS))  # recipes first: build_plan hashes them on disk
    recipes = {seed: _recipe_rel(seed) for seed in LOT_B_SEEDS}
    plans = {}
    for seed in LOT_B_SEEDS:
        rel = ds.build_plan(_component(LOT_B_SEEDS), seed, recipes[seed])
        plan = _read(Path(rel).name)
        plan["registration"] = {
            "path": f"configs/{REGISTRATION}",
            "role": "seed-lot-B replication of component F (927 trading days)",
            "note": ("a declared replication of the registered recipe and budget points; it may not by "
                     "itself escalate a dose claim or replace the incumbent, and it does not alter "
                     "component F's registered verdict"),
        }
        _write(Path(rel).name, plan)
        plans[seed] = rel

    record = {
        "schema_version": 1,
        "registration_id": "b4-scaling-927d-lotb-extension-20261028",
        "kind": "extension_of_registered_prereg",
        "owner_role": "captain",
        "parent_registration": f"configs/{PARENT_PREREG}",
        "parent_component": "F (927 trading days, seeds 11/29/47)",
        "registered_at_utc": _now(),
        "registered_by": "captain",
        "declared_before_any_score_of_these_seeds": True,
        "why": [
            "the rung was registered with three seeds; a three-seed paired mean has SE about 0.0012 "
            "while the promotion line is 0.0034, so the rung is judged at 2.8 SE",
            "the 441-day rung showed the *level* reproduces across independent seed lots (0.07424 vs "
            "0.07428) while the within-lot point ordering does not, so a second lot on the newest rung "
            "is the cheapest way to separate level from lot noise",
            "pooling six seeds is a variance reduction on the rung that currently scores highest; it "
            "does not add a new factor",
        ],
        "seeds": LOT_B_SEEDS,
        "seeds_already_registered": LOT_A_SEEDS,
        "rung_trading_days": DECLARED_TRADING_DAYS,
        "train_window": TRAIN_WINDOW,
        "points": POINTS,
        "steps_per_point": STEPS_PER_POINT,
        "unchanged": {
            "recipe_fields_other_than_seed": "byte-identical to component F (generator reproducibility is "
                                             "asserted before writing)",
            "budget_points": POINTS,
            "budget_only_overrides": "inherited unchanged from the component-F plans",
            "evaluation": "the same five-window delivery caliber (104 trading days) used for every rung",
            "thresholds": "none are introduced here",
        },
        "no_threshold_change": True,
        "does_not_alter_parent_decision": True,
        "pooling_rule": ("the lot-B seeds may be pooled with the registered lot-A seeds to evaluate a "
                         "six-seed blend of the rung; lot-B alone is not a pre-registered decision arm"),
        "cost": {
            "processes": len(LOT_B_SEEDS),
            "steps_per_seed": sum(STEPS_PER_POINT.values()),
            "wall_estimate_hours": 3.5,
            "host": "hz-4.matpool.com:26197 (2x RTX 3090, 172 GB)",
        },
        "artifacts": {
            "recipes": {recipes[s]: _sha(CONFIGS / Path(recipes[s]).name) for s in LOT_B_SEEDS},
            "plans": {plans[s]: _sha(CONFIGS / Path(plans[s]).name) for s in LOT_B_SEEDS},
        },
    }
    path = _write(REGISTRATION, record)
    return {"registration": str(path.relative_to(CONFIGS.parent)), "sha256": _sha(path),
            "recipes": list(record["artifacts"]["recipes"]), "plans": list(record["artifacts"]["plans"])}


def check() -> dict:
    record = _read(REGISTRATION)
    problems: list[str] = []
    for rel, digest in record["artifacts"]["recipes"].items():
        if _sha(CONFIGS / Path(rel).name) != digest:
            problems.append(f"{rel}: sha256 changed")
    for rel, digest in record["artifacts"]["plans"].items():
        if _sha(CONFIGS / Path(rel).name) != digest:
            problems.append(f"{rel}: sha256 changed")
            continue
        plan = _read(Path(rel).name)
        if _sha(CONFIGS / Path(plan["recipe_config"]).name) != plan["recipe_sha256"]:
            problems.append(f"{rel}: recipe_sha256 does not match {plan['recipe_config']}")
        if plan["registration"]["path"] != f"configs/{REGISTRATION}":
            problems.append(f"{rel}: registration path is {plan['registration']['path']!r}")
    problems.extend(verify_generator())
    return {"status": "ok" if not problems else "problems", "problems": problems,
            "plans": len(record["artifacts"]["plans"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = generate() if args.generate else check()
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("status", "ok") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
