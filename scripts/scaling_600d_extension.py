"""Register and generate the 600-day mid-rung of the data-scaling curve (component J).

The registered data-scaling document (``b4_data_scaling_prereg_20261027``) reads the
data axis at 60/132/240/441/927 trading days. Its two largest rungs are the endpoints
that carry the promotion decision, and there is no point between them: with the
endpoints alone the *shape* of the curve between 4 MB and 7.4 MB of training samples
is invisible, so a plateau or an interior peak cannot be distinguished from a straight
line.

This document adds one declared mid-rung: 600 trading days, 2021-05-13..2023-10-31.
It exists to *describe* where the curve turns, not to decide anything. It is written
and registered before any score of these three runs exists; it changes no recipe field
other than the training-window start (plus the fields derived from it), no budget
point, no evaluation window and no threshold; and it modifies nothing about the
registered rungs or the parent decision.

The recipes and plans are emitted by the registered generator itself
(``data_scaling_prereg.build_recipe`` / ``build_plan``), never hand-written, and the
rung is only created if that generator still reproduces the registered 927-day recipe
leaf by leaf.

Usage:
    python scripts/scaling_600d_extension.py --generate
    python scripts/scaling_600d_extension.py --check
    python scripts/scaling_600d_extension.py --check --reference <copy of the 927-day recipe>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_scaling_prereg as ds  # noqa: E402  (registered generator, reused verbatim)
from data_ladder_prereg import CONFIGS, _now, _read, _sha, _write  # noqa: E402

RUNG = "600d"
COMPONENT_ID = "J"
PARENT_PREREG = "b4_data_scaling_prereg_20261027.json"
REGISTRATION = "b4_data_scaling_600d_extension_20261028.registration.json"
TRAIN_WINDOW = ["2021-05-13", "2023-10-31"]
DECLARED_TRADING_DAYS = 600
SEEDS = [11, 29, 47]
POINTS = ["1.32", "2.64"]
POINT_ROLE_TEXT = "descriptive curve-shape point, not a promotion arm"
ANCHOR_RECIPE = "scaling_927d_csz_seed11.json"  # the registered 927-day recipe, seed 11

# Usable samples per trading day implied by the registered 927-day rung:
# 305_227 steps * 32 samples / 1.32 effective epochs / 927 trading days = 7_981.7.
REFERENCE_DENSITY_PER_DAY = 7982
DENSITY_TOLERANCE = 0.02

# The registered generator may only produce these leaf differences between the 927-day
# recipe and this rung: the training-window start and the identity/paths/budget fields
# derived from it. Anything else means the generator drifted and the rung is refused.
EXPECTED_DIFF = frozenset({
    "budget.max_train_samples",
    "data.cache_dir",
    "experiment_id",
    "hypothesis",
    "output_root",
    "splits.train.start",
})


def _component(seeds: list[int]) -> dict:
    """The new rung as a component record of the registered data-scaling document."""
    return {
        "id": COMPONENT_ID,
        "label": RUNG,
        "train_window": list(TRAIN_WINDOW),
        "declared_trading_days": DECLARED_TRADING_DAYS,
        "seeds": list(seeds),
        "points": list(POINTS),
        "raise_max_train_samples": True,
        "role": ("descriptive midpoint of the data axis between the registered 441-day and 927-day "
                 "rungs: it decides nothing, it only shows whether the curve bends in between"),
    }


def _recipe_rel(seed: int) -> str:
    return f"configs/scaling_{RUNG}_csz_seed{seed}.json"


def _capture(component: dict) -> dict[str, dict]:
    """Run the registered recipe generator without touching configs/ on disk."""
    captured: dict[str, dict] = {}
    real_write = ds._write

    def capture(name: str, payload: dict) -> Path:
        captured[name] = payload
        return CONFIGS / name

    ds._write = capture
    try:
        ds.build_recipe(component)
    finally:
        ds._write = real_write
    return captured


def _anchor_component() -> dict:
    """The registered 927-day component F, taken from the parent document itself."""
    for component in ds.COMPONENTS:
        if component["id"] == "F":
            return component
    raise SystemExit("the registered data-scaling document has no component F")


def _diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Recursive leaf-level difference of two JSON documents."""
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


def _expected_values(seed: int) -> dict[str, object]:
    """The values the window and the declared day count must force on this rung."""
    return {
        "seed": seed,
        "experiment_id": f"scaling-{RUNG}-seed{seed}",
        "budget.max_train_samples": 2 * ds._estimated_usable(DECLARED_TRADING_DAYS),
        "data.cache_dir": ds._cache(RUNG),
        "splits.train.start": f"{TRAIN_WINDOW[0]}T00:00:00",
        "splits.train.end": "2023-11-01T00:00:00",
    }


def _dig(record: dict, path: str) -> object:
    node: object = record
    for part in path.split("."):
        node = node[part]  # type: ignore[index]
    return node


def verify_generator(reference: Path | None = None) -> list[str]:
    """No drift: the generator must still reproduce 927d, and only move the window for 600d."""
    problems: list[str] = []

    anchor = _capture(_anchor_component())
    if ANCHOR_RECIPE not in anchor:
        return [f"generator did not emit {ANCHOR_RECIPE}"]
    captured = _capture(_component(SEEDS))
    registered_927 = (json.loads(reference.read_text(encoding="utf-8")) if reference is not None
                      else _read(ANCHOR_RECIPE))

    # (a) the registered 927-day recipe is byte-reproducible: no drift on the parent rung
    reproducible = _diff(anchor[ANCHOR_RECIPE], registered_927)
    if reproducible:
        problems.append("927-day recipe is no longer reproducible: " + "; ".join(reproducible))

    # (b) the 600-day recipe differs from the registered 927-day one only where the window does
    mid = captured[f"scaling_{RUNG}_csz_seed11.json"]
    diff = _diff(mid, registered_927)
    unexpected = sorted({line.split(":")[0] for line in diff} - set(EXPECTED_DIFF))
    missing = sorted(set(EXPECTED_DIFF) - {line.split(":")[0] for line in diff})
    if unexpected:
        problems.append("600-day recipe changed unexpected fields: " + ", ".join(unexpected))
    if missing:
        problems.append("600-day recipe kept fields the window must change: " + ", ".join(missing))

    # (c) the changed fields carry the values the window forces, not just some other value
    for seed in SEEDS:
        record = captured.get(f"scaling_{RUNG}_csz_seed{seed}.json")
        if record is None:
            problems.append(f"generator did not emit the seed-{seed} recipe")
            continue
        for path, expected in _expected_values(seed).items():
            actual = _dig(record, path)
            if actual != expected:
                problems.append(f"seed {seed}: {path} is {actual!r}, expected {expected!r}")

    # (d) the estimate this rung is built on stays on the registered density
    estimate = ds._estimated_usable(DECLARED_TRADING_DAYS)
    reference = DECLARED_TRADING_DAYS * REFERENCE_DENSITY_PER_DAY
    deviation = (estimate - reference) / reference
    if abs(deviation) > DENSITY_TOLERANCE:
        problems.append(
            f"estimated usable samples {estimate} deviate {deviation:+.2%} from "
            f"{DECLARED_TRADING_DAYS} x {REFERENCE_DENSITY_PER_DAY} = {reference}")
    return problems


def generate() -> dict:
    problems = verify_generator()
    if problems:
        raise SystemExit("generator no longer reproduces the registered recipe:\n  "
                         + "\n  ".join(problems))

    ds.build_recipe(_component(SEEDS))  # recipes first: build_plan hashes them on disk
    recipes = {seed: _recipe_rel(seed) for seed in SEEDS}
    plans: dict[int, str] = {}
    for seed in SEEDS:
        rel = ds.build_plan(_component(SEEDS), seed, recipes[seed])
        plan = _read(Path(rel).name)
        plan["registration"] = {
            "path": f"configs/{REGISTRATION}",
            "role": POINT_ROLE_TEXT,
            "note": ("a descriptive midpoint between the registered 441-day and 927-day rungs; it declares no "
                     "threshold, escalates no dose claim, replaces no incumbent and is not a submission candidate"),
        }
        _write(Path(rel).name, plan)
        plans[seed] = rel

    first = _read(Path(plans[SEEDS[0]]).name)
    usable = ds._estimated_usable(DECLARED_TRADING_DAYS)
    steps = first["derived_reference"]["steps_expected"]
    stamp = _now()
    record = {
        "schema_version": 1,
        "registration_id": "b4-data-scaling-600d-extension-20261028",
        "kind": "extension_of_registered_prereg",
        "owner_role": "captain",
        "parent_registration": f"configs/{PARENT_PREREG}",
        "parent_component": (
            "none: the parent declares rungs 60/132/240/441/927 and no rung between 441 and 927; this "
            "document adds rung J as a descriptive midpoint and does not extend a registered component"
        ),
        "purpose_zh": (
            f"在已注册的 441 与 927 交易日之间取一个中点档位（{DECLARED_TRADING_DAYS} 交易日，"
            f"{TRAIN_WINDOW[0]}..{TRAIN_WINDOW[1]}），只用于描述性判断数据量曲线在何处拐弯：此前只有端点，"
            "无法判断中间是否存在峰或平台。"
        ),
        "purpose": (
            "one descriptive interior point of the registered data axis; the endpoints 441 and 927 remain the "
            "decision rungs and the only ones that can move a verdict"
        ),
        "registered_at_utc": stamp,
        "registered_by": "captain",
        "declared_before_any_score_of_these_seeds": True,
        "declared_before_any_score_note": (
            "no run of these three (seed, rung) combinations has an out-of-sample score yet; the seeds are the "
            "registered anchor seeds reused at a new rung, and their scores at other rungs are irrelevant here "
            "because this document declares no threshold and no comparison against a fixed bar"
        ),
        "descriptive_only": True,
        "no_threshold_change": True,
        "does_not_alter_parent_decision": True,
        "why": [
            "the registered axis jumps 441 -> 927 (2.1x in samples): a plateau, a rising line and an interior peak "
            "are all consistent with the two endpoint scores",
            "a point in between costs one rung of compute and turns the curve's shape from unidentifiable into "
            "measured, at matched effective-epoch points and an unchanged evaluation caliber",
            "the rung is watched for shape only: it may not by itself escalate a dose claim, replace the "
            "incumbent or be reported as a candidate",
        ],
        "seeds": SEEDS,
        "seeds_reused_from_registered_rungs": True,
        "rung_trading_days": DECLARED_TRADING_DAYS,
        "train_window": TRAIN_WINDOW,
        "points": POINTS,
        "point_role": POINT_ROLE_TEXT,
        "window_provenance": {
            "source": "/root/data/cpt_jyc_2026_vwap (label archive, `date` column of all 58 month files)",
            "measured_927_window": {
                "range": ["2020-01-02", "2023-10-31"],
                "distinct_trading_days": 927,
                "label_rows": 7415987,
            },
            "measured_600_window": {
                "range": TRAIN_WINDOW,
                "distinct_trading_days": DECLARED_TRADING_DAYS,
                "label_rows": 4799995,
            },
            "anchor_day_index": 328,
            "note": "2021-05-13 is the 328th trading day of the 927-day window, i.e. the first of its last 600",
        },
        "derived": {
            "anchor": {"trading_days": 132, "usable_samples": 1054256},
            "usable_samples_estimate": usable,
            "steps_expected": steps,
            "usable_samples_per_day": round(usable / DECLARED_TRADING_DAYS, 1),
            "reference_density_per_day": REFERENCE_DENSITY_PER_DAY,
            "density_deviation": f"{(usable / DECLARED_TRADING_DAYS - REFERENCE_DENSITY_PER_DAY) / REFERENCE_DENSITY_PER_DAY:+.2%}",
            "min_coverage": first["exposure"]["min_coverage"],
            "estimate_not_budget": True,
            "note": ("the driver re-derives usable samples and steps from the pool it builds; the plan's "
                     "+/-5% deviation tolerance is the post-run monitoring rule, not an automatic gate"),
        },
        "unchanged": {
            "recipe": ("every field of the registered 927-day recipe except the training-window start and the "
                       "fields derived from it (experiment_id, output_root, data.cache_dir, "
                       "budget.max_train_samples, hypothesis, factor block of the plan)"),
            "train_window_end": "2023-10-31 (exclusive train end 2023-11-01T00:00:00)",
            "validation_split": ["2023-11-01T00:00:00", "2023-11-15T00:00:00"],
            "budget_points": "the same two points, 1.32 and 2.64 effective epochs, one continuous trajectory",
            "evaluation": "the same five-window delivery caliber (104 trading days) used for every rung",
            "thresholds": "none are introduced here",
        },
        "reading_rule": (
            "read only as a descriptive shape point between the 441-day and 927-day rungs; it declares no "
            "threshold, no verdict and no candidate, and the parent decision is taken from its own registered "
            "rungs whether or not this point is ever scored"
        ),
        "cost": {
            "processes": len(SEEDS),
            "steps_per_seed": sum(steps.values()),
            "wall_estimate_hours": 6.0,
            "host": "hz-t3.matpool.com:26155 (36 cores / 192 GB / 3x Tesla P100 16GB)",
            "basis": ("30.3 steps/s measured on hz-t3 by featnorm-control-seed11 "
                      "(/root/runs/featnorm-control-seed11/20261028T060829.804730Z_15124166eb98, 86 976 steps in "
                      "2 870.59 s at 132 trading days); the extrapolation assumes the same per-step cost at a "
                      "larger pool and three concurrent single-GPU processes"),
        },
        "artifacts": {
            "recipes": {recipes[s]: _sha(CONFIGS / Path(recipes[s]).name) for s in SEEDS},
            "plans": {plans[s]: _sha(CONFIGS / Path(plans[s]).name) for s in SEEDS},
        },
        "generation": {
            "script": "scripts/scaling_600d_extension.py",
            "sha256": _sha(Path(__file__).resolve()),
            "generated_at_utc": stamp,
            "generator": "data_scaling_prereg.build_recipe / build_plan (registered, reused verbatim)",
        },
    }
    path = _write(REGISTRATION, record)
    return {"registration": str(path.relative_to(CONFIGS.parent)),
            "sha256": _sha(path),
            "usable_samples_estimate": usable,
            "steps_expected": steps,
            "recipes": list(record["artifacts"]["recipes"]),
            "plans": list(record["artifacts"]["plans"])}


def check(reference: Path | None = None) -> dict:
    record = _read(REGISTRATION)
    problems: list[str] = []
    if not (CONFIGS / PARENT_PREREG).exists():
        problems.append(f"parent registration configs/{PARENT_PREREG} is missing")
    for flag in ("declared_before_any_score_of_these_seeds", "descriptive_only",
                 "no_threshold_change", "does_not_alter_parent_decision"):
        if record.get(flag) is not True:
            problems.append(f"{flag} is {record.get(flag)!r}, must be true")
    if record.get("kind") != "extension_of_registered_prereg":
        problems.append(f"kind is {record.get('kind')!r}")
    if record["parent_registration"] != f"configs/{PARENT_PREREG}":
        problems.append(f"parent_registration is {record['parent_registration']!r}")
    expected_steps = {point: int(round(float(point) * record["derived"]["usable_samples_estimate"] / 32))
                      for point in POINTS}
    if record["derived"]["steps_expected"] != expected_steps:
        problems.append(f"steps_expected {record['derived']['steps_expected']} != {expected_steps}")
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
        if plan["registration"]["role"] != POINT_ROLE_TEXT:
            problems.append(f"{rel}: registration role is {plan['registration']['role']!r}")
    script_sha = _sha(Path(__file__).resolve())
    if record["generation"]["sha256"] != script_sha:
        problems.append(f"generator script sha256 is {script_sha}, registered "
                        f"{record['generation']['sha256']}")
    problems.extend(verify_generator(reference))
    return {"status": "ok" if not problems else "problems",
            "problems": problems,
            "registration_sha256": _sha(CONFIGS / REGISTRATION),
            "recipes": len(record["artifacts"]["recipes"]),
            "plans": len(record["artifacts"]["plans"]),
            "usable_samples_estimate": record["derived"]["usable_samples_estimate"],
            "steps_expected": record["derived"]["steps_expected"],
            "min_coverage": record["derived"]["min_coverage"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--reference", type=Path, default=None,
                        help="compare against this copy of the registered 927-day recipe "
                             "instead of configs/scaling_927d_csz_seed11.json")
    args = parser.parse_args(argv)
    result = generate() if args.generate else check(args.reference)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result.get("status", "ok") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
