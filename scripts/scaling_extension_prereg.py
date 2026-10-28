"""Pre-register the 927-day rung's extension from 2.64 to 3.96 effective epochs.

The 927-day rung is registered as a two-point dose measurement (1.32, 2.64). Extending it to a
third point is cheap once the second checkpoint exists - ``run_training_trajectory.py
--resume-from`` continues from it instead of retraining - but "cheap" is exactly what makes an
after-the-fact choice possible. This document fixes the trigger before any 927 score exists: the
extension runs **only** when the seed-paired (2.64 - 1.32) RankIC difference is positive and its
5-day moving-block bootstrap interval excludes zero. Otherwise the rung stops at 2.64.

The three execution plans are the registered two-point plans with one appended budget point and
nothing else changed: the recipe and its ``recipe_sha256`` stay as they are, so the extension
trains the same data with the same configuration as the run it continues.

Usage::

    python scripts/scaling_extension_prereg.py --generate
    python scripts/scaling_extension_prereg.py --register
    python scripts/scaling_extension_prereg.py --check
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from jinyiquant.trajectory import derive_budget_steps  # noqa: E402
from scripts.run_training_trajectory import resume_position  # noqa: E402

PREREG_NAME = "b4_scaling_extension_prereg_20261028.json"
REGISTRATION_NAME = "b4_scaling_extension_prereg_20261028.registration.json"
SEEDS = (11, 29, 47)
BATCH_SIZE = 32
EXTENSION_POINT = {"name": "scaling-3.96", "effective_epochs": 3.96,
                   "role": ("candidate refinement only: the extension is triggered by the "
                            "pre-registered paired 2.64-vs-1.32 rule, and 3.96 never supports a "
                            "dose claim")}
# Measured by the 927-day boot probe (runs/scaling-927d-seed*/.../budget.json): the three seeds
# derive the same pool, so the extension's steps are known before it is triggered.
MEASURED_USABLE = 7399431

EXTENSION_RULE = ("927 日轨迹是否从 2.64 续到 3.96，当且仅当 927 日的种子配对 (2.64 − 1.32) RankIC "
                  "差值为正、且其 5 交易日移动块 bootstrap 95% 区间不含零时触发；否则停在 2.64，不续训。")
EVIDENCE_CALIBER = ("与 240→441 那次完全相同——同 5 个审计窗、104 个交易日、eval-v2、按交易日配对。")
NOT_A_DOSE_CLAIM = ("剂量结论只用已注册的 (1.32, 2.64) 对；3.96 是候选精化，不得用于\"更多数据更好\"的声明。")
HOMOGENEITY = "一旦触发，三个种子全部续到 3.96，不允许混合长度的 blend。"
COST_CAP = "3 种子 × 305,405 步增量，wall 上限 3 h。"
NO_THRESHOLD_CHANGE = "晋升阈值一个字都不改。"


def sha(path: Path) -> str:
    """Byte-exact digest; used for recipes, whose pins were computed over their exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_lf(path: Path) -> str:
    """Digest of the LF-normalised bytes, i.e. of what a ``git archive`` checkout contains."""
    pairs = Path(path).read_bytes().replace(bytes((13, 10)), bytes((10,)))
    return hashlib.sha256(pairs).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def write(name: str, document: dict) -> Path:
    path = CONFIGS / name
    # LF on purpose: git normalises on add, so a `git archive` checkout hashes identically.
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def base_plan(seed: int) -> str:
    return f"scaling_927d_seed{seed}_2point_plan.json"


def extension_plan(seed: int) -> str:
    return f"scaling_927d_seed{seed}_3point_plan.json"


def prefix_guard(usable: int = MEASURED_USABLE) -> dict:
    """Offline proof that the extension plans continue the registered two-point trajectory.

    No training: the derived steps of each plan are computed from the measured usable-sample
    count, and the driver's own resume guard (``resume_position``) is asked whether a source run
    that trained the two registered points may continue under the three-point plan.
    """
    seeds = {}
    for seed in SEEDS:
        two = read(base_plan(seed))
        three = read(extension_plan(seed))
        if three["budget_points"][:2] != two["budget_points"]:
            raise SystemExit(f"seed{seed}: the first two budget points changed")
        if three["recipe_sha256"] != sha(REPO / three["recipe_config"]):
            raise SystemExit(f"seed{seed}: the recipe no longer matches its pin")
        two_steps = [row["steps"] for row in derive_budget_steps(two["budget_points"], usable,
                                                                BATCH_SIZE)]
        three_steps = [row["steps"] for row in derive_budget_steps(three["budget_points"], usable,
                                                                  BATCH_SIZE)]
        if three_steps[:2] != two_steps:
            raise SystemExit(f"seed{seed}: the extension changes the registered steps")
        source = {"executed_steps": two_steps, "config_sha256": None, "plan_sha256": None}
        resume_position(three, derive_budget_steps(three["budget_points"], usable, BATCH_SIZE),
                        {"position": {"steps": two_steps[-1]}}, source)
        def refused(points: list[dict], payload_steps: int) -> bool:
            """True when the guard rejects this plan/payload pair (the negative controls)."""
            try:
                resume_position(three, derive_budget_steps(points, usable, BATCH_SIZE),
                                {"position": {"steps": payload_steps}}, source)
            except ValueError:
                return True
            return False

        shifted = copy.deepcopy(three["budget_points"])
        shifted[1]["effective_epochs"] += 0.01          # a point that no longer matches its source
        seeds[str(seed)] = {
            "registered_steps": two_steps, "extension_steps": three_steps,
            "increment": three_steps[-1] - three_steps[1], "prefix_accepted": True,
            "shifted_point_rejected": refused(shifted, two_steps[-1]),
            "finished_plan_rejected": refused(three["budget_points"], three_steps[-1])}
    return {"usable_samples": usable, "batch_size": BATCH_SIZE,
            "rule": ("the source run's trained points must be the head of the extension plan's "
                     "points (run_training_trajectory.resume_position)"),
            "seeds": seeds}


def generate() -> dict:
    for seed in SEEDS:
        plan = read(base_plan(seed))
        extended = copy.deepcopy(plan)
        extended["created_at_utc"] = now()
        extended["budget_points"] = plan["budget_points"] + [dict(EXTENSION_POINT)]
        write(extension_plan(seed), extended)
    guard = prefix_guard()
    prereg = {
        "schema_version": 1,
        "registration_id": "b4-scaling-extension-20261028",
        "owner_role": "captain",
        "phase": "B4 data-quantity research, 927-day rung extension (candidate refinement)",
        "status": "registered",
        "one_shot": True,
        "registered_at_source": ("written while the registered 1.32/2.64 trajectory of the 927-day "
                                 "rung was training and before any 927 score was read"),
        "purpose": ("Fix, before any 927-day score exists, the single condition under which the "
                    "927-day trajectory is extended from 2.64 to 3.96 effective epochs, so the "
                    "extension cannot be chosen after seeing the numbers."),
        "extension_rule": EXTENSION_RULE,
        "evidence_caliber": EVIDENCE_CALIBER,
        "not_a_dose_claim": NOT_A_DOSE_CLAIM,
        "homogeneity": HOMOGENEITY,
        "cost_cap": COST_CAP,
        "no_threshold_change": NO_THRESHOLD_CHANGE,
        "base_plans": [f"configs/{base_plan(seed)}" for seed in SEEDS],
        "extension_plans": [f"configs/{extension_plan(seed)}" for seed in SEEDS],
        "plan_shape": {
            "appended_point": dict(EXTENSION_POINT),
            "unchanged": ("recipe_config, recipe_sha256, both registered budget points, exposure, "
                          "resume, cloud and evaluation_scope are byte-identical to the registered "
                          "two-point plans"),
            "mechanism": ("run_training_trajectory.py --resume-from <finished 2.64 run> continues "
                          "from that run's last checkpoint; the 1.32 and 2.64 points are not "
                          "retrained"),
            "derived_reference": ("left as registered: it documents the two registered magnitudes; "
                                  "the driver re-derives every point from the pool actually built"),
        },
        "trigger_reading": {
            "seed_pairing": list(SEEDS),
            "paired_quantity": ("per-seed RankIC difference (2.64 - 1.32) at the registered points, "
                                "then the seed mean"),
            "interval": ("5-day moving-block bootstrap 95% interval, 4000 draws, paired by trading "
                         "day (the registered 240->441 estimator)"),
            "windows": "the same five audit windows and 104 paired trading days as the 240->441 comparison",
            "metric": "eval-v2 RankIC, paired by trading day",
            "trigger": ("seed-paired difference positive AND the 95% interval excludes zero; "
                        "otherwise the rung stops at 2.64"),
        },
        "prefix_guard": guard,
        "prohibitions": [
            "no extension when the registered trigger does not fire",
            "no mixed-length blend across seeds",
            "no dose claim from 3.96",
            "no change to any promotion threshold, secondary line, net5 limit or cost scenario",
            "no reading of the sealed interval",
            "no retraining of the 1.32 or 2.64 points",
        ],
    }
    path = write(PREREG_NAME, prereg)
    return {"prereg": str(path.relative_to(REPO)), "sha256": sha(path), "prefix_guard": guard}


def register() -> dict:
    prereg_path = CONFIGS / PREREG_NAME
    if not prereg_path.exists():
        raise SystemExit("generate the extension plans and the pre-registration first")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    plan_hashes, recipe_hashes = {}, {}
    for plan_rel in prereg["extension_plans"]:
        plan = json.loads((REPO / plan_rel).read_text(encoding="utf-8"))
        if plan["recipe_sha256"] != sha(REPO / plan["recipe_config"]):
            raise SystemExit(f"recipe changed after the plan was generated: {plan['recipe_config']}")
        plan_hashes[plan_rel] = sha(REPO / plan_rel)
        recipe_hashes[plan["recipe_config"]] = plan["recipe_sha256"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.strip().splitlines()
    record = {
        "schema_version": 1,
        "prereg_path": f"configs/{PREREG_NAME}",
        "prereg_sha256": sha(prereg_path),
        "registered_at": now(),
        "git_head": head,
        "plan_sha256": plan_hashes,
        "recipe_sha256": recipe_hashes,
        "ordering": {
            "registered_before_the_extension_run_started": True,
            "registered_before_any_927_score_was_read": True,
            "note": ("the registered 1.32/2.64 trajectory of the 927-day rung was already training; "
                     "the extension this document governs has neither started nor been read"),
        },
        "prefix_guard": prereg["prefix_guard"],
        "code": {
            "git_head": head,
            "uncommitted_paths": sorted(line.split(maxsplit=1)[-1] for line in dirty),
            "driver_sha256": sha_lf(REPO / "scripts" / "run_training_trajectory.py"),
            "adapter_sha256": sha_lf(REPO / "modified_tlob" / "adapter.py"),
            "digest_rule": "LF-normalised bytes, so the digests match the git-archive tree on the host",
        },
    }
    path = write(REGISTRATION_NAME, record)
    return {"registration": str(path.relative_to(REPO)), "sha256": sha(path), "record": record}


def check() -> dict:
    guard = prefix_guard()
    prereg = read(PREREG_NAME)
    registration = read(REGISTRATION_NAME)
    plans_ok = all(sha(REPO / rel) == registration["plan_sha256"][rel]
                   for rel in prereg["extension_plans"])
    recipes_ok = all(sha(REPO / rel) == registration["recipe_sha256"][rel]
                     for rel in registration["recipe_sha256"])
    return {"prereg_sha256_matches": sha(CONFIGS / PREREG_NAME) == registration["prereg_sha256"],
            "plan_sha256_matches": plans_ok, "recipe_sha256_matches": recipes_ok,
            "prefix_guard": guard}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true", help="write the plans and the prereg")
    parser.add_argument("--register", action="store_true", help="write the registration record")
    parser.add_argument("--check", action="store_true", help="re-verify digests and the prefix guard")
    args = parser.parse_args(argv)
    if args.generate:
        print(json.dumps(generate(), indent=1, ensure_ascii=False))
    if args.register:
        print(json.dumps(register(), indent=1, ensure_ascii=False))
    if args.check:
        result = check()
        print(json.dumps(result, indent=1, ensure_ascii=False))
        return 0 if all(value for key, value in result.items() if isinstance(value, bool)) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
