"""Register the 132-trading-day independent-information probe before any result exists.

The probe changes one factor only - how many trading days of history feed the frozen recipe -
and asks whether the extra independent information moves the in-sample/out-of-sample gap curve.
The escalation decision is not re-implemented here: it is the registered v2 ``still_improving``
state (configs/b4_budget_decision_prereg_v2_20261027.json) applied to C1 on the primary seed.
That rule module keeps its bytes: this module verifies them and refuses to proceed if they moved.

Everything structural that the probe may not improvise is checked mechanically: the pinned
132-trading-day window, the single escalation comparison, the inherited threshold/interval/repeat
triple, the +/-5% sample-and-step tolerance with its stop rule, the training-memory pre-flight as a
mandatory step, the primary/support seed split, and the bar that level 2 needs its own document.
No wall-clock figure is registered: the step time stays superseded_by t22.

Usage
    python scripts/data_probe_prereg.py --check
    python scripts/data_probe_prereg.py --register
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.budget_decision_prereg import (  # noqa: E402  (one source for hashing and stamping)
    GRID,
    MAIN_PREREG,
    MAIN_REGISTRATION,
    Refused,
    read_json,
    register,
    registration_for,
    round_up,
)
from scripts.budget_decision_prereg import (
    PREREG as RULE_V1_PREREG,
)
from scripts.budget_decision_prereg import (
    verdict as rule_verdict,
)
from scripts.budget_decision_prereg import (
    verify_prereg as verify_rule_document,
)
from scripts.budget_decision_prereg import (
    verify_registration as verify_rule_registration,
)
from scripts.promotion_gate import canonical_digest, rel, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/b4_data_probe_prereg_20261027.json"
DRAFT = ROOT / "configs/b4_data_probe_prereg_DRAFT_20261027.json"
RULE_PREREG = ROOT / "configs/b4_budget_decision_prereg_v2_20261027.json"
BASE_CONFIG = ROOT / "configs/mlplob_v3_60d_csz_19800.json"

REGISTRATION_ID = "b4-data-probe-132d-20261027"
WINDOW = ("2023-04-14", "2023-10-31")
TRAIN_DAYS = 132
ALTERNATIVE_DAYS = 120
EXTENSION_DAYS = 12
START_OF_VALIDATION = "2023-11-01"
PRIMARY_SEED = 11
SUPPORT_SEEDS = (29, 47)
PARALLEL_PROCESSES = 3
TOLERANCE = 0.05
ESTIMATE_STEP_SLACK = 2
ESTIMATE_SAMPLE_SLACK = 2
# Oldest step that may be filtered out of a registered run: every step here is mandatory.
REQUIRED_STEP_IDS = ("training_memory_preflight", "seed_and_arm_guard", "sample_and_step_guard",
                     "config_deviation_guard", "primary_seed_readout", "evaluation_caliber")
WALL_CLOCK_KEY_MARKERS = ("minute", "hour", "second")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Refused(message)


def _identity(document: dict) -> None:
    """Who this is, that it is registered, and which ratified draft it came from."""
    _require(document.get("schema_version") == 1, "unsupported schema_version")
    _require(document.get("registration_id") == REGISTRATION_ID,
             f"registration_id must be {REGISTRATION_ID}")
    _require(document.get("status") == "REGISTERED", "the document must be registered")
    _require(document.get("draft") is not True, "a registered document may not still be a draft")
    _require(document.get("one_shot") is True, "the registration is one-shot")
    _require(bool(document.get("owner_role")), "owner_role is required for the registration record")
    _require(document.get("registration_record") == rel(registration_for(PREREG)),
             "registration_record must name the record beside this document")
    source = document.get("derived_from") or {}
    _require(source.get("draft_revision") == 3, "the registration must derive from draft revision 3")
    _require(source.get("draft_commit") == "a3df84f", "the derivation must cite commit a3df84f")
    _require(source.get("draft_sha256") == sha256_file(DRAFT),
             "the recorded draft bytes are not the draft on disk")
    _require(bool(source.get("registration_time_changes")),
             "the registration-time changes must be listed, not implied")


def _window(document: dict, rule: dict) -> dict:
    """The window is pinned to 132 trading days ending 2023-10-31 and never touches evaluation."""
    design = document["design"]
    window = design["window"]
    _require(tuple(window["train_window"]) == WINDOW,
             f"the training window must be {list(WINDOW)}")
    _require(int(window["train_trading_days"]) == TRAIN_DAYS,
             f"the training window must be {TRAIN_DAYS} trading days")
    verified = window.get("trading_days_verified") or {}
    _require(int(verified.get("count", -1)) == TRAIN_DAYS,
             "the recorded trading-day count must be 132")
    _require(int(verified.get("extension_over_120d", -1)) == EXTENSION_DAYS,
             "the extension over the 120-day alternative must be 12 trading days")
    _require(bool(verified.get("basis")), "the trading-day count needs its basis recorded")
    alternatives = window.get("alternatives_measured") or {}
    _require(tuple(alternatives.get("132_trading_days") or ()) == WINDOW,
             "the 132-day alternative must be the registered window")
    short = tuple(alternatives.get(f"{ALTERNATIVE_DAYS}_trading_days") or ())
    _require(len(short) == 2 and short[0] > WINDOW[0] and short[1] == WINDOW[1],
             "the 120-day alternative must be a strict suffix of the registered window")
    _require(TRAIN_DAYS - int(alternatives[f"{ALTERNATIVE_DAYS}_trading_days_count"]) == EXTENSION_DAYS,
             "the recorded extension must be the difference between the two recorded windows")
    level1 = design["ladder"][0]
    _require(tuple(level1["train_window"]) == WINDOW and int(level1["train_trading_days"]) == TRAIN_DAYS,
             "level 1 must be the registered window")
    boundaries = design["boundaries"]
    _require(boundaries.get("train_window_upper_bound_exclusive") == START_OF_VALIDATION,
             "the training window ends the day before validation starts")
    _require(tuple(boundaries.get("validation_split") or ())[0] == START_OF_VALIDATION,
             "the validation split is the registered one")
    _require(boundaries.get("sealed_read") is False, "this probe may not read the sealed interval")
    _require(WINDOW[1] < START_OF_VALIDATION, "the training window reaches validation")
    evaluation = design["evaluation"]
    _require(list(evaluation["windows"]) == [list(item) for item in rule["windows"]["declared"]],
             "the evaluation windows must be the registered periods, unchanged")
    _require(list(evaluation["names"]) == list(rule["windows"]["names"]),
             "the evaluation period names must be the registered ones")
    _require(int(evaluation["days"]) == int(rule["windows"]["days_expected"]),
             "the evaluated day count must be the registered one")
    _require(evaluation.get("evaluation_version") == rule["estimator"]["evaluation_version"],
             "the evaluation version must be the registered one")
    _require(WINDOW[1] < min(start for start, _ in evaluation["windows"]),
             "the training window reaches an evaluation period")
    _require(WINDOW[0] >= str(boundaries["allowed_start"]), "the window starts before the allowed range")
    _require(WINDOW[1] < str(boundaries["allowed_end"]), "the window ends after the allowed range")
    return {"train_window": list(WINDOW), "train_window_days": TRAIN_DAYS,
            "trading_days_verified": dict(verified), "evaluated_days": int(evaluation["days"])}


def _seeds(document: dict, rule: dict) -> dict:
    """One primary decision seed, two descriptive support seeds, and no reselection."""
    design = document["design"]
    seeds = design["seeds"]
    _require(int(seeds["primary"]) == PRIMARY_SEED, f"the primary seed is {PRIMARY_SEED}")
    _require(tuple(int(item) for item in seeds["support"]) == SUPPORT_SEEDS,
             f"the support seeds are {list(SUPPORT_SEEDS)}")
    _require(seeds.get("primary_is_the_only_escalation_seed") is True,
             "only the primary seed may drive escalation")
    _require(seeds.get("support_may_not_escalate") is True, "support seeds may not escalate")
    _require(seeds.get("no_post_hoc_seed_selection") is True,
             "the primary seed may not be chosen after the fact")
    _require(bool(seeds.get("support_role")), "the support role must be stated")
    aggregate = seeds.get("three_seed_aggregate") or {}
    _require(aggregate.get("requires_own_preregistration") is True,
             "a three-seed aggregate needs its own pre-registration")
    escalation = design["escalation_rule"]["applies_to_comparison"]
    _require(int(escalation["seed"]) == PRIMARY_SEED, "the escalation comparison must use seed 11")

    parallel = design["parallel_execution"]
    _require(int(parallel["processes"]) == PARALLEL_PROCESSES,
             f"the three seeds run in {PARALLEL_PROCESSES} processes")
    _require(parallel.get("one_process_per_gpu") is True, "one process per GPU")
    _require(parallel.get("changes_the_caliber") is False,
             "parallelism may not change the read-out caliber")
    _require("may not lower any evidence requirement" in str(parallel.get("may_not", "")),
             "the parallel declaration must forbid lowering any evidence requirement")
    _require(bool(parallel.get("why_wall_clock_only")), "parallelism needs its stated purpose")

    recipe = read_json(BASE_CONFIG)
    inherits = document["inherits"]["frozen_recipe"]
    _require(inherits["sha256"] == sha256_file(BASE_CONFIG),
             "the frozen recipe bytes changed; this is no longer the registered factor")
    _require(int(recipe["seed"]) == PRIMARY_SEED and int(inherits["seed"]) == PRIMARY_SEED,
             "the frozen recipe already carries the primary seed")

    failure = design["failure_handling"]
    primary_failure = failure["primary_seed_failure"]
    _require(int(primary_failure["seed"]) == PRIMARY_SEED, "the primary-seed rule is about seed 11")
    _require(primary_failure.get("voids_the_round") is True,
             "a primary-seed failure voids the round")
    _require(primary_failure.get("may_reselect") is False, "no reselection after failure")
    support_failure = failure["support_seed_failure"]
    _require(tuple(int(item) for item in support_failure["seeds"]) == SUPPORT_SEEDS,
             "the support-seed rule covers 29 and 47")
    _require(support_failure.get("blocks_c1") is False,
             "a support-seed failure may not block C1")
    return {"primary_seed": PRIMARY_SEED, "support_seeds": list(SUPPORT_SEEDS),
            "parallel_processes": PARALLEL_PROCESSES,
            "primary_seed_only_escalates": True, "support_failure_blocks_c1": False}


def _escalation(document: dict, main: dict, rule: dict) -> dict:
    """The escalation bar is the registered v2 still_improving state, applied to C1 alone."""
    spec = document["design"]["escalation_rule"]
    _require(spec.get("rule_document") == rel(RULE_PREREG), "the escalation rule must be the v2 document")
    _require(spec.get("rule_sha256") == sha256_file(RULE_PREREG),
             "the cited escalation rule bytes are not the registered v2 document")
    _require(spec.get("no_post_hoc_switch") is True, "the escalation comparison may not be switched")

    thresholds = main["derived_thresholds"]
    min_improvement = float(thresholds["min_improvement"])
    grid = float(thresholds["grids"]["primary"])
    _require(abs(grid - GRID) < 1e-15, f"the registered primary grid is {grid}, not {GRID}")
    expected = round_up(grid, min_improvement / 2)
    derivation = spec["threshold_derivation"]
    _require(derivation.get("threshold_basis") == rel(MAIN_PREREG),
             "the threshold basis must be the registered b4 pre-registration")
    _require(derivation.get("threshold_basis_sha256") == sha256_file(MAIN_PREREG),
             "the threshold basis bytes do not reproduce")
    _require(abs(float(derivation["min_improvement"]) - min_improvement) < 1e-12,
             "the cited min_improvement is not the registered one")
    _require(abs(float(derivation["value"]) - expected) < 1e-12,
             f"the escalation floor must be min_improvement/2 rounded up on the grid ({expected})")
    _require(derivation.get("is_not_a_promotion_threshold") is True,
             "the screening floor is not a promotion threshold")
    rule_floor = float(rule["thresholds"]["improvement_floor"]["value"])
    _require(abs(rule_floor - expected) < 1e-12,
             "the registered v2 floor differs from the threshold basis this document cites")

    conditions = spec["conditions"]
    _require(abs(float(conditions["point_estimate_at_least"]) - expected) < 1e-12,
             "the declared point-estimate floor is not the registered floor")
    _require(conditions.get("interval_lower_bound_strictly_greater_than_zero") is True,
             "the 5-day block interval lower bound must be strictly above zero")
    repeat = rule["estimator"]["period_repeat"]
    periods = len(rule["windows"]["declared"])
    required_positive = int(np.ceil(float(main["evidence_rules"]["required_positive_fraction"])
                                    * periods - 1e-12))
    _require(int(conditions["declared_periods"]) == periods,
             "the declared period count is not the registered one")
    _require(int(conditions["required_positive_periods"]) == required_positive
             == int(repeat["required_positive"]),
             f"the repeat rule must be {required_positive} of {periods}")

    interval = spec["interval"]
    registered_interval = rule["estimator"]["interval"]
    for key in ("block_days", "bootstrap_samples", "confidence", "seed"):
        _require(str(interval[key]) == str(registered_interval[key]),
                 f"the interval {key} is not the registered one")

    pair = spec["applies_to_comparison"]
    _require(pair.get("sole_criterion") is True, "C1 must be the sole escalation criterion")
    comparisons = {item["id"]: item for item in document["design"]["comparisons"]}
    _require(len(comparisons) == len(document["design"]["comparisons"]), "a comparison is declared twice")
    _require(set(comparisons) == {"C1_pure_data_effect", "C2_optimum_shift", "C3_training_under_more_data"},
             f"unexpected comparisons {sorted(comparisons)}")
    c1 = comparisons["C1_pure_data_effect"]
    _require(c1["left"] == "132d@1.32" and c1["right"] == "60d@1.32",
             "C1 must compare 132d@1.32 against 60d@1.32")
    _require(pair["left"] == c1["left"] and pair["right"] == c1["right"],
             "the escalation comparison differs from the declared C1")
    driving = [key for key, item in comparisons.items() if item.get("in_escalation_decision")]
    _require(driving == ["C1_pure_data_effect"], f"only C1 may drive escalation, got {driving}")
    for key in ("C2_optimum_shift", "C3_training_under_more_data"):
        _require(comparisons[key].get("in_escalation_decision") is False,
                 f"{key} is reporting only and may not escalate")
    labels = document["design"]["arm_labels"]
    unknown = {side for item in comparisons.values() for side in (item["left"], item["right"])
               if side not in labels}
    _require(not unknown, f"the comparisons use undeclared arm labels {sorted(unknown)}")

    state_name = spec["escalation_state_name"]
    _require(state_name == "still_improving" and state_name in rule["verdicts"]["states"],
             "the escalation state must be the registered still_improving state")
    # Behavioural proof: the registered rule really does escalate a reading at the bar, and
    # really does refuse one whose interval covers zero.
    passing = {"paired_improvement": expected, "block_interval": {"low": 0.0002, "high": 0.004},
               "period_effects": [0.001, 0.002, 0.001, 0.002, 0.001]}
    crossing = {"paired_improvement": expected, "block_interval": {"low": -0.0001, "high": 0.004},
                "period_effects": [0.001, 0.002, 0.001, 0.002, 0.001]}
    thin = {"paired_improvement": expected, "block_interval": {"low": 0.0002, "high": 0.004},
            "period_effects": [0.001, -0.001, 0.002, -0.001, 0.003]}
    _require(rule_verdict(rule, passing)["state"] == state_name,
             "the registered rule does not escalate a reading at the bar")
    _require(rule_verdict(rule, crossing)["state"] != state_name,
             "a reading whose interval covers zero would escalate")
    _require(rule_verdict(rule, thin)["state"] != state_name,
             "a reading with too few positive periods would escalate")

    promotion = document["promotion"]
    _require(promotion.get("decided_here") is False, "this probe does not promote anything")
    _require(promotion.get("requires_own_pre_registration") is True,
             "promotion needs its own pre-registration")
    _require(abs(float(promotion["min_improvement"]) - min_improvement) < 1e-12,
             "the promotion floor must stay the registered min_improvement")
    _require(promotion.get("three_seed_aggregate") is True, "promotion needs the three-seed aggregate")
    _require(document["relation_to_registered_gate"].get("promotion_threshold_unchanged") is True,
             "this document must declare the promotion thresholds unchanged")
    return {"improvement_floor": expected, "min_improvement": min_improvement, "periods": periods,
            "required_positive": required_positive,
            "block_days": int(interval["block_days"]),
            "bootstrap_samples": int(interval["bootstrap_samples"]),
            "confidence": float(interval["confidence"]),
            "escalation_comparison": "C1_pure_data_effect (132d@1.32 vs 60d@1.32) on seed 11",
            "escalation_state": state_name, "sole_criterion": True, "sealed_read": False}


def _level_two(document: dict) -> dict:
    """Level 1 licenses nothing at level 2."""
    design = document["design"]
    ladder = {int(level["level"]): level for level in design["ladder"]}
    _require(set(ladder) == {1, 2}, "the ladder must declare exactly levels 1 and 2")
    for level in ladder.values():
        _require(level.get("needs_own_registration") is True,
                 "every level needs its own pre-registration")
    _require(ladder[2].get("authorized_by_level_1") is False,
             "level 2 may not be authorized by level 1")
    _require(int(ladder[2]["train_trading_days"]) == 240, "the provisional level-2 window is 240 days")
    authority = design["level_2_authorization"]
    _require(authority.get("authorized") is False, "level 2 is not authorized")
    _require(authority.get("authorized_by_level_1") is False, "level 1 does not authorize level 2")
    _require(bool(authority.get("requires")), "level 2 must state what it requires")
    _require("level 2" in " ".join(document["prohibitions"]).lower(),
             "the prohibitions must cover starting level 2")
    return {"level_2_authorized": False, "level_2_requires_own_registration": True}


def _execution(document: dict) -> dict:
    """The tolerance, the stop rule and the memory pre-flight are steps, not prose."""
    cost = document["design"]["steps_and_cost"]
    tolerance = cost["deviation_tolerance"]
    _require(abs(float(tolerance["relative"]) - TOLERANCE) < 1e-12,
             f"the deviation tolerance must be +/-{TOLERANCE}")
    _require(set(tolerance["applies_to"]) == {"usable_samples", "steps"},
             "the tolerance must cover usable samples and steps")
    _require("stop" in str(tolerance.get("rule", "")).lower()
             and "report" in str(tolerance.get("rule", "")).lower(),
             "the tolerance needs a stop-and-report rule")

    baseline = cost["validated_at_60_days"]
    base_days = int(baseline["trading_days"])
    base_samples = int(baseline["usable_samples"])
    base_steps = {float(key): int(value) for key, value in baseline["steps"].items()}
    _require(base_days == 60, "the estimate baseline is the measured 60-day pool")
    for label in ("level_1_132d", "level_2_240d"):
        estimate = cost[label]
        days = int(estimate["trading_days"])
        scale = days / base_days
        _require(abs(int(estimate["usable_samples"]) - base_samples * scale) <= ESTIMATE_SAMPLE_SLACK,
                 f"{label}: the usable-sample estimate does not reproduce the calibrated scale")
        for epochs, declared in estimate["steps"].items():
            expected = base_steps[float(epochs)] * scale
            _require(abs(int(declared) - expected) <= ESTIMATE_STEP_SLACK,
                     f"{label}: the {epochs}-epoch step estimate does not reproduce the "
                     f"calibrated scale ({expected})")
        _require(estimate.get("estimate_not_budget") is True,
                 f"{label}: an estimate may not be registered as a budget")

    steps = document["design"]["required_execution_steps"]
    _require(isinstance(steps, list) and steps, "the required execution steps must be a non-empty list")
    by_id = {}
    for step in steps:
        _require(bool(step.get("id")) and bool(step.get("what")) and bool(step.get("on_fail")),
                 "every required step needs an id, a what and an on_fail")
        _require(step.get("mandatory") is True, f"step {step.get('id')} must be mandatory")
        _require(step["id"] not in by_id, f"step {step['id']} is declared twice")
        by_id[step["id"]] = step
    missing = [step for step in REQUIRED_STEP_IDS if step not in by_id]
    _require(not missing, f"the required execution steps are missing {missing}")
    order = [step["id"] for step in steps]
    _require(order[0] == "training_memory_preflight",
             "the training-memory pre-flight runs before anything else")
    _require(order.index("training_memory_preflight") < order.index("seed_and_arm_guard"),
             "the pre-flight runs before the trajectories")

    preflight = by_id["training_memory_preflight"]
    window = preflight.get("steps_range") or []
    _require(len(window) == 2 and 200 <= int(window[0]) <= int(window[1]) <= 500,
             "the pre-flight must run 200-500 optimizer steps")
    _require("peak rss" in str(preflight["what"]).lower(),
             "the pre-flight must sample peak RSS")
    _require(preflight.get("aborts_the_round") is True, "the pre-flight can abort the round")
    _require(int(preflight.get("concurrent_processes", 0)) == PARALLEL_PROCESSES,
             "the pre-flight must account for all parallel processes")

    guard = by_id["sample_and_step_guard"]
    _require(abs(float(guard["tolerance"]) - TOLERANCE) < 1e-12,
             "the sample-and-step guard must carry the registered tolerance")
    _require(guard.get("stops_the_run") is True, "the sample-and-step guard must stop the run")

    prohibited = " ".join(document["prohibitions"]).lower()
    for phrase in ("run-it-and-see", "silent continuation", "sealed"):
        _require(phrase in prohibited, f"the prohibitions must cover '{phrase}'")
    return {"deviation_tolerance": TOLERANCE, "stop_and_report": True,
            "required_execution_steps": order,
            "memory_preflight_steps": [int(window[0]), int(window[1])],
            "memory_preflight_concurrency": PARALLEL_PROCESSES}


def _no_wall_clock(document: dict) -> dict:
    """No wall-clock figure is registered while t22 is open; the step time stays superseded."""
    cost = document["design"]["steps_and_cost"]
    _require(cost["wall_clock"]["registered"] is None, "no wall-clock figure may be registered")
    _require(bool(cost["wall_clock"].get("why")), "the absent wall-clock figure needs its reason")
    step_time = cost["step_time"]
    _require(step_time.get("superseded_by") == "t22", "the step time stays superseded_by t22")
    _require(step_time.get("not_a_commitment") is True, "the step time is not a commitment")

    def scan(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lower = key.lower()
                if any(marker in lower for marker in WALL_CLOCK_KEY_MARKERS) and isinstance(
                        value, (int, float)) and not isinstance(value, bool):
                    raise Refused(f"{path}/{key}: a wall-clock quantity may not be registered "
                                  f"before t22")
                scan(value, f"{path}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                scan(value, f"{path}[{index}]")

    scan(document, "")
    return {"wall_clock_registered": False, "step_time_superseded_by": "t22"}


def _unchanged_rules(document: dict, main: dict, rule: dict) -> dict:
    """The registered v1/v2 decision bytes are verified, never edited."""
    inherits = document["inherits"]
    v1 = verify_rule_registration(read_json(RULE_V1_PREREG), main)
    v2 = verify_rule_registration(rule, main, RULE_PREREG)
    _require(rule.get("revision_of", {}).get("sha256") == sha256_file(RULE_V1_PREREG),
             "v2 cites bytes that are not the registered v1 document")
    decision_rule = inherits["decision_rule"]
    _require(decision_rule["path"] == rel(RULE_PREREG), "the inherited rule must be the v2 document")
    _require(decision_rule["sha256"] == sha256_file(RULE_PREREG),
             "the inherited rule bytes changed")
    _require(decision_rule["registration_sha256"] == sha256_file(registration_for(RULE_PREREG)),
             "the inherited rule record changed")
    _require(decision_rule["registered_at"] == v2["registered_at"],
             "the cited registration time is not the recorded one")
    _require(inherits["threshold_basis"]["sha256"] == sha256_file(MAIN_PREREG),
             "the threshold basis bytes changed")
    _require(inherits["threshold_basis"]["registration_sha256"] == sha256_file(MAIN_REGISTRATION),
             "the threshold basis record changed")
    _require(document["relation_to_registered_gate"].get("reads_or_changes_the_v1_v2_bytes") is False,
             "this document may not change the registered decision bytes")
    return {"rules_unchanged": True,
            "v1_prereg_sha256": sha256_file(RULE_V1_PREREG),
            "v1_registered_at": v1["registered_at"],
            "v2_prereg_sha256": sha256_file(RULE_PREREG),
            "v2_registered_at": v2["registered_at"],
            "threshold_basis_sha256": sha256_file(MAIN_PREREG),
            "frozen_recipe_sha256": sha256_file(BASE_CONFIG),
            "draft_sha256": sha256_file(DRAFT),
            "promotion_threshold_untouched": True}


def verify_prereg(document: dict, main: dict, rule: dict) -> dict:
    """Every number the probe relies on is reproduced from a registered source, not trusted."""
    verify_rule_document(rule, main)  # the inherited rule still passes its own registered check
    _identity(document)
    checks: dict = {}
    checks.update(_window(document, rule))
    checks.update(_seeds(document, rule))
    checks.update(_escalation(document, main, rule))
    checks.update(_level_two(document))
    checks.update(_execution(document))
    checks.update(_no_wall_clock(document))
    checks.update(_unchanged_rules(document, main, rule))
    return checks


def verify_registration(document: dict, main: dict, rule: dict) -> dict:
    """The record must still describe these bytes, and its digest must reproduce."""
    registration_path = registration_for(PREREG)
    if not registration_path.is_file():
        raise Refused(f"{rel(registration_path)} is missing; run --register before using this rule")
    record = read_json(registration_path)
    if record["prereg_sha256"] != sha256_file(PREREG):
        raise Refused("the registered bytes of the pre-registration have changed")
    if record["registered_at"].startswith("0000"):
        raise Refused("registration timestamp is not set")
    checks = verify_prereg(document, main, rule)
    if record["checks"]["improvement_floor"] != checks["improvement_floor"]:
        raise Refused("the recorded escalation floor disagrees with the document")
    stored = dict(record)
    digest = stored.pop("canonical_digest", None)
    if digest != canonical_digest(stored):
        raise Refused("the registration record digest does not reproduce")
    return record


def escalation_state(reading: dict, rule: dict | None = None) -> dict:
    """Apply the registered v2 rule to one C1 reading; only still_improving escalates."""
    rule_document = rule if rule is not None else read_json(RULE_PREREG)
    state = rule_verdict(rule_document, reading)
    return {"state": state["state"], "escalates_to_level_2": state["state"] == "still_improving",
            "positive_periods": state["positive_periods"],
            "required_positive": state["required_positive"], "why": state["why"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prereg", type=Path, default=PREREG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)

    prereg_path = args.prereg if args.prereg.is_absolute() else ROOT / args.prereg
    document = read_json(prereg_path)
    main_document = read_json(MAIN_PREREG)
    rule = read_json(RULE_PREREG)
    registration_path = registration_for(prereg_path)
    try:
        if args.register:
            record = register(document, verify_prereg(document, main_document, rule), prereg_path)
            print(f"registered {record['registration_id']} at {record['registered_at']}")
            print(f"  prereg {rel(prereg_path)} sha256={record['prereg_sha256'][:16]}...")
            print(f"  window {document['design']['window']['train_window']} "
                  f"({document['design']['window']['train_trading_days']} trading days), "
                  f"floor={record['checks']['improvement_floor']} "
                  f"({record['checks']['min_improvement']}/2 on grid {GRID}), "
                  f"repeat {record['checks']['required_positive']}/{record['checks']['periods']}, "
                  f"blocks {record['checks']['block_days']}x{record['checks']['bootstrap_samples']}")
            print(f"  escalation C1 on seed {record['checks']['primary_seed']}, support "
                  f"{record['checks']['support_seeds']}; level 2 authorized="
                  f"{record['checks']['level_2_authorized']}")
            print(f"  record {rel(registration_path)}")
            return 0
        checks = verify_prereg(document, main_document, rule)
        record = verify_registration(document, main_document, rule) if registration_path.is_file() else None
        print(f"prereg ok: window {checks['train_window']} ({checks['train_window_days']} trading days), "
              f"floor={checks['improvement_floor']} ({checks['min_improvement']}/2), "
              f"repeat {checks['required_positive']}/{checks['periods']}, "
              f"blocks {checks['block_days']}x{checks['bootstrap_samples']}")
        print(f"  escalation: {checks['escalation_comparison']} -> state {checks['escalation_state']}")
        print(f"  seeds: primary {checks['primary_seed']} (only escalator), support "
              f"{checks['support_seeds']} in {checks['parallel_processes']} parallel processes")
        print(f"  tolerance +/-{checks['deviation_tolerance']}; pre-flight steps "
              f"{checks['memory_preflight_steps']} x {checks['memory_preflight_concurrency']} processes")
        print(f"  wall_clock_registered={checks['wall_clock_registered']} "
              f"(step time superseded_by {checks['step_time_superseded_by']}), "
              f"level_2_authorized={checks['level_2_authorized']}, sealed_read={checks['sealed_read']}")
        print(f"  rules unchanged: v1={checks['v1_prereg_sha256'][:16]}... "
              f"v2={checks['v2_prereg_sha256'][:16]}... recipe={checks['frozen_recipe_sha256'][:16]}...")
        if record is None:
            print("not registered yet: run --register to stamp the bytes and the timestamp")
        else:
            print(f"registered_at={record['registered_at']} prereg_sha256="
                  f"{record['prereg_sha256'][:16]}...")
        return 0
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
