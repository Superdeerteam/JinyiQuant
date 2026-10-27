"""The 132-day data probe may not be re-read, re-aimed or upgraded after the fact."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.budget_decision_prereg import MAIN_PREREG, Refused
from scripts.data_probe_prereg import (
    DRAFT,
    PREREG,
    RULE_PREREG,
    escalation_state,
    main,
    registration_for,
    verify_prereg,
    verify_registration,
)

ROOT = Path(__file__).resolve().parents[1]
V1_PREREG = ROOT / "configs/b4_budget_decision_prereg_20261027.json"
V1_SHA256 = "39008232e71f599642bfbbd17a57cece49a632edfe4e3749fb07aa3a371de0cb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def document() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def main_document() -> dict:
    return json.loads(MAIN_PREREG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rule() -> dict:
    return json.loads(RULE_PREREG.read_text(encoding="utf-8"))


def _tampered(document: dict, mutate) -> dict:
    tampered = copy.deepcopy(document)
    mutate(tampered)
    return tampered


def test_the_registered_probe_pins_its_window_and_inherits_the_escalation_bar(document, main_document, rule):
    checks = verify_prereg(document, main_document, rule)
    assert checks["train_window"] == ["2023-04-14", "2023-10-31"]
    assert checks["train_window_days"] == 132
    assert checks["trading_days_verified"]["count"] == 132
    assert checks["trading_days_verified"]["extension_over_120d"] == 12
    # The escalation bar is the registered v2 state, not a number this document may restate.
    assert checks["improvement_floor"] == pytest.approx(0.0017)
    assert checks["improvement_floor"] == pytest.approx(
        main_document["derived_thresholds"]["min_improvement"] / 2)
    assert checks["improvement_floor"] == pytest.approx(rule["thresholds"]["improvement_floor"]["value"])
    assert checks["periods"] == 5 and checks["required_positive"] == 4
    assert checks["block_days"] == 5 and checks["bootstrap_samples"] == 4000
    assert checks["escalation_comparison"].startswith("C1_pure_data_effect")
    assert checks["sole_criterion"] is True and checks["escalation_state"] == "still_improving"
    assert checks["sealed_read"] is False


def test_the_registration_record_stamps_the_registered_bytes(document, main_document, rule):
    record = verify_registration(document, main_document, rule)
    assert record["prereg_path"] == "configs/b4_data_probe_prereg_20261027.json"
    assert record["prereg_sha256"] == _sha(PREREG)
    assert record["registered_at"].startswith("2026-")
    assert record["thresholds_modified"] is False
    assert record["checks"]["train_window_days"] == 132
    assert record["checks"]["primary_seed"] == 11
    assert record["checks"]["level_2_authorized"] is False
    assert record["canonical_digest"]


def test_the_derivation_cites_the_ratified_draft_not_a_rewritten_one(document, main_document, rule):
    assert document["derived_from"]["draft_revision"] == 3
    assert document["derived_from"]["draft_sha256"] == _sha(DRAFT)
    tampered = _tampered(document, lambda d: d["derived_from"].__setitem__("draft_sha256", "0" * 64))
    with pytest.raises(Refused, match="draft bytes"):
        verify_prereg(tampered, main_document, rule)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["design"]["window"].__setitem__("train_window", ["2023-04-15", "2023-10-31"]),
         "training window"),
        (lambda d: d["design"]["window"].__setitem__("train_trading_days", 120), "trading days"),
        (lambda d: d["design"]["window"]["trading_days_verified"].__setitem__("count", 130),
         "trading-day count"),
        (lambda d: d["design"]["window"]["alternatives_measured"].__setitem__(
            "120_trading_days_count", 118), "extension"),
        (lambda d: d["design"]["boundaries"].__setitem__("sealed_read", True), "sealed"),
        (lambda d: d["design"]["evaluation"]["windows"][0].__setitem__(1, "2023-11-29"),
         "evaluation windows"),
    ],
)
def test_the_window_may_not_move(document, main_document, rule, mutate, message):
    with pytest.raises(Refused, match=message):
        verify_prereg(_tampered(document, mutate), main_document, rule)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["design"]["escalation_rule"]["applies_to_comparison"].__setitem__(
            "left", "132d@2.64"), "escalation comparison"),
        (lambda d: d["design"]["escalation_rule"]["applies_to_comparison"].__setitem__(
            "seed", 29), "seed 11"),
        (lambda d: d["design"]["comparisons"][1].__setitem__("in_escalation_decision", True),
         "only C1"),
        (lambda d: d["design"]["comparisons"][2].__setitem__("left", "132d@5.28"), "arm labels"),
        (lambda d: d["design"]["escalation_rule"]["threshold_derivation"].__setitem__(
            "value", 0.0018), "floor"),
        (lambda d: d["design"]["escalation_rule"]["conditions"].__setitem__(
            "required_positive_periods", 3), "repeat rule"),
        (lambda d: d["design"]["escalation_rule"]["interval"].__setitem__("block_days", 3),
         "block_days"),
        (lambda d: d["design"]["escalation_rule"].__setitem__("rule_sha256", "0" * 64),
         "v2 document"),
    ],
)
def test_the_escalation_bar_may_not_be_re_aimed(document, main_document, rule, mutate, message):
    with pytest.raises(Refused, match=message):
        verify_prereg(_tampered(document, mutate), main_document, rule)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["design"]["seeds"].__setitem__("primary", 29), "primary seed"),
        (lambda d: d["design"]["seeds"].__setitem__("support", [29, 31]), "support seeds"),
        (lambda d: d["design"]["seeds"].__setitem__("no_post_hoc_seed_selection", False),
         "after the fact"),
        (lambda d: d["design"]["seeds"].__setitem__("support_may_not_escalate", False),
         "may not escalate"),
        (lambda d: d["design"]["failure_handling"]["primary_seed_failure"].__setitem__(
            "voids_the_round", False), "voids the round"),
        (lambda d: d["design"]["failure_handling"]["support_seed_failure"].__setitem__(
            "blocks_c1", True), "not block C1"),
        (lambda d: d["design"]["parallel_execution"].__setitem__("may_not", "parallel is fine"),
         "lowering any evidence"),
        (lambda d: d["design"]["parallel_execution"].__setitem__("changes_the_caliber", True),
         "caliber"),
    ],
)
def test_the_seed_plan_may_not_be_rearranged(document, main_document, rule, mutate, message):
    with pytest.raises(Refused, match=message):
        verify_prereg(_tampered(document, mutate), main_document, rule)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["design"]["steps_and_cost"]["deviation_tolerance"].__setitem__(
            "relative", 0.10), "tolerance"),
        (lambda d: d["design"]["steps_and_cost"]["level_1_132d"]["steps"].__setitem__(
            "1.32", 50000), "calibrated scale"),
        (lambda d: d["design"]["required_execution_steps"].pop(0), "missing"),
        (lambda d: d["design"]["required_execution_steps"].insert(
            0, d["design"]["required_execution_steps"].pop(1)), "before anything else"),
        (lambda d: d["design"]["required_execution_steps"][0].__setitem__(
            "steps_range", [50, 100]), "200-500"),
        (lambda d: d["design"]["required_execution_steps"][0].__setitem__(
            "concurrent_processes", 1), "parallel processes"),
        (lambda d: d["design"]["required_execution_steps"][2].__setitem__(
            "stops_the_run", False), "stop the run"),
    ],
)
def test_the_execution_steps_may_not_be_softened(document, main_document, rule, mutate, message):
    with pytest.raises(Refused, match=message):
        verify_prereg(_tampered(document, mutate), main_document, rule)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["design"]["level_2_authorization"].__setitem__("authorized", True),
         "not authorized"),
        (lambda d: d["design"]["ladder"][1].__setitem__("authorized_by_level_1", True),
         "authorized by level 1"),
        (lambda d: d["design"]["ladder"][1].__setitem__("needs_own_registration", False),
         "own pre-registration"),
        (lambda d: d["promotion"].__setitem__("decided_here", True), "does not promote"),
        (lambda d: d["promotion"].__setitem__("min_improvement", 0.0017), "min_improvement"),
        (lambda d: d["relation_to_registered_gate"].__setitem__(
            "promotion_threshold_unchanged", False), "unchanged"),
    ],
)
def test_the_level_bar_and_the_promotion_line_may_not_be_touched(document, main_document, rule,
                                                                  mutate, message):
    with pytest.raises(Refused, match=message):
        verify_prereg(_tampered(document, mutate), main_document, rule)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["design"]["steps_and_cost"].__setitem__("estimate_minutes", 55.1),
         "wall-clock quantity"),
        (lambda d: d["design"]["steps_and_cost"]["wall_clock"].__setitem__("registered", 55.1),
         "wall-clock figure"),
        (lambda d: d["design"]["steps_and_cost"]["wall_clock"].__setitem__("registered", "55 min"),
         "wall-clock figure"),
        (lambda d: d["design"]["steps_and_cost"]["step_time"].__setitem__(
            "superseded_by", "t99"), "t22"),
        (lambda d: d["design"]["steps_and_cost"]["step_time"].__setitem__(
            "not_a_commitment", False), "not a commitment"),
    ],
)
def test_no_wall_clock_may_be_registered_before_t22(document, main_document, rule, mutate, message):
    with pytest.raises(Refused, match=message):
        verify_prereg(_tampered(document, mutate), main_document, rule)


def test_the_registered_v1_and_v2_rule_bytes_did_not_move(main_document, rule):
    checks = verify_prereg(json.loads(PREREG.read_text(encoding="utf-8")), main_document, rule)
    assert checks["v1_prereg_sha256"] == V1_SHA256 == _sha(V1_PREREG)
    assert checks["v2_prereg_sha256"] == _sha(RULE_PREREG) == "70adbc11d63f3cf4d01c0ebd44c23df0" \
                                                            "d950ad33c037921221e49071f92ab392"
    assert rule["revision_of"]["sha256"] == V1_SHA256
    # A doc that edited the v1/v2 chain would no longer match the registered digests.
    tampered = _tampered(json.loads(PREREG.read_text(encoding="utf-8")),
                         lambda d: d["inherits"]["decision_rule"].__setitem__("sha256", "0" * 64))
    with pytest.raises(Refused, match="inherited rule bytes"):
        verify_prereg(tampered, main_document, rule)


def test_the_escalation_state_is_read_through_the_registered_rule(rule):
    floor = rule["thresholds"]["improvement_floor"]["value"]
    at_the_bar = escalation_state({"paired_improvement": floor,
                                   "block_interval": {"low": 0.0002, "high": 0.004},
                                   "period_effects": [0.001, 0.002, 0.001, 0.002, 0.001]}, rule)
    assert at_the_bar["escalates_to_level_2"] is True
    assert at_the_bar["state"] == "still_improving"
    crossing = escalation_state({"paired_improvement": floor,
                                 "block_interval": {"low": -0.0001, "high": 0.004},
                                 "period_effects": [0.001, 0.002, 0.001, 0.002, 0.001]}, rule)
    assert crossing["escalates_to_level_2"] is False
    thin = escalation_state({"paired_improvement": floor,
                             "block_interval": {"low": 0.0002, "high": 0.004},
                             "period_effects": [0.001, -0.001, 0.002, -0.001, 0.003]}, rule)
    assert thin["escalates_to_level_2"] is False and thin["positive_periods"] == 3
    downgrade = escalation_state({"paired_improvement": -0.0256,
                                  "block_interval": {"low": -0.0295, "high": -0.0231},
                                  "period_effects": [-0.02, -0.03, -0.02, -0.03, -0.025]}, rule)
    assert downgrade["escalates_to_level_2"] is False
    assert downgrade["state"] == "significantly_degraded"


def test_the_registration_is_one_shot(document):
    assert registration_for(PREREG).is_file()
    assert main(["--register"]) == 2  # already registered: the entry point refuses a second stamp
