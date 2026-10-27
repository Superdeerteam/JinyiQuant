"""The budget decision rule must be three-state, inherited and impossible to read backwards."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.budget_decision_prereg import (
    MAIN_PREREG,
    PREREG,
    Refused,
    apply,
    decision,
    loss_trend,
    read_json,
    verdict,
    verify_prereg,
    verify_registration,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def document() -> dict:
    return read_json(PREREG)


@pytest.fixture(scope="module")
def main() -> dict:
    return read_json(MAIN_PREREG)


def test_registered_document_inherits_the_basis_without_touching_it(document, main):
    checks = verify_prereg(document, main)
    assert checks["improvement_floor"] == pytest.approx(0.0017)
    assert checks["improvement_floor"] == pytest.approx(main["derived_thresholds"]["min_improvement"] / 2)
    assert checks["periods"] == 5 and checks["required_positive"] == 4
    assert checks["sealed_read"] is False
    record = verify_registration(document, main)
    assert record["thresholds_modified"] is False
    assert record["candidate_state_at_registration"].startswith("no budget-point replay metric was read")
    # The publication thresholds are still exactly the registered ones.
    assert document["relation_to_registered_gate"]["promotion_threshold_unchanged"] is True
    assert "0.0034" in str(document["relation_to_registered_gate"]["a_candidate_that_clears_this_rule_still_needs"])


@pytest.mark.parametrize("value", [0.002, 0.001, 0.0034])
def test_a_floor_that_is_not_half_the_registered_minimum_is_refused(document, main, value):
    tampered = copy.deepcopy(document)
    tampered["thresholds"]["improvement_floor"]["value"] = value
    with pytest.raises(Refused, match="half|floor"):
        verify_prereg(tampered, main)


def test_declared_periods_must_be_the_registered_ones(document, main):
    tampered = copy.deepcopy(document)
    tampered["windows"]["declared"][0] = ["2023-11-01", "2023-11-29"]
    with pytest.raises(Refused, match="differ from the registered"):
        verify_prereg(tampered, main)


def test_the_undeclared_cell_may_not_also_be_a_declared_cell(document, main):
    tampered = copy.deepcopy(document)
    tampered["decision_table"]["cells"].append(
        {"id": "plateaued__improving", "loss_trend": "plateaued",
         "validation_verdict": "still_improving", "conclusion": "undertrained",
         "conclusion_label": "欠训", "next_step": "add epochs", "next_step_detail": "invented"})
    with pytest.raises(Refused, match="undeclared combination|declared 2x2"):
        verify_prereg(tampered, main)


def test_reading_a_sealed_window_is_refused(document, main):
    tampered = copy.deepcopy(document)
    tampered["windows"]["sealed_read"] = True
    with pytest.raises(Refused, match="sealed"):
        verify_prereg(tampered, main)


def _reading(improvement: float, low: float, high: float, effects: list[float]) -> dict:
    return {"paired_improvement": improvement,
            "block_interval": {"low": low, "high": high},
            "period_effects": effects}


def test_the_three_states_are_derived_exactly(document):
    floor = document["thresholds"]["improvement_floor"]["value"]
    improving = verdict(document, _reading(floor + 0.0005, 0.0002, 0.004,
                                           [0.001, 0.002, 0.001, 0.002, 0.001]))
    assert improving["state"] == "still_improving"

    saturated = verdict(document, _reading(-0.0004, -0.002, 0.001,
                                           [-0.001, 0.002, -0.001, 0.001, -0.001]))
    assert saturated["state"] == "saturated"

    # A positive improvement whose interval covers zero is not "still improving".
    crossing = verdict(document, _reading(floor + 0.0005, -0.0001, 0.004,
                                          [0.001, 0.002, 0.001, 0.002, 0.001]))
    assert crossing["state"] == "inconclusive"

    # Improvement >= floor with an interval above zero but only 3 positive periods is not enough.
    thin = verdict(document, _reading(floor + 0.0005, 0.0002, 0.004,
                                      [0.001, -0.001, 0.002, -0.001, 0.003]))
    assert thin["state"] == "inconclusive" and thin["positive_periods"] == 3

    # Improvement <= 0 whose interval excludes zero matches neither rule.
    negative = verdict(document, _reading(-0.0004, -0.002, -0.0001,
                                          [-0.001, -0.001, -0.001, -0.001, -0.001]))
    assert negative["state"] == "inconclusive"


def test_loss_trend_is_mechanical_and_undetermined_by_default(document):
    labels = [arm["label"] for arm in document["arms"]]
    falling = loss_trend(document, dict(zip(labels, [1.00, 0.95, 0.90])))
    assert falling["trend"] == "falling"
    plateaued = loss_trend(document, dict(zip(labels, [1.00, 1.001, 1.000])))
    assert plateaued["trend"] == "plateaued"
    mixed = loss_trend(document, dict(zip(labels, [1.00, 0.95, 0.949])))
    assert mixed["trend"] == "undetermined"
    assert loss_trend(document, None)["trend"] == "undetermined"
    assert loss_trend(document, {labels[0]: 1.0})["trend"] == "undetermined"


@pytest.mark.parametrize(
    "losses, state, expected_label, expected_cell",
    [
        ([1.00, 0.95, 0.90], "still_improving", "欠训", "falling__improving"),
        ([1.00, 0.95, 0.90], "saturated", "过拟合 60 日样本（数据受限）", "falling__saturated"),
        ([1.00, 1.001, 1.000], "saturated", "优化收敛（容量/结构受限）", "plateaued__saturated"),
        ([1.00, 1.001, 1.000], "still_improving", "证据不足", "plateaued__improving"),
        ([1.00, None, 0.90], "still_improving", "证据不足", None),
    ],
)
def test_the_table_never_invents_a_side(document, losses, state, expected_label, expected_cell):
    labels = [arm["label"] for arm in document["arms"]]
    values = None if None in losses else dict(zip(labels, losses))
    trend = loss_trend(document, values)
    outcome = decision(document, {"state": state, "why": "test"}, trend)
    assert outcome["state_label"] == expected_label
    assert outcome["table_cell"] == expected_cell
    if expected_label == "证据不足":
        assert outcome["state"] == "inconclusive"


def test_inconclusive_routes_to_a_named_probe(document):
    falling = loss_trend(document, dict(zip([arm["label"] for arm in document["arms"]],
                                            [1.00, 0.95, 0.90])))
    routed = decision(document, {"state": "inconclusive", "why": "test"}, falling)
    assert routed["route"]["first"] == "upper_bound"
    plateaued = loss_trend(document, dict(zip([arm["label"] for arm in document["arms"]],
                                              [1.00, 1.001, 1.000])))
    assert decision(document, {"state": "inconclusive", "why": "test"},
                    plateaued)["route"]["first"] == "capacity"
    assert decision(document, {"state": "inconclusive", "why": "test"},
                    loss_trend(document, None))["route"]["first"] == "none"


def test_applying_the_rule_refuses_an_analysis_that_predates_the_registration(document, main, tmp_path):
    record = verify_registration(document, main)
    early = tmp_path / "early.json"
    early.write_text(json.dumps({
        "created_at": "2000-01-01T00:00:00+00:00",
        "coverage": {"periods": document["windows"]["names"],
                     "days": document["windows"]["days_expected"]},
        "pairs": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Refused, match="before the registration"):
        apply(document, record, early, None, None, None)


def test_applying_the_rule_needs_the_declared_window(document, main, tmp_path):
    record = verify_registration(document, main)
    short = tmp_path / "short.json"
    short.write_text(json.dumps({
        "created_at": "2999-01-01T00:00:00+00:00",
        "coverage": {"periods": document["windows"]["names"], "days": 10},
        "pairs": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Refused, match="days"):
        apply(document, record, short, None, None, None)


def test_applying_the_rule_produces_one_of_three_states(document, main, tmp_path):
    record = verify_registration(document, main)
    primary = document["primary_pair"]
    analysis = {
        "created_at": "2999-01-01T00:00:00+00:00",
        "coverage": {"periods": document["windows"]["names"],
                     "days": document["windows"]["days_expected"]},
        "pairs": [{"baseline": primary["baseline"], "candidate": primary["candidate"],
                   "paired_improvement": 0.0025,
                   "block_interval": {"low": 0.0004, "high": 0.0041},
                   "period_effects": [0.001, 0.003, 0.002, 0.001, 0.003]}],
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
    losses = tmp_path / "losses.json"
    labels = [arm["label"] for arm in document["arms"]]
    losses.write_text(json.dumps(dict(zip(labels, [1.00, 0.95, 0.90]))), encoding="utf-8")

    outcome = apply(document, record, path, losses, None, None)
    assert outcome["validation"]["state"] == "still_improving"
    assert outcome["loss_trend"]["trend"] == "falling"
    assert outcome["outcome"]["state_label"] == "欠训"
    assert outcome["outcome"]["table_cell"] == "falling__improving"
    assert outcome["prereg"]["registered_at"] == record["registered_at"]
    assert outcome["no_threshold_relaxation"] is True
    assert "min_improvement 0.0034" in outcome["promotion"]

    # Without a loss record the same reading cannot resolve to a side.
    blind = apply(document, record, path, None, None, None)
    assert blind["loss_trend"]["trend"] == "undetermined"
    assert blind["outcome"]["state_label"] == "证据不足"


# --- v2: the fourth state and the axis collapse -----------------------------------------

V2_PREREG = ROOT / "configs/b4_budget_decision_prereg_v2_20261027.json"


@pytest.fixture(scope="module")
def v2() -> dict:
    return read_json(V2_PREREG)


def test_v2_is_a_superset_that_keeps_v1_classifications(v2, document):
    assert document["verdicts"]["states"] == ["still_improving", "saturated", "inconclusive"]
    assert v2["verdicts"]["states"] == ["still_improving", "saturated", "significantly_degraded",
                                        "inconclusive"]
    # Every reading v1 could classify gets the same label under v2.
    floor = v2["thresholds"]["improvement_floor"]["value"]
    improving = _reading(floor + 0.0005, 0.0002, 0.004, [0.001, 0.002, 0.001, 0.002, 0.001])
    assert verdict(document, improving)["state"] == verdict(v2, improving)["state"]
    saturated = _reading(-0.0004, -0.002, 0.001, [-0.001, 0.002, -0.001, 0.001, -0.001])
    assert verdict(document, saturated)["state"] == verdict(v2, saturated)["state"] == "saturated"


def test_v2_determines_a_downgrade_that_v1_had_to_call_inconclusive(v2, document):
    downgrade = _reading(-0.0256, -0.0295, -0.0231, [-0.02, -0.03, -0.02, -0.03, -0.025])
    assert verdict(document, downgrade)["state"] == "inconclusive"
    decided = verdict(v2, downgrade)
    assert decided["state"] == "significantly_degraded"
    assert "entirely below zero" in decided["why"]


def test_v2_projects_the_state_onto_the_two_valued_table_column(v2):
    labels = [arm["label"] for arm in v2["arms"]]
    falling = loss_trend(v2, dict(zip(labels, [1.00, 0.95, 0.90])))
    outcome = decision(v2, {"state": "significantly_degraded", "why": "test"}, falling)
    assert outcome["state"] == "determined"
    assert outcome["validation_axis"] == "not_improving"
    assert outcome["table_cell"] == "falling__not_improving"
    assert outcome["state_label"] == "过拟合 60 日样本（数据受限）"
    # A flat loss with a downgrade lands on the convergence cell instead.
    plateaued = loss_trend(v2, dict(zip(labels, [1.00, 1.001, 1.000])))
    assert decision(v2, {"state": "saturated", "why": "test"}, plateaued)["table_cell"] == \
        "plateaued__not_improving"


def test_v2_declares_the_action_separately_from_the_classification(v2):
    rule = v2["action_rule"]
    assert "still_improving" in rule["add_training_amount"]
    assert "saturated" in rule["refuse_training_amount"]
    assert "significantly_degraded" in rule["refuse_training_amount"]
    assert "upper bound is at or below zero" in rule["refuse_training_amount"]
    assert rule["undetermined_action"].startswith("only when the interval covers zero")


def test_v2_cannot_be_applied_to_an_earlier_analysis(v2, main, tmp_path):
    record = verify_registration(v2, main, V2_PREREG)
    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps({
        "created_at": "2026-10-27T14:03:55.947553+00:00",
        "coverage": {"periods": v2["windows"]["names"], "days": v2["windows"]["days_expected"]},
        "pairs": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Refused, match="before the registration"):
        apply(v2, record, earlier, None, None, None, V2_PREREG)
    assert v2["not_retroactive"] is True
    assert any("t5" in item for item in v2["may_not_be_used_to"])


def test_v2_registers_its_own_record_beside_it(v2, main):
    record = verify_registration(v2, main, V2_PREREG)
    assert record["prereg_path"] == "configs/b4_budget_decision_prereg_v2_20261027.json"
    assert record["thresholds_modified"] is False
    assert record["prereg_sha256"] != json.loads(
        (ROOT / "configs/b4_budget_decision_prereg_20261027.registration.json")
        .read_text(encoding="utf-8"))["prereg_sha256"]
