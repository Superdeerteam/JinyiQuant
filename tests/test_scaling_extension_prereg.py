"""The 927-day extension must stay a pre-registration, not a choice made after the numbers.

Three things make that true and are pinned here: the extension plans differ from the registered
two-point plans only by the appended point, the pre-registration states the trigger verbatim, and
the driver's resume guard accepts exactly the continuation the plans describe. The guard is checked
offline - the derived steps come from the measured usable-sample count, no training runs.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from jinyiquant.trajectory import derive_budget_steps  # noqa: E402
from scripts import scaling_extension_prereg as prereg  # noqa: E402
from scripts.run_training_trajectory import resume_position  # noqa: E402

SEEDS = (11, 29, 47)


def read(name: str) -> dict:
    return json.loads((prereg.CONFIGS / name).read_text(encoding="utf-8"))


def test_the_extension_plans_only_append_a_point():
    for seed in SEEDS:
        two, three = read(prereg.base_plan(seed)), read(prereg.extension_plan(seed))
        assert three["budget_points"][:2] == two["budget_points"]
        assert three["budget_points"][2]["effective_epochs"] == 3.96
        assert three["budget_points"][2]["name"] == "scaling-3.96"
        differing = [key for key in two if two[key] != three[key]]
        assert differing == ["created_at_utc", "budget_points"]
        assert three["recipe_sha256"] == two["recipe_sha256"]
        assert three["recipe_sha256"] == prereg.sha(prereg.REPO / three["recipe_config"])


def test_the_registration_pins_the_prereg_and_the_plans():
    registration = read(prereg.REGISTRATION_NAME)
    document = read(prereg.PREREG_NAME)
    assert registration["prereg_sha256"] == prereg.sha(prereg.CONFIGS / prereg.PREREG_NAME)
    assert registration["plan_sha256"] == {rel: prereg.sha(prereg.REPO / rel)
                                           for rel in document["extension_plans"]}
    assert registration["recipe_sha256"] == {read(prereg.extension_plan(seed))["recipe_config"]:
                                             read(prereg.extension_plan(seed))["recipe_sha256"]
                                             for seed in SEEDS}
    assert registration["ordering"]["registered_before_the_extension_run_started"] is True
    assert registration["ordering"]["registered_before_any_927_score_was_read"] is True
    assert registration["code"]["driver_sha256"] == prereg.sha_lf(
        prereg.REPO / "scripts" / "run_training_trajectory.py")


def test_the_prereg_states_the_trigger_without_softening_it():
    document = read(prereg.PREREG_NAME)
    assert document["extension_rule"] == prereg.EXTENSION_RULE
    assert "当且仅当" in document["extension_rule"]
    assert "区间不含零" in document["extension_rule"]
    assert document["no_threshold_change"] == "晋升阈值一个字都不改。"
    assert document["homogeneity"] == "一旦触发，三个种子全部续到 3.96，不允许混合长度的 blend。"
    assert document["cost_cap"] == "3 种子 × 305,405 步增量，wall 上限 3 h。"
    assert len(document["extension_plans"]) == len(SEEDS) == 3


def test_the_prefix_guard_accepts_the_registered_head_and_rejects_anything_else():
    guard = prereg.prefix_guard()
    for seed in SEEDS:
        entry = guard["seeds"][str(seed)]
        assert entry["registered_steps"] == [305_227, 610_453]
        assert entry["extension_steps"] == [305_227, 610_453, 915_680]
        assert entry["increment"] == 305_227
        assert entry["prefix_accepted"] is True
        assert entry["shifted_point_rejected"] is True
        assert entry["finished_plan_rejected"] is True


def test_the_guard_refuses_a_source_that_trained_a_point_wrongly():
    """A source whose 1.32 point used a different step count cannot continue this plan."""
    plan = read(prereg.extension_plan(11))
    budget = derive_budget_steps(plan["budget_points"], prereg.MEASURED_USABLE, prereg.BATCH_SIZE)
    source = {"executed_steps": [305_226, 610_453], "config_sha256": None, "plan_sha256": None}
    with pytest.raises(ValueError, match="not the head of this plan's points"):
        resume_position(plan, budget, {"position": {"steps": 610_453}}, source)
