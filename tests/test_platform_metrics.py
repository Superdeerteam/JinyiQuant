"""Behavior checks for the platform-style five-metric report helpers.

The percentile composite is a frozen legacy-caliber artifact: these tests pin that
it is descriptive only and cannot be used as a promotion/elimination input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import select_epoch_share as schedule
from scripts.compare_grouped_arms import PromotionDisabledError, legacy_guardrail, promotion_decision
from scripts.report_platform_metrics import (
    COMPONENTS,
    LegacyCaliberError,
    composite,
    legacy_record_block,
    percentile,
    promotion_input_guard,
    stress_buckets,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_MARK = "旧口径结论——不得用于晋升"


def five_component_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ic": [0.05, 0.06, 0.07, 0.08],
        "ic_ir": [1.0, 2.0, 3.0, 4.0],
        "sharpe": [10.0, 20.0, 30.0, 40.0],
        "stress": [0.01, 0.02, 0.03, 0.04],
        "turnover": [1.4, 1.3, 1.2, 1.1],
    })


def test_percentile_direction_and_ties():
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(percentile(values, higher=True), [0.0, 0.5, 1.0])
    assert np.allclose(percentile(values, higher=False), [1.0, 0.5, 0.0])
    assert np.allclose(percentile(np.array([2.0, 2.0, 2.0]), higher=True), [0.5, 0.5, 0.5])


def test_composite_is_the_mean_percentile_and_gap_decomposes():
    scored = composite(five_component_frame(), use="description")
    assert np.allclose(scored["composite"], scored[[f"p_{name}" for name in COMPONENTS]].mean(axis=1))
    assert np.allclose(scored["gap_to_target"], 0.85 - scored["composite"])
    contributions = scored[[f"gap_contrib_{name}" for name in COMPONENTS]].sum(axis=1)
    assert np.allclose(contributions, scored["gap_to_target"])
    assert scored.loc[3, "p_turnover"] == 1.0
    assert scored.loc[0, "p_turnover"] == 0.0


def test_percentile_composite_refuses_promotion_input():
    for use in ("promotion", "elimination", "selection", "gate"):
        with pytest.raises(LegacyCaliberError):
            composite(five_component_frame(), use=use)
        with pytest.raises(LegacyCaliberError):
            promotion_input_guard(use)
    descriptive = promotion_input_guard("description")
    assert descriptive["promotion_allowed"] is False
    assert descriptive["legacy_mark"] == LEGACY_MARK
    record = legacy_record_block()
    assert record["promotion_allowed"] is False
    assert record["legacy_mark"] == LEGACY_MARK


def test_grouped_arm_guardrail_cannot_promote():
    left = {"metrics": {"turnover_mean": 1.1}}
    deltas = {"rank_ic": {"mean": 0.01, "interval": [0.001, 0.02]}}
    row = legacy_guardrail(left, deltas, {"min_rank_ic_gain": 0.0023, "turnover_max": 1.3211})
    assert row["pass"] is True
    assert row["usable_for_promotion"] is False
    assert row["usable_for_elimination"] is False
    assert row["legacy"] is True
    assert row["legacy_status"] == LEGACY_MARK
    with pytest.raises(PromotionDisabledError):
        promotion_decision(row)


def test_schedule_guardrail_cannot_promote():
    summary = {"mean": {"turnover_mean": 1.1}}
    diffs = {"rank_ic": {"mean": 0.01, "interval": [0.001, 0.02]}}
    row = schedule.legacy_guardrail(summary, diffs, {"min_rank_ic_gain": 0.0046, "turnover_max": 1.3211})
    assert row["pass"] is True
    assert row["usable_for_promotion"] is False
    assert row["usable_for_elimination"] is False
    assert row["legacy"] is True
    assert row["legacy_status"] == LEGACY_MARK
    with pytest.raises(schedule.PromotionDisabledError):
        schedule.promotion_decision(row)


def test_legacy_plan_annotations_are_disabled():
    cases = {
        "configs/grouped_ranking_20261025_plan.json":
            ("guardrail_usable_for_promotion",),
        "configs/round5_official_five_20261026_plan.json":
            ("promotion_rule_usable_for_promotion",),
        "configs/epoch_share_selection_20261025_plan.json":
            ("guardrail_usable_for_promotion",),
        "configs/mlplob_v3_60d_csz_20261026_plan.json":
            ("guardrail_usable_for_promotion",),
        "configs/dual_patch_v3_60d_csz_20261026_plan.json":
            ("guardrail_usable_for_promotion",),
    }
    for relative, keys in cases.items():
        document = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        block = document["legacy_caliber"]
        assert block["status"] == "disabled"
        assert block["legacy_mark"] == LEGACY_MARK
        assert block["decision_usable_for_promotion"] is False
        for key in keys:
            assert block[key] is False
        assert block["replaced_by"] == "eval-v2"


def test_stress_buckets_cover_every_scored_day():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "rank_ic": [0.01, 0.02, 0.03, 0.04],
    })
    conditions = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "vol_ratio": [0.5, 1.0, 1.5, 0.9],
        "amount_ratio": [0.8, 1.2, 0.7, 1.1],
    })
    buckets = stress_buckets(daily, conditions)
    assert buckets["high_vol"]["days"] + buckets["low_vol"]["days"] == len(daily)
    assert buckets["high_amount"]["days"] + buckets["low_amount"]["days"] == len(daily)
    assert buckets["high_vol"]["days"] == 2
    assert buckets["high_amount"]["days"] == 2
    with pytest.raises(ValueError):
        stress_buckets(daily, conditions.iloc[:-1])
