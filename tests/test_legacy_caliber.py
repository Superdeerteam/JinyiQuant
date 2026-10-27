"""Behavior checks for the frozen legacy evaluation caliber."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_multiscale_replay import define_arms
from scripts.legacy_caliber import (
    CHECKPOINTS,
    LEGACY_MARK,
    build_record,
    mark_legacy,
    panel_16,
    promotion_conditions,
    subset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_panel_16_is_nine_checkpoints_plus_seven_combinations():
    panel = panel_16()
    assert panel["n_arms"] == 16
    assert panel["frozen_checkpoints"]["count"] == 9
    assert panel["pre_declared_combinations"]["count"] == 7
    assert len(set(panel["arms"])) == 16
    weights = {row["arm"]: row["weights"] for row in panel["pre_declared_combinations"]["arms"]}
    assert sum(weights["blend_all_equal"].values()) == pytest.approx(1.0)
    assert weights["tlob_3seed"]["tlob_s11"] == pytest.approx(1 / 3)
    assert panel["target"] == 0.85
    assert panel["final_interval"] == ["2024-10-08", "2025-01-01"]
    assert panel["final_interval_opened"] is False


def test_panel_arms_match_the_report_script_definition():
    singles = {name for names in CHECKPOINTS.values() for name in names}
    assert set(panel_16()["arms"]) == set(define_arms(dict(CHECKPOINTS))) | singles


def test_mark_legacy_cannot_be_re_enabled():
    marked = mark_legacy({"id": "x", "usable_for_promotion": True, "usable_for_elimination": True})
    assert marked["usable_for_promotion"] is False
    assert marked["usable_for_elimination"] is False
    assert marked["legacy_status"] == LEGACY_MARK
    assert marked["replaced_by"] == "eval-v2"


def test_every_promotion_condition_is_disabled():
    conditions = promotion_conditions()
    ids = {row["id"] for row in conditions}
    assert {"round5-composite-min", "grouped-min-rank-ic-gain", "old-decision-floor"} <= ids
    for row in conditions:
        assert row["status"] == "disabled"
        assert row["usable_for_promotion"] is False
        assert row["usable_for_elimination"] is False
        assert row["source"]
        assert row["legacy_status"] == LEGACY_MARK


def test_build_record_forces_the_legacy_mark_on_every_conclusion():
    record = build_record([{"id": "kept", "result": "reject"},
                           {"id": "sneaky", "result": "accept", "usable_for_promotion": True}])
    assert record["schema_version"] == 1
    assert record["caliber_id"] == "legacy-pre-eval-v2"
    assert len(record["conclusions"]) == 2
    for entry in record["conclusions"]:
        assert entry["usable_for_promotion"] is False
        assert entry["usable_for_elimination"] is False
        assert entry["legacy_status"] == LEGACY_MARK
    assert record["time_boundary"]["opened"] is False
    assert record["replacement"]["status"] == "pending"
    assert record["disabled_code_paths"]
    for row in record["disabled_code_paths"]:
        assert row["status"] in {"descriptive_only", "disabled", "disabled_annotation"}
        assert row["path"] and row["behavior"] and row["test"]


def test_subset_keeps_verbatim_values_and_marks_missing_paths():
    document = {"a": {"b": [1, 2]}, "c": 3}
    assert subset(document, ["a.b", "c", "missing"]) == {"a.b": [1, 2], "c": 3, "missing": None}


def test_legacy_record_artifact_when_present_holds_the_frozen_flags():
    path = ROOT / "runs/legacy_caliber.json"
    if not path.exists():
        pytest.skip("runs/legacy_caliber.json is a local frozen artifact")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["legacy_mark"] == LEGACY_MARK
    assert record["panel_16"]["n_arms"] == 16
    assert record["conclusions"]
    for entry in record["conclusions"]:
        assert entry["usable_for_promotion"] is False
        assert entry["legacy_status"] == LEGACY_MARK
    for row in record["promotion_conditions"]:
        assert row["status"] == "disabled"
    for entry in record["conclusions"]:
        for relative in entry.get("run_dirs", []):
            assert (ROOT / relative).is_dir(), relative
    for row in record["archived_decision_records"]:
        assert (ROOT / row["path"]).is_file(), row["path"]
    assert record["time_boundary"]["closed_interval"] == ["2024-10-08", "2025-01-01"]
