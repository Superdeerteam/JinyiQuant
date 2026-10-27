"""Sealed final-check extension: the template is landable and refuses the four peeks.

The template inherits the measured threshold basis from the main registration and
changes only the periods and the repeat condition. Nothing here reads the sealed
interval; the validator reads the run index's determination fields only.
"""

from __future__ import annotations

import json

import pytest

from scripts.promotion_gate import Refused, result_label, validate_sealed_scope
from scripts.sealed_final_prereg import MAIN_PREREG, TEMPLATE, template_document


def test_sealed_extension_template_is_landable_and_refuses_the_four_failure_modes():
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    main = json.loads(MAIN_PREREG.read_text(encoding="utf-8"))

    # The committed template is byte-identical to the builder, so it can be landed directly.
    assert template == template_document()
    scope = validate_sealed_scope(template, require_registered=False)
    assert scope["periods"] == [["2024-10-08", "2024-12-31"]]
    assert scope["sealed_interval"] == ["2024-10-08", "2025-01-01"]
    assert scope["minimum_periods"] == 1
    assert scope["index_guard"]["intersects_sealed_true"] == 0
    assert template["periods"]["no_more_months_requirement"] is True
    assert template["evidence_rules"]["minimum_periods"] == 1
    assert template["verdict"]["no_threshold_relaxation"] is True
    assert template["verdict"]["labels"] == {"accept": "通过", "reject": "未通过",
                                             "inconclusive": "证据不足"}
    assert set(template["verdict"]["conclusion_templates"]) == {"accept", "reject", "inconclusive"}
    # The gate prints the sealed wording, so the three states cannot drift into a fourth.
    assert [result_label(template, status) for status in ("accept", "reject", "inconclusive")] == \
        ["通过", "未通过", "证据不足"]
    assert [result_label(main, status) for status in ("accept", "reject", "inconclusive")] == \
        ["保留", "回退", "证据不足"]
    # Same calibers as the main registration: only periods and the repeat condition differ.
    assert template["object_policy_template"] == main["object_policy_template"]
    assert template["net5_limit"] == main["net5_limit"]
    assert template["derived_thresholds"] == main["derived_thresholds"]
    changed = {key for key in main["evidence_rules"] if template["evidence_rules"].get(key)
               != main["evidence_rules"][key]}
    assert changed == {"minimum_periods"}

    def variant(**changes) -> dict:
        document = json.loads(json.dumps(template))
        for path, value in changes.items():
            target = document
            *parents, leaf = path.split(".")
            for parent in parents:
                target = target[parent]
            target[leaf] = value
        return document

    # (1) splitting the sealed window into several periods is a repeated peek.
    with pytest.raises(Refused, match="exactly one period"):
        validate_sealed_scope(variant(**{"periods.declared": [["2024-10-08", "2024-11-15"],
                                                             ["2024-11-18", "2024-12-31"]]}),
                              require_registered=False)
    # (2) a period inside an already-declared research window is refused.
    with pytest.raises(Refused, match="overlaps the declared research period"):
        validate_sealed_scope(variant(**{"periods.declared": [["2024-06-03", "2024-06-28"]]}),
                              require_registered=False)
    # (3) a candidate that started before the sealed registration is refused.
    registered = variant(registered_at="2026-10-27T09:00:00+00:00")
    assert validate_sealed_scope(registered, require_registered=True,
                                 candidate_started_at="2026-10-27T09:30:00+00:00")["registered_at"]
    with pytest.raises(Refused, match="predates the sealed registration"):
        validate_sealed_scope(registered, require_registered=True,
                             candidate_started_at="2026-10-27T08:30:00+00:00")
    with pytest.raises(Refused, match="still a template"):
        validate_sealed_scope(template, require_registered=True)
    # (4) relaxing the improvement threshold, a secondary line or the 5bp line is refused.
    with pytest.raises(Refused, match="changes 'min_improvement'"):
        validate_sealed_scope(variant(**{"object_policy_template.min_improvement": 0.002}),
                              require_registered=False)
    with pytest.raises(Refused, match="changes 'max_metric_regressions'"):
        validate_sealed_scope(variant(**{"object_policy_template.max_metric_regressions.turnover_mean":
                                         0.5}), require_registered=False)
    with pytest.raises(Refused, match="5bp net regression line"):
        validate_sealed_scope(variant(net5_limit=0.5), require_registered=False)
