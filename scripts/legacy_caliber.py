"""Freeze the pre-eval-v2 evaluation caliber and re-label its conclusions as legacy.

This module records, without evaluating anything or opening data:

* the 16-arm percentile panel (9 frozen checkpoints + 7 pre-declared equal-weight
  combinations) that ``scripts/report_platform_metrics.py`` scores against;
* the promotion conditions derived from that panel and from the old guardrails;
* every accept/reject/promotion conclusion produced under the old caliber, with
  its source record, run directories and verbatim original metrics;
* the code paths where the caliber is now disabled.

Every conclusion and condition is forced to ``usable_for_promotion = false`` with
the mark ``旧口径结论——不得用于晋升``. ``python scripts/legacy_caliber.py`` writes
``runs/legacy_caliber.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
CALIBER_ID = "legacy-pre-eval-v2"
LEGACY_MARK = "旧口径结论——不得用于晋升"
FINAL_INTERVAL = ["2024-10-08", "2025-01-01"]
COMPONENTS = ["ic", "ic_ir", "sharpe", "stress", "turnover"]
TARGET = 0.85
FROZEN_INDEX = "runs/frozen/a-20261026/sha256_index.json"
RUN_MARKERS = ("manifest.json", "metrics.json", "decision.json", "spec.json", "summary_metrics.json")

CHECKPOINTS = {
    "tlob": ["tlob_s11", "tlob_s29", "tlob_s47"],
    "mlp": ["mlp_s11", "mlp_s29", "mlp_s47"],
    "patch": ["patch_s11", "patch_s29", "patch_s47"],
}
COMBINATIONS = (
    ("tlob_3seed", {"tlob_s11": 1 / 3, "tlob_s29": 1 / 3, "tlob_s47": 1 / 3}),
    ("mlp_3seed", {"mlp_s11": 1 / 3, "mlp_s29": 1 / 3, "mlp_s47": 1 / 3}),
    ("patch_3seed", {"patch_s11": 1 / 3, "patch_s29": 1 / 3, "patch_s47": 1 / 3}),
    ("blend_tlob_mlp_50_50", {**{name: 1 / 6 for name in CHECKPOINTS["tlob"]},
                              **{name: 1 / 6 for name in CHECKPOINTS["mlp"]}}),
    ("blend_tlob_patch_50_50", {**{name: 1 / 6 for name in CHECKPOINTS["tlob"]},
                                **{name: 1 / 6 for name in CHECKPOINTS["patch"]}}),
    ("blend_mlp_patch_50_50", {**{name: 1 / 6 for name in CHECKPOINTS["mlp"]},
                               **{name: 1 / 6 for name in CHECKPOINTS["patch"]}}),
    ("blend_all_equal", {name: 1 / 9 for names in CHECKPOINTS.values() for name in names}),
)

PROMOTION_CONDITIONS = (
    {
        "id": "round5-composite-min",
        "source": "configs/round5_official_five_20261026_plan.json#promotion_rule.composite_min",
        "condition": "equal-weight mean of the five component percentiles against the fixed 15-arm "
                     "reference panel must be >= 0.85",
        "derived_from": "16-arm percentile panel (target already used by the baseline at 0.813)",
    },
    {
        "id": "round5-composite-interval",
        "source": "configs/round5_official_five_20261026_plan.json#promotion_rule.composite_interval",
        "condition": "the 5-day moving-block 95% interval of the composite difference excludes zero",
        "derived_from": "16-arm percentile panel",
    },
    {
        "id": "round5-robustness",
        "source": "configs/round5_official_five_20261026_plan.json#promotion_rule.robustness",
        "condition": "IC_IR improves in both screen months and both verify months",
        "derived_from": "16-arm percentile panel",
    },
    {
        "id": "round5-guardrails",
        "source": "configs/round5_official_five_20261026_plan.json#promotion_rule.guardrails",
        "condition": "ic >= 0.05991, ic_ir >= 2.680903, sharpe >= 38.993601, stress >= 0.052985, "
                     "turnover <= 1.111947",
        "derived_from": "baseline five values of the nine-model equal blend",
    },
    {
        "id": "grouped-min-rank-ic-gain",
        "source": "configs/grouped_ranking_20261025_plan.json#guardrail.min_rank_ic_gain "
                  "(carried into configs/mlplob_v3_60d_csz_20261026_plan.json and "
                  "configs/dual_patch_v3_60d_csz_20261026_plan.json)",
        "condition": "paired daily RankIC gain >= 0.0023",
        "derived_from": "legacy seed-dispersion floor used as a promotion threshold",
    },
    {
        "id": "grouped-turnover-max",
        "source": "configs/grouped_ranking_20261025_plan.json#guardrail.turnover_max",
        "condition": "ensemble mean turnover <= 1.3211",
        "derived_from": "60d/9000 baseline seed mean plus one cross-seed sd",
    },
    {
        "id": "grouped-interval-above-zero",
        "source": "scripts/compare_grouped_arms.py#compare",
        "condition": "the 5-day moving-block 95% interval of the RankIC delta has a lower bound > 0",
        "derived_from": "legacy bootstrap guardrail",
    },
    {
        "id": "grouped-decision",
        "source": "configs/grouped_ranking_20261025_plan.json#decision",
        "condition": "promote only when the gain, interval and turnover checks all pass; otherwise "
                     "retain the previous baseline",
        "derived_from": "combination of the three guardrails above",
    },
    {
        "id": "epoch-share-floor",
        "source": "configs/epoch_share_selection_20261025_plan.json#guardrail",
        "condition": "schedule gain >= 0.0046 paired daily RankIC over the 60d/9000 baseline with a "
                     "95% interval above zero and turnover <= 1.3211",
        "derived_from": "legacy schedule-selection gate that promoted the V3 baseline",
    },
    {
        "id": "old-decision-floor",
        "source": "runs/*/decision.json (predeclared minimum_mean_gain gate)",
        "condition": "seed-mean paired RankIC delta >= the predeclared floor 0.01164665094206816 plus "
                     "both-window, gross, turnover and worst-day gates",
        "derived_from": "previous accept/reject protocol; records now re-labelled as legacy",
    },
)


def panel_16() -> dict:
    """Return the frozen 16-arm panel definition used by the legacy proxy."""
    arms = [name for names in CHECKPOINTS.values() for name in names]
    arms.extend(name for name, _ in COMBINATIONS)
    return {
        "definition": "9 frozen checkpoints (TLOB/MLPLOB/dual_patch x seeds 11/29/47) + 7 pre-declared "
                      "equal-weight combinations",
        "n_arms": len(arms),
        "frozen_checkpoints": {"count": sum(len(names) for names in CHECKPOINTS.values()),
                               "arms": [name for names in CHECKPOINTS.values() for name in names]},
        "pre_declared_combinations": {
            "count": len(COMBINATIONS),
            "arms": [{"arm": name, "weights": weights} for name, weights in COMBINATIONS],
        },
        "arms": sorted(arms),
        "components": COMPONENTS,
        "percentile_formula": "(worse + 0.5*ties) / (n-1) inside the panel; higher is better except "
                              "turnover, which is reversed",
        "composite": "equal-weight mean of the five component percentiles",
        "target": TARGET,
        "target_note": "0.85 is the official competition target; the local panel composite is an "
                       "approximation of the unobserved platform score, not the platform score itself",
        "source": "scripts/report_platform_metrics.py",
        "evidence_windows": ["2023-11", "2024-01", "2024-03", "2024-04", "2024-06"],
        "final_interval": FINAL_INTERVAL,
        "final_interval_opened": False,
    }


def promotion_conditions() -> list[dict]:
    """Return every legacy promotion condition, all disabled for promotion use."""
    rows = []
    for raw in PROMOTION_CONDITIONS:
        row = dict(raw)
        row["status"] = "disabled"
        row["legacy_status"] = LEGACY_MARK
        row["usable_for_promotion"] = False
        row["usable_for_elimination"] = False
        rows.append(row)
    return rows


def mark_legacy(entry: dict) -> dict:
    """Force a conclusion entry into the legacy caliber (flags cannot be re-enabled)."""
    marked = dict(entry)
    marked["legacy_status"] = LEGACY_MARK
    marked["usable_for_promotion"] = False
    marked["usable_for_elimination"] = False
    marked["replaced_by"] = "eval-v2"
    return marked


CONCLUSIONS = (
    {
        "id": "round5-percentile-protocol",
        "experiment_id": "round5-official-five-20261026",
        "record": "configs/round5_official_five_20261026_plan.json",
        "metric_record": "modified_tlob/round5_platform_five_20261026.json",
        "sections": ["status", "promotion_rule", "decision", "local_approximation.today_proxy",
                     "access_policy"],
        "metric_paths": ["target_score", "baseline", "proxy", "windows", "evidence_boundary"],
        "run_dirs": ["runs/multiscale-replay-20261026", "runs/multiscale-replay-verify-20261026",
                     "runs/submission-best-20261026/build"],
        "result": "protocol_pending_candidate",
    },
    {
        "id": "model-v3-baseline",
        "experiment_id": "model-v3-20261025",
        "record": "modified_tlob/model_v3_20261025.json",
        "sections": ["decision", "evidence_boundary"],
        "metric_paths": ["selection", "ensembles_validation", "replay.window_ensembles", "matrix"],
        "run_dirs": ["runs/v2-60d-csz-19800", "runs/v2-60d-csz-19800-seed29",
                     "runs/v2-60d-csz-19800-seed47", "runs/model-versions/model-v3-20261025",
                     "runs/multi-window-final-20261025"],
        "result": "accepted_as_baseline",
    },
    {
        "id": "mlplob-v3",
        "experiment_id": "mlplob-v3-60d-csz-20261026",
        "record": "modified_tlob/mlplob_v3_20261026.json",
        "sections": ["decision", "conclusion"],
        "metric_paths": ["arms", "pairs", "submission"],
        "runs_field": "runs",
        "run_dirs": [],
        "result": "not_promoted",
    },
    {
        "id": "dual-patch-v3",
        "experiment_id": "dual-patch-v3-60d-csz-20261026",
        "record": "modified_tlob/dual_patch_v3_20261026.json",
        "sections": ["decision"],
        "metric_paths": ["arms", "pairs"],
        "run_dirs": ["runs/dual-patch-v3-60d-csz-19800", "runs/dual-patch-v3-60d-csz-19800-seed29",
                     "runs/dual-patch-v3-60d-csz-19800-seed47"],
        "result": "not_promoted",
    },
    {
        "id": "dual-patch-multiscale",
        "experiment_id": "dual-patch-multiscale-20261026",
        "record": "modified_tlob/dual_patch_multiscale_20261026.json",
        "sections": ["decision", "evidence_boundary"],
        "metric_paths": ["training.ensemble_validation", "training.ensemble_pair",
                         "comparison.validation_10d", "comparison.screen_3m.raw",
                         "comparison.screen_3m.neutralized", "comparison.screen_3m.paired",
                         "comparison.verify_2m.raw", "comparison.verify_2m.neutralized",
                         "comparison.verify_2m.paired"],
        "runs_field": "artifacts.runs",
        "run_dirs": [],
        "result": "not_promoted",
    },
    {
        "id": "grouped-ranking",
        "experiment_id": "grouped-ranking-20261025",
        "record": "modified_tlob/grouped_ranking_20261025.json",
        "sections": ["decision", "design"],
        "metric_paths": ["arms", "pairs"],
        "plan_runs": "configs/grouped_ranking_20261025_plan.json",
        "run_dirs": [],
        "result": "no_arm_promoted",
    },
    {
        "id": "tlob-mlplob-blend",
        "experiment_id": "tlob-mlplob-fixed-blend-20261026",
        "record": "modified_tlob/tlob_mlplob_blend_20261026.json",
        "sections": ["decision", "conclusion", "evidence_boundary"],
        "metric_paths": ["arms", "pairs", "blend", "submission", "inference_cost_local"],
        "runs_field": "runs",
        "run_dirs": [],
        "result": "not_promoted",
    },
    {
        "id": "model-versions-20261023",
        "experiment_id": "model-versions-20261023",
        "record": "modified_tlob/model_versions_20261023.json",
        "sections": ["policy", "archived_multi_seed_evidence", "limits", "reproduction"],
        "metric_paths": ["versions", "comparison", "style_adjusted.windows"],
        "run_dirs": ["runs/model-versions/v0", "runs/model-versions/v1",
                     "runs/fixed-seed-ensemble-20261017"],
        "result": "not_promoted",
    },
    {
        "id": "model-versions-multi-seed-20261023",
        "experiment_id": "model-versions-multi-seed-20261023",
        "record": "modified_tlob/model_versions_multi_seed_20261023.json",
        "sections": ["gates", "scope", "limits", "sources"],
        "metric_paths": ["cells", "intervals", "pooled_seed_mean_delta", "version_pair", "windows",
                         "worst_day", "identity_checks"],
        "run_dirs": ["runs/model-versions/multi-seed-20261023",
                     "runs/label-cszscore-20261018/20261018T104425.466761Z_1b8039d5f4e1",
                     "runs/fixed-seed-ensemble-20261017/20261017T064944.073834Z_193b03ead7ff"],
        "result": "not_promoted",
    },
    {
        "id": "model-v1.1-july",
        "experiment_id": "model-v1.1-july-experiment-20261023",
        "record": "modified_tlob/model_v1_1_july_20261023.json",
        "sections": ["decision", "evidence_boundary"],
        "metric_paths": ["comparison", "v1_1_run", "baseline_v2", "supplement_v0_local_oct"],
        "run_dirs": ["runs/model-versions/v1.1-july", "runs/model-v1.1-replay-20261023",
                     "runs/model-versions/v2-cloud-20261023"],
        "result": "not_promoted",
    },
    {
        "id": "label-centering-20261018",
        "experiment_id": "label-centering-20261018",
        "record": "modified_tlob/label_centering_20261018.json",
        "sections": ["conclusion", "candidate_status", "retained_version", "gates", "limitations"],
        "decision_records": ["runs/label-centering-20261018/*/decision.json"],
        "decision_paths": ["result", "action", "model_promoted", "mean_rank_ic_delta",
                           "minimum_mean_gain", "checks", "period_mean_deltas", "interpretation"],
        "run_dirs": ["runs/label-centering-20261018"],
        "result": "reject",
    },
    {
        "id": "label-cszscore-20261018",
        "experiment_id": "label-cszscore-20261018",
        "record": "modified_tlob/label_cszscore_20261018.json",
        "sections": ["conclusion", "candidate_status", "retained_version", "gates", "limitations"],
        "decision_records": ["runs/label-cszscore-20261018/*/decision.json"],
        "decision_paths": ["result", "action", "model_promoted", "mean_rank_ic_delta",
                           "minimum_mean_gain", "checks", "period_mean_deltas", "interpretation"],
        "run_dirs": ["runs/label-cszscore-20261018"],
        "result": "reject",
    },
    {
        "id": "label-contrast-20261024",
        "experiment_id": "label-contrast-20261024",
        "record": "modified_tlob/label_contrast_20261024.json",
        "sections": ["decision.framework", "final_consolidation.decision",
                     "final_consolidation.recommendation", "limitations"],
        "metric_paths": ["decision.candidates", "final_consolidation.paired_summary"],
        "run_dirs": ["runs/label-contrast-20261024", "runs/label-contrast-eval-20261024"],
        "result": "no_promotion",
    },
    {
        "id": "coverage-date-weighting-20261018",
        "experiment_id": "coverage-date-weighting-20261018",
        "record": "modified_tlob/coverage_date_weighting_20261018.json",
        "sections": ["conclusion", "candidate_status", "retained_version", "gates", "limitations"],
        "decision_records": ["runs/coverage-date-weighting-20261018/*/decision.json"],
        "decision_paths": ["result", "action", "model_promoted", "mean_rank_ic_delta",
                           "minimum_mean_gain", "checks", "period_mean_deltas", "interpretation"],
        "run_dirs": ["runs/coverage-date-weighting-20261018"],
        "result": "reject",
    },
    {
        "id": "coverage-step-scaling-20261018",
        "experiment_id": "coverage-step-scaling-20261018",
        "record": "modified_tlob/coverage_step_scaling_20261018.json",
        "sections": ["conclusion", "candidate_status", "retained_version", "gates", "limitations"],
        "decision_records": ["runs/coverage-step-scaling-20261018/*/decision.json"],
        "decision_paths": ["result", "action", "model_promoted", "mean_rank_ic_delta",
                           "minimum_mean_gain", "checks", "period_mean_deltas", "interpretation"],
        "run_dirs": ["runs/coverage-step-scaling-20261018"],
        "result": "reject",
    },
    {
        "id": "training-coverage-arms-20261018",
        "experiment_id": "training-coverage-arms-20261018",
        "record": "modified_tlob/training_coverage_arms_20261018.json",
        "sections": ["conclusion", "candidate_status", "retained_version", "limitations"],
        "decision_records": ["runs/training-coverage-arms-20261018/*/decision.json"],
        "decision_paths": ["result", "action", "model_promoted", "mean_rank_ic_delta",
                           "minimum_mean_gain", "checks", "period_mean_deltas", "interpretation"],
        "run_dirs": ["runs/training-coverage-arms-20261018"],
        "result": "reject",
    },
    {
        "id": "fixed-seed-ensemble-20261017",
        "experiment_id": "fixed-seed-ensemble-20261017",
        "record": "modified_tlob/fixed_seed_ensemble_20261017.json",
        "sections": ["decision", "policy", "limitations", "validation"],
        "metric_paths": ["statistics", "inference_cost", "paired_intervals", "windows", "runs"],
        "run_dirs": ["runs/fixed-seed-ensemble-20261017"],
        "result": "not_promoted",
    },
    {
        "id": "monthly-coverage-20261025",
        "experiment_id": "monthly-coverage-vs-volume-20261025",
        "record": "modified_tlob/monthly_coverage_20261025.json",
        "sections": ["decision", "evidence_boundary"],
        "metric_paths": ["arms", "paired_comparison", "seed_noise"],
        "run_dirs": ["runs/monthly-132d-csz", "runs/v2-60d-csz-19800"],
        "result": "not_promoted",
    },
    {
        "id": "training-duration-20261015",
        "experiment_id": "training-duration-20261015",
        "record": "modified_tlob/training_duration_20261015.json",
        "sections": ["conclusion_zh", "retained_budget", "screening", "replication", "limitations"],
        "metric_paths": ["means_by_budget", "terminal_metrics", "evidence"],
        "decision_records": ["runs/training-duration-20261015/screen-analysis/decision.json"],
        "decision_paths": ["result", "accepted", "action", "model_promoted"],
        "run_dirs": ["runs/training-duration-20261015"],
        "result": "not_promoted",
    },
    {
        "id": "training-progression-20261015",
        "experiment_id": "training-progression-20261015",
        "record": "modified_tlob/training_progression_20261015.json",
        "sections": ["decisions", "final_validation", "limitations"],
        "metric_paths": ["step_table"],
        "decision_records": ["runs/training-progression/step*-decision.json"],
        "decision_paths": ["result", "accepted", "action", "model_promoted"],
        "run_dirs": ["runs/training-progression"],
        "result": "not_promoted",
    },
)

DISABLED_CODE_PATHS = (
    {
        "path": "scripts/report_platform_metrics.py",
        "surface": "16-arm percentile score / five-component composite",
        "status": "descriptive_only",
        "behavior": "composite() rejects any non-descriptive use; promotion_input_guard() raises "
                    "LegacyCaliberError for promotion/elimination requests; the written record sets "
                    "promotion_allowed=false",
        "test": "tests/test_platform_metrics.py::test_percentile_composite_refuses_promotion_input",
    },
    {
        "path": "scripts/compare_grouped_arms.py",
        "surface": "guardrail pass / plan decision",
        "status": "disabled",
        "behavior": "legacy_guardrail() marks every gate usable_for_promotion=false; the record sets "
                    "promotion_allowed=false; promotion_decision() raises PromotionDisabledError",
        "test": "tests/test_platform_metrics.py::test_grouped_arm_guardrail_cannot_promote",
    },
    {
        "path": "scripts/select_epoch_share.py",
        "surface": "schedule-selection guardrail pass",
        "status": "disabled",
        "behavior": "legacy_guardrail() marks every gate usable_for_promotion=false; the record sets "
                    "promotion_allowed=false; promotion_decision() raises PromotionDisabledError",
        "test": "tests/test_platform_metrics.py::test_schedule_guardrail_cannot_promote",
    },
    {
        "path": "configs/grouped_ranking_20261025_plan.json",
        "surface": "guardrail and decision",
        "status": "disabled_annotation",
        "behavior": "legacy_caliber block marks the pre-registered guardrail/decision as legacy; values "
                    "are preserved verbatim as history",
        "test": "tests/test_platform_metrics.py::test_legacy_plan_annotations_are_disabled",
    },
    {
        "path": "configs/round5_official_five_20261026_plan.json",
        "surface": "promotion_rule (composite >= 0.85) and decision",
        "status": "disabled_annotation",
        "behavior": "legacy_caliber block marks the composite promotion rule as legacy; values are "
                    "preserved verbatim as history",
        "test": "tests/test_platform_metrics.py::test_legacy_plan_annotations_are_disabled",
    },
    {
        "path": "configs/epoch_share_selection_20261025_plan.json",
        "surface": "guardrail and decision rule",
        "status": "disabled_annotation",
        "behavior": "legacy_caliber block marks the schedule-selection guardrail as legacy",
        "test": "tests/test_platform_metrics.py::test_legacy_plan_annotations_are_disabled",
    },
    {
        "path": "configs/mlplob_v3_60d_csz_20261026_plan.json",
        "surface": "carried-over guardrail and decision",
        "status": "disabled_annotation",
        "behavior": "legacy_caliber block marks the carried-over 0.0023/1.3211 gate as legacy",
        "test": "tests/test_platform_metrics.py::test_legacy_plan_annotations_are_disabled",
    },
    {
        "path": "configs/dual_patch_v3_60d_csz_20261026_plan.json",
        "surface": "carried-over guardrail and decision",
        "status": "disabled_annotation",
        "behavior": "legacy_caliber block marks the carried-over 0.0023/1.3211 gate as legacy",
        "test": "tests/test_platform_metrics.py::test_legacy_plan_annotations_are_disabled",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(document, dotted: str):
    node = document
    for part in dotted.split("."):
        if isinstance(node, dict):
            node = node[part]
        elif isinstance(node, list):
            node = node[int(part)]
        else:
            raise KeyError(dotted)
    return node


def subset(document, paths: list[str]) -> dict:
    """Return ``{dotted_path: verbatim value}`` for the paths present in the document."""
    out: dict = {}
    for path in paths:
        try:
            out[path] = _lookup(document, path)
        except (KeyError, IndexError, TypeError, ValueError):
            out[path] = None
    return out


def _flatten_runs(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_flatten_runs(item))
        return rows
    if isinstance(value, dict):
        rows = []
        for item in value.values():
            rows.extend(_flatten_runs(item))
        return rows
    return []


def _is_run_dir(path: Path) -> bool:
    if any((path / name).exists() for name in RUN_MARKERS):
        return True
    children = [child for child in path.iterdir() if child.is_dir()]
    return bool([p for p in path.iterdir() if p.is_file()]) and not any(
        any((child / name).exists() for name in RUN_MARKERS) for child in children)


def run_dirs(declared: list[str], extra: list[str]) -> tuple[list[str], list[str]]:
    """Resolve declared paths/globs to existing run directories; report what is missing."""
    found: list[str] = []
    missing: list[str] = []
    for raw in [*declared, *extra]:
        candidates = [p for p in sorted(ROOT.glob(raw))] if "*" in raw else [ROOT / raw]
        if not candidates:
            missing.append(raw)
            continue
        for candidate in candidates:
            if not candidate.exists():
                missing.append(raw)
                continue
            if candidate.is_file():
                continue
            if _is_run_dir(candidate):
                found.append(candidate.relative_to(ROOT).as_posix())
                continue
            children = [child for child in sorted(candidate.iterdir())
                        if child.is_dir() and _is_run_dir(child)]
            if children:
                found.extend(child.relative_to(ROOT).as_posix() for child in children)
            else:
                missing.append(raw)
    ordered = list(dict.fromkeys(found))
    return ordered, sorted(set(missing))


def decision_inventory(exclude_prefix: str = "runs/frozen/") -> list[dict]:
    """Inventory every pre-existing decision.json accept/reject record."""
    rows = []
    for path in sorted(ROOT.glob("runs/**/decision.json")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(exclude_prefix):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        rows.append(mark_legacy({
            "path": relative,
            "sha256": _sha256(path),
            "result": document.get("result"),
            "accepted": document.get("accepted"),
            "action": document.get("action"),
            "model_promoted": document.get("model_promoted"),
            "category": ("engineering_fixture"
                         if any(token in relative for token in (
                             "acceptance/", "/suites/", "infrastructure-audit",
                             "single-factor-engineering", "sync-20261015", "f1-f4-validation"))
                         else "model_promotion"),
        }))
    return rows


def conclusion_entry(spec: dict) -> dict:
    """Build one verbatim, forced-legacy conclusion entry from a declaration."""
    record = ROOT / spec["record"]
    entry: dict = {
        "id": spec["id"],
        "category": "model_promotion",
        "experiment_id": spec["experiment_id"],
        "record": {"path": spec["record"], "sha256": _sha256(record) if record.exists() else None},
        "result": spec["result"],
    }
    if not record.exists():
        entry["missing_record"] = True
        return mark_legacy(entry)
    document = json.loads(record.read_text(encoding="utf-8"))
    metric_record = ROOT / spec.get("metric_record", spec["record"])
    metric_document = (json.loads(metric_record.read_text(encoding="utf-8"))
                       if metric_record.exists() else document)
    if "sections" in spec:
        entry["original_conclusion"] = subset(document, spec["sections"])
    if "metric_paths" in spec:
        entry["original_metrics"] = subset(metric_document, spec["metric_paths"])
    extra_runs: list[str] = []
    if spec.get("runs_field"):
        extra_runs = _flatten_runs(_lookup(document, spec["runs_field"]))
    if spec.get("plan_runs"):
        plan = json.loads((ROOT / spec["plan_runs"]).read_text(encoding="utf-8"))
        for arm in plan.get("arms", {}).values():
            extra_runs.extend(arm.get("runs", []))
    resolved, missing = run_dirs(spec.get("run_dirs", []), extra_runs)
    entry["run_dirs"] = resolved
    if missing:
        entry["missing_run_dirs"] = missing
    decision_rows = []
    missing_decisions = []
    for pattern in spec.get("decision_records", []):
        matched = sorted(ROOT.glob(pattern))
        if not matched:
            missing_decisions.append(pattern)
        for path in matched:
            document = json.loads(path.read_text(encoding="utf-8"))
            decision_rows.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "metrics": subset(document, spec.get("decision_paths", [])),
            })
    if decision_rows:
        entry["decision_records"] = decision_rows
    if missing_decisions:
        entry["missing_decision_records"] = missing_decisions
    return mark_legacy(entry)


def _frozen_snapshot() -> dict | None:
    path = ROOT / FROZEN_INDEX
    if not path.exists():
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    return {"path": FROZEN_INDEX, "sha256": _sha256(path), "totals": index.get("totals"),
            "sources": [row["source"] for row in index.get("sources", [])]}


def build_record(conclusions: list[dict] | None = None, created_at: str | None = None) -> dict:
    """Assemble the legacy-caliber document; every conclusion is forced to not-for-promotion."""
    entries = [mark_legacy(entry) for entry in (conclusions or [])]
    panel = panel_16()
    conditions = promotion_conditions()
    if panel["n_arms"] != 16:
        raise ValueError("legacy panel must contain exactly 16 arms")
    if any(entry.get("usable_for_promotion") for entry in entries):
        raise ValueError("a legacy conclusion cannot remain usable for promotion")
    missing_runs = {entry["id"]: entry["missing_run_dirs"]
                    for entry in entries if entry.get("missing_run_dirs")}
    missing_records = [entry["id"] for entry in entries if entry.get("missing_record")]
    missing_decisions = {entry["id"]: entry["missing_decision_records"]
                         for entry in entries if entry.get("missing_decision_records")}
    return {
        "schema_version": SCHEMA_VERSION,
        "caliber_id": CALIBER_ID,
        "created_at_utc": created_at or datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/legacy_caliber.py",
        "legacy_mark": LEGACY_MARK,
        "status": "frozen_legacy",
        "scope": "pre-eval-v2 evaluation and promotion caliber: the 16-arm percentile proxy, the "
                 "derived guardrails and every accept/reject/promotion conclusion recorded under them. "
                 "Engineering fixture acceptance suites are indexed only as archived decisions; they "
                 "are not model-promotion evidence.",
        "panel_16": panel,
        "promotion_conditions": conditions,
        "conclusions": entries,
        "archived_decision_records": decision_inventory(),
        "disabled_code_paths": [dict(row) for row in DISABLED_CODE_PATHS],
        "frozen_snapshot": _frozen_snapshot(),
        "unfrozen": {
            "missing_run_dirs": missing_runs,
            "missing_records": missing_records,
            "missing_decision_records": missing_decisions,
            "note": "these archives are absent from the workspace; their conclusions remain legacy and "
                    "cannot be reproduced from local run directories. Nothing is re-created or patched.",
        },
        "time_boundary": {
            "closed_interval": FINAL_INTERVAL,
            "opened": False,
            "note": "stage A does not read [2024-10-08, 2025-01-01); the legacy windows are all "
                    "already-used development data",
        },
        "replacement": {
            "contract": "eval-v2",
            "status": "pending",
            "note": "eval-v2 (competition-metric approximation plus cost=cU economics) is not final; "
                    "until it lands no legacy conclusion may authorise promotion or elimination",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/legacy_caliber.json")
    args = parser.parse_args()
    entries = [conclusion_entry(spec) for spec in CONCLUSIONS]
    record = build_record(entries)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    unfrozen = record["unfrozen"]
    print(f"wrote {output}")
    print(f"conclusions={len(record['conclusions'])} decisions={len(record['archived_decision_records'])} "
          f"conditions={len(record['promotion_conditions'])}")
    print(f"missing_records={unfrozen['missing_records']}")
    print(f"missing_run_dirs={unfrozen['missing_run_dirs']}")
    print(f"missing_decision_records={unfrozen['missing_decision_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
