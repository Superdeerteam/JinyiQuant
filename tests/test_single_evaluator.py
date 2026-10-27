"""Single-evaluator wiring: import/call sites, cross-entry equality, no local formulas.

The eval-v2 contract in ``src/jinyiquant/evaluation.py`` is frozen by t2/t5. These
tests pin that every production entry that reports metrics reaches that one
implementation, that the same predictions produce identical metrics through
different entries, and that no local IC/Sharpe/turnover/cost formula remains.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jinyiquant import runner
from jinyiquant.comparison import archived_cost_scenarios, declared_cost_scenarios
from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import EVALUATION_VERSION, evaluate
from scripts import compare_multiscale_replay as multiscale
from scripts import report_platform_metrics as platform

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SHA256 = "0786b9d8238f0dc4dab9ffd062cdd84368f6bbc50c40bd05f80da3d025ad4560"
CONFIG = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
          "constant_portfolios": "undefined", "single_side_rate": 0.0005}

# file -> (names that must be imported from jinyiquant.evaluation,
#          contract functions that must be called directly in that file)
WIRED = {
    "src/jinyiquant/runner.py": ({"evaluate", "EVALUATION_VERSION"}, {"evaluate"}),
    "src/jinyiquant/baselines.py": ({"EVALUATION_VERSION"}, set()),
    "src/jinyiquant/comparison.py": ({"evaluate", "aggregate_metric", "EVALUATION_VERSION"},
                                     {"evaluate", "aggregate_metric"}),
    "modified_tlob/diagnostics.py": ({"evaluate", "EVALUATION_VERSION"}, {"evaluate"}),
    "modified_tlob/infer.py": ({"EVALUATION_VERSION"}, set()),
    "scripts/report_platform_metrics.py": ({"aggregate_metric", "EVALUATION_VERSION"},
                                           {"aggregate_metric", "declared_cost_scenarios"}),
    "scripts/report_model_versions_multi_seed.py": ({"evaluate", "EVALUATION_VERSION"},
                                                    {"evaluate", "declared_cost_scenarios"}),
    "scripts/compare_multiscale_replay.py": ({"evaluate", "EVALUATION_VERSION"},
                                             {"evaluate", "declared_cost_scenarios"}),
    "scripts/compare_blend_replay.py": ({"evaluate", "EVALUATION_VERSION"},
                                        {"evaluate", "declared_cost_scenarios"}),
    "scripts/sweep_blend_ratios.py": ({"EVALUATION_VERSION"}, {"declared_cost_scenarios"}),
    "scripts/compare_coverage_volume.py": ({"evaluate", "EVALUATION_VERSION"},
                                           {"evaluate", "archived_cost_scenarios"}),
    "scripts/compare_grouped_arms.py": ({"evaluate", "EVALUATION_VERSION"},
                                        {"evaluate", "declared_cost_scenarios"}),
    "scripts/compare_recipe_reference.py": ({"evaluate", "EVALUATION_VERSION"},
                                            {"archived_cost_scenarios"}),
    "scripts/compare_signal_smoothing.py": ({"evaluate", "EVALUATION_VERSION"},
                                            {"evaluate", "archived_cost_scenarios"}),
    "scripts/compare_training_round.py": ({"evaluate", "EVALUATION_VERSION"},
                                          {"evaluate", "archived_cost_scenarios"}),
    "scripts/audit_seed_ensemble.py": ({"evaluate", "EVALUATION_VERSION"},
                                       {"evaluate", "declared_cost_scenarios"}),
    "scripts/analyze_annual_baseline.py": ({"evaluate", "EVALUATION_VERSION"}, {"evaluate"}),
    "scripts/analyze_training_duration.py": ({"EVALUATION_VERSION"}, set()),
    "scripts/diagnose_training_coverage.py": ({"EVALUATION_VERSION"}, set()),
    "scripts/replay_platform_submission.py": ({"EVALUATION_VERSION"}, set()),
    "scripts/annual_baseline.py": ({"EVALUATION_VERSION"}, set()),
    "scripts/analyze_learning_diagnostics.py": ({"evaluate", "EVALUATION_VERSION"}, {"evaluate"}),
    "scripts/replay_model_versions.py":
        ({"evaluate", "EVALUATION_VERSION", "aggregate_metric"}, {"evaluate", "aggregate_metric"}),
}

# Files whose metrics are produced elsewhere but that must still carry the version.
VERSION_ONLY = {
    "src/jinyiquant/diagnosis.py": "evaluation_version",
}

FORBIDDEN_FORMULAS = {
    r"np\.corrcoef\(": "local correlation",
    r"method\s*=\s*['\"]spearman": "local rank-IC formula",
    r"\bic\.mean\(\)\s*/\s*ic\.std\(": "local IC_IR formula",
    r"\bls\.mean\(\)\s*/\s*ls\.std\(": "local Sharpe formula",
    r"np\.sqrt\(252\)": "local Sharpe annualization",
    r"0\.5\s*\*\s*np\.abs|0\.5 \* [a-z_]*\.abs\(": "local turnover formula",
}
# Explicit, machine-readable exemptions from the duplicate-formula scan. They are
# data, not a silent ``continue``: every entry needs a reason and a live/disabled
# status, and only the frozen contract itself may be a live exemption.
FORMULA_EXEMPTIONS = (
    {
        "path": "src/jinyiquant/evaluation.py",
        "reason": "the frozen eval-v2 contract implementation itself",
        "status": "live",
        "role": "contract implementation",
    },
    {
        "path": "scripts/select_epoch_share.py",
        "reason": "retired selection path; its local summary formulas are not on any live "
                  "report or comparison path",
        "status": "disabled",
        "disabled_by": "t1 legacy-caliber freeze (旧口径结论——不得用于晋升)",
        "replacement": "eval-v2 contract metrics; no live promotion decision",
    },
)
# Non-metric diagnostics that may use percentile ranks over prediction sets.
FORMULA_DIAGNOSTIC_ALLOWLIST = (
    {
        "path": "scripts/compare_blend_replay.py",
        "pattern": "rank\\(pct=True\\)",
        "reason": "top-quintile score overlap between two prediction sets; a score-agreement "
                  "diagnostic, not an eval-v2 return/turnover metric",
        "status": "live",
    },
)
FORMULA_EXEMPT_PATHS = frozenset(entry["path"] for entry in FORMULA_EXEMPTIONS)
FORMULA_ALLOWLIST = frozenset((entry["path"], entry["pattern"])
                              for entry in FORMULA_DIAGNOSTIC_ALLOWLIST)
FORBIDDEN_TOKENS = ("percentile", "composite", "rank_pct", "综合")


def file_text(path: Path) -> str:
    """Read source tolerating a UTF-8 BOM on a few legacy scripts."""
    return path.read_text(encoding="utf-8-sig")


def source(relative: str) -> str:
    return file_text(ROOT / relative)


def imported_contract_names(relative: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source(relative))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("evaluation"):
            names.update(alias.name for alias in node.names)
    return names


def called_names(relative: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source(relative))):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                names.add(function.id)
            elif isinstance(function, ast.Attribute):
                names.add(function.attr)
    return names


def nested_keys(value) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(nested_keys(item))
    return keys


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def synthetic_frames():
    """Five official-grid trading days, twelve stocks, deterministic scores."""
    times = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
    instruments = [f"S{index:02d}" for index in range(12)]
    targets = np.linspace(-0.04, 0.05, len(instruments))
    rows = []
    for day in (2, 3, 6, 7, 8):
        for step, clock in enumerate(times):
            rng = np.random.default_rng(day * 100 + step)
            scores = rng.permutation(np.arange(len(instruments), dtype=float))
            for index, instrument in enumerate(instruments):
                rows.append({"date": f"2025-01-{day:02d} {clock}:00", "instrument": instrument,
                             "score": scores[index], "target": targets[index]})
    frame = pd.DataFrame(rows)
    return frame[["date", "instrument", "score"]], frame[["date", "instrument", "target"]]


def contract_text_sha256() -> str:
    import hashlib

    text = (ROOT / "src/jinyiquant/evaluation.py").read_text(encoding="utf-8")
    normalized = text.replace(chr(13) + chr(10), chr(10))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def test_eval_contract_source_is_frozen_and_wired_by_import_and_call_site():
    # Line-ending normalized so a fresh checkout cannot fail the freeze pin.
    assert contract_text_sha256() == EVALUATION_SHA256
    for relative, (expected_imports, expected_calls) in WIRED.items():
        imported = imported_contract_names(relative)
        missing = expected_imports - imported
        assert not missing, f"{relative}: missing contract imports {sorted(missing)}"
        called = called_names(relative)
        missing_calls = expected_calls - called
        assert not missing_calls, f"{relative}: missing contract calls {sorted(missing_calls)}"
    for relative, token in VERSION_ONLY.items():
        assert token in source(relative), f"{relative}: metric products must carry {token}"
    # Comparison reports that reuse the shared loader still reach evaluate there.
    recipe_imports = {alias.name for node in ast.walk(ast.parse(source(
        "scripts/compare_recipe_reference.py"))) if isinstance(node, ast.ImportFrom)
        for alias in node.names}
    assert "_load" in recipe_imports
    # Coverage analysis reaches the contract through the annual evaluator helper.
    coverage_imports = {alias.name for node in ast.walk(ast.parse(source(
        "scripts/diagnose_training_coverage.py"))) if isinstance(node, ast.ImportFrom)
        for alias in node.names}
    assert "continuous_sections" in coverage_imports
    assert "evaluate(" in source("scripts/analyze_annual_baseline.py")


def test_no_duplicate_metric_formulas_outside_the_contract():
    import re

    scanned = sorted(
        list((ROOT / "src/jinyiquant").glob("*.py"))
        + list((ROOT / "modified_tlob").glob("*.py"))
        + list((ROOT / "scripts").glob("*.py")))
    for path in scanned:
        relative = path.relative_to(ROOT).as_posix()
        if relative in FORMULA_EXEMPT_PATHS:
            continue
        text = file_text(path)
        for pattern, label in FORBIDDEN_FORMULAS.items():
            if (relative, pattern) in FORMULA_ALLOWLIST:
                continue
            assert not re.search(pattern, text), f"{relative}: duplicate {label}"
        assert not re.search(r"rank\(pct=True\)", text) or (
            relative, r"rank\(pct=True\)") in FORMULA_ALLOWLIST, f"{relative}: local percentile"


def test_single_side_rate_is_an_explicit_config_field_never_a_silent_zero():
    assert EvaluationConfig().single_side_rate is None
    assert EvaluationConfig(single_side_rate=0.0005).single_side_rate == 0.0005
    predictions, targets = synthetic_frames()
    undeclared = evaluate(predictions, targets, {**CONFIG, "single_side_rate": None},
                          undefined_constant_portfolios=True)[0]
    economics = undeclared["economic_metrics"]
    assert economics["single_side_rate"] is None
    assert economics["cost_mean"] is None and economics["net_return_mean"] is None
    assert economics["status"] == "undefined_single_side_rate_not_configured"
    configured = evaluate(predictions, targets, CONFIG, undefined_constant_portfolios=True)[0]
    economic = configured["economic_metrics"]
    assert economic["single_side_rate"] == CONFIG["single_side_rate"]
    assert economic["status"] == "computed"
    assert economic["cost_mean"] is not None and economic["cost_mean"] > 0
    assert economic["cost_mean"] == pytest.approx(
        CONFIG["single_side_rate"] * economic["total_abs_weight_change_mean"], rel=0, abs=1e-15)


def test_same_predictions_agree_across_runner_report_and_packaging_entries(tmp_path):
    predictions, targets = synthetic_frames()
    direct, direct_daily, _ = evaluate(predictions, targets, CONFIG, return_sections=True,
                                       undefined_constant_portfolios=True)
    shared = ("rank_ic_mean", "rank_ic_ir", "ic_mean", "ic_ir", "long_short_return_mean",
              "long_short_sharpe", "turnover_mean", "evaluation_version")

    report_metrics = multiscale.evaluate_scores(
        predictions, targets, EvaluationConfig(**CONFIG), "score")[0]
    assert canonical({key: direct[key] for key in shared}) == canonical(
        {key: report_metrics[key] for key in shared})

    curve = runner._curve_evaluation(predictions, targets, SimpleNamespace(**CONFIG))
    runner_block = {"rank_ic_mean": curve["validation_rank_ic_mean"],
                    "ic_mean": curve["validation_ic_mean"],
                    "long_short_return_mean": curve["validation_long_short_return_mean"],
                    "turnover_mean": curve["validation_turnover_mean"],
                    "evaluation_version": curve["validation_evaluation_version"]}
    assert canonical(runner_block) == canonical(
        {key: direct[key] for key in runner_block})

    # Packaging entry: infer writes predictions.parquet with a `factor` column but
    # no evaluation block. The archived re-evaluation entry must fail clearly, then
    # reproduce the same contract block when the training evaluation is supplied.
    run = tmp_path / "packaged"
    run.mkdir()
    predictions.rename(columns={"score": "factor"}).to_parquet(run / "predictions.parquet", index=False)
    targets.to_parquet(run / "targets.parquet", index=False)
    (run / "config.json").write_text(json.dumps(
        {"schema_version": 1, "weights": "model.pt", "training": False, "label_input": False,
         "evaluation_version": EVALUATION_VERSION}), encoding="utf-8")
    with pytest.raises(ValueError, match="no evaluation block"):
        archived_cost_scenarios(run)
    packaged = archived_cost_scenarios(
        run, scenarios=(CONFIG["single_side_rate"],), evaluation=EvaluationConfig(**CONFIG))
    scenario = packaged["scenarios"]["5bp"]
    assert packaged["evaluation_version"] == EVALUATION_VERSION
    assert packaged["economic_status"] == "computed"
    assert canonical(scenario["competition_metrics"]) == canonical(direct["competition_metrics"])
    assert canonical(scenario["economic_metrics"]) == canonical(direct["economic_metrics"])

    five = platform.five(direct_daily)
    assert five["evaluation_version"] == EVALUATION_VERSION
    assert five["turnover_conversion_factor"] == 0.5
    for report_key, contract_key in (("ic", "ic_mean"), ("ic_ir", "ic_ir"),
                                     ("sharpe", "long_short_sharpe"),
                                     ("turnover", "turnover_mean"), ("rank_ic", "rank_ic_mean")):
        assert five[report_key] == pytest.approx(direct[contract_key], rel=0, abs=1e-15)


def test_reports_pair_approximate_competition_and_economic_cost_scenarios():
    predictions, targets = synthetic_frames()
    block = declared_cost_scenarios(predictions, targets, CONFIG,
                                    undefined_constant_portfolios=True)
    assert block["evaluation_version"] == EVALUATION_VERSION
    assert block["turnover_conversion_factor"] == 0.5
    assert block["comparison_usable"] is True
    assert block["competition_status"] == "usable"
    assert block["economic_status"] == "computed"
    assert block["predictive_usable"] is True and block["economic_usable"] is True
    assert set(block["scenarios"]) == {"0bp", "5bp"}
    zero, five = block["scenarios"]["0bp"], block["scenarios"]["5bp"]
    for scenario in (zero, five):
        assert scenario["evaluation_version"] == EVALUATION_VERSION
        competition = scenario["competition_metrics"]
        economic = scenario["economic_metrics"]
        assert competition["approximate"] is True
        assert "stress_stability" in competition["unavailable_metrics"]
        for key in ("gross_return_mean", "cost_mean", "net_return_mean",
                    "turnover_mean", "breakeven_single_side_rate",
                    "total_abs_weight_change_mean", "turnover_conversion_factor"):
            assert key in economic
        assert economic["turnover_conversion_factor"] == 0.5
        assert economic["turnover_mean"] == pytest.approx(
            economic["total_abs_weight_change_mean"] / 2, rel=0, abs=1e-15)
    assert zero["single_side_rate"] == 0.0 and zero["economic_metrics"]["cost_mean"] == 0.0
    assert zero["economic_metrics"]["net_return_mean"] == pytest.approx(
        zero["economic_metrics"]["gross_return_mean"], rel=0, abs=1e-15)
    assert five["single_side_rate"] == 0.0005
    economic_five = five["economic_metrics"]
    assert economic_five["cost_mean"] == pytest.approx(
        0.0005 * economic_five["total_abs_weight_change_mean"], rel=0, abs=1e-15)
    assert economic_five["net_return_mean"] == pytest.approx(
        economic_five["gross_return_mean"] - economic_five["cost_mean"], rel=0, abs=1e-15)
    assert economic_five["breakeven_single_side_rate"] == pytest.approx(
        economic_five["gross_return_mean"] / economic_five["total_abs_weight_change_mean"],
        rel=0, abs=1e-15)
    keys = [key.lower() for key in nested_keys(block)]
    assert not any(token in key for key in keys for token in FORBIDDEN_TOKENS)
    json.dumps(block, allow_nan=False)

def overlapping_frames():
    """Consecutive 15-minute labels overlap the 30-minute horizon."""
    instruments = [f"S{index:02d}" for index in range(5)]
    rows = []
    for day in (2, 3):
        for clock, scores in (("10:00:00", [0, 1, 2, 3, 4]),
                              ("10:15:00", [4, 3, 2, 1, 0]),
                              ("10:50:00", [0, 1, 2, 3, 4])):
            for index, instrument in enumerate(instruments):
                rows.append({"date": f"2025-01-{day:02d} {clock}", "instrument": instrument,
                             "score": float(scores[index]),
                             "target": [0.01, -0.02, 0.03, -0.04, 0.05][index]})
    frame = pd.DataFrame(rows)
    return frame[["date", "instrument", "score"]], frame[["date", "instrument", "target"]]


def test_overlapping_or_descriptive_status_gates_comparison_and_reports():
    from jinyiquant.comparison import evaluation_comparison_gate

    predictions, targets = synthetic_frames()
    normal = evaluate(predictions, targets, CONFIG, undefined_constant_portfolios=True)[0]
    normal_gate = evaluation_comparison_gate(normal)
    assert normal_gate["passed"] is True and normal_gate["reasons"] == []
    assert normal_gate["competition_status"] == "usable"
    assert normal_gate["economic_status"] == "computed"
    assert normal_gate["predictive_usable"] is True and normal_gate["economic_usable"] is True
    overlap_predictions, overlap_targets = overlapping_frames()
    overlapping = evaluate(overlap_predictions, overlap_targets, CONFIG,
                           undefined_constant_portfolios=True)[0]
    gate = evaluation_comparison_gate(overlapping)
    assert gate["passed"] is False and any("overlapping" in reason for reason in gate["reasons"])
    block = declared_cost_scenarios(overlap_predictions, overlap_targets, CONFIG,
                                    undefined_constant_portfolios=True)
    assert block["comparison_usable"] is False
    assert any("overlapping" in reason for reason in block["comparison_gate_reasons"])
    for scenario in block["scenarios"].values():
        assert scenario["comparison_gate"]["passed"] is False
    normal_block = declared_cost_scenarios(predictions, targets, CONFIG,
                                           undefined_constant_portfolios=True)
    assert normal_block["comparison_usable"] is True
    comparison_source = source("src/jinyiquant/comparison.py")
    assert "evaluation_comparison_gate(run.metrics)" in comparison_source
    assert '"evidence_gates"' in comparison_source
    json.dumps(block, allow_nan=False)

def test_formula_exemption_ledger_is_explicit_machine_readable_data():
    ledger = json.loads(json.dumps(FORMULA_EXEMPTIONS))
    assert isinstance(ledger, list) and ledger
    for entry in ledger:
        assert set(entry) >= {"path", "reason", "status"}
        assert (ROOT / entry["path"]).is_file(), entry["path"]
        assert entry["reason"].strip()
        assert entry["status"] in {"live", "disabled"}
        if entry["status"] == "disabled":
            assert entry["disabled_by"].strip()
    live = {entry["path"] for entry in ledger if entry["status"] == "live"}
    assert live == {"src/jinyiquant/evaluation.py"}, live
    disabled = {entry["path"]: entry for entry in ledger if entry["status"] == "disabled"}
    assert "scripts/select_epoch_share.py" in disabled
    assert "t1" in disabled["scripts/select_epoch_share.py"]["disabled_by"]
    # The disabled selection path must not be imported by any non-test production module.
    for path in (list((ROOT / "src/jinyiquant").glob("*.py"))
                 + list((ROOT / "modified_tlob").glob("*.py"))
                 + list((ROOT / "scripts").glob("*.py"))):
        if path.name == "select_epoch_share.py":
            continue
        tree = ast.parse(file_text(path))
        modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names]
        assert not any(module.endswith("select_epoch_share") for module in modules), path
    for entry in FORMULA_DIAGNOSTIC_ALLOWLIST:
        assert (ROOT / entry["path"]).is_file()
        assert entry["reason"].strip() and entry["status"] == "live"


def test_live_learning_audit_producer_uses_the_contract():
    from scripts.analyze_learning_diagnostics import independent_daily

    predictions, targets = synthetic_frames()
    direct, direct_daily, _ = evaluate(predictions, targets, CONFIG, return_sections=True,
                                       undefined_constant_portfolios=False)
    daily, metrics = independent_daily(predictions, targets, EvaluationConfig(**CONFIG))
    assert daily["rank_ic"].tolist() == direct_daily["rank_ic"].tolist()
    assert daily["ic"].tolist() == direct_daily["ic"].tolist()
    assert metrics["rank_ic_mean"] == direct["rank_ic_mean"]
    assert metrics["evaluation_version"] == EVALUATION_VERSION


def test_replay_report_ic_ir_uses_the_contract_with_legacy_fallback():
    # replay_model_versions imports torch through the adapter at module import; skip
    # cleanly in the torch-free dev environment (.venv) instead of failing it.
    pytest.importorskip("torch")
    from jinyiquant.evaluation import aggregate_metric
    from scripts.replay_model_versions import contract_ic_ir

    predictions, targets = synthetic_frames()
    _, daily, _ = evaluate(predictions, targets, CONFIG, return_sections=True,
                           undefined_constant_portfolios=False)
    ic_ir, source_text = contract_ic_ir(
        {"evaluation_version": EVALUATION_VERSION, "ic_ir": 0.9}, daily)
    assert ic_ir == 0.9 and source_text == "eval-v2 metrics.json"
    fallback, fallback_source = contract_ic_ir({}, daily)
    assert fallback == aggregate_metric(daily["ic"].to_numpy(dtype=float), "ir")
    assert "aggregate_metric" in fallback_source

def test_archived_reeval_honours_archived_constant_portfolio_mode(tmp_path):
    instruments = [f"S{index:02d}" for index in range(5)]
    rows = []
    for clock in ("10:00:00", "10:30:00"):
        for index, instrument in enumerate(instruments):
            rows.append({"date": f"2025-01-02 {clock}", "instrument": instrument,
                         "score": 7.0, "target": [0.01, -0.02, 0.03, -0.04, 0.05][index]})
    frame = pd.DataFrame(rows)
    run = tmp_path / "infer_run"
    run.mkdir()
    frame[["date", "instrument", "score"]].rename(columns={"score": "factor"}).to_parquet(
        run / "predictions.parquet", index=False)
    frame[["date", "instrument", "target"]].to_parquet(run / "targets.parquet", index=False)
    base = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
            "single_side_rate": 0.0005}
    (run / "config.json").write_text(json.dumps({"evaluation": {
        **base, "constant_portfolios": "neutral"}}), encoding="utf-8")
    neutral = archived_cost_scenarios(run)
    assert neutral["economic_status"] == "computed"
    assert neutral["comparison_usable"] is True
    assert neutral["scenarios"]["5bp"]["economic_metrics"]["gross_return_mean"] == 0.0
    assert neutral["scenarios"]["5bp"]["economic_metrics"]["net_return_mean"] == 0.0
    (run / "config.json").write_text(json.dumps({"evaluation": {
        **base, "constant_portfolios": "undefined"}}), encoding="utf-8")
    undefined = archived_cost_scenarios(run)
    assert undefined["economic_status"] == "undefined_no_defined_rebalance"
    assert undefined["comparison_usable"] is False
    assert undefined["scenarios"]["5bp"]["economic_metrics"]["gross_return_mean"] is None
    assert any("undefined" in reason for reason in undefined["comparison_gate_reasons"])


def test_no_rate_and_single_timestamp_economics_fail_closed():
    from jinyiquant.comparison import evaluation_comparison_gate

    predictions, targets = synthetic_frames()
    no_rate = {key: value for key, value in CONFIG.items() if key != "single_side_rate"}
    metrics = evaluate(predictions, targets, no_rate, undefined_constant_portfolios=True)[0]
    gate = evaluation_comparison_gate(metrics)
    assert gate["passed"] is False and gate["predictive_usable"] is True
    assert gate["economic_status"] == "undefined_single_side_rate_not_configured"
    assert any("undefined/descriptive" in reason for reason in gate["economic_reasons"])
    single = predictions[predictions.date.eq(predictions.date.iloc[0])]
    single_targets = targets[targets.date.eq(targets.date.iloc[0])]
    block = declared_cost_scenarios(single, single_targets, CONFIG,
                                    undefined_constant_portfolios=True)
    assert block["comparison_usable"] is False
    assert block["economic_status"] == "undefined_no_defined_rebalance"
    assert block["competition_status"] == "usable"
    assert block["comparison_gate_reasons"]


def test_runner_and_baselines_enforce_the_contract_version(monkeypatch):
    from jinyiquant import baselines, runner

    assert baselines.require_contract_metrics is runner.require_contract_metrics
    good = {"evaluation_version": EVALUATION_VERSION}
    assert runner.require_contract_metrics(good) is good
    for bad in ({}, {"evaluation_version": "eval-v1"}, {"evaluation_version": None}):
        with pytest.raises(ValueError, match="Evaluation contract version mismatch"):
            runner.require_contract_metrics(bad)
    # The runner's learning-curve evaluation path must reject a foreign metric product.
    predictions, targets = synthetic_frames()
    monkeypatch.setattr(runner, "evaluate",
                        lambda *args, **kwargs: ({"evaluation_version": "eval-v1"}, None, None))
    with pytest.raises(ValueError, match="Evaluation contract version mismatch"):
        runner._curve_evaluation(predictions, targets, SimpleNamespace(**CONFIG))
