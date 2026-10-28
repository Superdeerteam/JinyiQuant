"""Contract tests for the fusion-increment gate (scripts/fusion_increment_gate.py).

The gate is the judge for every R6..R18 route, so the properties that must not drift are:
the registration and the code agree on the thresholds, a plan cannot run against a different
registration, the read is (window, trading day) paired and scored by the house block interval,
a route that merely reproduces the trunk gets no allocation, an increment that lives only in
the weight-fit block is caught by the post-freeze guard, and the confirmation window is a
one-shot that needs the captain's release.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gate_module():
    spec = importlib.util.spec_from_file_location(
        "fusion_increment_gate", ROOT / "scripts/fusion_increment_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_stub(*, mean, ci_low, days=165, positive=7):
    pooled = {"days": days, "mean": mean, "ci_low": ci_low, "ci_high": mean + 0.01}
    return {
        "pooled": {"rank_ic": pooled,
                   "turnover": {**pooled, "mean": 0.01},
                   "long_short_return": {**pooled, "mean": 0.002}},
        "per_window": {},
    }


def fused_plan():
    return {"_path": "memory", "_sha256": "0" * 64, "route": "R99", "arm_kind": "fused",
            "candidate": {"label": "synthetic"}}


# --------------------------------------------------------------------------------------- #
# registration / plan integrity (no predictions are touched)
# --------------------------------------------------------------------------------------- #
def test_registration_agrees_with_the_pinned_constants():
    gate = gate_module()
    registration = gate.load_registration()
    decision = registration["decision"]
    assert decision["min_improvement"] == gate.MIN_IMPROVEMENT == 0.003
    assert decision["min_positive_windows"] == gate.MIN_POSITIVE_WINDOWS == 7
    assert decision["windows_total"] == gate.WINDOWS_TOTAL == 8
    assert registration["window_roles"]["screen_days"] == gate.SCREEN_DAYS == 165
    assert registration["window_roles"]["confirmation_days"] == gate.CONFIRMATION_DAYS == 38
    assert registration["window_roles"]["post_freeze_days"] == gate.POST_FREEZE_DAYS == 155
    assert registration["estimator"]["draws"] == gate.BOOTSTRAP_DRAWS == 20000
    assert registration["estimator"]["seed"] == gate.BOOTSTRAP_SEED == 20261028
    assert tuple(registration["fusion_increment_definition"]["weight_fit"]["grid"]) == \
        gate.WEIGHT_GRID
    assert gate.gate_script_state(registration)["matches"] is True


def test_registration_rejects_a_changed_threshold(tmp_path):
    gate = gate_module()
    body = json.loads((ROOT / "configs" / "fusion_increment_registration_20261028.json")
                      .read_text(encoding="utf-8"))
    body["decision"]["min_improvement"] = 0.001
    path = tmp_path / "loosened.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        gate.load_registration(path)
    except SystemExit as error:
        assert "min_improvement" in str(error)
    else:
        raise AssertionError("a loosened threshold must abort the gate")


def test_registration_rejects_a_swapped_estimator(tmp_path):
    gate = gate_module()
    body = json.loads((ROOT / "configs" / "fusion_increment_registration_20261028.json")
                      .read_text(encoding="utf-8"))
    body["estimator"]["implementation_sha256"] = "0" * 64
    path = tmp_path / "swapped.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        gate.load_registration(path)
    except SystemExit as error:
        assert "sha256 differs" in str(error)
    else:
        raise AssertionError("a swapped block-interval implementation must abort the gate")


def test_plan_pinned_to_another_registration_is_refused(tmp_path):
    gate = gate_module()
    registration = gate.load_registration()
    plan = {"schema_version": gate.PLAN_SCHEMA, "route": "R18", "arm_kind": "fused",
            "registration": {"path": "configs/fusion_increment_registration_20261028.json",
                             "sha256": "f" * 64},
            "incumbent": {"run_dir": "x", "members": [{"version": "r441_s11"}]},
            "candidate": {"label": "R18", "run_dir": "y",
                          "members": [{"version": "mr_s11"}]}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    try:
        gate.load_plan(path, registration)
    except SystemExit as error:
        assert "re-author the plan" in str(error)
    else:
        raise AssertionError("a plan pinned to another registration must not run")


def test_audit_reads_no_prediction_and_lists_the_registered_roles():
    gate = gate_module()
    audit = gate.run_audit(gate.load_registration(), ledger=ROOT / "runs" / "fusion-gate" /
                           "does-not-exist.jsonl")
    assert audit["reads_any_prediction"] is False
    assert len(audit["window_roles"]["screen"]) == 8
    assert audit["window_roles"]["confirmation"] == ["jun2024", "sep2024"]
    assert audit["weight_fit"]["trading_days"] == 10
    assert audit["weight_fit"]["start"] >= "2023-11-01"
    assert audit["gate_script"]["matches"] is True


# --------------------------------------------------------------------------------------- #
# decision ladder (pure inputs, no data)
# --------------------------------------------------------------------------------------- #
def test_decision_ladder_labels_every_failed_condition():
    gate = gate_module()
    registration = gate.load_registration()
    plan = fused_plan()
    weak = gate.decide_screen(registration=registration, plan=plan,
                              fit={"weight": 0.3},
                              read=read_stub(mean=0.001, ci_low=-0.002),
                              guard={"mean": -0.004},
                              positives={"count": 5, "of": 8},
                              cost={"available": True})
    assert weak["verdict"] == "below_threshold"
    assert set(weak["fail_reasons"]) == {"below_threshold", "interval_includes_zero",
                                         "positive_windows_below_7",
                                         "post_freeze_mean_not_positive"}
    assert weak["clears"] is False


def test_decision_requires_the_cost_report_even_when_the_statistics_pass():
    gate = gate_module()
    registration = gate.load_registration()
    blocked = gate.decide_screen(registration=registration, plan=fused_plan(),
                                 fit={"weight": 0.25},
                                 read=read_stub(mean=0.01, ci_low=0.004),
                                 guard={"mean": 0.008},
                                 positives={"count": 8, "of": 8},
                                 cost={"available": False})
    assert blocked["verdict"] == "blocked_missing_cost_report"
    assert blocked["clears"] is False

    cleared = gate.decide_screen(registration=registration, plan=fused_plan(),
                                 fit={"weight": 0.25},
                                 read=read_stub(mean=0.01, ci_low=0.004),
                                 guard={"mean": 0.008},
                                 positives={"count": 8, "of": 8},
                                 cost={"available": True})
    assert cleared["verdict"] == "clears_fusion_gate"
    assert cleared["clears"] is True


def test_net5_is_required_for_a_clear():
    """缺 5bp 净收益旁证时即使统计量全过也不许 clears（换手/net5 只报，但缺了就是读数不完整）。"""
    gate = gate_module()
    registration = gate.load_registration()
    kwargs = dict(registration=registration, plan=fused_plan(), fit={"weight": 0.25},
                  read=read_stub(mean=0.01, ci_low=0.004), guard={"mean": 0.008},
                  positives={"count": 8, "of": 8}, cost={"available": True})
    assert gate.decide_screen(**kwargs)["verdict"] == "clears_fusion_gate"
    incomplete = gate.decide_screen(**kwargs, side_complete=False)
    assert incomplete["verdict"] == "blocked_missing_cost_report"
    assert incomplete["clears"] is False


def test_turnover_attention_is_an_annotation_not_a_gate():
    """Δturnover 超过 0.0318 只打旗；两个方向的 verdict 必须完全相同。"""
    gate = gate_module()
    registration = gate.load_registration()
    base = read_stub(mean=0.01, ci_low=0.004)
    quiet = {"pooled": {**base["pooled"],
                        "turnover": {**base["pooled"]["turnover"], "mean": 0.0}},
             "per_window": {}}
    loud = {"pooled": {**base["pooled"],
                       "turnover": {**base["pooled"]["turnover"],
                                    "mean": gate.TURNOVER_ATTENTION_THRESHOLD + 0.01}},
            "per_window": {}}
    kwargs = dict(registration=registration, plan=fused_plan(), fit={"weight": 0.25},
                  guard={"mean": 0.008}, positives={"count": 8, "of": 8},
                  cost={"available": True})
    quiet_decision = gate.decide_screen(read=quiet, **kwargs)
    loud_decision = gate.decide_screen(read=loud, **kwargs)
    assert quiet_decision["annotations"]["turnover_attention"] is False
    assert loud_decision["annotations"]["turnover_attention"] is True
    assert loud_decision["annotations"]["does_not_gate"] is True
    assert quiet_decision["verdict"] == loud_decision["verdict"] == "clears_fusion_gate"
    assert quiet_decision["clears"] is True and loud_decision["clears"] is True
    assert quiet_decision["fail_reasons"] == loud_decision["fail_reasons"] == []


def test_single_model_is_descriptive_only():
    gate = gate_module()
    registration = gate.load_registration()
    plan = dict(fused_plan(), arm_kind="single_model")
    decision = gate.decide_screen(registration=registration, plan=plan, fit={"weight": 1.0},
                                  read=read_stub(mean=0.05, ci_low=0.04),
                                  guard={"mean": 0.05},
                                  positives={"count": 8, "of": 8},
                                  cost={"available": True})
    assert decision["verdict"] == "descriptive_only"
    assert decision["clears"] is False


def test_post_freeze_guard_drops_only_the_weight_fit_block():
    gate = gate_module()
    registration = gate.load_registration()
    windows = ["nov2023", "nov2023", "nov2023", "jan2024"]
    days = ["2023-11-02", "2023-11-13", "2023-11-15", "2024-01-05"]
    deltas = [1.0, 1.0, 0.0, 0.0]
    read = {"series": {"windows": windows, "days": days, "delta_rank_ic": deltas}}
    guard = gate.post_freeze_guard(read, registration)
    assert guard["days_excluded"] == 2
    assert guard["days"] == 2
    assert guard["mean"] == 0.0
    assert guard["excluded_block"].startswith("nov2023 2023-11-01..2023-11-14")


def test_positive_windows_counts_only_positive_window_means():
    gate = gate_module()
    read = {"per_window": {name: {"rank_ic": {"delta_mean": value}}
                           for name, value in (("a", 0.001), ("b", -0.001), ("c", 0.0))}}
    bucket = gate.positive_windows(read, ["a", "b", "c"])
    assert bucket == {"unit": "window", "count": 1, "of": 3, "windows": ["a"]}


# --------------------------------------------------------------------------------------- #
# cost report
# --------------------------------------------------------------------------------------- #
def test_cost_report_converts_seconds_into_per_trading_day_cost(tmp_path):
    gate = gate_module()
    path = tmp_path / "inference_cost.json"
    path.write_text(json.dumps({
        "arm": "R99",
        "chunks": [{"chunk": 0, "dates": ["2023-11-01", "2023-11-02"],
                    "per_seed_seconds": {"11": 10.0},
                    "resources": {"gpu_peak_memory_mb": 128.0},
                    "shared_preprocessing_seconds": 2.0},
                   {"chunk": 1, "dates": ["2023-11-03"],
                    "per_seed_seconds": {"11": 10.0},
                    "resources": {"gpu_peak_memory_mb": 64.0},
                    "shared_preprocessing_seconds": 2.0}],
        "measurement": "unit test", "platform_verified": True,
        "total_seconds": 24.0,
    }), encoding="utf-8")
    report = gate.cost_report({"paths": [str(path)]}, arm="candidate")
    assert report["available"] is True
    assert report["days_covered"] == 3
    assert report["seconds_per_trading_day"] == 8.0
    assert report["gpu_peak_memory_mb"] == 128.0
    assert report["platform_verified"] is True

    missing = gate.cost_report({"paths": [str(tmp_path / "nope.json")]}, arm="candidate")
    assert missing["available"] is False
    assert "missing inference cost" in missing["reason"]


# --------------------------------------------------------------------------------------- #
# end-to-end reads on synthetic replays
# --------------------------------------------------------------------------------------- #
def synthetic_read(tmp_path, variant, route):
    """One full screen read on a synthetic replay that mirrors the window layout."""
    gate = gate_module()
    registration = gate.load_registration()
    case = gate.synthetic_case(tmp_path, variant=variant, route=route,
                               registration_sha256=registration["_sha256"])
    ledger = tmp_path / "ledger.jsonl"
    plan = gate.load_plan(case["plan"], registration)
    result = gate.run_gate(plan, registration, mode="screen", ledger=ledger)
    return gate, registration, case, ledger, result


def test_screen_read_pairs_days_and_reuses_the_registered_estimator(tmp_path):
    gate, _, _, ledger, result = synthetic_read(tmp_path, "positive", "R91")
    assert len(result["read"]["series"]["days"]) == gate.SCREEN_DAYS
    assert result["decision"]["verdict"] == "clears_fusion_gate"
    assert 0.0 < result["weight"]["weight"] <= gate.MAX_FUSION_WEIGHT

    pooled = result["read"]["pooled"]["rank_ic"]
    direct = gate.block_interval(np.asarray(result["read"]["series"]["delta_rank_ic"]),
                                 block=gate.BLOCK_DAYS, draws=gate.BOOTSTRAP_DRAWS,
                                 seed=gate.BOOTSTRAP_SEED)
    assert pooled["mean"] == direct["mean"]
    assert pooled["ci_low"] == direct["ci_low"]
    # the retired single-block interval is reported side by side but never gates
    assert pooled["archived_ci_low"] is not None
    assert pooled["ci_excludes_zero"] is True

    side = result["side_reports"]
    assert side["complete"] is True
    assert side["long_short_daily_mean"]["fused"] is not None
    assert side["turnover"]["trunk"] is not None
    assert side["net_return_5bp"]["fused"] is not None
    # 5bp 净收益在"没有前一组合"的交易日上未定义：区间只跑在有定义的那几天，并报出计数。
    net5_defined = side["net_return_5bp"]["days_defined"]
    assert net5_defined is not None and 0 < net5_defined <= gate.SCREEN_DAYS
    assert side["net_return_5bp"]["paired_delta"]["days"] == net5_defined
    assert (net5_defined + side["net_return_5bp"]["days_undefined"]) == gate.SCREEN_DAYS
    assert side["annotations"]["turnover_attention"] in (True, False)
    assert side["inference_cost"]["fused_marginal_seconds_per_day"] is not None
    assert side["inference_cost"]["candidate"]["available"] is True

    rows = gate.ledger_entries(ledger)
    assert [row["mode"] for row in rows] == ["screen"]
    assert rows[0]["headline"]["weight"] == result["weight"]["weight"]
    assert rows[0]["headline"]["net5_delta"] is not None
    assert rows[0]["headline"]["turnover_attention"] in (True, False)
    assert rows[0]["gate_sha256"] == result["gate"]["sha256"]


def test_candidate_that_reproduces_the_trunk_gets_no_allocation(tmp_path):
    _, _, _, _, result = synthetic_read(tmp_path, "duplicate", "R92")
    assert result["weight"]["weight"] == 0.0
    assert result["decision"]["verdict"] == "no_allocation"
    assert result["decision"]["clears"] is False


def test_increment_confined_to_the_fit_block_is_rejected(tmp_path):
    _, _, _, _, result = synthetic_read(tmp_path, "fit_only", "R94")
    assert result["weight"]["weight"] > 0.0
    assert result["post_freeze"]["mean"] <= 0.0
    assert "post_freeze_mean_not_positive" in result["decision"]["fail_reasons"]
    assert result["decision"]["clears"] is False


def test_confirmation_is_a_one_shot_that_needs_the_captains_release(tmp_path):
    gate, registration, case, ledger, screening = synthetic_read(tmp_path, "positive", "R96")
    plan = gate.load_plan(case["plan"], registration)
    screening_id = screening["ledger"]["read_id"]
    release = tmp_path / "release.json"
    release.write_text(json.dumps({"route": "R96", "released_by": "captain",
                                   "screening_read_id": screening_id}), encoding="utf-8")

    confirm = gate.run_confirm(plan, registration, ledger=ledger, release_path=release)
    assert confirm["confirmation"]["passes"] is True
    assert confirm["confirmation"]["positive_days"] >= gate.CONFIRMATION_MIN_POSITIVE_DAYS
    assert confirm["paired_days"] == gate.CONFIRMATION_DAYS

    release.write_text(json.dumps({"route": "R96", "released_by": "engineer",
                                   "screening_read_id": screening_id}), encoding="utf-8")
    try:
        gate.run_confirm(plan, registration, ledger=tmp_path / "fresh.jsonl",
                         release_path=release)
    except SystemExit as error:
        assert "released_by" in str(error)
    else:
        raise AssertionError("only the captain may release a confirmation read")

    release.write_text(json.dumps({"route": "R96", "released_by": "captain",
                                   "screening_read_id": screening_id}), encoding="utf-8")
    try:
        gate.run_confirm(plan, registration, ledger=ledger, release_path=release)
    except SystemExit as error:
        assert "already spent" in str(error)
    else:
        raise AssertionError("the confirmation window is one-shot per direction")


def test_refit_after_a_read_is_refused(tmp_path):
    gate, registration, case, ledger, screening = synthetic_read(tmp_path, "positive", "R97")
    plan = gate.load_plan(case["plan"], registration)
    try:
        gate.run_gate(plan, registration, mode="confirm",
                      ledger=tmp_path / "refit.jsonl", frozen_weight=0.0)
    except SystemExit as error:
        assert "refit detected" in str(error)
    else:
        raise AssertionError("the weight must stay frozen after the screening read")
