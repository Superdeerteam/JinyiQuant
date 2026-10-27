"""Guards that must fail closed: empty cost scenarios, foreign metric versions, sealed reads."""

import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from jinyiquant import baselines, data, tracking
from jinyiquant.comparison import declared_cost_scenarios
from jinyiquant.config import EvaluationConfig, Period


def test_empty_cost_scenarios_are_not_usable_evidence():
    result = declared_cost_scenarios(pd.DataFrame(), pd.DataFrame(), EvaluationConfig(), scenarios=())
    assert result["scenarios"] == {}
    assert result["comparison_usable"] is False
    assert any("no cost scenarios declared" in reason for reason in result["comparison_gate_reasons"])


def test_declared_scenarios_report_usable_economics():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"] * 3)
    # Day two reverses the book, so the rebalance is real: U > 0 and cost must bite.
    predictions = pd.DataFrame({"date": dates,
                                "instrument": ["A", "A", "B", "B", "C", "C"],
                                "score": [3.0, 1.1, 2.0, 2.1, 1.0, 3.2]})
    targets = predictions.drop(columns=["score"]).assign(target=[0.01, 0.012, 0.0, 0.001, -0.01, -0.011])
    result = declared_cost_scenarios(predictions, targets, EvaluationConfig(), scenarios=(0.0, 0.0005))
    assert sorted(result["scenarios"]) == ["0bp", "5bp"]
    assert result["comparison_usable"] is True
    zero, five = (result["scenarios"][key]["economic_metrics"] for key in ("0bp", "5bp"))
    assert five["status"] == "computed"
    assert five["cost_mean"] > 0
    assert zero["cost_mean"] == 0
    assert five["net_return_mean"] == pytest.approx(five["gross_return_mean"] - five["cost_mean"])


def test_baselines_metric_loader_rejects_a_foreign_evaluation_version(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"evaluation_version": "eval-v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        baselines.load_contract_metrics(run)
    (run / "metrics.json").write_text(json.dumps({"evaluation_version": "eval-v2"}), encoding="utf-8")
    assert baselines.load_contract_metrics(run)["evaluation_version"] == "eval-v2"


def test_baselines_summary_actually_routes_through_the_guard(tmp_path, monkeypatch):
    attempt = {"run": str(tmp_path / "run"), "period": ["2023-11-01", "2023-11-15"],
               "seed": 1, "model": "mean"}
    (tmp_path / "attempts.json").write_text(json.dumps([attempt]), encoding="utf-8")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(baselines, "RunConfig",
                        SimpleNamespace(model_validate=staticmethod(lambda document: object())))

    def boom(path):
        raise ValueError("foreign evaluation version")

    monkeypatch.setattr(baselines, "load_contract_metrics", boom)
    with pytest.raises(ValueError, match="foreign evaluation version"):
        baselines.summarize(tmp_path, {})


def final_config(tmp_path, *, max_views=1):
    final = Period(start=datetime(2024, 10, 8), end=datetime(2025, 1, 1))
    return SimpleNamespace(output_root=tmp_path,
                           data=SimpleNamespace(source="source", version="v1"),
                           splits=SimpleNamespace(final=final),
                           final_access=SimpleNamespace(max_views=max_views, purpose="unit test"))


def test_sealed_read_requires_a_recorded_claim_token(tmp_path):
    start, end = data.SEALED_START, data.SEALED_END
    with pytest.raises(ValueError, match="Sealed final interval"):
        data.assert_development_read(start, end)
    with pytest.raises(TypeError, match="FinalAccessClaim"):
        data.assert_development_read(start, end, sealed_access=True)
    config = final_config(tmp_path)
    claim = tracking.claim_final_access(config, "unit-run")
    assert isinstance(claim, tracking.FinalAccessClaim)
    assert claim.scope and claim.run_id == "unit-run"
    data.assert_development_read(start, end, sealed_access=claim)


def test_claim_is_audited_and_the_view_limit_is_enforced(tmp_path):
    config = final_config(tmp_path)
    tracking.claim_final_access(config, "first")
    with sqlite3.connect(tmp_path / "audit.sqlite3") as db:
        rows = db.execute("SELECT scope, run_id, purpose FROM final_access").fetchall()
    assert len(rows) == 1 and rows[0][1] == "first" and rows[0][2] == "unit test"
    with pytest.raises(ValueError, match="limit reached"):
        tracking.claim_final_access(config, "second")


def test_load_data_signature_keeps_the_claim_keyword():
    import inspect

    parameters = inspect.signature(data.load_data).parameters
    # Bounded bar loading is additive and opt-in: no read changes unless ``stream_bars`` is
    # passed, and every keyword stays keyword-only.
    assert list(parameters) == ["config", "start", "end", "sealed_access", "stream_bars", "chunk_rows"]
    assert parameters["sealed_access"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["stream_bars"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["stream_bars"].default is False
