"""Behavioral checks for fixed raw-scale averaging and forward research decisions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jinyiquant.io import file_sha256, write_json
from modified_tlob.ensemble import (
    SEEDS,
    FixedSeedEnsemble,
    average_predictions,
    paired_block_intervals,
    research_decision,
)
from scripts.run_seed_ensemble import claim_training, complete_correlations, guarded_data, process_tree_usage


def frames():
    base = pd.DataFrame({"date": pd.to_datetime(["2023-11-01 09:30"] * 3), "instrument": list("ABC")})
    return {s: base.assign(score=np.array([1., 3., 2.]) * scale) for s, scale in zip(SEEDS, [1, 2, 9])}


def test_key_aligned_equal_raw_average_no_rank_or_std_normalization():
    values = frames()
    values[29] = values[29].iloc[::-1]
    result = average_predictions(values)
    np.testing.assert_array_equal(result.score, [4., 12., 8.])
    values[47].loc[0, "score"] = 100
    expected = (1 + 2 + 100) / 3
    assert average_predictions(values).score.iloc[0] == expected


@pytest.mark.parametrize("failure", ["missing_seed", "extra_seed", "missing_stock", "duplicate", "nan"])
def test_averaging_fails_closed(failure):
    values = frames()
    if failure == "missing_seed":
        values.pop(47)
    elif failure == "extra_seed":
        values[99] = values[11]
    elif failure == "missing_stock":
        values[29] = values[29].iloc[:2]
    elif failure == "duplicate":
        values[29] = pd.concat([values[29], values[29].iloc[:1]])
    else:
        values[29].loc[0, "score"] = np.nan
    with pytest.raises(ValueError):
        average_predictions(values)


def test_live_ensemble_uses_each_models_unscaled_prediction_and_fallback():
    models = {}
    for seed, scale in zip(SEEDS, [.001, .002, .009]):
        def predict(batch, scale=scale):
            # Same network output, distinct fitted training scales and training-mean fallback.
            return np.array([2 * scale, -scale, .003])
        models[seed] = SimpleNamespace(seed=seed, config={"lr": 1e-4}, preprocessing={"lookback": 128},
                                       predict=predict)
    np.testing.assert_allclose(FixedSeedEnsemble(models).predict(None), [.008, -.004, .003])
    models[29].preprocessing = {"lookback": 64}
    with pytest.raises(ValueError, match="identical"):
        FixedSeedEnsemble(models)


def test_three_checkpoint_roundtrip_restores_scales_and_missing_history_fallback(tmp_path):
    pytest.importorskip("torch")
    from test_modified_tlob import batch, data_config

    from jinyiquant.config import ModelConfig
    from modified_tlob.adapter import TLOBAdapter

    cfg = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=3,
                      epochs=2, device="cpu", target_scaling="train_std")
    sample = batch()
    sample.observed[-1] = False
    paths, predictions = {}, []
    for seed, scale in zip(SEEDS, [.001, .002, .009]):
        model = TLOBAdapter(cfg, data_config(), seed=seed)
        model.trained, model.target_scale, model.intercept = True, scale, scale / 2
        paths[seed] = tmp_path / f"{seed}.pt"
        model.save(paths[seed])
        predictions.append(model.predict(sample).astype(float))
    restored = FixedSeedEnsemble.load(paths, device="cpu")
    np.testing.assert_allclose(restored.predict(sample), np.mean(predictions, axis=0), atol=1e-12, rtol=1e-7)
    assert restored.predict(sample)[-1] == pytest.approx(.002)


def paired():
    return pd.DataFrame({"period": ["a"] * 10 + ["b"] * 10,
        "date": pd.date_range("2023-11-01", periods=10).tolist() + pd.date_range("2024-03-01", periods=10).tolist(),
        "rank_ic": .01, "long_short_return": .00001, "turnover": -.01})


def decision(data, *, summaries=None, cost=True):
    intervals = [paired_block_intervals(data, block_days=b, samples=100) for b in (5, 2)]
    summary = summaries or [{"period": p, "worst_day_delta": 0} for p in ("a", "b")]
    return research_decision(data, summary, intervals, {"minimum_mean_gain": .00707,
        "noise": {"long_short_return_mean": .000047, "turnover_mean": .087,
                  "min_daily_long_short_return": .000068}}, cost_measured=cost)


def test_blocks_are_within_windows_and_pair_days_not_stocks_or_seeds():
    data = paired()
    data.loc[data.period == "b", "rank_ic"] = .02
    result = paired_block_intervals(data, block_days=5, samples=100)
    pooled = next(r for r in result["rows"] if r["period"] == "all" and r["metric"] == "rank_ic")
    np.testing.assert_allclose(pooled["interval"], [.015, .015])
    assert pooled["days"] == 20
    assert result == paired_block_intervals(data.sample(frac=1, random_state=9), block_days=5, samples=100)
    assert decision(data)["result"] == "accept"


@pytest.mark.parametrize("failure", ["gain", "window", "gross", "turnover", "worst", "cost"])
def test_point_gates_reject_failure_without_search(failure):
    data, summary, cost = paired(), None, True
    if failure == "gain":
        data["rank_ic"] = .007
    elif failure == "window":
        data.loc[data.period == "a", "rank_ic"] = -.001
        data.loc[data.period == "b", "rank_ic"] = .03
    elif failure == "gross":
        data["long_short_return"] = -.00001
    elif failure == "turnover":
        data["turnover"] = .001
    elif failure == "worst":
        summary = [{"period": "a", "worst_day_delta": -.00007}, {"period": "b", "worst_day_delta": 0}]
    else:
        cost = False
    result = decision(data, summaries=summary, cost=cost)
    assert result["result"] == ("inconclusive" if failure == "cost" else "reject")
    assert result["action"] == "retain_seed11_reference"
    assert not result["model_promoted"]


def test_positive_mean_with_uncertain_blocks_is_inconclusive():
    data = paired()
    data["rank_ic"] = np.tile([-.09] * 5 + [.11] * 5, 2)
    assert decision(data)["result"] == "inconclusive"
    with pytest.raises(ValueError, match="complete"):
        decision(data.iloc[:19])


def test_role_guard_rejects_future_roles_and_inference_never_supplies_label_path(tmp_path, monkeypatch):
    registry = tmp_path / "roles.json"
    write_json(registry, {"active": "research"})
    plan = {"registry_path": str(registry), "registry_sha256": file_sha256(registry)}
    window = {"train_start": "2023-10-18", "train_end_exclusive": "2023-11-01",
              "evaluation_start": "2023-11-01", "evaluation_end_exclusive": "2023-11-15"}
    config = SimpleNamespace(data=SimpleNamespace(model_dump=lambda **kw: {"labels_path": "secret"}))
    captured = []
    def read(data, **kwargs):
        captured.append((data, kwargs))
        return SimpleNamespace(labels=pd.DataFrame())
    monkeypatch.setattr("scripts.run_seed_ensemble.load_data", read)
    for start, end in [("2024-07-01", "2024-07-02"), ("2023-10-18", "2023-11-02")]:
        with pytest.raises(ValueError, match="outside"):
            guarded_data(plan, config, window, start, end, labels=True)
    assert not captured
    guarded_data(plan, config, window, "2023-11-01", "2023-11-02", labels=False)
    assert captured[0][0]["labels_path"] is None
    write_json(registry, {"active": "changed"})
    with pytest.raises(ValueError, match="registration changed"):
        guarded_data(plan, config, window, "2023-11-01", "2023-11-02", labels=False)


def test_watchdog_counts_real_worker_below_windows_launcher(monkeypatch):
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4000 * 2**20), cpu_times=lambda: (10., 2.))
    parent = SimpleNamespace(children=lambda recursive: [child],
        memory_info=lambda: SimpleNamespace(rss=4 * 2**20), cpu_times=lambda: (0., 0.))
    monkeypatch.setattr("scripts.run_seed_ensemble.psutil.Process", lambda pid: parent)
    assert process_tree_usage(1) == {"rss_mb": 4004., "cpu_seconds": 12., "process_count": 2}


def test_coverage_checks_named_correlations_ignoring_schema_metadata():
    coverage = {"schema_version": 1, **{name: {"eligible_sections_sha256": "same", "valid_sections_sha256": "same",
        "eligible_section_count": 80, "valid_section_count": 80} for name in ("rank_ic", "ic")}}
    assert complete_correlations({"correlation_coverage": coverage}, 80)
    coverage["ic"]["valid_section_count"] = 79
    assert not complete_correlations({"correlation_coverage": coverage}, 80)
    coverage["ic"]["valid_section_count"] = 80
    coverage["rank_ic"]["valid_sections_sha256"] = "different"
    assert not complete_correlations({"correlation_coverage": coverage}, 80)


def test_claim_can_recover_only_proven_zero_step_pretraining_interruption(tmp_path):
    plan = tmp_path / "plan.json"
    write_json(plan, {"max_training_attempts": 6})
    claim = tmp_path / "claim.json"
    first = tmp_path / "first"
    first.mkdir()
    (first / "events.jsonl").write_text("", encoding="utf-8")
    claim_training(claim, first, "a", 11, plan)
    with pytest.raises(ValueError, match="already spent"):
        claim_training(claim, tmp_path / "second", "a", 11, plan)
    from jinyiquant.io import read_json

    record = read_json(claim) | {"status": "pretraining_interrupted", "training_steps": 0}
    write_json(claim, record)
    (first / "events.jsonl").write_text('{"stage":"training","event_type":"started"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="after training started"):
        claim_training(claim, tmp_path / "second", "a", 11, plan)
    (first / "events.jsonl").write_text("", encoding="utf-8")
    claim_training(claim, tmp_path / "second", "a", 11, plan)
    assert read_json(claim)["previous_pretraining_attempt"]["run"] == str(first)
