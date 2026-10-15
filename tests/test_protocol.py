"""Factor exemptions never extend to evaluation, labels, budget or source identity."""

from copy import deepcopy

import pytest

from jinyiquant.protocol import fair_config_pair, fair_data_pair, validate_change


def configs():
    baseline = {"data": {"lookback": 30, "sample_times": ["10:00"], "source": "frozen", "version": "1"},
                "model": {"alpha": 1}, "budget": {"wall_seconds": 60}, "evaluation": {"quantile": .2},
                "splits": {"validation": {"start": "2020-01-01", "end": "2020-01-02"}}}
    candidate = deepcopy(baseline)
    candidate["data"]["lookback"] = 60
    change = {"factor": "data.lookback", "baseline": 30, "candidate": 60, "hypothesis": "Longer history helps"}
    return baseline, candidate, change


@pytest.mark.parametrize("section,field,value", [
    ("data", "sample_times", ["14:00"]), ("data", "version", "2"), ("data", "source", "other"),
    ("model", "alpha", 2), ("budget", "wall_seconds", 120), ("evaluation", "quantile", .1),
    ("splits", "validation", {"start": "2020-01-01", "end": "2020-01-03"}),
])
def test_only_named_factor_can_change(section, field, value):
    baseline, candidate, change = configs()
    fair_config_pair(baseline, candidate, change)
    candidate[section][field] = value
    with pytest.raises(ValueError, match="unfair"):
        fair_config_pair(baseline, candidate, change)


@pytest.mark.parametrize("factor", ["data", "data.*", "data.sample_times", "budget.wall_seconds", "splits.train"])
def test_factor_cannot_be_a_wildcard_or_change_the_evaluation_contract(factor):
    _, _, change = configs()
    change["factor"] = factor
    with pytest.raises(ValueError, match="factor"):
        validate_change(change)


def test_feature_metadata_exemption_keeps_raw_source_checks():
    change = {"factor": "data.feature_columns", "baseline": ["bid"], "candidate": ["bid", "volume"],
              "hypothesis": "Test complementarity"}
    baseline = {"feature_columns": ["bid"], "files": {"bars": {"sha256": "abc"}}, "label_rows": 20}
    candidate = {**baseline, "feature_columns": ["bid", "volume"]}
    fair_data_pair(baseline, candidate, change)
    candidate["files"] = {"bars": {"sha256": "changed"}}
    with pytest.raises(ValueError, match="identity"):
        fair_data_pair(baseline, candidate, change)
    candidate["files"] = baseline["files"]
    candidate["feature_columns"] = ["unexpected"]
    with pytest.raises(ValueError, match="contradicts"):
        fair_data_pair(baseline, candidate, change)


def test_suite_rejects_second_factor_before_any_training(tmp_path, monkeypatch):
    from jinyiquant import experiments
    from jinyiquant.demo import create_demo
    from jinyiquant.io import digest, read_json, write_json

    path = create_demo(tmp_path / "fixture")
    baseline = read_json(path)
    candidate = deepcopy(baseline)
    candidate["data"]["lookback"] += 1
    candidate["model"]["alpha"] = 2
    write_json(path.parent / "candidate.json", candidate)
    report = {"status": "completed"}
    write_json(path.parent / "calibration.json", report)
    policy = {"primary_metric": "rank_ic_mean", "direction": "maximize", "min_improvement": .01,
        "max_metric_regressions": {"turnover_mean": .05}, "max_wall_ratio": 2, "max_memory_ratio": 2,
        "min_seeds": 2, "min_periods": 2, "bootstrap_samples": 100, "block_days": 1, "confidence": .95,
        "seed": 1, "threshold_rationale": "Measured noise and practical floor",
        "change": {"factor": "data.lookback", "baseline": 5, "candidate": 6, "hypothesis": "Longer history"},
        "threshold_basis": {"calibration_report": "calibration.json", "calibration_sha256": digest(report),
                            "practical_improvement": .001, "noise_multiplier": 1}}
    plan = {"baseline_config": "config.json", "candidate_config": "candidate.json", "policy": policy,
        "calibration_report": "calibration.json", "seeds": [11, 29],
        "periods": [{"start": "2020-01-20", "end": "2020-01-27"},
                    {"start": "2020-01-27", "end": "2020-02-03"}]}
    write_json(path.parent / "suite.json", plan)

    def forbidden(*args, **kwargs):
        pytest.fail("A rejected plan must not launch a child run")

    monkeypatch.setattr(experiments, "run", forbidden)
    with pytest.raises(ValueError, match="undeclared"):
        experiments.suite(path.parent / "suite.json")
