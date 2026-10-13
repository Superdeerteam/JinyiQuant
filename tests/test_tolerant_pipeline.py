"""Small gaps may continue; recorded coverage must still expose whole-section outages."""

import numpy as np
import pandas as pd
import pytest

from jinyiquant.config import RunConfig
from jinyiquant.io import read_json, write_json
from jinyiquant.models import LinearModel
from jinyiquant.runner import execute_pipeline


def _fixture(tmp_path, *, missing_stock=False, minimum_coverage=0.99):
    bars, labels, universe = [], [], []
    for day in ("2020-01-02", "2020-01-03"):
        for section in ("10:00", "10:30"):
            cutoff = pd.Timestamp(f"{day} {section}")
            for stock in range(100):
                key = {"date": cutoff, "instrument": f"{stock:06d}.SZ"}
                universe.append(key)
                if missing_stock and stock == 99:
                    continue
                for lag in range(3):
                    bars.append({**key, "date": cutoff - pd.Timedelta(minutes=lag),
                                 "close": 100 + stock / 10 + lag / 100})
                labels.append({**key, "target": (stock - 49.5) / 10000,
                               "label_end": cutoff + pd.Timedelta(minutes=30)})
    for name, rows in (("bars", bars), ("labels", labels), ("universe", universe)):
        pd.DataFrame(rows).to_parquet(tmp_path / f"{name}.parquet", index=False)
    return RunConfig.model_validate({
        "experiment_id": "tolerant-pipeline-test", "hypothesis": "Isolated gaps remain explicit",
        "data": {"bars_path": str(tmp_path / "bars.parquet"),
                 "labels_path": str(tmp_path / "labels.parquet"),
                 "instruments_path": str(tmp_path / "universe.parquet"),
                 "source": "synthetic", "version": "isolated-gap-v1",
                 "feature_columns": ["close"], "lookback": 3, "min_history": 2,
                 "sample_times": ["10:00", "10:30"]},
        "splits": {"train": {"start": "2020-01-02", "end": "2020-01-03"},
                   "validation": {"start": "2020-01-03", "end": "2020-01-04"},
                   "final": {"start": "2020-01-04", "end": "2020-01-05"}},
        "acceptance": {"min_coverage": minimum_coverage},
    })


def _destination(tmp_path, name):
    destination = tmp_path / name
    destination.mkdir()
    write_json(destination / "manifest.json", {})
    return destination


def test_isolated_missing_stock_preserves_prediction_universe_and_reports_fallback(tmp_path):
    config = _fixture(tmp_path, missing_stock=True)
    destination = _destination(tmp_path, "missing-answer")
    execute_pipeline(config, destination)
    predictions = pd.read_parquet(destination / "predictions.parquet")
    assert len(predictions) == 200
    assert not predictions.duplicated(["date", "instrument"]).any()
    missing = predictions.instrument.eq("000099.SZ")
    model = LinearModel.load(destination / "model.npz")
    np.testing.assert_array_equal(predictions.loc[missing, "score"], [model.intercept] * 2)
    assert predictions.loc[~missing, "score"].nunique() > 1
    assert len(pd.read_parquet(destination / "scored_predictions.parquet")) == 198
    assert read_json(destination / "checks.json")["passed"]
    checks = {item["name"]: item for item in read_json(destination / "checks.json")["checks"]}
    assert checks["train_expected_universe_coverage"]["details"]["minimum"] == 0.99
    coverage = read_json(destination / "metrics.json")["coverage"]
    assert coverage["expected_prediction_samples"] == 200
    assert coverage["prediction_samples"] == 200
    assert coverage["scored_samples"] == 198
    assert coverage["missing_answer_samples"] == 2
    assert coverage["no_history_prediction_samples"] == 2
    assert coverage["no_history_prediction_policy"] == "training_mean"
    daily = pd.read_csv(destination / "daily.csv")
    assert daily["expected_sample_count"].tolist() == [200]
    assert daily["prediction_sample_count"].tolist() == [200]
    assert daily["missing_answer_count"].tolist() == [2]
    assert daily["no_history_prediction_count"].tolist() == [2]

    # Future answer availability and extreme values cannot alter any prediction.
    labels = pd.read_parquet(config.data.labels_path)
    labels = pd.concat([labels, predictions.loc[missing, ["date", "instrument"]].assign(target=12345.0)],
                       ignore_index=True)
    labels["label_end"] = labels.date + pd.Timedelta(minutes=30)
    labels.to_parquet(config.data.labels_path, index=False)
    changed = _destination(tmp_path, "answer-added")
    execute_pipeline(config, changed)
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(changed / "predictions.parquet"),
                                  check_exact=True)
    assert len(pd.read_parquet(changed / "scored_predictions.parquet")) == 200


@pytest.mark.parametrize("date,check_name", [
    ("2020-01-02 10:30", "train_expected_universe_coverage"),
    ("2020-01-03 10:30", "expected_universe_coverage"),
])
def test_entire_missing_answer_section_remains_visible_and_fails(tmp_path, date, check_name):
    config = _fixture(tmp_path)
    labels = pd.read_parquet(config.data.labels_path)
    labels.loc[~labels.date.eq(pd.Timestamp(date))].to_parquet(config.data.labels_path, index=False)
    destination = _destination(tmp_path, "missing-section")
    with pytest.raises(ValueError, match="Data acceptance failed"):
        execute_pipeline(config, destination)
    checks = {item["name"]: item for item in read_json(destination / "checks.json")["checks"]}
    assert checks[check_name]["passed"] is False
    assert checks[check_name]["details"]["minimum"] == 0.0
    assert checks[check_name]["details"]["missing_answers"] == 100
    assert not (destination / "metrics.json").exists()


def test_gap_exceeding_declared_tolerance_is_not_reported_successful(tmp_path):
    config = _fixture(tmp_path, missing_stock=True, minimum_coverage=0.995)
    destination = _destination(tmp_path, "below-coverage")
    with pytest.raises(ValueError, match="Data acceptance failed"):
        execute_pipeline(config, destination)
    checks = read_json(destination / "checks.json")
    assert checks["passed"] is False
    assert any(not item["passed"] for item in checks["checks"] if "coverage" in item["name"])
    assert not (destination / "model.npz").exists()


def test_no_history_training_answers_do_not_contaminate_the_fitted_model(tmp_path):
    config = _fixture(tmp_path, missing_stock=True)
    before = _destination(tmp_path, "training-without-answer")
    execute_pipeline(config, before)
    labels = pd.read_parquet(config.data.labels_path)
    universe = pd.read_parquet(config.data.instruments_path)
    missing_train_keys = universe.loc[universe.instrument.eq("000099.SZ")
                                      & universe.date.lt(pd.Timestamp("2020-01-03"))]
    extra = missing_train_keys.assign(target=99999.0,
                                     label_end=missing_train_keys.date + pd.Timedelta(minutes=30))
    pd.concat([labels, extra], ignore_index=True).to_parquet(config.data.labels_path, index=False)
    after = _destination(tmp_path, "training-without-history")
    execute_pipeline(config, after)
    pd.testing.assert_frame_equal(pd.read_parquet(before / "predictions.parquet"),
                                  pd.read_parquet(after / "predictions.parquet"), check_exact=True)
    original, changed = LinearModel.load(before / "model.npz"), LinearModel.load(after / "model.npz")
    assert original.intercept == changed.intercept
    np.testing.assert_array_equal(original.mean, changed.mean)
    np.testing.assert_array_equal(original.weights, changed.weights)
