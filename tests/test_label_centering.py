"""Full-timestamp label centering: full cross-section mean, raw std scale, relative inference."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402
from scripts.run_label_centering import assert_only_centering_changes, label_centering_stats  # noqa: E402

TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
TARGET = np.array([-.0040, -.0030, -.0020, -.0010, .0200, .0210, .0220, .0230], dtype=np.float32)


def setup(*, timestamps=2, per=4):
    dates = []
    for index in range(timestamps):
        dates.extend([pd.Timestamp("2020-01-02") + pd.Timedelta(minutes=30 * index)] * per)
    keys = pd.DataFrame({"date": pd.to_datetime(dates),
                         "instrument": [f"S{item}" for item in range(per)] * timestamps})
    x = np.random.default_rng(7).normal(size=(len(keys), 8, 12)).astype(np.float32)
    observed = np.ones_like(x, dtype=bool)
    history_dates = np.full((len(x), 8), np.datetime64("NaT", "ns"))
    batch = Batch(keys, x, observed, history_dates, {})
    data = DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic",
                      version="1", feature_columns=lob_columns(3), lookback=8, min_history=2,
                      sample_times=TIMES)
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=1, batch_size=4,
                         target_scaling="train_std", target_centering="timestamp_mean_train",
                         device="cpu", log_every_steps=100)
    target = np.resize(TARGET, len(keys)).astype(np.float32)
    return TLOBAdapter(config, data), batch, target, data


def full_means(target, per=4):
    return np.repeat(target.astype(np.float64).reshape(-1, per).mean(axis=1), per)


def test_centering_uses_full_timestamp_mean_and_raw_std():
    model, batch, target, _ = setup()
    mask = np.ones(len(target), dtype=bool)
    selected, stats = model._training_targets(batch.keys, target, mask)
    centered = target.astype(np.float64) - full_means(target)
    assert stats["timestamps"] == 2 and stats["rows"] == len(target)
    assert stats["raw_std"] == pytest.approx(target.astype(np.float64).std())
    assert stats["centered_std"] == pytest.approx(centered.std())
    assert stats["max_abs_centered_timestamp_mean"] < 1e-12
    np.testing.assert_array_equal(selected, centered.astype(np.float32))

    model.fit(batch, target)
    assert model.target_scale == pytest.approx(target.astype(np.float64).std())
    assert model.target_scale != pytest.approx(centered.std())
    assert model.intercept == pytest.approx(centered.mean(), abs=1e-8)


def test_mask_selects_rows_but_centering_uses_the_full_timestamp_cross_section():
    model, batch, target, _ = setup(timestamps=3)
    mask = np.arange(len(target)) >= 4
    selected, stats = model._training_targets(batch.keys, target, mask)
    centered = target.astype(np.float64) - full_means(target)
    assert stats["mask_excluded_rows"] == 4 and stats["timestamps"] == 3
    assert not np.isclose(centered[0], 0.0)
    np.testing.assert_array_equal(selected, centered[mask].astype(np.float32))


@pytest.mark.parametrize("filename", ["weights.pt", "weights.json"])
def test_centered_checkpoint_keeps_relative_scores_without_add_back(tmp_path, filename):
    model, batch, target, _ = setup()
    model.fit(batch, target)
    expected = model.predict(batch)
    path = tmp_path / filename
    (model.export_text if filename.endswith("json") else model.save)(path)
    restored = TLOBAdapter.load(path)
    assert restored.config["target_centering"] == "timestamp_mean_train"
    assert restored.target_centering_stats == model.target_centering_stats
    np.testing.assert_allclose(restored.predict(batch), expected, atol=1e-12, rtol=1e-9)
    payload = (torch.load(path, weights_only=True) if filename.endswith("pt")
               else json.loads(path.read_text(encoding="utf-8")))
    assert payload["prediction_units"] == "relative_return_score"
    assert "timestamp_means" not in payload
    with torch.inference_mode():
        parts = [model.network(torch.from_numpy(batch.x[offset:offset + model.config["batch_size"]])).cpu().numpy()
                 for offset in range(0, len(batch.x), model.config["batch_size"])]
    relative = np.concatenate(parts).astype(np.float64) * model.target_scale
    np.testing.assert_allclose(expected, relative, atol=1e-12, rtol=1e-9)


def test_plan_statistics_match_the_fitted_cross_section():
    model, batch, target, _ = setup()
    frame = pd.DataFrame({"date": batch.keys["date"], "instrument": batch.keys["instrument"],
                          "target": target})
    stats = label_centering_stats(frame)
    model.fit(batch, target)
    assert stats["timestamps"] == 2 and stats["rows"] == len(target)
    assert stats["raw_std"] == pytest.approx(model.target_scale)
    assert stats["max_abs_centered_timestamp_mean"] < 1e-12
    assert stats["centered_std"] == pytest.approx(model.target_centering_stats["centered_std"])


def test_default_is_off_and_unknown_centering_is_rejected():
    assert ModelConfig().target_centering == "none"
    with pytest.raises(ValueError):
        ModelConfig(target_centering="timestamp_mean_validation")


def test_plan_guard_accepts_only_the_centering_change():
    reference = {"experiment_id": "old", "hypothesis": "old", "output_root": "old",
                 "model": {"target_centering": "none", "target_scaling": "train_std", "max_steps": 1500},
                 "budget": {"wall_seconds": 3600.0, "max_rss_mb": 6144.0, "max_train_samples": 100000}}
    candidate = deepcopy(reference)
    candidate.update(experiment_id="new", hypothesis="new", output_root="new",
                     budget={"wall_seconds": 900.0, "max_rss_mb": 9472.0, "max_train_samples": 8})
    candidate["model"]["target_centering"] = "timestamp_mean_train"
    assert assert_only_centering_changes(reference, candidate, "window") is None
    candidate["model"]["target_scaling"] = "none"
    with pytest.raises(ValueError):
        assert_only_centering_changes(reference, candidate, "window")
