"""Fixed ten-trading-day recency weights: formula, normalization and unchanged sampling."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import DATE_DECAY_HALF_LIFE_TRADING_DAYS, TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")


def setup(*, days=40, weighting="date_decay_10d", batch_size=4):
    clock = pd.to_timedelta([f"{time}:00" for time in TIMES])
    dates = np.repeat(pd.bdate_range("2020-01-02", periods=days).to_numpy(), len(TIMES)) + np.tile(clock, days)
    keys = pd.DataFrame({"date": pd.to_datetime(dates), "instrument": "A"})
    x = np.random.default_rng(31).normal(size=(len(keys), 8, 12)).astype(np.float32)
    batch = Batch(keys, x, np.ones_like(x, dtype=bool),
                  np.full((len(x), 8), np.datetime64("NaT", "ns")), {})
    data = DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic", version="1",
                      feature_columns=lob_columns(3), lookback=8, min_history=2, sample_times=TIMES)
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=1, batch_size=batch_size,
                         target_scaling="train_std", time_loss_weighting=weighting, device="cpu",
                         log_every_steps=100)
    model = TLOBAdapter(config, data)
    target = np.linspace(-.004, .004, len(keys), dtype=np.float32)
    return model, batch, target


def test_fixed_halflife_mean_one_and_recent_ten_day_share():
    model, batch, target = setup()
    progress = model.fit(batch, target)
    stats = model.time_loss_weighting_stats
    assert stats["method"] == "date_decay_10d"
    assert stats["half_life_trading_days"] == DATE_DECAY_HALF_LIFE_TRADING_DAYS == 10
    assert stats["days"] == 40 and stats["samples"] == 320
    assert [row["age_trading_days"] for row in stats["by_date"]] == list(range(39, -1, -1))
    weights = np.asarray([row["weight"] for row in stats["by_date"]], dtype=np.float64)
    raw_mean = np.mean(2.0 ** (-np.arange(39, -1, -1) / 10))
    assert weights[-1] == pytest.approx(1.0 / raw_mean)
    assert weights[0] / weights[-1] == pytest.approx(2.0 ** (-39 / 10))
    assert np.repeat(weights, len(TIMES)).mean() == pytest.approx(1.0)
    assert stats["recent_weight_share"] == pytest.approx(np.repeat(weights, len(TIMES))[-80:].sum() / 320)
    assert stats["recent_weight_share"] == pytest.approx(0.533, abs=5e-4)
    assert stats["mean_weight"] == pytest.approx(1.0)
    assert np.isfinite(progress["weighted_loss"]) and progress["weighted_loss"] > 0
    assert progress["optimizer"] == "Adam_date_weighted_MSE"
    assert model.target_scale == pytest.approx(float(target.astype(np.float64).std()))


def test_unit_weights_reproduce_unweighted_training_and_batch_order():
    plain, batch, target = setup(days=1, weighting="none")
    weighted, _, _ = setup(days=1, weighting="date_decay_10d")
    left, right = plain.fit(batch, target), weighted.fit(batch, target)
    assert set(right) >= {"loss", "weighted_loss", "optimization_loss", "samples_seen", "steps"}
    assert right["steps"] == left["steps"] == 2
    assert right["samples_seen"] == left["samples_seen"] == 8
    assert right["loss"] == pytest.approx(left["loss"])
    assert right["weighted_loss"] == pytest.approx(left["weighted_loss"])
    assert weighted.time_loss_weighting_stats["days"] == 1
    assert weighted.time_loss_weighting_stats["by_date"][0]["weight"] == pytest.approx(1.0)
    for name, value in weighted.network.state_dict().items():
        torch.testing.assert_close(value, plain.network.state_dict()[name], atol=1e-8, rtol=1e-6)


def test_ages_start_from_the_selected_pool_not_the_full_pool():
    model, batch, target = setup(days=3)
    mask = np.arange(len(target)) >= 16
    model.fit(batch, target, sample_mask=mask)
    stats = model.time_loss_weighting_stats
    assert stats["days"] == 1 and stats["samples"] == 8
    assert stats["by_date"][0]["age_trading_days"] == 0
    assert stats["by_date"][0]["weight"] == pytest.approx(1.0)
    assert stats["recent_weight_share"] == pytest.approx(1.0)


@pytest.mark.parametrize("filename", ["weights.pt", "weights.json"])
def test_date_weight_metadata_roundtrip(tmp_path, filename):
    model, batch, target = setup(days=2)
    model.fit(batch, target)
    path = tmp_path / filename
    (model.export_text if filename.endswith("json") else model.save)(path)
    restored = TLOBAdapter.load(path)
    assert restored.time_loss_weighting_stats == model.time_loss_weighting_stats
    assert restored.config["time_loss_weighting"] == "date_decay_10d"
    np.testing.assert_array_equal(restored.predict(batch), model.predict(batch))


def test_other_halflives_are_not_configurable():
    with pytest.raises(ValueError):
        ModelConfig(time_loss_weighting="date_decay_5d")
