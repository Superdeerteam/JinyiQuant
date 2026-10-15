"""Train-only intraday weights, raw-unit reporting, and inference independence."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")


def setup(*, target_scaling="train_std", days=2, recorder=None):
    dates = pd.to_datetime([f"2020-01-{day:02d} {time}" for day in range(2, 2 + days) for time in TIMES])
    keys = pd.DataFrame({"date": dates, "instrument": "A"})
    x = np.random.default_rng(31).normal(size=(len(keys), 8, 12)).astype(np.float32)
    batch = Batch(keys, x, np.ones_like(x, dtype=bool), np.full((len(x), 8), np.datetime64("NaT", "ns")), {})
    data = DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic", version="1",
                      feature_columns=lob_columns(3), lookback=8, min_history=2, sample_times=TIMES)
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=1, batch_size=32,
                         target_scaling=target_scaling, time_loss_weighting="train_variance", device="cpu",
                         log_every_steps=1)
    model = TLOBAdapter(config, data, recorder=recorder)
    amplitude = np.arange(1, 9, dtype=np.float32) * .001
    target = np.concatenate([-amplitude, amplitude] + [amplitude * 100] * (days - 2))
    return model, batch, target


@pytest.mark.parametrize("target_scaling", ["none", "train_std"])
def test_weighted_update_matches_formula_and_logs_unweighted_original_loss(target_scaling):
    events = []
    recorder = SimpleNamespace(event=lambda stage, event, **context: events.append((stage, event, context)))
    model, batch, target = setup(target_scaling=target_scaling, recorder=recorder)
    reference, _, _ = setup(target_scaling=target_scaling)
    variance = target.astype(np.float64).var()
    by_time_variance = np.array([target[[i, i + 8]].astype(np.float64).var() for i in range(8)])
    inverse = 1 / (by_time_variance + .05 * variance)
    weights = np.tile(inverse / inverse.mean(), 2).astype(np.float32)
    scale = target.astype(np.float64).std() if target_scaling == "train_std" else 1.0
    scaled = (target.astype(np.float64) / scale).astype(np.float32)
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(model.seed)).numpy()
    optimizer = torch.optim.Adam(reference.network.parameters(), lr=model.config["learning_rate"])
    prediction = reference.network(torch.from_numpy(batch.x[order]))
    errors = (prediction - torch.from_numpy(scaled[order])).square()
    expected_raw = errors.mean().item() * scale ** 2
    weighted = (torch.from_numpy(weights[order]) * errors).sum() / torch.from_numpy(weights[order]).sum()
    expected_weighted = weighted.item() * scale ** 2
    weighted.backward()
    optimizer.step()
    progress = model.fit(batch, target, validation=(batch, target))
    assert progress["loss"] == pytest.approx(expected_raw)
    assert progress["weighted_loss"] == pytest.approx(expected_weighted)
    assert expected_raw != pytest.approx(expected_weighted)
    assert progress["optimization_loss"] == pytest.approx(weighted.item())
    for name, value in model.network.state_dict().items():
        torch.testing.assert_close(value, reference.network.state_dict()[name], atol=0, rtol=0)
    stats = model.time_loss_weighting_stats
    assert stats["samples"] == 16 and stats["days"] == 2
    assert stats["epsilon"] == pytest.approx(.05 * variance)
    for i, row in enumerate(stats["by_time"].values()):
        assert row["samples"] == row["days"] == 2
        assert row["weight"] == pytest.approx(weights[i])
    assert sum(row["samples"] * row["weight"] for row in stats["by_time"].values()) == pytest.approx(16)
    for stage, event, context in events:
        if stage == "training" and event in ("progress", "epoch_completed"):
            assert context["loss"] == pytest.approx(expected_raw)
            assert context["weighted_loss"] == pytest.approx(expected_weighted)
        elif stage == "validation" and event == "epoch_completed":
            assert context["loss"] == pytest.approx(np.square(model.predict(batch) - target).mean(), rel=1e-6)


def test_weights_and_updates_ignore_excluded_and_validation_targets():
    first, batch, target = setup(days=3)
    second, _, _ = setup(days=3)
    mask = np.arange(len(target)) < 16
    changed = target.copy()
    changed[~mask] = -1000
    first.fit(batch, target, sample_mask=mask, validation=(batch, target))
    second.fit(batch, changed, sample_mask=mask, validation=(batch, target + 100))
    assert first.time_loss_weighting_stats == second.time_loss_weighting_stats
    assert first.time_loss_weighting_stats["samples"] == 16
    assert first.time_loss_weighting_stats["days"] == 2
    for name, value in first.network.state_dict().items():
        torch.testing.assert_close(value, second.network.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("constant", [False, True])
def test_equal_variances_and_constant_targets_have_unit_weights(constant):
    model, batch, target = setup()
    target[:] = 0 if constant else np.repeat([-.003, .003], 8)
    progress = model.fit(batch, target)
    stats = model.time_loss_weighting_stats
    assert stats["constant_target_fallback"] is constant
    assert all(row["weight"] == pytest.approx(1) for row in stats["by_time"].values())
    assert np.isfinite(progress["loss"]) and np.isfinite(model.predict(batch)).all()


def test_missing_configured_training_time_fails_after_mask():
    model, batch, target = setup()
    mask = np.arange(len(target)) % 8 != 0
    with pytest.raises(ValueError, match="every configured training sample time"):
        model.fit(batch, target, sample_mask=mask)
    assert not model.trained
    with pytest.raises(ValueError):
        ModelConfig(time_loss_weighting="validation_variance")


@pytest.mark.parametrize("filename", ["weights.pt", "weights.json"])
def test_weight_metadata_roundtrip_and_inference_does_not_use_weights(tmp_path, filename):
    model, batch, target = setup()
    model.fit(batch, target)
    path = tmp_path / filename
    (model.export_text if filename.endswith("json") else model.save)(path)
    restored = TLOBAdapter.load(path)
    assert restored.time_loss_weighting_stats == model.time_loss_weighting_stats
    assert restored.training_sample_times == TIMES
    np.testing.assert_array_equal(restored.predict(batch), model.predict(batch))
    restored.time_loss_weighting_stats = None
    restored.training_sample_times = ()
    np.testing.assert_array_equal(restored.predict(batch), model.predict(batch))
