"""Same-timestamp grouped sampling and the MSE plus pairwise-hinge ranking objective."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import (  # noqa: E402
    TLOBAdapter,
    grouped_epoch_batches,
    pairwise_hinge_ranking_loss,
    timestamp_groups,
)
from modified_tlob.schema import lob_columns  # noqa: E402

TIMES = ("10:00", "10:30")
LOOKBACK = 8


def keys_and_batch(timestamps=4, per=10, seed=3):
    dates = []
    for index in range(timestamps):
        dates.extend([pd.Timestamp("2020-01-02 10:00") + pd.Timedelta(minutes=30 * index)] * per)
    keys = pd.DataFrame({"date": pd.to_datetime(dates),
                         "instrument": [f"S{item}" for item in range(per)] * timestamps})
    x = np.random.default_rng(seed).normal(size=(len(keys), LOOKBACK, 12)).astype(np.float32)
    batch = Batch(keys, x, np.ones_like(x, dtype=bool),
                  np.full((len(x), LOOKBACK), np.datetime64("NaT", "ns")), {})
    target = np.random.default_rng(seed + 1).normal(size=len(keys)).astype(np.float32) * 1e-4
    return batch, target


def data_config():
    return DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic",
                      version="1", feature_columns=lob_columns(3), lookback=LOOKBACK, min_history=1,
                      sample_times=TIMES)


def model_config(**changes):
    values = dict(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=2, batch_size=4, max_steps=6,
                  target_scaling="train_std", target_centering="timestamp_zscore_train_rms_matched",
                  sampling="timestamp_grouped", device="cpu", log_every_steps=1)
    values.update(changes)
    return ModelConfig(**values)


def test_timestamp_groups_hold_one_timestamp_and_skip_masked_rows():
    batch, _ = keys_and_batch(timestamps=3, per=5)
    mask = np.ones(len(batch.keys), dtype=bool)
    mask[[0, 7, 14]] = False
    groups = timestamp_groups(batch.keys, mask)
    assert len(groups) == 3
    for group, timestamp in zip(groups, sorted(pd.unique(batch.keys["date"]))):
        assert set(batch.keys["date"].iloc[group]) == {timestamp}
        assert len(group) == 4
    assert sorted(np.concatenate(groups).tolist()) == np.flatnonzero(mask).tolist()


def test_grouped_epoch_batches_are_exactly_sized_same_timestamp_and_deterministic():
    batch, _ = keys_and_batch(timestamps=4, per=10)
    mask = np.ones(len(batch.keys), dtype=bool)
    groups = timestamp_groups(batch.keys, mask)
    first = grouped_epoch_batches(groups, 4, torch.Generator().manual_seed(5))
    second = grouped_epoch_batches(groups, 4, torch.Generator().manual_seed(5))
    assert len(first) == 8  # four timestamps x two full batches; the two-row tail is dropped
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
        assert len(left) == 4
        assert batch.keys["date"].iloc[left].nunique() == 1
    used = np.concatenate(first)
    assert len(np.unique(used)) == len(used) == 32
    assert set(used.tolist()) <= set(np.flatnonzero(mask).tolist())


def test_pairwise_hinge_is_zero_for_ordered_scores_and_positive_when_reversed():
    targets = torch.tensor([-1.0, 0.0, 1.0])
    assert pairwise_hinge_ranking_loss(torch.tensor([-1.0, 0.0, 1.0]), targets).item() == 0.0
    assert pairwise_hinge_ranking_loss(torch.tensor([1.0, 0.0, -1.0]), targets).item() > 0.0


def test_pairwise_hinge_gradient_raises_the_score_of_the_better_stock():
    scores = torch.tensor([0.0, 0.3], requires_grad=True)
    targets = torch.tensor([1.0, 0.0])
    loss = pairwise_hinge_ranking_loss(scores, targets)
    loss.backward()
    assert scores.grad[0] < 0 and scores.grad[1] > 0
    before = loss.item()
    with torch.no_grad():
        scores -= 0.1 * scores.grad
    assert pairwise_hinge_ranking_loss(scores, targets).item() < before


def test_config_rejects_ranking_without_grouped_sampling():
    with pytest.raises(ValueError):
        model_config(sampling="shuffle", ranking_loss="pairwise_hinge", ranking_loss_weight=1.0)
    with pytest.raises(ValueError):
        model_config(ranking_loss_weight=1.0)
    with pytest.raises(ValueError):
        model_config(ranking_loss="pairwise_hinge", ranking_loss_weight=0.0)
    with pytest.raises(ValueError):
        model_config(ranking_loss="pairwise_hinge", ranking_loss_weight=1.0, batch_size=3)


def test_grouped_fit_logs_objective_terms_and_keeps_controls_identical():
    batch, target = keys_and_batch(timestamps=4, per=10)
    events = []
    ranking_model = TLOBAdapter(model_config(ranking_loss="pairwise_hinge", ranking_loss_weight=1.0),
                                data_config(), seed=11)
    ranking_model.recorder = type("Recorder", (), {"event": staticmethod(
        lambda stage, event, **values: events.append((event, values)))})()
    progress = ranking_model.fit(batch, target)
    assert progress["steps"] == 6 and progress["sampling"] == "timestamp_grouped"
    assert progress["ranking_loss_weight"] == 1.0 and progress["ranking_loss"] is not None
    assert np.isfinite(progress["ranking_loss"]) and progress["ranking_loss"] > 0
    assert progress["objective_loss"] > progress["optimization_loss"]
    assert any(name == "grouped_sampling_fitted" for name, _ in events)
    log_rows = [values for name, values in events if name == "progress"]
    assert log_rows and all(row["ranking_loss"] is not None for row in log_rows)

    control = TLOBAdapter(model_config(ranking_loss="none", ranking_loss_weight=0.0), data_config(), seed=11)
    control_progress = control.fit(batch, target)
    assert control_progress["ranking_loss"] is None
    assert control_progress["objective_loss"] == pytest.approx(control_progress["optimization_loss"])


def test_grouped_ranking_checkpoint_round_trips_through_text(tmp_path):
    batch, target = keys_and_batch(timestamps=4, per=10)
    model = TLOBAdapter(model_config(ranking_loss="pairwise_hinge", ranking_loss_weight=1.0),
                        data_config(), seed=11)
    model.fit(batch, target)
    path = tmp_path / "modified_tlob_weights.json"
    model.export_text(path)
    loaded = TLOBAdapter.load(path, device="cpu")
    assert loaded.config["sampling"] == "timestamp_grouped"
    assert loaded._loss_label() == "MSE+1x_pairwise_hinge"
    assert np.allclose(model.predict(batch), loaded.predict(batch), atol=1e-6, rtol=1e-5)
