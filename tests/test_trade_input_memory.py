"""Reduce temporary trade-input arrays without changing training or fail-closed inputs."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns, lob_trade_columns  # noqa: E402


class LegacyInputAdapter(TLOBAdapter):
    """Frozen pre-optimization transform-then-select path for numerical comparison."""

    def _input(self, batch, *, sample_mask=None):
        x = np.asarray(batch.x, dtype=np.float32)
        expected = (len(batch.keys), self.architecture["seq_size"], self.architecture["num_features"])
        if x.shape != expected or batch.observed.shape != x.shape or not np.isfinite(x).all():
            raise ValueError("Batch requires aligned finite preprocessed LOB windows")
        if self.new_input_contract:
            x = x.copy()
            if self.config["book_representation"] == "relative_mid_bps":
                prices = x[..., :self.lob_feature_count:2].astype(np.float64)
                best = prices[:, -1, :2]
                mid = best.mean(axis=1)[:, None, None]
                valid = (prices > 0) & (best > 0).all(axis=1)[:, None, None]
                relative = np.zeros_like(prices)
                np.divide(prices, mid, out=relative, where=valid)
                x[..., :self.lob_feature_count:2] = np.where(valid, (relative - 1) * 10000, 0)
            if self.architecture["num_features"] == 15:
                flows = x[..., 12:]
                if (flows < 0).any():
                    raise ValueError("Trade flow inputs must be nonnegative after preprocessing")
                x[..., 12:] = np.log1p(flows) if self.config["trade_features"] == "log1p" else 0
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite transformed LOB inputs")
        return x if sample_mask is None else x[sample_mask]


def fixture(trades=True):
    fields = lob_trade_columns() if trades else lob_columns()
    data = DataConfig(bars_path="unused.parquet", labels_path="unused.parquet", source="synthetic", version="1",
                      feature_columns=fields, lookback=8, min_history=2, sample_times=("10:00",))
    rng = np.random.default_rng(91)
    x = rng.uniform(1, 1000, size=(7, 8, len(fields))).astype(np.float32)
    x[..., :12:2] = rng.uniform(9, 11, size=(7, 8, 6))
    observed = np.ones_like(x, dtype=bool)
    observed[0, :, :12] = False
    observed[1, :-1, :12] = False
    keys = pd.DataFrame({"date": pd.date_range("2020-01-02 10:00", periods=7), "instrument": "A"})
    batch = Batch(keys, x, observed, np.full((7, 8), np.datetime64("NaT", "ns")), {}, 12)
    target = np.array([900, -700, -.03, -.01, 0, .02, .04], dtype=np.float32)
    return data, batch, target, batch.effective_rows.sum(axis=1) >= data.min_history


def models(data, representation="raw", trade_features="zero"):
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=2, max_steps=3,
                         batch_size=3, log_every_steps=1, device="cpu", target_scaling="train_std",
                         book_representation=representation, trade_features=trade_features)
    return tuple(cls(config, data, seed=11) for cls in (LegacyInputAdapter, TLOBAdapter))


@pytest.mark.parametrize("trades,representation,trade_features", [
    (True, "raw", "zero"), (True, "raw", "log1p"),
    (True, "relative_mid_bps", "zero"), (True, "relative_mid_bps", "log1p"),
    (False, "raw", "zero"), (False, "relative_mid_bps", "zero")])
def test_every_optimizer_update_and_prediction_matches_legacy(trades, representation, trade_features):
    data, batch, target, mask = fixture(trades)
    old, new = models(data, representation, trade_features)
    original = batch.x.copy(), batch.observed.copy(), target.copy()
    np.testing.assert_array_equal(old._input(batch, sample_mask=mask), new._input(batch, sample_mask=mask))
    histories, curves, progress = [], [], []
    for model in (old, new):
        updates, predictions = [], []

        class Recorder:
            def event(self, stage, event, **values):
                if stage == "training" and event == "progress":
                    updates.append({key: value.detach().clone() for key, value in model.network.state_dict().items()})

        model.recorder = Recorder()
        progress.append(model.fit(batch, target, sample_mask=mask, validation=(batch, target),
            observer=lambda adapter, epoch, step: predictions.append(adapter._predict(batch).copy())))
        histories.append(updates)
        curves.append(predictions)
    assert progress[0] == progress[1]
    assert len(histories[0]) == len(histories[1]) == 3
    for left, right in zip(histories[0], histories[1]):
        assert left.keys() == right.keys()
        for name in left:
            assert torch.equal(left[name], right[name]), name
    np.testing.assert_array_equal(curves[0], curves[1])
    np.testing.assert_array_equal(old.predict(batch), new.predict(batch))
    np.testing.assert_array_equal(new.predict(batch)[~mask], np.full((~mask).sum(), new.intercept))
    for actual, before in zip((batch.x, batch.observed, target), original):
        np.testing.assert_array_equal(actual, before)


def test_log1p_reuses_owned_selected_flow_storage(monkeypatch):
    data, batch, _, mask = fixture()
    _, model = models(data, trade_features="log1p")
    before = batch.x.copy()
    log1p, calls = np.log1p, []

    def observe(values, *, out=None):
        calls.append((values.shape, out is values, np.shares_memory(values, batch.x)))
        return log1p(values, out=out)

    monkeypatch.setattr(np, "log1p", observe)
    transformed = model._input(batch, sample_mask=mask)
    assert calls == [((int(mask.sum()), 8, 3), True, False)]
    assert transformed.flags.owndata and not np.shares_memory(transformed, batch.x)
    np.testing.assert_array_equal(batch.x, before)


@pytest.mark.parametrize("fault,message", [
    ("nonfinite_window", "aligned finite preprocessed"), ("negative_flow", "nonnegative"),
    ("nonfinite_target", "finite targets"), ("observed_shape", "aligned finite preprocessed"),
    ("relative_overflow", "Nonfinite transformed")])
def test_excluded_rows_still_receive_full_validation(fault, message):
    data, batch, target, mask = fixture()
    representation = "relative_mid_bps" if fault == "relative_overflow" else "raw"
    old, new = models(data, representation, "log1p")
    assert not mask[0]
    if fault == "nonfinite_window":
        batch.x[0, 0, 0] = np.nan
    elif fault == "negative_flow":
        batch.x[0, 0, 12] = -1
    elif fault == "nonfinite_target":
        target[0] = np.nan
    elif fault == "observed_shape":
        batch.observed = batch.observed[:-1]
    else:
        batch.x[0, :, :12:2] = np.finfo(np.float32).max
        batch.x[0, -1, [0, 2]] = np.finfo(np.float32).tiny
    for model in (old, new):
        with np.errstate(over="ignore"), pytest.raises(ValueError, match=message):
            model.fit(batch, target, sample_mask=mask)
        assert not model.trained
