"""Streaming pool equivalence: per-batch windows must equal the materialised pool rows.

The training pool used to expand every window into one float32 tensor (24.9 GB for the
132-day window). ``jinyiquant.pool.StreamingPool`` keeps the sample keys and one canonical bar
matrix instead and builds windows per batch. These tests hold it to the only acceptance that
matters for training: the same keys in the same order must produce the same bytes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jinyiquant.data import build_batch, history_diagnostics, lob_trade_columns
from jinyiquant.pool import StreamingPool, training_pool

FEATURES = list(lob_trade_columns())


def config(**kwargs):
    defaults = dict(feature_columns=FEATURES, lookback=16, min_history=1, missing_policy="ffill",
                    max_ffill_steps=5, sample_times=[], price_scale=1.0, missing_sentinel=None,
                    source="streaming-pool-test", version="v1", history_days=1,
                    allowed_start="2020-01-01", allowed_end="2024-10-08", bars_path=None,
                    labels_path=None, instruments_path=None, instrument_map_path=None)
    return SimpleNamespace(**(defaults | kwargs))


def fixture(instruments: int = 6, minutes: int = 120, seed: int = 11) -> pd.DataFrame:
    """Bars with long missing runs, including gaps wider than the fill limit.

    Missing runs are deliberately contiguous so a window that straddles a gap is only
    reproducible if the fill is computed from the same raw bars, never from batch state.
    """
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2020-01-02 09:30", periods=minutes, freq="min")
    frames = []
    for name in [f"S{index:02d}" for index in range(instruments)]:
        values = rng.normal(size=(minutes, len(FEATURES))).cumsum(axis=0) + 1000.0
        values[rng.random((minutes, len(FEATURES))) < 0.02] = np.nan
        for start in (5, 21, 40, 77, 101):  # 8-row gaps: wider than max_ffill_steps=5
            values[start:start + 8, :9] = np.nan
        frames.append(pd.DataFrame({"date": stamps, "instrument": name,
                                    **dict(zip(FEATURES, values.T))}))
    return pd.concat(frames, ignore_index=True)


def keys(instruments: int = 6, first: int = 20, count: int = 20) -> pd.DataFrame:
    """Consecutive sample timestamps, so neighbouring keys share most of their window."""
    stamps = pd.date_range("2020-01-02 09:30", periods=120, freq="min")[first:first + count]
    return pd.DataFrame([{"date": stamp, "instrument": f"S{number:02d}"}
                         for stamp in stamps for number in range(instruments)])


def materialised_and_streamed(**kwargs):
    settings = config(**kwargs)
    bars, source = fixture(), keys()
    return settings, build_batch(bars, source, settings), StreamingPool.build(bars, source, settings)


def draws(size: int = 8, count: int = 50, total: int = 120, seed: int = 1):
    """Sampler-shaped draws: one permutation, chunked, never re-sorted.

    ``TLOBAdapter.fit`` draws ``torch.randperm(len(y))`` once per epoch and hands over
    ``order[offset:offset + batch_size]`` unchanged, so a real batch is an unsorted slice of a
    permutation. Sorting here would weaken the test: the pool would only have to agree row-wise,
    not in the order the optimizer actually sees.
    """
    order = np.random.default_rng(seed).permutation(total)
    return [order[start:start + size] for start in range(0, size * count, size)]


@pytest.mark.parametrize("policy,limit,minimum,lookback", [
    ("ffill", 5, 1, 128),
    ("ffill", 2, 3, 32),
    ("ffill", 0, 1, 16),
    ("zero", 0, 1, 16),
])
def test_streaming_batches_are_bitwise_identical_to_the_materialised_pool(policy, limit, minimum, lookback):
    settings, materialised, pool = materialised_and_streamed(
        missing_policy=policy, max_ffill_steps=limit, min_history=minimum, lookback=lookback)
    for positions in draws(total=len(materialised.keys)):
        batch = pool.batch(positions)
        assert batch.keys.equals(materialised.keys.iloc[positions].reset_index(drop=True))
        assert batch.x.tobytes() == materialised.x[positions].tobytes()
        np.testing.assert_array_equal(batch.observed, materialised.observed[positions])
        np.testing.assert_array_equal(batch.history_dates, materialised.history_dates[positions])
        assert batch.history_feature_count == materialised.history_feature_count
        assert batch.coverage == materialised.coverage


def test_streaming_keys_keep_the_materialised_order_and_the_sampler_draw():
    """The pool may not reorder keys: its key table is the materialised one, row for row."""
    _, materialised, pool = materialised_and_streamed()
    assert pool.keys.equals(materialised.keys)
    assert np.array_equal(pool.keys["date"].to_numpy(), materialised.keys["date"].to_numpy())
    assert pool.keys["instrument"].tolist() == materialised.keys["instrument"].tolist()
    for positions in draws(count=10, total=len(materialised.keys), seed=9):
        assert pool.batch(positions).keys["instrument"].tolist() == \
            materialised.keys.iloc[positions]["instrument"].tolist()


def test_streaming_coverage_and_fit_mask_match_the_materialised_pool():
    _, materialised, pool = materialised_and_streamed(min_history=2)
    assert pool.coverage == materialised.coverage
    assert pool.fit_mask.tolist() == (materialised.effective_rows.sum(axis=1) >= 2).tolist()
    assert pool.history_diagnostics().equals(history_diagnostics(materialised))


def test_streamed_batches_survive_missing_gaps_that_span_batch_boundaries():
    """A gap shared by neighbouring keys must fill identically in whichever batch it lands."""
    _, materialised, pool = materialised_and_streamed(missing_policy="ffill", max_ffill_steps=5)
    keys_by_date = pool.keys.groupby("date").groups
    neighbouring = [date for date in keys_by_date
                    if (date - pd.Timedelta(minutes=1)) in keys_by_date]
    assert neighbouring, "fixture must contain consecutive timestamps whose windows overlap"
    filled = (~materialised.observed) & (materialised.x != 0)
    assert filled.any(), "fixture must contain values only a forward fill can produce"
    for date in neighbouring:
        batch = np.asarray(keys_by_date[date])
        previous = np.asarray(keys_by_date[date - pd.Timedelta(minutes=1)])
        assert pool.batch(batch).x.tobytes() == materialised.x[batch].tobytes()
        assert pool.batch(previous).x.tobytes() == materialised.x[previous].tobytes()


def test_the_equality_assertion_is_not_vacuous():
    """Mutating the index or the key order must break the comparison, or it proves nothing."""
    _, materialised, pool = materialised_and_streamed()
    positions = np.arange(8)
    assert pool.batch(positions).x.tobytes() == materialised.x[positions].tobytes()

    pool.codes, pool.begin, pool.stop = pool.codes[::-1].copy(), pool.begin[::-1].copy(), pool.stop[::-1].copy()
    assert pool.batch(positions).x.tobytes() != materialised.x[positions].tobytes()

    _, fresh_materialised, fresh_pool = materialised_and_streamed()
    fresh_pool.keys = fresh_pool.keys.iloc[[1, 0, *range(2, len(fresh_pool.keys))]].reset_index(drop=True)
    assert not fresh_pool.batch(positions).keys.equals(
        fresh_materialised.keys.iloc[positions].reset_index(drop=True))

    _, third_materialised, third_pool = materialised_and_streamed()
    third_pool.begin = third_pool.begin + 1
    assert third_pool.batch(positions).x.tobytes() != third_materialised.x[positions].tobytes()


def test_default_training_path_stays_materialised():
    settings, materialised, _ = materialised_and_streamed()
    pool = training_pool(fixture(), keys(), settings)
    assert type(pool) is not StreamingPool
    assert pool.x.shape == materialised.x.shape
    assert type(training_pool(fixture(), keys(), settings, streaming=True)) is StreamingPool


def test_streaming_batch_validates_its_draw_and_approved_times():
    _, _, pool = materialised_and_streamed()
    with pytest.raises(ValueError, match="1-D integer"):
        pool.batch(np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="1-D integer"):
        pool.batch(np.array([[0, 1]]))
    with pytest.raises(ValueError, match="outside the pool"):
        pool.batch(np.array([len(pool.keys)]))
    with pytest.raises(ValueError, match="outside the pool"):
        pool.batch(np.array([-1]))

    settings = config(sample_times=["09:31", "09:32"])
    bars = fixture()
    source = keys(first=1, count=2)          # 09:31 and 09:32
    approved = build_batch(bars, source, settings)
    streamed = StreamingPool.build(bars, source, settings)
    assert streamed.batch(np.arange(4)).x.tobytes() == approved.x[:4].tobytes()
    streamed.keys = streamed.keys.assign(date=streamed.keys.date + pd.Timedelta(hours=3))
    with pytest.raises(ValueError, match="unapproved sampling time"):
        streamed.batch(np.arange(4))


def test_streaming_pool_reports_a_bounded_footprint():
    _, materialised, pool = materialised_and_streamed(lookback=128)
    window_bytes = materialised.x.nbytes + materialised.observed.nbytes + materialised.history_dates.nbytes
    pool_bytes = sum(pool.nbytes().values())
    assert pool_bytes < window_bytes / 10, (pool_bytes, window_bytes)
    assert "bars" in pool.nbytes() and "window_geometry" in pool.nbytes()


def training_case(instruments: int = 6, days: int = 4, seed: int = 5):
    """Bars over consecutive days plus 10:00 sample keys: the recipe's shape, 16-bar windows."""
    rng = np.random.default_rng(seed)
    frames, sample_keys = [], []
    for day in range(days):
        date = pd.Timestamp("2020-01-02") + pd.Timedelta(days=day)
        stamps = pd.date_range(date + pd.Timedelta(hours=9, minutes=30), periods=60, freq="min")
        for number in range(instruments):
            values = rng.normal(size=(len(stamps), len(FEATURES))).cumsum(axis=0) + 1000.0
            values[rng.random(values.shape) < .01] = np.nan
            frames.append(pd.DataFrame({"date": stamps, "instrument": f"S{number:02d}",
                                        **dict(zip(FEATURES, values.T))}))
            sample_keys.append({"date": date + pd.Timedelta(hours=10), "instrument": f"S{number:02d}"})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(sample_keys)


def test_streamed_training_reproduces_the_materialised_trajectory():
    """``fit`` on a ``StreamingPool`` must reproduce the materialised run bit for bit.

    The switch is the pool type: a materialised ``Batch`` indexes its prefetched window tensor,
    a ``StreamingPool`` builds the windows of each sampler draw. Steps, loss trace, fitted
    transforms and every parameter have to match, and shifting the pool's window geometry must
    break that, or the comparison proves nothing.
    """
    torch = pytest.importorskip("torch")
    from jinyiquant.config import DataConfig, ModelConfig
    from modified_tlob.adapter import TLOBAdapter

    settings = DataConfig(bars_path="unused", labels_path="unused", source="streaming-training-test",
                          version="v1", feature_columns=lob_trade_columns(), lookback=16, min_history=1,
                          history_days=1, sample_times=("10:00",))
    bars, source = training_case()
    materialised, pool = build_batch(bars, source, settings), StreamingPool.build(bars, source, settings)
    assert pool.keys.equals(materialised.keys)
    target = np.random.default_rng(7).normal(scale=.01, size=len(source)).astype(np.float32)
    mask = np.zeros(len(source), dtype=bool)
    mask[::3] = True            # a strict subset, so the sampler's masked space != the pool space
    model_config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=2,
                               epochs=3, device="cpu", log_every_steps=1, target_scaling="train_std")

    def run(batch):
        model = TLOBAdapter(model_config, settings, seed=42)
        events = []
        model.recorder = SimpleNamespace(event=lambda stage, event, **values: events.append((event, values)))
        progress = model.fit(batch, target, sample_mask=mask, step_budget=10)
        state = {name: value.detach().cpu().contiguous().numpy().tobytes()
                 for name, value in model.network.state_dict().items()}
        return model, progress, [values["loss"] for event, values in events if event == "progress"], state

    left, left_progress, left_losses, left_state = run(materialised)
    draws, draw_window = [], pool.batch

    def counting(indices):
        draws.append(np.asarray(indices).copy())
        return draw_window(indices)

    pool.batch = counting
    right, right_progress, right_losses, right_state = run(pool)
    assert left_progress == right_progress and left_progress["steps"] == 10
    assert len(right_losses) == 10 and left_losses == right_losses
    assert left_state.keys() == right_state.keys()
    differing = [name for name in left_state if left_state[name] != right_state[name]]
    assert not differing, differing
    np.testing.assert_array_equal(left.predict(materialised), right.predict(materialised))
    # One draw for the compile probe plus one per step, each mapped through the sample mask.
    selected = np.flatnonzero(mask)
    assert len(draws) == 11 and np.array_equal(draws[0], selected[:2])
    assert all(set(draw) <= set(selected) and len(draw) <= 2 for draw in draws)

    # The same code on a pool whose windows are one bar later must train a different model.
    pool.begin = pool.begin + 1
    _, shifted_progress, shifted_losses, shifted_state = run(pool)
    assert shifted_losses != left_losses and shifted_progress["steps"] == 10
    assert [name for name in left_state if left_state[name] != shifted_state[name]]

