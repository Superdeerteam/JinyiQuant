"""Opening-residual contracts: zero-initialised identity, 09:30-only gating, frozen trunk, scale."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import (  # noqa: E402
    TLOBAdapter,
    apply_frozen_timestamp_zscore,
    frozen_stats_sha256,
    open_market_mask,
    timestamp_zscore_train_rms_matched,
)
from modified_tlob.model import ModifiedTLOB  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

torch.set_num_threads(1)
OPENING = np.array([True, True, True, False, False, False])


def data_config(levels=3):
    return DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic", version="1",
                      feature_columns=lob_columns(levels), lookback=8, min_history=2, history_days=1,
                      sample_times=("09:30", "10:00"))


def batch(levels=3):
    """Two sections x three instruments; the first three keys are 09:30."""
    rng = np.random.default_rng(31)
    x = rng.normal(size=(6, 8, 4 * levels)).astype(np.float32)
    keys = pd.DataFrame({"date": pd.to_datetime(
        ["2023-08-01 09:30"] * 3 + ["2023-08-01 10:00"] * 3),
        "instrument": ["A", "B", "C", "A", "B", "C"]})
    return Batch(keys, x, np.ones_like(x, dtype=bool), np.full((6, 8), np.datetime64("NaT", "ns")), {})


def adapter():
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=3, epochs=1,
                         device="cpu", log_every_steps=1, target_scaling="train_std")
    return TLOBAdapter(config, data_config())


def target():
    return np.linspace(-.01, .01, 6, dtype=np.float32)


def fitted():
    model, b = adapter(), batch()
    model.fit(b, target())
    return model, b


def test_forward_features_width_and_output_equivalence():
    model = ModifiedTLOB()  # V2 architecture: hidden 40, 128 bars, 12 book columns
    assert model.feature_dim == 320
    x = torch.zeros(2, 128, 12)
    features = model.forward_features(x)
    assert features.shape == (2, 320)
    torch.testing.assert_close(model.forward_output(features), model(x), atol=0, rtol=0)
    with pytest.raises(ValueError):
        model.forward_features(torch.zeros(1, 8, 12))


def test_zero_initialised_head_reproduces_frozen_trunk_exactly():
    model, b = fitted()
    baseline = model.predict(b)
    model.attach_open_residual()
    np.testing.assert_array_equal(model.predict(b), baseline)
    with torch.inference_mode():
        correction = model.open_residual_head(torch.from_numpy(model._trunk_features(model._input(b))))
    assert torch.equal(correction, torch.zeros(6))


def test_opening_marker_parses_exchange_local_time():
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-08-01 09:30", "2023-08-01 10:00"]),
                         "instrument": ["A", "B"]})
    np.testing.assert_array_equal(open_market_mask(keys), [True, False])
    aware = keys.assign(date=pd.to_datetime(keys.date).dt.tz_localize("Asia/Shanghai"))
    with pytest.raises(ValueError, match="timezone-naive"):
        open_market_mask(aware)
    with pytest.raises(ValueError, match="HH:MM"):
        open_market_mask(keys, "0930")


def test_only_opening_rows_are_corrected_in_scaled_target_units():
    model, b = fitted()
    baseline = model.predict(b)
    model.attach_open_residual()
    progress = model.fit_open_residual(b, target() / model.target_scale, max_steps=25)
    after = model.predict(b)
    np.testing.assert_array_equal(after[~OPENING], baseline[~OPENING])
    assert not np.array_equal(after[OPENING], baseline[OPENING])
    assert progress["final_mse"] < progress["initial_mse"] and progress["steps"] == 25
    with torch.inference_mode():
        correction = model.open_residual_head(
            torch.from_numpy(model._trunk_features(model._input(b))[OPENING])).numpy().astype(np.float64)
    np.testing.assert_allclose(after[OPENING] - baseline[OPENING], correction * model.target_scale,
                               atol=1e-12, rtol=1e-5)  # float32 correction addition, float64 rescale


def test_training_rejects_non_opening_or_history_less_samples():
    model, b = fitted()
    model.attach_open_residual()
    with pytest.raises(ValueError, match="09:30"):
        model.fit_open_residual(b, target(), sample_mask=~OPENING, max_steps=1)
    closed = Batch(b.keys[~OPENING].reset_index(drop=True), b.x[~OPENING], b.observed[~OPENING],
                   b.history_dates[~OPENING], {})
    with pytest.raises(ValueError, match="09:30"):
        model.fit_open_residual(closed, target()[:3], max_steps=1)
    b.observed[OPENING] = False
    with pytest.raises(ValueError, match="09:30"):
        model.fit_open_residual(b, target(), max_steps=1)


def test_trunk_stays_frozen_and_parameter_isolation_holds():
    model, b = fitted()
    before = {name: tensor.clone() for name, tensor in model.network.state_dict().items()}
    model.attach_open_residual()
    assert all(parameter.requires_grad for parameter in model.open_residual_head.parameters())
    model.fit_open_residual(b, target() / model.target_scale, max_steps=10)
    for name, tensor in model.network.state_dict().items():
        assert torch.equal(tensor, before[name]), name
    assert not any(parameter.grad is not None for parameter in model.network.parameters())


@pytest.mark.parametrize("filename", ["model.pt", "weights.json"])
def test_checkpoint_roundtrip_preserves_trunk_and_head(tmp_path, filename):
    model, b = fitted()
    model.attach_open_residual()
    model.fit_open_residual(b, target() / model.target_scale, max_steps=10)
    expected = model.predict(b)
    trunk = {name: tensor.clone() for name, tensor in model.network.state_dict().items()}
    path = tmp_path / filename
    model.save(path) if path.suffix == ".pt" else model.export_text(path)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else torch.load(
        path, weights_only=True)
    assert payload["schema_version"] == 4 and set(payload["state_dict"]) == set(trunk)
    restored = TLOBAdapter.load(path, device="cpu")
    assert restored.open_residual_head is not None
    assert restored.open_residual_metadata["sample_time"] == "09:30"
    assert restored.open_residual_metadata["base_state_sha256"] == payload["open_residual"]["base_state_sha256"]
    np.testing.assert_array_equal(restored.predict(b), expected)
    for name, tensor in restored.network.state_dict().items():
        assert torch.equal(tensor, trunk[name]), name


def test_checkpoint_rejects_head_from_another_trunk(tmp_path):
    model, b = fitted()
    model.attach_open_residual()
    model.fit_open_residual(b, target() / model.target_scale, max_steps=5)
    path = tmp_path / "weights.json"
    model.export_text(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["open_residual"]["base_state_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen trunk"):
        TLOBAdapter.load(path)


def test_legacy_checkpoint_without_head_still_loads(tmp_path):
    model, b = fitted()
    expected = model.predict(b)
    path = tmp_path / "legacy.json"
    model.export_text(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2 and "open_residual" not in payload
    restored = TLOBAdapter.load(path)
    assert restored.open_residual_head is None and restored.open_residual_metadata is None
    np.testing.assert_array_equal(restored.predict(b), expected)
    payload["schema_version"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version=4"):
        TLOBAdapter.load(path)


def test_frozen_cross_sectional_transform_reuses_saved_statistics():
    instruments = ["A", "B", "C", "D"]
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-08-01 09:30"] * 4 + ["2023-08-02 09:30"] * 4),
                         "instrument": instruments * 2})
    raw = np.array([-.004, -.001, .002, .005, -.003, .001, .003, .007], dtype=np.float32)
    transformed, stats = timestamp_zscore_train_rms_matched(keys, raw, None)
    reused, summary = apply_frozen_timestamp_zscore(keys, raw, stats)
    np.testing.assert_array_equal(reused, transformed)
    assert summary["stats_sha256"] == frozen_stats_sha256(stats)
    assert summary["checkpoint_per_timestamp_sha256"] == stats["per_timestamp_sha256"]
    subset = np.array([0, 2, 5, 6])  # partial cross-sections must not be re-standardised
    partial, _ = apply_frozen_timestamp_zscore(keys.iloc[subset], raw[subset], stats)
    np.testing.assert_array_equal(partial, transformed[subset])
    with pytest.raises(ValueError, match="no statistics"):
        apply_frozen_timestamp_zscore(keys.assign(date=keys.date + pd.Timedelta(days=7)), raw, stats)
