"""Fixed-weight score blend: checkpoint loading, exact averaging and contract rejections."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import TLOBAdapter, TLOBBlend  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402
from scripts import build_blend_weights  # noqa: E402

torch.set_num_threads(1)


def data_config(**overrides):
    values = dict(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic",
                  version="1", feature_columns=lob_columns(3), lookback=8, min_history=2,
                  history_days=1, sample_times=("10:00",))
    values.update(overrides)
    return DataConfig(**values)


def batch(levels=3, lookback=8):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(7, lookback, 4 * levels)).astype(np.float32)
    keys = pd.DataFrame({"date": pd.date_range("2020-01-02 10:00", periods=7), "instrument": "A"})
    return Batch(keys, x, np.ones_like(x, dtype=bool),
                 np.full((7, lookback), np.datetime64("NaT", "ns")), {})


def trained(seed, *, preprocessing=None, kind="modified_tlob", **model):
    values = dict(kind=kind, hidden_dim=8, num_layers=2, batch_size=3, epochs=1, device="cpu",
                  log_every_steps=1, target_scaling="train_std",
                  target_centering="timestamp_zscore_train_rms_matched")
    values.update(model)
    cfg = ModelConfig(**values)
    adapter = TLOBAdapter(cfg, preprocessing or data_config(), seed=seed)
    adapter.intercept = 0.001 * seed
    adapter.target_scale = 0.003
    adapter.target_centering_stats = {"method": "timestamp_zscore_train_rms_matched"}
    adapter.trained = True
    return adapter


def export(model, path):
    model.export_text(path)
    return json.loads(path.read_text(encoding="utf-8"))


def blend_payload(left, right, weights=(0.75, 0.25), names=("left", "right")):
    return {
        "schema_version": 4, "kind": "modified_tlob_blend",
        "members": [{"name": names[0], "weight": weights[0], "payload": left},
                    {"name": names[1], "weight": weights[1], "payload": right}],
    }


def write(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return path


def test_blend_predicts_the_fixed_weighted_average(tmp_path):
    left, right = trained(1), trained(2)
    payload = blend_payload(export(left, tmp_path / "left.json"), export(right, tmp_path / "right.json"))
    model = TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")
    assert isinstance(model, TLOBBlend) and model.names == ("left", "right")
    expected = 0.75 * left.predict(batch()) + 0.25 * right.predict(batch())
    np.testing.assert_allclose(model.predict(batch()), expected, rtol=0, atol=1e-12)
    assert model.intercept == pytest.approx(0.75 * 0.001 + 0.25 * 0.002)
    assert model.preprocessing == left.preprocessing
    assert model.target_scale == pytest.approx(0.003)


def test_blend_honors_the_configured_predict_batch_size(tmp_path):
    left, right = trained(1), trained(2)
    payload = blend_payload(export(left, tmp_path / "left.json"), export(right, tmp_path / "right.json"))
    model = TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")
    model.config["batch_size"] = 2
    model.predict(batch())
    assert {member.config["batch_size"] for member in model.members} == {2}


def test_blend_rejects_weights_that_do_not_sum_to_one(tmp_path):
    payload = blend_payload(export(trained(1), tmp_path / "left.json"),
                            export(trained(2), tmp_path / "right.json"), weights=(0.75, 0.5))
    with pytest.raises(ValueError, match="sum to one"):
        TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")


def test_blend_rejects_mismatched_preprocessing(tmp_path):
    left = export(trained(1), tmp_path / "left.json")
    right = export(trained(2, preprocessing=data_config(history_days=2)), tmp_path / "right.json")
    with pytest.raises(ValueError, match="preprocessing differs"):
        TLOBAdapter.load(write(tmp_path / "blend.json", blend_payload(left, right)), device="cpu")


def test_blend_scores_each_member_on_its_own_trailing_lookback(tmp_path):
    short = trained(1, preprocessing=data_config(lookback=8))
    long = trained(2, preprocessing=data_config(lookback=16))
    payload = blend_payload(export(short, tmp_path / "short.json"), export(long, tmp_path / "long.json"))
    model = TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")
    assert model.preprocessing["lookback"] == 16 and model.architecture["seq_size"] == 16
    full = batch(lookback=16)
    tail = Batch(full.keys, full.x[:, -8:], full.observed[:, -8:], full.history_dates[:, -8:], {})
    expected = 0.75 * short.predict(tail) + 0.25 * long.predict(full)
    np.testing.assert_allclose(model.predict(full), expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="requires 16"):
        model.predict(batch(lookback=8))


def test_blend_rejects_mismatched_prediction_scale(tmp_path):
    left = export(trained(1), tmp_path / "left.json")
    right_model = trained(2)
    right_model.target_scale = 0.004
    right = export(right_model, tmp_path / "right.json")
    with pytest.raises(ValueError, match="prediction scale"):
        TLOBAdapter.load(write(tmp_path / "blend.json", blend_payload(left, right)), device="cpu")


def test_blend_rejects_a_single_member(tmp_path):
    payload = {"schema_version": 4, "kind": "modified_tlob_blend",
               "members": [{"name": "only", "weight": 1.0,
                            "payload": export(trained(1), tmp_path / "only.json")}]}
    with pytest.raises(ValueError, match="at least two members"):
        TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")


def test_mlplob_blend_reuses_the_same_trunk_contract(tmp_path):
    left, right = trained(1, kind="mlplob"), trained(2, kind="mlplob")
    payload = blend_payload(export(left, tmp_path / "left.json"), export(right, tmp_path / "right.json"),
                            weights=(0.5, 0.5))
    model = TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")
    assert isinstance(model, TLOBBlend) and model.config["kind"] == "mlplob"
    expected = 0.5 * left.predict(batch()) + 0.5 * right.predict(batch())
    np.testing.assert_allclose(model.predict(batch()), expected, rtol=0, atol=1e-12)


def test_blend_mixes_both_trunks_with_the_fixed_weighted_average(tmp_path):
    left, right = trained(1, kind="modified_tlob"), trained(2, kind="mlplob")
    payload = blend_payload(export(left, tmp_path / "left.json"), export(right, tmp_path / "right.json"))
    model = TLOBAdapter.load(write(tmp_path / "blend.json", payload), device="cpu")
    assert isinstance(model, TLOBBlend) and model.kinds == ("modified_tlob", "mlplob")
    expected = 0.75 * left.predict(batch()) + 0.25 * right.predict(batch())
    np.testing.assert_allclose(model.predict(batch()), expected, rtol=0, atol=1e-12)
    assert model.preprocessing == left.preprocessing == right.preprocessing
    assert model.target_scale == pytest.approx(0.003)


def test_mixed_blend_rejects_a_different_common_input_transform(tmp_path):
    left = export(trained(1), tmp_path / "left.json")
    right = export(trained(2, kind="mlplob", book_representation="relative_mid_bps"),
                   tmp_path / "right.json")
    with pytest.raises(ValueError, match="book_representation"):
        TLOBAdapter.load(write(tmp_path / "blend.json", blend_payload(left, right)), device="cpu")


def test_blend_rejects_trunk_specific_recipe_mismatch_within_one_trunk(tmp_path):
    left = export(trained(1, kind="modified_tlob", num_heads=1), tmp_path / "left.json")
    right = export(trained(2, kind="modified_tlob", num_heads=2), tmp_path / "right.json")
    with pytest.raises(ValueError, match="num_heads"):
        TLOBAdapter.load(write(tmp_path / "blend.json", blend_payload(left, right)), device="cpu")


def test_builder_records_mixed_trunks_and_keeps_common_checks(tmp_path):
    left, right = trained(1, kind="modified_tlob"), trained(2, kind="mlplob")
    export(left, tmp_path / "left.json")
    export(right, tmp_path / "right.json")
    payload = build_blend_weights.build(
        [("tlob", tmp_path / "left.json", 0.5), ("mlplob", tmp_path / "right.json", 0.5)],
        {}, experiment_id="mixed-blend-test")
    assert payload["kind"] == "modified_tlob_blend"
    assert payload["blend"]["kind"] == "mixed"
    assert payload["blend"]["member_kinds"] == {"tlob": "modified_tlob", "mlplob": "mlplob"}
    other = trained(3, kind="mlplob", target_centering="timestamp_mean_train")
    export(other, tmp_path / "other.json")
    with pytest.raises(ValueError, match="target_centering"):
        build_blend_weights.build([("tlob", tmp_path / "left.json", 0.5),
                                   ("mlplob", tmp_path / "other.json", 0.5)], {},
                                  experiment_id="bad")


def test_builder_accepts_mixed_lookbacks_but_not_other_preprocessing_drift(tmp_path):
    short = trained(1, preprocessing=data_config(lookback=8))
    long = trained(2, preprocessing=data_config(lookback=16))
    export(short, tmp_path / "short.json")
    export(long, tmp_path / "long.json")
    payload = build_blend_weights.build(
        [("short", tmp_path / "short.json", 0.5), ("long", tmp_path / "long.json", 0.5)],
        {}, experiment_id="mixed-lookback-test")
    assert payload["kind"] == "modified_tlob_blend"
    other = trained(3, preprocessing=data_config(history_days=2))
    export(other, tmp_path / "other.json")
    with pytest.raises(ValueError, match="preprocessing differs"):
        build_blend_weights.build([("short", tmp_path / "short.json", 0.5),
                                   ("other", tmp_path / "other.json", 0.5)], {},
                                  experiment_id="bad")


def test_builder_validates_members_and_writes_schema_four(tmp_path):
    left, right = trained(1, kind="mlplob"), trained(2, kind="mlplob")
    export(left, tmp_path / "left.json")
    export(right, tmp_path / "right.json")
    payload = build_blend_weights.build(
        [("s1", tmp_path / "left.json", 0.5), ("s2", tmp_path / "right.json", 0.5)],
        {}, experiment_id="mlplob-blend-test")
    assert payload["schema_version"] == 4 and payload["kind"] == "modified_tlob_blend"
    assert payload["blend"]["kind"] == "mlplob"
    assert [entry["name"] for entry in payload["members"]] == ["s1", "s2"]
    with pytest.raises(ValueError, match="sum to one"):
        build_blend_weights.build([("s1", tmp_path / "left.json", 0.9),
                                   ("s2", tmp_path / "right.json", 0.5)], {},
                                  experiment_id="bad")
    other = trained(3, kind="mlplob", target_centering="timestamp_mean_train")
    export(other, tmp_path / "other.json")
    with pytest.raises(ValueError, match="target_centering"):
        build_blend_weights.build([("s1", tmp_path / "left.json", 0.5),
                                   ("s3", tmp_path / "other.json", 0.5)], {},
                                  experiment_id="bad")
