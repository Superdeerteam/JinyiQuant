"""Tests for the optional ``data.feature_transform`` window transform (default OFF)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from jinyiquant.config import DataConfig, load_config
from jinyiquant.data import build_batch, lob_columns, relative_price_log_volume
from jinyiquant.pool import StreamingPool

ROOT = Path(__file__).resolve().parents[1]
FEATURES = lob_columns(3)
PRICE = [index for index, name in enumerate(FEATURES) if "price" in name]
VOLUME = [index for index, name in enumerate(FEATURES) if "volume" in name]

# Two time steps x two instruments. A ends at mid (101 + 99) / 2 = 100; B never has a finite
# top-of-book price, so after forward fill its reference mid stays NaN and its prices go to 0.
BARS = [
    ("09:30", "A", [100, 3, 99, 0, 102, 7, 98, 1, 104, 0, 96, 2]),
    ("09:31", "A", [101, 4, 99, 0.5, 102, 8, 98, 2, 104, 1, 96, 3]),
    ("09:30", "B", [np.nan, 1, np.nan, 1, 200, 1, 198, 1, 202, 1, 196, 1]),
    ("09:31", "B", [np.nan, 2, np.nan, 2, 200, 2, 198, 2, 202, 2, 196, 2]),
]

# Bytes the unchanged builder produced on bars_frame()/keys_frame() before the key existed.
EXPECTED_HEX = (
    "0000c842000040400000c642000000000000cc420000e040"
    "0000c4420000803f0000d042000000000000c04200000040"
    "0000ca42000080400000c6420000003f0000cc4200000041"
    "0000c442000000400000d0420000803f0000c04200004040"
    "000000000000803f000000000000803f000048430000803f"
    "000046430000803f00004a430000803f000044430000803f"
    "000000000000004000000000000000400000484300000040"
    "000046430000004000004a43000000400000444300000040"
)

REGISTERED = {
    "configs/scaling_441d_csz_seed11.json":
        "c4c5353c24caab28585b7e8129b07c9773a603084b2662d444374306f7471ac4",
    "configs/scaling_927d_csz_seed11.json":
        "a6199b5cc1981b92d95bad1a4346ef0f71b6a312a4229004a7de818a3c41098e",
    "configs/scaling_927d_seed11_2point_plan.json":
        "20826d6714b57f848da8ed2ff8fc911d9e2ab06c5cff9016027de2c25e7613e8",
}


def bars_frame(price_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for minute, instrument, values in BARS:
        scaled = [value * (price_scale if index in PRICE else 1.0) for index, value in enumerate(values)]
        rows.append([pd.Timestamp(f"2020-01-02 {minute}"), instrument, *scaled])
    return pd.DataFrame(rows, columns=["date", "instrument", *FEATURES])


def keys_frame() -> pd.DataFrame:
    return pd.DataFrame({"date": [pd.Timestamp("2020-01-02 09:31")] * 2, "instrument": ["A", "B"]})


def config(**overrides) -> SimpleNamespace:
    values = dict(feature_columns=FEATURES, lookback=2, min_history=1, missing_policy="ffill",
                  max_ffill_steps=5, sample_times=[], price_scale=1.0, missing_sentinel=None,
                  source="feature-transform-test", version="v1", history_days=1)
    values.update(overrides)
    return SimpleNamespace(**values)


def transformed(price_scale: float = 1.0):
    return build_batch(bars_frame(price_scale), keys_frame(),
                       config(feature_transform="price_rel_vol_log"))


def test_transform_matches_hand_computed_window():
    batch = transformed()
    # A, first step, ref = (101 + 99) / 2 = 100: ask_price1 100 -> 0, bid_price1 99 -> -0.01,
    # ask_price2 102 -> 0.02, bid_price2 98 -> -0.02, ask_price3 104 -> 0.04, bid_price3 96 -> -0.04.
    prices = np.array([0.0, -0.01, 0.02, -0.02, 0.04, -0.04], dtype=np.float32)
    volumes = np.log1p(np.array([3, 0, 7, 1, 0, 2], dtype=np.float32))
    np.testing.assert_array_equal(batch.x[0, 0], np.column_stack([prices, volumes]).ravel())
    # A, last step: (101 - 100) / 100 = 0.01, volumes log1p(4), log1p(0.5), log1p(8), ...
    prices = np.array([0.01, -0.01, 0.02, -0.02, 0.04, -0.04], dtype=np.float32)
    volumes = np.log1p(np.array([4, 0.5, 8, 2, 1, 3], dtype=np.float32))
    np.testing.assert_array_equal(batch.x[0, 1], np.column_stack([prices, volumes]).ravel())
    # B has no reference mid: every price channel is zero, volumes are still log1p.
    assert not batch.x[1, :, PRICE].any()
    np.testing.assert_array_equal(batch.x[1, 0, VOLUME], np.log1p(np.ones(6, dtype=np.float32)))
    np.testing.assert_array_equal(batch.x[1, 1, VOLUME], np.log1p(np.full(6, 2, dtype=np.float32)))
    assert np.isfinite(batch.x).all()


def test_non_positive_reference_zeroes_price_channels():
    window = np.full((1, 2, len(FEATURES)), np.float32(5.0))
    window[0, :, VOLUME] = np.float32(-2.0)      # negative volume is clamped to 0 before log1p
    window[0, -1, 0] = np.float32(0.0)           # last-step ask_price1
    window[0, -1, 2] = np.float32(0.0)           # last-step bid_price1 -> mid 0
    relative_price_log_volume(window, FEATURES)
    assert not window[0, :, PRICE].any()
    assert not window[0, :, VOLUME].any()


def test_absent_key_keeps_pre_change_bytes():
    batch = build_batch(bars_frame(), keys_frame(), config())
    assert batch.x.tobytes() == bytes.fromhex(EXPECTED_HEX)


def test_bitwise_identity_check_is_not_vacuous():
    assert build_batch(bars_frame(), keys_frame(), config()).x.tobytes() == bytes.fromhex(EXPECTED_HEX)
    mutated = bars_frame()
    mutated.loc[0, "ask_price1"] = 100.5
    assert build_batch(mutated, keys_frame(), config()).x.tobytes() != bytes.fromhex(EXPECTED_HEX)


def test_absent_key_equals_explicit_none():
    absent = build_batch(bars_frame(), keys_frame(), config())
    explicit = build_batch(bars_frame(), keys_frame(), config(feature_transform="none"))
    assert absent.x.tobytes() == explicit.x.tobytes()


def test_price_channels_are_scale_invariant():
    np.testing.assert_array_equal(transformed().x, transformed(price_scale=50.0).x)


def test_streaming_pool_matches_materialised_windows():
    settings = config(feature_transform="price_rel_vol_log")
    pool = StreamingPool.build(bars_frame(), keys_frame(), settings, history_quality_summary=False)
    streamed = pool.batch(np.arange(len(keys_frame())))
    np.testing.assert_array_equal(streamed.x, build_batch(bars_frame(), keys_frame(), settings).x)


def test_default_transform_is_omitted_from_config_dumps():
    data = DataConfig(bars_path=Path("bars.parquet"), labels_path=Path("labels.parquet"),
                      source="s", version="v")
    assert "feature_transform" not in data.model_dump()
    assert "feature_transform" not in data.model_dump(mode="json")
    assert DataConfig.model_validate(data.model_dump()).feature_transform == "none"
    snapshot = load_config(ROOT / "configs/scaling_441d_csz_seed11.json").snapshot()
    assert "feature_transform" not in snapshot["data"]


def test_explicit_transform_survives_a_dump_roundtrip():
    data = DataConfig(bars_path=Path("bars.parquet"), labels_path=Path("labels.parquet"),
                      source="s", version="v", feature_transform="price_rel_vol_log")
    dumped = data.model_dump(mode="json")
    assert dumped["feature_transform"] == "price_rel_vol_log"
    assert DataConfig.model_validate(dumped).feature_transform == "price_rel_vol_log"


def test_registered_configs_stay_hash_pinned():
    for relative, expected in REGISTERED.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    plan = json.loads((ROOT / "configs/scaling_927d_seed11_2point_plan.json").read_text(encoding="utf-8"))
    assert plan["recipe_sha256"] == REGISTERED[plan["recipe_config"]]
    assert "feature_transform" not in json.loads(
        (ROOT / plan["recipe_config"]).read_text(encoding="utf-8"))["data"]


def test_protocol_accepts_transform_as_a_declared_factor():
    from jinyiquant.protocol import fair_config_pair

    def recipe():
        return {"data": {"lookback": 8, "source": "s", "version": "v"}, "model": {"kind": "mlplob"}}

    baseline, candidate = recipe(), recipe()
    candidate["data"]["feature_transform"] = "price_rel_vol_log"
    change = {"factor": "data.feature_transform", "baseline": "none",
              "candidate": "price_rel_vol_log", "hypothesis": "scale-invariant prices rank better"}
    fair_config_pair(baseline, candidate, change)


def test_adapter_preprocessing_hides_the_default_transform():
    import pytest

    pytest.importorskip("torch")
    from jinyiquant.config import ModelConfig
    from modified_tlob.adapter import TLOBAdapter

    model = ModelConfig(kind="mlplob", hidden_dim=8, num_layers=2, epochs=1, batch_size=2, device="cpu")
    paths = dict(bars_path=Path("bars.parquet"), labels_path=Path("labels.parquet"),
                 source="s", version="v", feature_columns=FEATURES)
    assert "feature_transform" not in TLOBAdapter(model, DataConfig(**paths)).preprocessing
    enabled = DataConfig(feature_transform="price_rel_vol_log", **paths)
    assert TLOBAdapter(model, enabled).preprocessing["feature_transform"] == "price_rel_vol_log"
