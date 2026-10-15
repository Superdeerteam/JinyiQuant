"""Causal input experiments preserve labels, primary history and checkpoint meaning."""

import json
import os
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig, RunConfig, load_config  # noqa: E402
from jinyiquant.data import build_batch, history_diagnostics, load_data, preprocess_bars  # noqa: E402
from jinyiquant.io import read_json  # noqa: E402
from jinyiquant.runner import run  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.diagnostics import subset_batch  # noqa: E402
from modified_tlob.package import build_package  # noqa: E402
from modified_tlob.schema import TRADE_COLUMNS, lob_columns, lob_trade_columns  # noqa: E402
from scripts.tlob_fixture import create_fixture  # noqa: E402


def data_config(trades=False, **changes):
    values = dict(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic", version="1",
                  feature_columns=lob_trade_columns() if trades else lob_columns(), lookback=8, min_history=2,
                  sample_times=("10:00",), history_days=1)
    return DataConfig(**(values | changes))


def market():
    rows = []
    for stock in ("A", "B", "C"):
        for index, date in enumerate(pd.date_range("2020-01-02 09:55", periods=7, freq="min")):
            mid = 10 + index / 100
            row = {"date": date, "instrument": stock, "volume": 100 + index,
                   "amount": 1000 + index * 10, "deal_number": 5 + index}
            for level in range(1, 4):
                row.update({f"ask_price{level}": mid + level / 100, f"bid_price{level}": mid - level / 100,
                            f"ask_volume{level}": 200 + index, f"bid_volume{level}": 180 + index})
            if stock == "B" or (stock == "C" and index != 5):
                row.update(dict.fromkeys(lob_columns(), np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def keys():
    return pd.DataFrame({"date": pd.to_datetime(["2020-01-02 10:00"] * 3), "instrument": ["A", "B", "C"]})


def adapter(data, **changes):
    values = dict(kind="modified_tlob", hidden_dim=8, num_layers=1, epochs=1, batch_size=3, device="cpu")
    return TLOBAdapter(ModelConfig(**(values | changes)), data, seed=11)


def test_relative_mid_formula_scale_invariance_padding_and_invalid_anchor():
    data = data_config()
    bars = market()
    batch = build_batch(preprocess_bars(bars, data), keys(), data)
    model = adapter(data, book_representation="relative_mid_bps")
    before = batch.x.copy()
    transformed = model._input(batch)
    anchor = before[0, -1, [0, 2]].astype(np.float64).mean()
    np.testing.assert_allclose(transformed[0, 2:, :12:2],
                               (before[0, 2:, :12:2].astype(np.float64) / anchor - 1) * 10000,
                               atol=1e-5, rtol=1e-6)
    assert not transformed[:, :2].any()  # Left padding is zero, not -10000 bp.
    np.testing.assert_array_equal(batch.x, before)
    changed = bars.copy()
    changed[list(lob_columns()[::2])] *= 2
    np.testing.assert_array_equal(model._input(build_batch(preprocess_bars(changed, data), keys(), data)), transformed)
    changed.loc[(changed.instrument == "A") & (changed.date == keys().date.iloc[0]), "ask_price1"] = 0
    invalid = build_batch(preprocess_bars(changed, data), keys(), data)
    assert np.isfinite(model._input(invalid)).all()
    assert not model._input(invalid)[0, :, :12:2].any()
    pd.testing.assert_frame_equal(invalid.keys, batch.keys)
    np.testing.assert_array_equal(invalid.effective_rows, batch.effective_rows)


def test_trade_missing_values_never_ffill_and_do_not_create_primary_history():
    data = data_config(trades=True)
    bars = market()
    missing = (bars.instrument == "A") & (bars.date == pd.Timestamp("2020-01-02 09:59"))
    zero = (bars.instrument == "A") & (bars.date == pd.Timestamp("2020-01-02 10:00"))
    bars.loc[missing, list(TRADE_COLUMNS)] = [-1, np.nan, -2]
    bars.loc[missing, "ask_price2"] = np.nan
    bars.loc[zero, list(TRADE_COLUMNS)] = 0
    batch = build_batch(preprocess_bars(bars, data), keys(), data)
    original = build_batch(preprocess_bars(bars, data_config()), keys(), data_config())
    assert not batch.x[0, -2:, 12:].any()
    assert not batch.observed[0, -2, 12:].any()
    assert batch.observed[0, -1, 12:].all()
    assert batch.x[0, -2, 4] == batch.x[0, -3, 4]  # Quote filling remains active.
    np.testing.assert_array_equal(batch.effective_rows, original.effective_rows)
    subset = subset_batch(batch, [1, 2])
    assert subset.history_feature_count == 12
    np.testing.assert_array_equal(subset.effective_rows.sum(axis=1), [0, 1])
    assert batch.coverage["usable_samples"] == original.coverage["usable_samples"] == 1
    np.testing.assert_array_equal(history_diagnostics(batch).effective_history_length, [6, 0, 1])
    left, right = adapter(data), adapter(data, trade_features="log1p")
    assert left.architecture == right.architecture
    assert not left._input(batch)[..., 12:].any()
    np.testing.assert_allclose(right._input(batch)[..., 12:], np.log1p(batch.x[..., 12:]))
    for model in (left, right):
        model.fit(batch, [.01, 500, 1000], sample_mask=batch.effective_rows.sum(axis=1) >= data.min_history)
        assert model.intercept == pytest.approx(.01)
        np.testing.assert_array_equal(model.predict(batch)[1:], np.full(2, model.intercept))


def test_currency_units_future_bars_and_same_named_label_columns_are_isolated(tmp_path):
    data = data_config(trades=True, bars_path=tmp_path / "bars.parquet", labels_path=tmp_path / "labels.parquet")
    bars = market()
    labels = keys().assign(target=[.01, .02, .03], volume=1e12, amount=1e15, deal_number=1e9)
    bars.to_parquet(data.bars_path, index=False)
    labels.to_parquet(data.labels_path, index=False)
    bundle = load_data(data, start="2020-01-02", end="2020-01-03")
    batch = build_batch(bundle.bars, keys(), data)
    model = adapter(data, book_representation="relative_mid_bps", trade_features="log1p")
    model.fit(batch, labels.target, sample_mask=batch.effective_rows.sum(axis=1) >= data.min_history)
    before = model.predict(batch)
    changed = bars.copy()
    changed.loc[changed.date > keys().date.iloc[0], list(lob_trade_columns())] = 1e9
    changed.to_parquet(data.bars_path, index=False)
    labels.loc[:, ["target", *TRADE_COLUMNS]] *= -999
    labels.to_parquet(data.labels_path, index=False)
    altered = load_data(data, start="2020-01-02", end="2020-01-03")
    rebuilt = build_batch(altered.bars, keys(), data)
    np.testing.assert_array_equal(batch.x, rebuilt.x)
    np.testing.assert_array_equal(model.predict(rebuilt), before)
    assert set(bundle.labels) == {"date", "instrument", "target"}
    cents = bars.copy()
    cents[[*lob_columns()[::2], "amount"]] *= 100
    cents_data = DataConfig.model_validate(data.model_dump() | {"price_scale": 100})
    cent_batch = build_batch(preprocess_bars(cents, cents_data), keys(), cents_data)
    np.testing.assert_allclose(model._input(cent_batch), model._input(batch), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("trades,representation,trade_features", [
    (False, "relative_mid_bps", "zero"), (True, "raw", "zero"), (True, "raw", "log1p")])
def test_new_input_checkpoints_roundtrip_and_reject_version_downgrade(
        tmp_path, trades, representation, trade_features):
    data = data_config(trades=trades)
    batch = build_batch(preprocess_bars(market(), data), keys(), data)
    model = adapter(data, book_representation=representation, trade_features=trade_features, target_scaling="train_std")
    model.fit(batch, [.01, .02, .03], sample_mask=batch.effective_rows.sum(axis=1) >= data.min_history)
    assert model._payload()["schema_version"] == 3
    for name in ("model.pt", "weights.json"):
        path = tmp_path / name
        model.save(path) if path.suffix == ".pt" else model.export_text(path)
        restored = TLOBAdapter.load(path)
        np.testing.assert_array_equal(model.predict(batch), restored.predict(batch))
        assert restored.config["book_representation"] == representation
        assert restored.config["trade_features"] == trade_features
    payload = read_json(tmp_path / "weights.json")
    missing = deepcopy(payload)
    missing["model_config"].pop("book_representation")
    (tmp_path / "missing.json").write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="input transformation contract"):
        TLOBAdapter.load(tmp_path / "missing.json")
    payload["schema_version"] = 2
    (tmp_path / "weights.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version=3"):
        TLOBAdapter.load(tmp_path / "weights.json")


@pytest.mark.parametrize("scaling,version", [("none", 1), ("train_std", 2)])
def test_legacy_checkpoints_without_new_fields_keep_predictions(tmp_path, scaling, version):
    data = data_config()
    batch = build_batch(preprocess_bars(market(), data), keys(), data)
    model = adapter(data, target_scaling=scaling)
    model.fit(batch, [.01, .02, .03], sample_mask=batch.effective_rows.sum(axis=1) >= data.min_history)
    payload = model._payload()
    assert payload["schema_version"] == version
    payload["model_config"] = deepcopy(payload["model_config"])
    for key in ("book_representation", "trade_features"):
        payload["model_config"].pop(key)
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    np.testing.assert_array_equal(model.predict(batch), TLOBAdapter.load(path).predict(batch))


def test_config_requires_ordered_trade_schema_and_runner_uses_only_book_history(tmp_path):
    path = create_fixture(tmp_path / "fixture")
    bars = pd.read_parquet(path.parent / "bars.parquet")
    bars[list(TRADE_COLUMNS)] = [100, 1000, 5]
    bars.loc[bars.instrument == "000005.SZ", list(lob_columns())] = np.nan
    bars.to_parquet(path.parent / "bars.parquet", index=False)
    raw = load_config(path).snapshot()
    raw["data"]["feature_columns"] = list(lob_trade_columns())
    raw["model"].update(device="cpu", max_steps=2, trade_features="log1p", book_representation="relative_mid_bps")
    raw["acceptance"]["min_coverage"] = .8
    config = RunConfig.model_validate(raw)
    output, code = run(config)
    assert code == 0, (output / "worker.log").read_text(encoding="utf-8")
    coverage = read_json(output / "coverage.json")
    assert coverage["training_samples_used"] == 8
    assert coverage["training_samples_excluded_insufficient_history"] == 2
    assert coverage["training_mean_fallback_samples"] == 2
    assert coverage["no_history_prediction_samples"] == 2
    assert read_json(output / "checks.json")["passed"]
    package = build_package(output, tmp_path / "offline_package")
    cli = subprocess.run([sys.executable, "-m", "jinyiquant", "--help"], cwd=package,
        env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True, text=True, timeout=45)
    assert cli.returncode == 0, cli.stderr + cli.stdout
    assert "validate-config" in cli.stdout
    expected = pd.read_parquet(output / "submission.parquet")
    key_path = tmp_path / "inference_keys.parquet"
    expected[["date", "instrument"]].to_parquet(key_path, index=False)
    # Only packaged source is importable in this fresh inference process.
    result = subprocess.run([sys.executable, str(package / "modified_tlob/infer.py"),
        "--weights", str(package / "modified_tlob_weights.json"), "--bars", str(path.parent / "bars.parquet"),
        "--keys", str(key_path), "--output-root", str(tmp_path / "offline_predictions"),
        "--device", "cpu", "--min-coverage", "0.8"], cwd=package,
        env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr + result.stdout
    offline = pd.read_parquet(json.loads(result.stdout)["predictions"])
    pd.testing.assert_frame_equal(offline[["date", "instrument"]], expected[["date", "instrument"]])
    np.testing.assert_allclose(offline.factor, expected.factor, atol=1e-6, rtol=1e-5)
    for fields in (list(lob_columns()), list(reversed(lob_trade_columns())), list(lob_columns(5) + TRADE_COLUMNS)):
        invalid = deepcopy(raw)
        invalid["data"]["feature_columns"] = fields
        with pytest.raises(ValueError):
            RunConfig.model_validate(invalid)
