"""Behavioral contracts for the project's dual-scale PatchTST trunk."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.model import DualPatchTLOB  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

torch.set_num_threads(1)


def data_config(lookback=128):
    return DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic", version="1",
                      feature_columns=lob_columns(3), lookback=lookback, min_history=2, history_days=1,
                      sample_times=("10:00",))


def batch(lookback=128, count=7):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(count, lookback, 12)).astype(np.float32)
    keys = pd.DataFrame({"date": pd.date_range("2020-01-02 10:00", periods=count), "instrument": "A"})
    return Batch(keys, x, np.ones_like(x, dtype=bool),
                 np.full((count, lookback), np.datetime64("NaT", "ns")), {})


def test_dual_scale_geometry_and_gradients_reach_both_branches():
    torch.manual_seed(0)
    model = DualPatchTLOB(hidden_dim=8, num_layers=1, seq_size=128, num_features=12, num_heads=2)
    assert [(branch.patch_count, branch.patch_length) for branch in model.branches] == [(4, 32), (16, 8)]
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for x in (torch.randn(3, 128, 12), torch.zeros(1, 128, 12)):
        optimizer.zero_grad()
        prediction = model(x)
        assert prediction.shape == (len(x),) and torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        named = {name: parameter.grad for name, parameter in model.named_parameters()}
        for prefix in ("branches.0", "branches.1"):
            assert any(gradient is not None and gradient.abs().sum() > 0
                       for name, gradient in named.items() if name.startswith(prefix))
        optimizer.step()
    changed = [name for name, parameter in model.named_parameters()
               if not torch.equal(before[name], parameter.detach())]
    assert any(name.startswith("branches.0") for name in changed)
    assert any(name.startswith("branches.1") for name in changed)


def test_long_history_geometry_keeps_the_32_and_8_bar_scales():
    torch.manual_seed(0)
    model = DualPatchTLOB(hidden_dim=8, num_layers=1, seq_size=256, num_features=12, num_heads=2,
                          patch_counts=(8, 32))
    assert [(branch.patch_count, branch.patch_length) for branch in model.branches] == [(8, 32), (32, 8)]
    prediction = model(torch.randn(2, 256, 12))
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()


def test_wrong_patch_or_input_contract_fails_closed():
    for kwargs in ({"patch_counts": (3, 16)}, {"patch_counts": (4, 4)}, {"patch_counts": (16, 4)},
                   {"patch_counts": (1, 16)}, {"seq_size": 10, "patch_counts": (4, 16)},
                   {"hidden_dim": 7, "num_heads": 4}):
        with pytest.raises(ValueError):
            DualPatchTLOB(**{"hidden_dim": 8, "num_layers": 1, "num_features": 12, "num_heads": 2, **kwargs})
    model = DualPatchTLOB(hidden_dim=8, num_layers=1, seq_size=128, num_features=12, num_heads=2)
    for x in (torch.zeros(8, 12), torch.zeros(1, 128, 11), torch.full((1, 128, 12), float("nan"))):
        with pytest.raises(ValueError):
            model(x)
    with pytest.raises(ValueError):
        ModelConfig(kind="dual_patch", patch_counts=(4, 4))
    with pytest.raises(ValueError):
        ModelConfig(kind="dual_patch", patch_counts=(16, 4))
    with pytest.raises(ValueError):
        ModelConfig(kind="dual_patch", hidden_dim=7, num_heads=4)


def test_dual_patch_adapter_checkpoint_roundtrip(tmp_path):
    cfg = ModelConfig(kind="dual_patch", hidden_dim=8, num_layers=1, num_heads=2, batch_size=3,
                      epochs=1, device="cpu", log_every_steps=1)
    model, b = TLOBAdapter(cfg, data_config()), batch()
    progress = model.fit(b, np.linspace(-.01, .01, len(b.x)))
    assert progress["steps"] == 3 and progress["samples"] == 7
    assert isinstance(model.network, DualPatchTLOB)
    assert model.parameter_groups == ("norm_layer", "branches", "channel_heads", "head")
    before = model.predict(b)
    for filename in ("model.pt", "weights.json"):
        path = tmp_path / filename
        model.save(path) if path.suffix == ".pt" else model.export_text(path)
        payload = json.loads(path.read_text()) if path.suffix == ".json" else torch.load(path, weights_only=True)
        assert payload["kind"] == "dual_patch"
        assert tuple(payload["architecture"]["patch_counts"]) == model.architecture["patch_counts"]
        assert ({key: value for key, value in payload["architecture"].items() if key != "patch_counts"}
                == {key: value for key, value in model.architecture.items() if key != "patch_counts"})
        restored = TLOBAdapter.load(path, device="cpu")
        assert isinstance(restored.network, DualPatchTLOB)
        assert restored.architecture == model.architecture
        np.testing.assert_allclose(restored.predict(b), before, atol=1e-6, rtol=1e-5)


def test_dual_patch_end_to_end_runner_keeps_csz_label_and_evaluation_contract(tmp_path):
    from jinyiquant.config import RunConfig, load_config
    from jinyiquant.io import read_json
    from jinyiquant.runner import run
    from scripts.tlob_fixture import create_fixture

    config_path = create_fixture(tmp_path / "fixture")
    raw = load_config(config_path).snapshot()
    raw["model"].update(kind="dual_patch", hidden_dim=8, num_layers=1, num_heads=2, patch_counts=[2, 4],
                        epochs=1, max_steps=5, batch_size=3, device="cpu", target_scaling="train_std",
                        target_centering="timestamp_zscore_train_rms_matched")
    output, code = run(RunConfig.model_validate(raw))
    assert code == 0, (output / "worker.log").read_text(encoding="utf-8")
    assert read_json(output / "checks.json")["passed"]
    payload = json.loads((output / "modified_tlob_weights.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "dual_patch"
    assert payload["architecture"]["patch_counts"] == [2, 4]
    assert payload["label"] == "official_vwap_return_timestamp_zscore_train_rms_matched"
    assert payload["prediction_units"] == "relative_return_score"
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert len(predictions) == 10 and np.isfinite(predictions.score).all()
