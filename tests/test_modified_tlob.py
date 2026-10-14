"""Behavioral regression contracts, optional in the original CPU baseline environment."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, ModelConfig  # noqa: E402
from jinyiquant.data import Batch, build_batch, preprocess_bars  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.device import select_device  # noqa: E402
from modified_tlob.model import ModifiedTLOB  # noqa: E402
from modified_tlob.normalization import BiN  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def data_config(levels=3):
    return DataConfig(bars_path="bars.parquet", labels_path="labels.parquet", source="synthetic", version="1",
                      feature_columns=lob_columns(levels), lookback=8, min_history=2, history_days=1,
                      sample_times=("10:00",))


def batch(levels=3):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(7, 8, 4 * levels)).astype(np.float32)
    keys = pd.DataFrame({"date": pd.date_range("2020-01-02 10:00", periods=7), "instrument": "A"})
    return Batch(keys, x, np.ones_like(x, dtype=bool), np.full((7, 8), np.datetime64("NaT", "ns")), {})


def adapter(levels=3):
    cfg = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=3,
                      epochs=2, device="cpu", log_every_steps=1)
    return TLOBAdapter(cfg, data_config(levels))


@pytest.mark.parametrize("levels", [3, 5])
@pytest.mark.parametrize("count", [1, 4])
def test_shapes_gradients_optimizer_and_constant_windows(levels, count):
    torch.manual_seed(3)
    model = ModifiedTLOB(hidden_dim=8, seq_size=8, num_layers=2, num_heads=2, num_features=levels * 4)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    for x in (torch.randn(count, 8, levels * 4), torch.zeros(count, 8, levels * 4)):
        optim.zero_grad()
        y = model(x)
        assert y.shape == (count,) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optim.step()
    assert not torch.equal(before["emb_layer.weight"], model.emb_layer.weight)
    assert not torch.equal(before["final_layers.0.weight"], model.final_layers[0].weight)


def test_bin_parameters_not_replaced_and_optimizer_still_owns_them():
    norm = BiN(12, 8)
    optimizer = torch.optim.Adam(norm.parameters())
    ids = [id(p) for p in norm.parameters()]
    with torch.no_grad():
        norm.y1.fill_(-1)
        norm.y2.fill_(-2)
    x = torch.ones(1, 12, 8, requires_grad=True)
    norm(x).sum().backward()
    assert ids == [id(p) for p in norm.parameters()]
    assert ids == [id(p) for p in optimizer.param_groups[0]["params"]]
    assert torch.isfinite(x.grad).all()
    assert norm.y1.item() == pytest.approx(.01)


@pytest.mark.parametrize("kwargs", [{"hidden_dim": 5}, {"seq_size": 3}, {"num_layers": 0}, {"num_heads": 0}])
def test_invalid_architecture_fails(kwargs):
    with pytest.raises(ValueError):
        ModifiedTLOB(**kwargs)


def test_wrong_shapes_and_nonfinite_input_fail():
    model = ModifiedTLOB(seq_size=8)
    for x in (torch.zeros(8, 12), torch.zeros(1, 8, 20), torch.full((1, 8, 12), float("nan"))):
        with pytest.raises(ValueError):
            model(x)


def test_regression_head_is_unbounded():
    model = ModifiedTLOB(hidden_dim=8, seq_size=8, num_layers=1)
    with torch.no_grad():
        model.final_layers[-1].weight.zero_()
        model.final_layers[-1].bias.fill_(100)
        torch.testing.assert_close(model(torch.zeros(1, 8, 12)), torch.tensor([100.]))


def test_devices_preflight_and_no_runtime_fallback(monkeypatch):
    from modified_tlob import device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    assert str(select_device("cpu")) == "cpu"
    with pytest.raises(RuntimeError, match="Explicit CUDA"):
        select_device("cuda")
    def broken_preflight():
        raise RuntimeError("simulated driver compute failure")
    monkeypatch.setattr(device, "_cuda_preflight", broken_preflight)
    assert str(select_device("auto")) == "cpu"
    with pytest.raises(RuntimeError, match="compute failure"):
        select_device("cuda")
    m = adapter()
    def oom(*args):
        raise torch.OutOfMemoryError("simulated runtime OOM")
    monkeypatch.setattr(m.network, "forward", oom)
    with pytest.raises(torch.OutOfMemoryError):
        m.fit(batch(), np.zeros(7))


def test_binary_and_text_roundtrip_frozen_preprocessing_and_history_fallback(tmp_path):
    m, b = adapter(), batch()
    m.fit(b, np.linspace(-.01, .01, len(b.x)))
    before = m.predict(b)
    for filename in ("model.pt", "weights.json"):
        path = tmp_path / filename
        m.save(path) if path.suffix == ".pt" else m.export_text(path)
        restored = TLOBAdapter.load(path, device="cpu")
        np.testing.assert_allclose(restored.predict(b), before, atol=1e-6, rtol=1e-5)
        assert restored.preprocessing == m.preprocessing
    b.observed[0] = False
    b.observed[1, 1:] = False
    assert m.predict(b)[0] == m.intercept
    assert m.predict(b)[1] == m.intercept


def test_missing_and_future_input_behaviors():
    cfg = data_config()
    keys = pd.DataFrame({"date": [pd.Timestamp("2020-01-03 10:00")], "instrument": ["A"]})
    bars = pd.concat([keys.assign(date=pd.Timestamp("2020-01-03 09:59")), keys,
                      keys.assign(date=pd.Timestamp("2020-01-03 10:01"))], ignore_index=True)
    for f in cfg.feature_columns:
        bars[f] = [1, -1, 2]
    clean = preprocess_bars(bars, cfg)
    before = build_batch(clean, keys, cfg)
    changed = clean.copy()
    changed.loc[changed.date > keys.date.iloc[0], list(cfg.feature_columns)] = 1e9
    after = build_batch(changed, keys, cfg)
    assert np.isfinite(before.x).all()
    np.testing.assert_array_equal(before.x, after.x)
    m = adapter()
    m.fit(batch(), np.zeros(7))
    np.testing.assert_array_equal(m.predict(before), m.predict(after))


def test_original_baselines_do_not_import_optional_torch():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from jinyiquant.models import LinearModel; from jinyiquant.config import ModelConfig; "
        "assert 'torch' not in sys.modules; assert ModelConfig().kind == 'ridge'"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_torch_fails_instead_of_device_fallback():
    code = """
import sys
from types import SimpleNamespace
sys.modules['torch'] = None
from jinyiquant.models import create_model
try:
    create_model(SimpleNamespace(model=SimpleNamespace(kind='modified_tlob')))
except RuntimeError as exc:
    assert 'optional Python 3.11/PyTorch environment' in str(exc)
else:
    raise AssertionError('Missing torch masqueraded as CPU fallback')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_platform_errors_are_logged_and_never_train(tmp_path, monkeypatch):
    from jinyiquant.io import file_sha256, write_json
    from modified_tlob.submission import predict_range
    m = adapter()
    m.fit(batch(), np.zeros(7))
    m.export_text(tmp_path / "modified_tlob_weights.json")
    write_json(tmp_path / "submission_config.json", {"device": "cpu", "platform_price_scale": 1,
        "weights_sha256": file_sha256(tmp_path / "modified_tlob_weights.json"),
        "min_coverage": .99, "wall_seconds": 10})
    monkeypatch.setattr(TLOBAdapter, "fit", lambda *a, **kw: pytest.fail("Public inference tried to train"))
    with pytest.raises(ValueError, match="simple SQL table"):
        predict_range({"bar1m": "bad; SELECT vwap"}, "2020-01-03", "2020-01-04",
                      package_root=tmp_path, query=lambda *a, **kw: pytest.fail("Unexpected query"))
    archives = list((tmp_path / "inference_runs").glob("*/events.jsonl"))
    events = [json.loads(line) for line in archives[0].read_text(encoding="utf-8").splitlines()]
    assert events[-1]["context"]["status"] == "failed"
    assert any(e["event_type"] == "failed" and "traceback" in e["context"] for e in events)
    (tmp_path / "modified_tlob_weights.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        predict_range({"bar1m": "bars"}, "2020-01-03", "2020-01-04", package_root=tmp_path,
                      query=lambda *a, **kw: pytest.fail("Should not query after corrupted weights"))


def test_five_level_adapter_training_and_bad_field_order():
    m = adapter(5)
    m.fit(batch(5), np.linspace(-.01, .01, 7))
    assert np.isfinite(m.predict(batch(5))).all()
    data = data_config().model_dump(mode="json")
    data["feature_columns"] = list(reversed(data["feature_columns"]))
    with pytest.raises(ValueError, match="field order"):
        TLOBAdapter(m.config, data)


def upstream_namespace():
    from einops import rearrange
    scope = {"torch": torch, "nn": torch.nn, "cst": SimpleNamespace(DEVICE="cpu"), "rearrange": rearrange}
    provenance = json.loads((ROOT / "modified_tlob/provenance.json").read_text(encoding="utf-8"))
    for name, classes in (("bin", {"BiN"}), ("mlplob", {"MLP"}),
                          ("tlob", {"ComputeQKV", "TransformerLayer", "TLOB", "sinusoidal_positional_embedding"})):
        path = ROOT / "modified_tlob/upstream" / f"{name}.py.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["files"][f"models/{name}.py"]["sha256"]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in classes]
        exec(compile(tree, str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("seq_size,hidden_dim,num_heads,sinusoidal,num_layers", [
    (8, 8, 1, True, 2), (128, 40, 2, True, 2), (8, 8, 1, False, 2), (128, 40, 1, True, 4)])
def test_same_weights_input_upstream_intermediate_and_output_parity(seq_size, hidden_dim, num_heads,
                                                                  sinusoidal, num_layers):
    scope = upstream_namespace()
    upstream = scope["TLOB"](hidden_dim, num_layers, seq_size, 12, num_heads, sinusoidal, "FI_2010").eval()
    model = ModifiedTLOB(hidden_dim=hidden_dim, num_layers=num_layers, seq_size=seq_size,
                         num_features=12, num_heads=num_heads, is_sin_emb=sinusoidal).eval()
    state = upstream.state_dict()
    state.pop("order_type_embedder.weight")
    for k in list(state):
        if k.startswith(f"final_layers.{len(upstream.final_layers) - 1}."):
            state[k] = state[k][:1].clone()
    if sinusoidal:
        state["pos_encoder"] = upstream.pos_encoder
    model.load_state_dict(state)
    reference, candidate = {}, {}
    handles = []
    for name in ("norm_layer", "emb_layer", *[f"layers.{i}" for i in range(num_layers * 2)]):
        def hook(store, key):
            return lambda module, args, output: store.update({key: (output[0] if isinstance(output, tuple)
                                                                   else output).detach().clone()})
        handles.append(upstream.get_submodule(name).register_forward_hook(hook(reference, name)))
        handles.append(model.get_submodule(name).register_forward_hook(hook(candidate, name)))
    # Nondegenerate data isolates exact trunk parity from documented BiN safety repairs.
    x = torch.randn(3, seq_size, 12)
    with torch.inference_mode():
        expected, actual = upstream(x)[:, 0], model(x)
    for name in reference:
        torch.testing.assert_close(candidate[name], reference[name], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for handle in handles:
        handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No real CUDA compute environment")
def test_gpu_trained_weights_cpu_load(tmp_path):
    m = adapter()
    m.device = select_device("cuda")
    m.network.to(m.device)
    b = batch()
    m.fit(b, np.linspace(-.01, .01, 7))
    m.save(tmp_path / "gpu.pt")
    restored = TLOBAdapter.load(tmp_path / "gpu.pt", device="cpu")
    np.testing.assert_allclose(restored.predict(b), m.predict(b), atol=1e-6, rtol=1e-5)
