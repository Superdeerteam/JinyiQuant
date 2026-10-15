"""Behavioral regression contracts, optional in the original CPU baseline environment."""

import ast
import hashlib
import json
import os
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


def test_update_cap_stops_inside_epoch_and_reports_processed_sample_loss(monkeypatch):
    model = adapter()
    model.config.update(epochs=10, max_steps=4)
    events, checkpoints, optimizer_calls = [], [], []
    model.recorder = SimpleNamespace(event=lambda stage, event, **values: events.append((event, values)))
    original = torch.optim.Adam.step
    def count_step(optimizer, *args, **kwargs):
        optimizer_calls.append(1)
        return original(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", count_step)
    progress = model.fit(batch(), np.linspace(-.01, .01, 7),
                         observer=lambda _, epoch, steps: checkpoints.append((epoch, steps)))
    assert len(optimizer_calls) == progress["steps"] == 4
    assert progress["samples_seen"] == 10 and progress["epochs_completed"] == 1
    assert progress["epochs"] == pytest.approx(1 + 3 / 7)
    assert progress["stop_reason"] == "max_steps"
    assert [steps for _, steps in checkpoints] == [0, 3, 4]
    assert checkpoints[-1][0] == progress["epochs"]
    last_batch = [values for event, values in events if event == "progress"][-1]
    assert progress["loss"] == pytest.approx(last_batch["loss"])
    stopped = [values for event, values in events if event == "epoch_stopped"]
    assert len(stopped) == 1 and stopped[0]["epoch_samples"] == 3


def test_absent_update_cap_preserves_full_epoch_training():
    uncapped, capped = adapter(), adapter()
    capped.config["max_steps"] = 6
    target = np.linspace(-.01, .01, 7)
    left, right = uncapped.fit(batch(), target), capped.fit(batch(), target)
    assert left["steps"] == right["steps"] == 6
    assert left["samples_seen"] == right["samples_seen"] == 14
    assert left["epochs"] == right["epochs"] == 2
    np.testing.assert_array_equal(uncapped.predict(batch()), capped.predict(batch()))
    with pytest.raises(ValueError):
        ModelConfig(max_steps=0)


def test_training_time_subset_keeps_full_validation_and_correct_training_denominator(tmp_path):
    from jinyiquant.config import RunConfig, load_config
    from jinyiquant.io import read_json
    from jinyiquant.runner import run
    from scripts.tlob_fixture import create_fixture
    path = create_fixture(tmp_path / "fixture")
    for filename in ("bars.parquet", "labels.parquet", "instruments.parquet"):
        original = pd.read_parquet(path.parent / filename)
        later = original.copy()
        later["date"] += pd.Timedelta(hours=4, minutes=30)
        if "label_end" in later:
            later["label_end"] += pd.Timedelta(hours=4, minutes=30)
        pd.concat([original, later], ignore_index=True).to_parquet(path.parent / filename, index=False)
    raw = load_config(path).snapshot()
    raw["data"].update(sample_times=["10:00", "14:30"], training_sample_times=["14:30"])
    raw["model"].update(device="cpu", max_steps=5, time_loss_weighting="train_variance")
    config = RunConfig.model_validate(raw)
    output, code = run(config)
    assert code == 0, (output / "worker.log").read_text(encoding="utf-8")
    training = pd.read_parquet(output / "train_history.parquet")
    validation = pd.read_parquet(output / "evaluation_history.parquet")
    assert len(training) == 10 and set(training.date.dt.strftime("%H:%M")) == {"14:30"}
    assert len(validation) == 20 and set(validation.date.dt.strftime("%H:%M")) == {"10:00", "14:30"}
    check = next(row for row in read_json(output / "checks.json")["checks"]
                 if row["name"] == "train_expected_universe_coverage")
    assert check["passed"] and check["details"]["expected_keys"] == 10
    assert check["details"]["minimum"] == 1
    summary = next(event["context"] for event in map(json.loads, (output / "events.jsonl").read_text().splitlines())
                   if event["stage"] == "training" and event["event_type"] == "progress"
                   and "epochs_completed" in event["context"])
    assert summary["samples_seen"] > len(training)
    assert summary["throughput_samples_per_second"] * summary["elapsed_seconds"] == pytest.approx(summary["samples_seen"])
    from modified_tlob.diagnostics import _prepare
    destination = tmp_path / "diagnostic-data"
    recorder = SimpleNamespace(event=lambda *args, **kwargs: None)
    _, _, targets, _, validation_batch, _ = _prepare(config, destination, recorder)
    assert len(targets) == 10 and set(targets.date.dt.strftime("%H:%M")) == {"14:30"}
    assert len(validation_batch.keys) == 20
    coverage = next(row for row in read_json(destination / "checks.json")["checks"]
                    if row["name"] == "train_answer_coverage")
    assert coverage["passed"] and coverage["details"]["expected_keys"] == 10
    for times in ([], ["11:00"], ["10:00", "10:00"]):
        with pytest.raises(ValueError, match="training_sample_times"):
            DataConfig.model_validate({**config.data.model_dump(), "training_sample_times": times})


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


@pytest.mark.parametrize("configured", [None, ":16:8", ":4096:8"])
def test_cuda_workspace_prepared_before_preflight_and_logged(monkeypatch, configured):
    from modified_tlob import device
    name = "CUBLAS_WORKSPACE_CONFIG"
    if configured is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, configured)
    expected = configured or ":4096:8"
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    observed, events = [], []
    monkeypatch.setattr(device, "_cuda_preflight", lambda: observed.append(os.environ[name]))
    monkeypatch.setattr(device, "device_resources", lambda selected: {"device": str(selected)})
    recorder = SimpleNamespace(event=lambda *args, **context: events.append(context))
    assert str(select_device("auto", recorder)) == "cuda"
    assert observed == [expected]
    assert events[-1]["cublas_workspace_config"] == expected


@pytest.mark.parametrize("requested", ["auto", "cuda"])
@pytest.mark.parametrize("configured", ["", ":1:1", "4096:8"])
def test_invalid_cuda_workspace_fails_before_preflight(monkeypatch, requested, configured):
    from modified_tlob import device
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", configured)
    monkeypatch.setattr(device, "_cuda_preflight", lambda: pytest.fail("Invalid config reached CUDA"))
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG must be"):
        select_device(requested)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == configured


def test_missing_workspace_after_cuda_initialization_requires_fresh_process(monkeypatch):
    from modified_tlob import device
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(device, "_cuda_preflight", lambda: pytest.fail("Late config reached CUDA"))
    with pytest.raises(RuntimeError, match="fresh process"):
        select_device("auto")


def test_adapter_determinism_event_records_actual_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    events = []
    recorder = SimpleNamespace(event=lambda stage, event, **context: events.append((event, context)))
    TLOBAdapter(adapter().config, data_config(), recorder=recorder)
    determinism = next(context for event, context in events if event == "determinism")
    assert determinism["cuda_matmul_workspace"] == ":16:8"


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


def test_default_target_scaling_matches_raw_return_adam_update():
    b = batch()
    target = np.linspace(-.01, .01, len(b.x), dtype=np.float32)
    cfg = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=len(b.x),
                      epochs=1, device="cpu")
    legacy = cfg.model_dump()
    legacy.pop("target_scaling")
    legacy.pop("time_loss_weighting")
    m, reference = TLOBAdapter(legacy, data_config()), TLOBAdapter(cfg, data_config())
    optimizer = torch.optim.Adam(reference.network.parameters(), lr=cfg.learning_rate)
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(m.seed)).numpy()
    loss = torch.nn.functional.mse_loss(reference.network(torch.from_numpy(b.x[order])),
                                        torch.from_numpy(target[order]))
    loss.backward()
    optimizer.step()
    progress = m.fit(b, target)
    assert m.config["target_scaling"] == "none" and m.target_scale == 1.0
    assert m.config["time_loss_weighting"] == "none" and m.time_loss_weighting_stats is None
    assert progress["loss"] == pytest.approx(loss.item())
    assert progress["loss"] == progress["weighted_loss"]
    for name, parameter in m.network.state_dict().items():
        torch.testing.assert_close(parameter, reference.network.state_dict()[name], atol=0, rtol=0)


def test_target_scale_uses_only_selected_training_targets_and_logs_raw_units():
    b = batch()
    target = np.array([-.003, -.001, .001, .003, .004, .006, 900.], dtype=np.float32)
    mask = np.arange(len(target)) < 6
    cfg = {**adapter().config, "target_scaling": "train_std"}
    events, observed_scales = [], []
    recorder = SimpleNamespace(event=lambda stage, event, **context: events.append((stage, event, context)))
    model = TLOBAdapter(cfg, data_config(), recorder=recorder)
    validation_target = np.linspace(-.01, .01, len(target), dtype=np.float32)
    progress = model.fit(b, target, sample_mask=mask, validation=(b, validation_target),
                         observer=lambda m, *_: observed_scales.append(m.target_scale))
    expected_scale = float(target[mask].astype(np.float64).std())
    assert model.target_scale == expected_scale
    assert observed_scales == [expected_scale] * (cfg["epochs"] + 1)
    changed = target.copy()
    changed[~mask] = -900.
    other = TLOBAdapter(cfg, data_config())
    other.fit(b, changed, sample_mask=mask, validation=(b, validation_target * 1000))
    assert other.target_scale == model.target_scale and other.intercept == model.intercept
    for name, parameter in model.network.state_dict().items():
        torch.testing.assert_close(parameter, other.network.state_dict()[name], atol=0, rtol=0)
    assert progress["loss"] == pytest.approx(progress["optimization_loss"] * expected_scale ** 2)
    training_events = [context for stage, _, context in events if stage == "training" and "loss" in context]
    assert training_events
    for context in training_events:
        assert context["loss"] == pytest.approx(context["optimization_loss"] * expected_scale ** 2)
    validation_events = [context for stage, event, context in events
                         if stage == "validation" and event == "epoch_completed"]
    assert validation_events[-1]["loss"] == pytest.approx(
        np.square(model.predict(b) - validation_target).mean(), rel=1e-6)


@pytest.mark.parametrize("filename", ["model.pt", "weights.json"])
def test_scaled_predictions_observers_diagnostics_and_checkpoints_share_raw_units(tmp_path, filename):
    from modified_tlob.diagnostics import _network_scores

    b = batch()
    m = TLOBAdapter({**adapter().config, "target_scaling": "train_std"}, data_config())
    observations = []

    def observe(model, epoch, steps):
        with torch.inference_mode():
            expected = model.network(torch.from_numpy(b.x)).double().numpy() * model.target_scale
        np.testing.assert_allclose(model._predict(b), expected, atol=1e-8, rtol=1e-6)
        np.testing.assert_array_equal(_network_scores(model, b), model._predict(b))
        observations.append(epoch)

    m.fit(b, np.linspace(-.01, .02, len(b.x)), observer=observe)
    assert observations == [0, 1, 2]
    b.observed[0] = False
    expected = m.predict(b)
    assert expected[0] == m.intercept
    path = tmp_path / filename
    m.save(path) if path.suffix == ".pt" else m.export_text(path)
    payload = json.loads(path.read_text()) if path.suffix == ".json" else torch.load(path, weights_only=True)
    assert payload["schema_version"] == 2  # The old loader accepts only v1, so cannot misread scaled units.
    restored = TLOBAdapter.load(path)
    assert restored.target_scale == m.target_scale
    np.testing.assert_array_equal(restored.predict(b), expected)
    np.testing.assert_array_equal(_network_scores(restored, b), expected)


@pytest.mark.parametrize("filename", ["model.pt", "weights.json"])
def test_legacy_checkpoint_without_target_scale_keeps_unit_scale(tmp_path, filename):
    m, b = adapter(), batch()
    m.fit(b, np.linspace(-.01, .01, len(b.x)))
    path = tmp_path / filename
    m.save(path) if path.suffix == ".pt" else m.export_text(path)
    payload = json.loads(path.read_text()) if path.suffix == ".json" else torch.load(path, weights_only=True)
    assert payload["schema_version"] == 1
    payload.pop("target_scale")
    payload["model_config"].pop("target_scaling")
    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        torch.save(payload, path)
    restored = TLOBAdapter.load(path)
    assert restored.target_scale == 1.0 and restored.config["target_scaling"] == "none"
    np.testing.assert_array_equal(restored.predict(b), m.predict(b))


def test_scaled_checkpoint_rejects_invalid_or_missing_scale_and_downgraded_version(tmp_path):
    m, b = TLOBAdapter({**adapter().config, "target_scaling": "train_std"}, data_config()), batch()
    m.fit(b, np.linspace(-.01, .01, len(b.x)))
    path = tmp_path / "scaled.json"
    m.export_text(path)
    original = json.loads(path.read_text())
    for changes in ({"target_scale": 0}, {"target_scale": -1}, {"target_scale": float("inf")},
                    {"target_scale": float("nan")}, {"schema_version": 1}, {"target_scale": None}):
        payload = {**original, **changes}
        if payload["target_scale"] is None:
            payload.pop("target_scale")
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="scale|scaling"):
            TLOBAdapter.load(path)


@pytest.mark.parametrize("constant", [0., .003])
def test_constant_training_targets_have_finite_unit_scale_fallback(constant):
    b = batch()
    events = []
    recorder = SimpleNamespace(event=lambda stage, event, **context: events.append((event, context)))
    model = TLOBAdapter({**adapter().config, "target_scaling": "train_std"}, data_config(), recorder=recorder)
    result = model.fit(b, np.full(len(b.x), constant))
    assert model.target_scale == 1.0 and np.isfinite(result["loss"])
    assert np.isfinite(model.predict(b)).all()
    assert next(context for event, context in events if event == "target_scaling_fitted")["constant_target_fallback"]


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
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT)))})
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
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT)))})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", ["model.pt", "weights.json"])
def test_local_inference_entry_never_trains_and_preserves_predictions(tmp_path, monkeypatch, filename):
    from jinyiquant.checks import verify_artifacts
    from modified_tlob.infer import infer
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    b, m = batch(), adapter()
    m.fit(b, np.zeros(7))
    path = tmp_path / filename
    m.save(path) if path.suffix == ".pt" else m.export_text(path)
    rows = []
    for i, row in enumerate(b.keys.itertuples(index=False)):
        for lag in range(8):
            rows.append({"date": row.date - pd.Timedelta(minutes=7 - lag), "instrument": row.instrument,
                         **dict(zip(data_config().feature_columns, b.x[i, lag]))})
    bars_path, keys_path = tmp_path / "bars.parquet", tmp_path / "keys.parquet"
    pd.DataFrame(rows).assign(vwap_return=999).to_parquet(bars_path, index=False)
    b.keys.to_parquet(keys_path, index=False)
    monkeypatch.setattr(TLOBAdapter, "fit", lambda *a, **kw: pytest.fail("Inference attempted training"))
    destination = infer(path, bars_path, keys_path, tmp_path / "inference", device="cpu", min_coverage=1)
    result = pd.read_parquet(destination / "predictions.parquet")
    assert list(result) == ["date", "instrument", "factor"]
    np.testing.assert_allclose(result.factor, m.predict(b), atol=1e-6, rtol=1e-5)
    assert verify_artifacts(destination)["passed"]
    assert json.loads((destination / "config.json").read_text(encoding="utf-8"))["data"]["labels_path"] is None
    # Direct file execution must work from another cwd without PYTHONPATH.
    import os
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, str(ROOT / "modified_tlob/infer.py"), "--weights", str(path),
               "--bars", str(bars_path), "--keys", str(keys_path), "--device", "cpu",
               "--output-root", str(tmp_path / "cli"), "--start-date", "2020-01-03", "--end-date", "2020-01-04"]
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    cli = pd.read_parquet(json.loads(completed.stdout)["predictions"])
    assert cli.date.eq(pd.Timestamp("2020-01-03 10:00")).all() and len(cli) == 1


def test_local_inference_failure_archived_without_output(tmp_path):
    from modified_tlob.infer import infer
    with pytest.raises(FileNotFoundError):
        infer(tmp_path / "missing.pt", tmp_path / "bars.parquet", tmp_path / "keys.parquet", tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not (run_dir / "predictions.parquet").exists()


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
