"""Step diagnostics preserve optimization and never expose sealed final data."""

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from jinyiquant.config import DataConfig, LoggingConfig, ModelConfig, RunConfig, load_config  # noqa: E402
from jinyiquant.data import Batch  # noqa: E402
from jinyiquant.io import read_json, write_json  # noqa: E402
from jinyiquant.runner import execute_pipeline  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402
from scripts.tlob_fixture import create_fixture  # noqa: E402


def _training_case():
    data = DataConfig(bars_path="unused", labels_path="unused", source="synthetic", version="1",
                      feature_columns=lob_columns(3), lookback=8, min_history=2, sample_times=("10:00",))
    config = ModelConfig(kind="modified_tlob", hidden_dim=8, num_layers=2, batch_size=3,
                         epochs=3, max_steps=7, device="cpu", target_scaling="train_std")
    x = np.random.default_rng(31).normal(size=(7, 8, 12)).astype(np.float32)
    keys = pd.DataFrame({"date": pd.date_range("2020-01-02 10:00", periods=7), "instrument": "A"})
    batch = Batch(keys, x, np.ones_like(x, dtype=bool), np.full((7, 8), np.datetime64("NaT", "ns")), {})
    return data, config, batch, np.linspace(-.01, .02, 7)


@pytest.mark.parametrize("interval,expected", [
    (None, [0, 3, 6, 7]), (1, list(range(8))), (2, [0, 2, 3, 4, 6, 7]),
    (3, [0, 3, 6, 7]), (7, [0, 3, 6, 7]),
])
def test_step_observations_preserve_parameters_predictions_rng_and_training_mode(interval, expected):
    data, config, batch, target = _training_case()
    results, checkpoints = [], []
    for observed in (False, True):
        np.random.seed(123)
        random.seed(123)
        model = TLOBAdapter(config, data, seed=42)
        training_modes = []
        hook = model.network.register_forward_pre_hook(
            lambda module, _: training_modes.append(module.training) if torch.is_grad_enabled() else None)

        def observer(adapter, epoch, step):
            checkpoints.append((epoch, step))
            adapter._predict(batch)
            # A diagnostic may draw randomness, but it must not perturb subsequent updates.
            torch.rand(4)
            np.random.random(4)
            random.random()

        progress = model.fit(batch, target, observer=observer if observed else None,
                             observer_every_steps=interval)
        hook.remove()
        assert all(training_modes) and len(training_modes) == 7
        assert model.network.training
        results.append((model, progress, torch.get_rng_state(), np.random.random(4), random.random()))
    assert [step for _, step in checkpoints] == expected
    assert checkpoints[-1][0] == results[0][1]["epochs"] == pytest.approx(2 + 3 / 7)
    assert results[0][1] == results[1][1]
    for key, value in results[0][0].network.state_dict().items():
        assert torch.equal(value, results[1][0].network.state_dict()[key])
    np.testing.assert_array_equal(results[0][0].predict(batch), results[1][0].predict(batch))
    assert torch.equal(results[0][2], results[1][2])
    np.testing.assert_array_equal(results[0][3], results[1][3])
    assert results[0][4] == results[1][4]


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_step_interval_requires_positive_integer(value):
    with pytest.raises(ValueError):
        LoggingConfig(learning_curve_every_steps=value)
    data, config, batch, target = _training_case()
    with pytest.raises(ValueError, match="observer_every_steps"):
        TLOBAdapter(config, data).fit(batch, target, observer_every_steps=value)


@pytest.mark.parametrize("evaluation_split", ["validation", "final"])
def test_step_curve_metrics_match_final_evaluator_and_keep_final_sealed(tmp_path, monkeypatch, evaluation_split):
    from jinyiquant import runner

    path = create_fixture(tmp_path / "fixture")
    # Add opening sections and one missing validation answer; inference must retain its key.
    for name in ("bars", "labels", "instruments"):
        frame = pd.read_parquet(path.parent / f"{name}.parquet")
        opening = frame.copy()
        opening["date"] -= pd.Timedelta(minutes=30)
        if "label_end" in opening:
            opening["label_end"] -= pd.Timedelta(minutes=30)
        combined = pd.concat([frame, opening], ignore_index=True)
        if name == "labels":
            combined = combined.loc[~(combined.date.eq(pd.Timestamp("2020-01-06 09:30"))
                                      & combined.instrument.eq("000005.SZ"))]
        combined.to_parquet(path.parent / f"{name}.parquet", index=False)
    raw = load_config(path).snapshot()
    raw.update(evaluation_split=evaluation_split, final_access={"purpose": "Synthetic isolation test"})
    raw["data"]["sample_times"] = ["09:30", "10:00"]
    raw["acceptance"]["min_coverage"] = .8
    raw["model"].update(device="cpu", max_steps=5, epochs=3)
    raw["logging"].update(learning_curves=True, learning_curve_every_steps=2)
    config = RunConfig.model_validate(raw)
    original_load, original_save, original_evaluate = runner.load_data, TLOBAdapter.save, runner.evaluate
    checkpoint_saved, reads, evaluation_before_save = False, [], []

    def guarded_load(data, start=None, end=None):
        reads.append((end, checkpoint_saved))
        if evaluation_split == "final" and not checkpoint_saved:
            assert end == config.splits.train.end
        return original_load(data, start=start, end=end)

    def tracked_save(self, destination):
        nonlocal checkpoint_saved
        original_save(self, destination)
        checkpoint_saved = True

    def tracked_evaluate(*args, **kwargs):
        evaluation_before_save.append(not checkpoint_saved)
        if evaluation_split == "final":
            assert checkpoint_saved
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "load_data", guarded_load)
    monkeypatch.setattr(TLOBAdapter, "save", tracked_save)
    monkeypatch.setattr(runner, "evaluate", tracked_evaluate)
    destination = tmp_path / "run"
    destination.mkdir()
    write_json(destination / "manifest.json", {"environment": {"determinism": {}}})
    execute_pipeline(config, destination)
    curve = read_json(destination / "learning_curve.json")
    assert [row["steps"] for row in curve] == [0, 2, 4, 5]
    assert read_json(destination / "checks.json")["passed"]
    if evaluation_split == "final":
        assert reads == [(config.splits.train.end, False), (config.splits.final.end, True)]
        assert evaluation_before_save == [False]
        assert all(row["validation_mse"] is None and "validation_rank_ic_mean" not in row for row in curve)
    else:
        assert evaluation_before_save == [True, True, True, True, False]
        metrics = read_json(destination / "metrics.json")
        for metric in ("rank_ic_mean", "ic_mean", "long_short_return_mean", "turnover_mean",
                       "min_daily_long_short_return"):
            assert curve[-1]["validation_" + metric] == pytest.approx(metrics[metric])
        assert curve[-1]["validation_evaluation_mse"] == pytest.approx(metrics["mse"])
        predictions = pd.read_parquet(destination / "predictions.parquet")
        targets = pd.read_parquet(destination / "targets.parquet")
        aligned = predictions.merge(targets, on=["date", "instrument"], validate="one_to_one")
        assert len(predictions) == 20 and len(aligned) == curve[-1]["validation_scored_samples"] == 19
        assert curve[-1]["validation_mse"] == pytest.approx(np.square(aligned.score - aligned.target).mean())
        sections = pd.read_csv(destination / "by_timestamp.csv")
        opening = sections.loc[pd.to_datetime(sections.timestamp).dt.strftime("%H:%M").eq("09:30")]
        assert curve[-1]["validation_opening_0930_sections"] == 2
        assert curve[-1]["validation_opening_0930_rank_ic_mean"] == pytest.approx(opening.rank_ic.mean())
        assert curve[-1]["validation_opening_0930_long_short_return_mean"] == pytest.approx(
            opening.long_short_return.mean())
