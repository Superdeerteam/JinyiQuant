"""Research validation may give epoch feedback; final checks must not."""

import pandas as pd
import pytest

from jinyiquant.config import RunConfig, load_config
from jinyiquant.io import read_json, write_json
from jinyiquant.runner import execute_pipeline
from scripts.tlob_fixture import create_fixture


@pytest.mark.parametrize("evaluation_split", ["validation", "final"])
def test_training_receives_only_research_validation(tmp_path, monkeypatch, evaluation_split):
    pytest.importorskip("torch")
    from modified_tlob.adapter import TLOBAdapter

    config = load_config(create_fixture(tmp_path / "fixture"))
    raw = config.snapshot()
    raw.update(evaluation_split=evaluation_split, final_access={"purpose": "synthetic isolation test"})
    config = RunConfig.model_validate(raw)
    original_fit = TLOBAdapter.fit
    calls = []
    from jinyiquant import runner

    original_load = runner.load_data
    original_save = TLOBAdapter.save
    checkpoint_saved = False
    reads = []

    def guarded_load(data, start=None, end=None):
        reads.append((end, checkpoint_saved))
        if evaluation_split == "final" and not checkpoint_saved:
            assert end <= config.splits.train.end
        return original_load(data, start=start, end=end)

    def tracked_save(self, path):
        nonlocal checkpoint_saved
        original_save(self, path)
        checkpoint_saved = True

    monkeypatch.setattr(runner, "load_data", guarded_load)
    monkeypatch.setattr(TLOBAdapter, "save", tracked_save)

    def tracked_fit(self, batch, target, **options):
        calls.append(options.get("validation"))
        return original_fit(self, batch, target, **options)

    monkeypatch.setattr(TLOBAdapter, "fit", tracked_fit)
    directory = tmp_path / "run"
    directory.mkdir()
    write_json(directory / "manifest.json", {"environment": {"determinism": {}}})
    execute_pipeline(config, directory)
    assert len(calls) == 1
    if evaluation_split == "final":
        assert calls == [None]
        assert reads == [(config.splits.train.end, False), (config.splits.final.end, True)]
    else:
        batch, targets = calls[0]
        assert len(batch.keys) == len(targets) > 0
        assert batch.keys.date.ge(config.splits.validation.start).all()
        assert batch.keys.date.lt(config.splits.validation.end).all()
    predictions = pd.read_parquet(directory / "predictions.parquet")
    split = getattr(config.splits, evaluation_split)
    assert predictions.date.ge(split.start).all() and predictions.date.lt(split.end).all()
    assert read_json(directory / "checks.json")["passed"]


def _ridge_fixture(tmp_path):
    """Deterministic labeled fixture; ridge keeps this boundary test free of torch."""
    from modified_tlob.schema import lob_columns

    config = load_config(create_fixture(tmp_path / "fixture"))
    raw = config.snapshot()
    raw["model"] = {**raw["model"], "kind": "ridge", "alpha": 1.0}
    raw["data"]["feature_columns"] = list(lob_columns(3))
    return RunConfig.model_validate(raw)


def _execute(config, directory):
    directory.mkdir()
    write_json(directory / "manifest.json", {"environment": {"determinism": {}}})
    execute_pipeline(config, directory)
    return directory


def test_future_market_and_later_split_answers_never_move_earlier_predictions(tmp_path):
    """Train-only normalization/labels and causal windows: later data cannot rewrite earlier outputs."""
    config = _ridge_fixture(tmp_path)
    baseline = _execute(config, tmp_path / "baseline")
    original = pd.read_parquet(baseline / "predictions.parquet")
    weights = (baseline / "model.npz").read_bytes()
    assert read_json(baseline / "checks.json")["passed"]

    cutoff = pd.Timestamp("2020-01-06 10:00")
    for name in ("bars", "labels"):
        frame = pd.read_parquet(tmp_path / "fixture" / f"{name}.parquet")
        later = frame.date > cutoff
        assert later.any()
        if name == "bars":
            for column in config.data.feature_columns:
                frame.loc[later, column] = 1e9
        else:
            frame.loc[later, "vwap_return"] = -1234.0
        frame.to_parquet(tmp_path / "fixture" / f"{name}.parquet", index=False)
    changed = _execute(config, tmp_path / "changed")
    altered = pd.read_parquet(changed / "predictions.parquet")

    assert read_json(changed / "checks.json")["passed"]
    earlier = original.date <= cutoff
    assert earlier.any() and (~earlier).any()
    pd.testing.assert_frame_equal(original.loc[earlier].reset_index(drop=True),
                                  altered.loc[earlier].reset_index(drop=True), check_exact=True)
    # The fitted model only saw training-split rows: later answers changed no weight or scale.
    assert (changed / "model.npz").read_bytes() == weights
    # The mutation did reach the later keys, so unchanged earlier predictions are not vacuous.
    later = ~earlier
    assert not altered.loc[later, "score"].equals(original.loc[later, "score"].reset_index(drop=True))
