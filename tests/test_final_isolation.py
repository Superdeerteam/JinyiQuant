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
    else:
        batch, targets = calls[0]
        assert len(batch.keys) == len(targets) > 0
        assert batch.keys.date.ge(config.splits.validation.start).all()
        assert batch.keys.date.lt(config.splits.validation.end).all()
    predictions = pd.read_parquet(directory / "predictions.parquet")
    split = getattr(config.splits, evaluation_split)
    assert predictions.date.ge(split.start).all() and predictions.date.lt(split.end).all()
    assert read_json(directory / "checks.json")["passed"]
