"""V0/V1 version contracts: only the training label cross-sectional transform changes."""

import json
from pathlib import Path

import pandas as pd
import pytest

from jinyiquant.config import RunConfig, load_config
from jinyiquant.io import read_json

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "configs/model_v0.json"
V1 = ROOT / "configs/model_v1.json"
CSZ = "timestamp_zscore_train_rms_matched"
IDENTITY = {"experiment_id", "hypothesis", "output_root"}
SHARED = ("data", "splits", "budget", "acceptance", "evaluation", "logging", "seed", "evaluation_split")


def test_versions_change_only_the_label_transform():
    v0, v1 = load_config(V0).snapshot(), load_config(V1).snapshot()
    assert {key for key in v0 if v0[key] != v1.get(key)} == IDENTITY | {"model"}
    model_keys = {key for key in v0["model"] if v0["model"][key] != v1["model"].get(key)}
    assert model_keys == {"target_centering"}
    assert (v0["model"]["target_centering"], v1["model"]["target_centering"]) == ("none", CSZ)


def test_versions_keep_network_inputs_dates_and_budget_frozen():
    v0, v1 = load_config(V0).snapshot(), load_config(V1).snapshot()
    for key in SHARED:
        assert v0[key] == v1[key], key
    assert v0["splits"] == {"train": {"start": "2023-10-18T00:00:00", "end": "2023-11-01T00:00:00"},
                            "validation": {"start": "2023-11-01T00:00:00", "end": "2023-11-15T00:00:00"},
                            "final": {"start": "2024-10-08T00:00:00", "end": "2025-01-01T00:00:00"}}
    assert v0["model"]["max_steps"] == 1500 and v1["model"]["max_steps"] == 1500
    assert v0["model"]["target_scaling"] == v1["model"]["target_scaling"] == "train_std"
    assert v0["data"]["history_days"] == v1["data"]["history_days"] == 10


def test_label_csz_trains_through_the_unified_runner(tmp_path):
    pytest.importorskip("torch")
    from jinyiquant.runner import run
    from scripts.tlob_fixture import create_fixture

    raw = load_config(create_fixture(tmp_path / "fixture")).snapshot()
    raw["model"].update(target_centering=CSZ, target_scaling="train_std", max_steps=5, device="cpu")
    output, code = run(RunConfig.model_validate(raw))
    assert code == 0, (output / "worker.log").read_text(encoding="utf-8")
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    fitted = next(event["context"] for event in events
                  if event["stage"] == "training" and event["event_type"] == "target_centering_fitted")
    assert fitted["method"] == CSZ and fitted["c"] > 0
    assert fitted["transformed_std"] == pytest.approx(fitted["centered_std"], rel=1e-9)
    assert any(event["event_type"] == "validation_diagnostic_skipped" for event in events)
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert len(predictions) and predictions.score.abs().max() > 0
    # Inference keeps network(x) * raw-label target_scale: it never re-applies c or labels.
    assert read_json(output / "metrics.json")["coverage"]["scored_samples"] > 0
