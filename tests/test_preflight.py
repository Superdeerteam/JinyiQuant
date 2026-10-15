"""Read-only input-location checks are distinct from data or model acceptance."""

import json
from pathlib import Path

import pytest

from jinyiquant.cli import main
from jinyiquant.config import load_config
from jinyiquant.io import write_json


def make_config(tmp_path, *, optional=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "bars").mkdir()
    (inputs / "labels.parquet").write_bytes(b"unparsed input")
    data = {"bars_path": "../inputs/bars", "labels_path": "../inputs/labels.parquet",
            "source": "preflight-fixture", "version": "1"}
    if optional:
        (inputs / "instruments.parquet").write_bytes(b"unparsed input")
        (inputs / "mapping.csv").write_bytes(b"unparsed input")
        data.update(instruments_path="../inputs/instruments.parquet", instrument_map_path="../inputs/mapping.csv")
    config = tmp_path / "configs" / "run.json"
    write_json(config, {"experiment_id": "preflight", "hypothesis": "Check locations only", "data": data,
                       "splits": {"train": {"start": "2020-01-02", "end": "2020-01-03"},
                                  "validation": {"start": "2020-01-03", "end": "2020-01-04"},
                                  "final": {"start": "2020-01-04", "end": "2020-01-05"}}})
    return config


def test_check_inputs_resolves_relative_paths_across_working_directories_without_reading_data(
        tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path, optional=True)
    before = sorted(tmp_path.rglob("*"))
    original_open = Path.open

    def no_data_open(path, *args, **kwargs):
        assert not path.resolve().is_relative_to(tmp_path / "inputs"), "Preflight must not read data"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_data_open)
    monkeypatch.chdir(tmp_path / "inputs")
    assert main(["validate-config", "../configs/run.json", "--check-inputs"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["inputs_present"]
    assert len(report["checks"]) == 4
    assert all(check["passed"] and Path(check["path"]).is_absolute() for check in report["checks"])
    assert {check["kind"] for check in report["checks"]} == {"file", "directory"}
    assert sorted(tmp_path.rglob("*")) == before
    assert not (config.parent / "runs").exists()


@pytest.mark.parametrize("missing", ["bars_path", "labels_path", "instruments_path", "instrument_map_path"])
def test_check_inputs_reports_missing_required_or_configured_optional_path(tmp_path, capsys, missing):
    config = make_config(tmp_path, optional=True)
    path = getattr(load_config(config).data, missing)
    path.rmdir() if path.is_dir() else path.unlink()
    assert main(["validate-config", str(config), "--check-inputs"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["inputs_present"] is False
    failed = [check for check in report["checks"] if not check["passed"]]
    assert failed == [{"name": missing, "path": str(path), "kind": "missing_or_other", "passed": False}]


def test_check_inputs_accepts_label_directory_and_unconfigured_optional_paths(tmp_path, capsys):
    config = make_config(tmp_path)
    labels = load_config(config).data.labels_path
    labels.unlink()
    labels.mkdir()
    assert main(["validate-config", str(config), "--check-inputs"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inputs_present"] is True
    assert [check["name"] for check in report["checks"]] == ["bars_path", "labels_path"]
    assert all(check["kind"] == "directory" for check in report["checks"])


def test_default_validate_config_output_stays_compatible_with_missing_input(tmp_path, capsys):
    config = make_config(tmp_path)
    load_config(config).data.labels_path.unlink()
    assert main(["validate-config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out) == {"valid": True, "config_sha256": load_config(config).checksum}
