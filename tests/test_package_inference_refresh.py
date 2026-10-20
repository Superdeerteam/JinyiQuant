"""Refreshing execution code must preserve the fitted submission byte for byte."""

import sys

import pytest

from jinyiquant.io import file_sha256, read_json, write_json
from modified_tlob import package


@pytest.fixture
def existing_package(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    contents = {
        package.INFERENCE_ENTRY: "def main(*args):\n    return 'old'\n",
        "modified_tlob/device.py": "def device_resources(device):\n    return {}\n",
        "modified_tlob/adapter.py": "class TLOBAdapter:\n    pass\n",
        "jinyiquant/data.py": "# frozen preprocessing\n",
        "modified_tlob_weights.json": '{"state_dict": {"test": [1.25, 2.5]}}\n',
        "training_config.json": '{"recipe": "unchanged"}\n',
        "requirements-cpu.lock.txt": "torch==2.5.0\n",
        "inference.ipynb": '{"cells": []}\n',
    }
    for name, content in contents.items():
        path = old / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())
    write_json(old / "submission_config.json", {
        "weights_sha256": file_sha256(old / "modified_tlob_weights.json"), "device": "auto"})
    inventory = {p.relative_to(old).as_posix(): file_sha256(p) for p in old.rglob("*") if p.is_file()}
    write_json(old / "package_manifest.json", {
        "schema_version": 1, "files": inventory, "training_run_id": "frozen-training-001",
        "scope": "development", "platform_acceptance": "previous_status",
        "custom_training_provenance": {"commit": "original"}})
    source_root = tmp_path / "source"
    entry = source_root / package.INFERENCE_ENTRY
    entry.parent.mkdir(parents=True)
    entry.write_text("from .device import device_resources\nfrom .adapter import TLOBAdapter\n"
                     "from jinyiquant.data import build_batch\n"
                     "def main(*args):\n    return 'new'\n", encoding="utf-8")
    monkeypatch.setattr(package, "ROOT", source_root)
    return old, tmp_path / "new"


def test_refresh_preserves_every_other_upload_file_and_training_provenance(existing_package):
    old, destination = existing_package
    original = read_json(old / "package_manifest.json")
    old_manifest_hash = file_sha256(old / "package_manifest.json")
    for name in ("inference_runs/run/events.jsonl", "training_runs/run/metrics.json",
                 "modified_tlob/__pycache__/adapter.pyc"):
        path = old / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime only")
    assert package.refresh_inference_package(old, destination) == destination
    refreshed = read_json(destination / "package_manifest.json")
    assert set(refreshed["files"]) == set(original["files"])
    for name, digest in original["files"].items():
        assert file_sha256(old / name) == digest
        assert file_sha256(destination / name) == (
            file_sha256(package.ROOT / name) if name == package.INFERENCE_ENTRY else digest)
        assert refreshed["files"][name] == file_sha256(destination / name)
    assert file_sha256(old / "package_manifest.json") == old_manifest_hash
    assert refreshed["training_run_id"] == original["training_run_id"]
    assert refreshed["custom_training_provenance"] == original["custom_training_provenance"]
    assert refreshed["scope"] == original["scope"]
    assert refreshed["platform_acceptance"] == "not_verified"
    update = refreshed["inference_update"]
    assert update["previous_package_manifest_sha256"] == old_manifest_hash
    assert update["previous_submission_sha256"] == original["files"][package.INFERENCE_ENTRY]
    assert update["submission_sha256"] == refreshed["files"][package.INFERENCE_ENTRY]
    assert update["source"] == "current_repository_working_tree"
    assert not (destination / "inference_runs").exists()
    assert not (destination / "training_runs").exists()
    assert not (destination / "modified_tlob" / "__pycache__").exists()


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra", "wrong_weight_contract"])
def test_refresh_rejects_unverified_payload_before_creating_destination(existing_package, mutation):
    old, destination = existing_package
    if mutation == "tamper":
        (old / "modified_tlob_weights.json").write_bytes(b"tampered")
    elif mutation == "missing":
        (old / "requirements-cpu.lock.txt").unlink()
    elif mutation == "extra":
        (old / "unlisted.py").write_text("print('unexpected')", encoding="utf-8")
    else:
        write_json(old / "submission_config.json", {"weights_sha256": "0" * 64})
        manifest = read_json(old / "package_manifest.json")
        manifest["files"]["submission_config.json"] = file_sha256(old / "submission_config.json")
        write_json(old / "package_manifest.json", manifest)
    with pytest.raises(ValueError, match="checksum|missing|Unlisted"):
        package.refresh_inference_package(old, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "C:/escape.py",
                                  "modified_tlob\\submission.py", "modified_tlob//submission.py",
                                  "inference_runs/run.py"])
def test_refresh_rejects_unsafe_inventory_paths(existing_package, name):
    old, destination = existing_package
    manifest = read_json(old / "package_manifest.json")
    manifest["files"][name] = "0" * 64
    write_json(old / "package_manifest.json", manifest)
    with pytest.raises(ValueError, match="Unsafe"):
        package.refresh_inference_package(old, destination)
    assert not destination.exists()


def test_refresh_rejects_existing_or_nested_destination(existing_package):
    old, destination = existing_package
    destination.mkdir()
    marker = destination / "keep"
    marker.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        package.refresh_inference_package(old, destination)
    assert marker.read_bytes() == b"preserved"
    with pytest.raises(FileExistsError):
        package.refresh_inference_package(old, old)
    with pytest.raises(ValueError, match="outside"):
        package.refresh_inference_package(old, old / "nested")


def test_refresh_rejects_links(existing_package, tmp_path):
    old, destination = existing_package
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    link = old / "linked"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("This Windows account cannot create symbolic links")
    with pytest.raises(ValueError, match="links"):
        package.refresh_inference_package(old, destination)
    assert not destination.exists()


@pytest.mark.parametrize("missing_symbol", [False, True])
def test_refresh_checks_new_entry_local_dependencies(existing_package, missing_symbol):
    old, destination = existing_package
    manifest = read_json(old / "package_manifest.json")
    device = old / "modified_tlob/device.py"
    if missing_symbol:
        device.write_text("def unrelated():\n    pass\n", encoding="utf-8")
        manifest["files"]["modified_tlob/device.py"] = file_sha256(device)
    else:
        device.unlink()
        del manifest["files"]["modified_tlob/device.py"]
    write_json(old / "package_manifest.json", manifest)
    with pytest.raises(ValueError, match="dependency|device_resources"):
        package.refresh_inference_package(old, destination)
    assert not destination.exists()


def test_refresh_cli_dispatches_without_training(existing_package, monkeypatch, capsys):
    old, destination = existing_package
    monkeypatch.setattr(sys, "argv", ["package", str(old), str(destination), "--refresh-inference"])
    monkeypatch.setattr(package, "build_package", lambda *args: pytest.fail("Refresh must not build or train"))
    package.main()
    assert str(destination) in capsys.readouterr().out
    assert read_json(destination / "package_manifest.json")["training_run_id"] == "frozen-training-001"
