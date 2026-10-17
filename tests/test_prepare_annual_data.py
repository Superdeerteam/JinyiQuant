import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts import prepare_annual_data as prep


def _sources(root, bad=None):
    root.mkdir()
    for name in (prep.LABEL_ARCHIVE, *prep.STATIC_FILES):
        (root / name).write_bytes(b"fixture")
    with tarfile.open(root / prep.BAR_ARCHIVE, "w:gz") as archive:
        for name in (*prep.BAR_MEMBERS, "cpt_jyc_2026_e2e_bar1m/202407.0.feather", "../escape.feather"):
            member = tarfile.TarInfo(name)
            if bad == "link" and name == prep.BAR_MEMBERS[0]:
                member.type, member.linkname = tarfile.SYMTYPE, "../../escape"
                archive.addfile(member)
            else:
                member.size = 3
                archive.addfile(member, io.BytesIO(b"bar"))
        if bad == "duplicate":
            member = tarfile.TarInfo(prep.BAR_MEMBERS[0])
            member.size = 3
            archive.addfile(member, io.BytesIO(b"bar"))


def _fake_labels(monkeypatch, calls):
    monkeypatch.setattr(prep.shutil, "which", lambda name: "/usr/bin/bsdtar")
    monkeypatch.setattr(prep.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, "\n".join((*prep.LABEL_MEMBERS, "cpt_jyc_2026_vwap/cpt_jyc_2026_vwap_2024-07.parquet")), ""))

    def extract(executable, archive, member, target):
        calls.append(member)
        artifact, partial = prep._copy_artifact(io.BytesIO(b"label"), target, 1024)
        partial.rename(target)
        return artifact

    monkeypatch.setattr(prep, "_extract_label", extract)


def test_exact_months_hashes_and_no_overwrite(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    _sources(source)
    calls = []
    _fake_labels(monkeypatch, calls)
    result = prep.prepare(source, output)
    assert result["status"] == "succeeded"
    assert calls == list(prep.LABEL_MEMBERS)
    assert sorted(p.name for p in (output / "bars").iterdir()) == sorted(Path(x).name for x in prep.BAR_MEMBERS)
    assert sorted(p.name for p in (output / "labels").iterdir()) == sorted(Path(x).name for x in prep.LABEL_MEMBERS)
    assert len(result["artifacts"]) == 28
    assert not (tmp_path / "escape.feather").exists()
    for artifact in result["artifacts"]:
        payload = Path(artifact["path"]).read_bytes()
        assert artifact["bytes"] == len(payload)
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        prep.prepare(source, output)


@pytest.mark.parametrize("bad, match", [("link", "regular file"), ("duplicate", "Duplicate")])
def test_invalid_selected_bar_fails_before_labels(tmp_path, monkeypatch, bad, match):
    source, output = tmp_path / "source", tmp_path / "output"
    _sources(source, bad)
    calls = []
    _fake_labels(monkeypatch, calls)
    with pytest.raises(ValueError, match=match):
        prep.prepare(source, output)
    assert calls == []
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "traceback" in manifest["error"]


def test_extractor_failure_does_not_publish_partial_label(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    _sources(source)
    monkeypatch.setattr(prep.shutil, "which", lambda name: "/usr/bin/bsdtar")
    monkeypatch.setattr(prep.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0, "\n".join(prep.LABEL_MEMBERS), ""))

    class FailedExtractor:
        def __init__(self, command, stdout, stderr):
            self.stdout = io.BytesIO(b"partial")

        def wait(self, timeout=None):
            return 2

        def poll(self):
            return 2

    monkeypatch.setattr(prep.subprocess, "Popen", FailedExtractor)
    with pytest.raises(RuntimeError, match="Label extraction failed"):
        prep.prepare(source, output)
    assert list((output / "labels").glob("*.partial"))
    assert not list((output / "labels").glob("*.parquet"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
