import hashlib
import io
import json
import tarfile

import pytest

from jinyiquant.preparation import prepare_data


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries:
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


def test_prepares_exactly_one_member_without_overwriting(tmp_path):
    path = tmp_path / "bars.tar.gz"
    make_archive(path, [("bars/202001.feather", b"selected"), ("bars/202002.feather", b"unrelated")])
    first = prepare_data(path, "bars/202001.feather", tmp_path / "prepared")
    second = prepare_data(path, "bars/202001.feather", tmp_path / "prepared")
    assert first != second
    assert (first / "202001.feather").read_bytes() == b"selected"
    assert not (first / "202002.feather").exists()
    manifest = json.loads((first / "preparation.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(b"selected").hexdigest()
    assert manifest["source"]["sha256"] is None
    assert manifest["source"]["verification"] == "stat_and_selected_member_only"


@pytest.mark.parametrize("member", ["../escape.feather", "/absolute.feather", "C:/bad.feather", "bars\\bad.feather"])
def test_rejects_traversal_before_creating_output(tmp_path, member):
    with pytest.raises(ValueError):
        prepare_data(tmp_path / "absent.tar.gz", member, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_budget_failure_is_recorded_and_no_complete_data_is_published(tmp_path):
    archive_path = tmp_path / "bars.tar.gz"
    make_archive(archive_path, [("bars/202001.feather", b"too large")])
    with pytest.raises(ValueError, match="budget"):
        prepare_data(archive_path, "bars/202001.feather", tmp_path / "prepared", max_member_bytes=1)
    manifest_path = next((tmp_path / "prepared").glob("*/preparation.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "traceback" in manifest["error"]
    assert not list(manifest_path.parent.glob("*.feather"))


def test_rejects_symbolic_link_member(tmp_path):
    archive_path = tmp_path / "bars.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("bars/202001.feather")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member)
    with pytest.raises(ValueError, match="regular file"):
        prepare_data(archive_path, "bars/202001.feather", tmp_path / "prepared")
