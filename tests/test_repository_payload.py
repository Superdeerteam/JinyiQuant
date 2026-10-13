import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_repository_payload.py"
SPEC = importlib.util.spec_from_file_location("check_repository_payload", SCRIPT)
payload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(payload)


def stage(repository, path, content=b"example\n"):
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    subprocess.run(["git", "-C", str(repository), "add", "--", path], check=True, capture_output=True)
    return target


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    return tmp_path


@pytest.mark.parametrize("path", [
    "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv", "nested/bars.parquet", "nested/BARS.CSV",
    "nested/backup.tar.gz", "nested/samples.npz", "runs/run-1/metrics.json", "nested/数据/input.txt",
    "nested/.venv/settings.json", "events.jsonl", "nested/.env.local", "model.pth",
])
def test_rejects_data_even_when_explicitly_staged(repository, path):
    stage(repository, path)
    result = payload.check_repository(repository)
    assert result["passed"] is False
    assert result["violations"][0]["path"] == path


def test_allows_source_configs_and_existing_competition_documents(repository):
    for path in ("src/jinyiquant/demo.py", "tests/test_data.py", "configs/real.json",
                 "比赛说明/数据平台文档.json", "官方demo/Transformer_modelsave_predict.txt", "uv.lock"):
        stage(repository, path)
    result = payload.check_repository(repository, "tracked")
    assert result["passed"]
    assert result["files_checked"] == 6


def test_checks_staged_blob_even_when_working_copy_is_safe(repository):
    target = stage(repository, "disguised.txt", b"PAR1\0binary")
    target.write_text("safe working copy", encoding="utf-8")
    result = payload.check_repository(repository)
    assert not result["passed"]
    assert "binary_content" in result["violations"][0]["reasons"]


def test_oversized_text_is_rejected(repository):
    stage(repository, "oversized.txt", b"a" * (payload.MAX_FILE_BYTES + 1))
    result = payload.check_repository(repository)
    assert not result["passed"]
    assert "file_exceeds_2_mib" in result["violations"][0]["reasons"]


def test_deleting_tracked_dataset_is_allowed_but_tracked_mode_blocks_it(repository):
    stage(repository, "bars.csv")
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Payload test", "-c",
                    "user.email=payload@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    assert payload.check_repository(repository)["passed"]
    assert not payload.check_repository(repository, "tracked")["passed"]
    subprocess.run(["git", "-C", str(repository), "rm", "bars.csv"], check=True, capture_output=True)
    assert payload.check_repository(repository)["passed"]
    assert payload.check_repository(repository, "tracked")["passed"]


def test_rejects_regular_file_changed_into_symbolic_link(repository):
    stage(repository, "README.md")
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Payload test", "-c",
                    "user.email=payload@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    # Construct the index entry directly so Windows needs no symlink privileges.
    oid = subprocess.run(["git", "-C", str(repository), "hash-object", "-w", "--stdin"],
                         input=b"../private", check=True, capture_output=True).stdout.decode().strip()
    subprocess.run(["git", "-C", str(repository), "update-index", "--cacheinfo", f"120000,{oid},README.md"],
                   check=True, capture_output=True)
    result = payload.check_repository(repository)
    assert not result["passed"]
    assert "non_regular_file" in result["violations"][0]["reasons"]
