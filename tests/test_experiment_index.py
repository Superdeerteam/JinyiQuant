"""Behavior checks for the manifest.json experiment index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from scripts.build_experiment_index import (
    INDEX_FILENAME,
    REQUIRED_FIELDS,
    SQLITE_FILENAME,
    SUMMARY_FILENAME,
    build_index,
    discover_manifests,
    field_value,
    main,
    query_records,
    read_entries,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_BYTES = b'{"rank_ic_mean": 0.09, "days": 21}\n'
SOURCE_BYTES = b"frozen source zip"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def complete_manifest(**overrides) -> dict:
    document = {
        "schema_version": 1,
        "run_id": "run-1",
        "experiment_id": "exp-a",
        "status": "completed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:10:00+00:00",
        "config_sha256": "c" * 64,
        "plan_sha256": "p" * 64,
        "seed": 11,
        "code": {
            "commit": "a" * 40,
            "snapshot": "source.zip",
            "snapshot_sha256": "d" * 64,
            "content_sha256": "e" * 64,
            "working_tree_status": " M src/x.py\n",
            "files": {"src/x.py": "f" * 64},
        },
        "data_version": {
            "source": "official_local_download",
            "version": "v1",
            "data_sha256": "1" * 64,
            "requested_start": "2023-08-01 00:00:00",
            "requested_end": "2023-11-01 00:00:00",
            "bars_start": "2023-07-22 00:00:00",
        },
        "evaluation_split": "validation",
        "artifacts": [
            {"path": "metrics.json", "sha256": sha256(METRICS_BYTES), "bytes": len(METRICS_BYTES)},
            {"path": "source.zip", "sha256": sha256(SOURCE_BYTES), "bytes": len(SOURCE_BYTES)},
        ],
    }
    document.update(overrides)
    return document


def write_run(root: Path, relative: str, document: dict,
              metrics: bytes | None = METRICS_BYTES, extra: dict[str, bytes] | None = None) -> Path:
    run_dir = root / relative
    run_dir.mkdir(parents=True, exist_ok=True)
    if metrics is not None:
        (run_dir / "metrics.json").write_bytes(metrics)
    (run_dir / "source.zip").write_bytes(SOURCE_BYTES)
    for name, payload in (extra or {}).items():
        (run_dir / name).write_bytes(payload)
    (run_dir / "manifest.json").write_text(json.dumps(document, ensure_ascii=False),
                                           encoding="utf-8")
    return run_dir


def fixture_tree(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    write_run(root, "exp-a/run1", complete_manifest())
    write_run(root, "exp-a/run2", complete_manifest(
        run_id="run-2", status="failed", started_at="2026-01-02T00:00:00+00:00",
        ended_at="2026-01-02T00:05:00+00:00",
        plan_sha256=None, seed=None, data_version=None, artifacts=None,
        error={"reason": "full scoring checks failed"}))
    write_run(root, "exp-b/run3", complete_manifest(
        run_id="run-3", experiment_id="exp-b", status="interrupted",
        ended_at=None, code=None, config_sha256=None, plan_sha256=None,
        data_version=None, evaluation_split=None, artifacts=None),
        extra={"failure.json": json.dumps({"reason": "container unreachable"}).encode()})
    write_run(root, "exp-a/run4", complete_manifest(
        run_id="run-4", started_at="2026-01-02T00:00:00+00:00",
        ended_at="2026-01-02T00:10:00+00:00"))
    write_run(root, "frozen/copy", complete_manifest(run_id="frozen-copy"))
    write_run(root, "index/junk", complete_manifest(run_id="index-junk"))
    write_run(root, "project-review/pytest-base-temp/test_x0/run1",
              complete_manifest(run_id="fixture-run", experiment_id=None))
    return root


def test_discovery_is_recursive_and_skips_snapshot_and_index(tmp_path):
    root = fixture_tree(tmp_path)
    discovered = [path.relative_to(root).as_posix() for path in discover_manifests(root)]
    assert discovered == [
        "exp-a/run1/manifest.json",
        "exp-a/run2/manifest.json",
        "exp-a/run4/manifest.json",
        "exp-b/run3/manifest.json",
        "project-review/pytest-base-temp/test_x0/run1/manifest.json",
    ]


def test_rebuild_is_byte_identical_and_counts_match(tmp_path):
    root = fixture_tree(tmp_path)
    first = build_index(root, tmp_path / "index-a")
    second = build_index(root, tmp_path / "index-b")
    assert first["status_counts"] == second["status_counts"]
    assert first["artifacts"] == second["artifacts"]
    assert (tmp_path / "index-a" / INDEX_FILENAME).read_bytes() == \
        (tmp_path / "index-b" / INDEX_FILENAME).read_bytes()
    assert (tmp_path / "index-a" / SQLITE_FILENAME).read_bytes() == \
        (tmp_path / "index-b" / SQLITE_FILENAME).read_bytes()
    assert (tmp_path / "index-a" / SUMMARY_FILENAME).read_bytes() == \
        (tmp_path / "index-b" / SUMMARY_FILENAME).read_bytes()
    assert first["artifacts"]["jsonl"]["sha256"] == sha256(
        (tmp_path / "index-a" / INDEX_FILENAME).read_bytes())
    assert first["artifacts"]["sqlite"]["sha256"] == sha256(
        (tmp_path / "index-a" / SQLITE_FILENAME).read_bytes())
    entries = read_entries(tmp_path / "index-a")
    assert len(entries) == 5
    assert first["entries"]["total"] == 5
    assert first["entries"]["test_fixtures"] == 1
    assert first["status_counts"] == {"completed": 3, "failed": 1, "interrupted": 1}
    assert first["status_group_counts"] == {"failed": 1, "interrupted": 1, "success": 3}


def test_required_fields_present_and_gaps_are_not_backfilled(tmp_path):
    root = fixture_tree(tmp_path)
    build_index(root, tmp_path / "index")
    entries = {entry["index_key"].split("/")[-2]: entry for entry in read_entries(tmp_path / "index")}
    failed = entries["run2"]
    for name in ("run_id", "experiment_id", "status", "started_at", "ended_at", "code",
                 "config_sha256", "plan_sha256", "seed", "data", "splits", "metrics",
                 "artifacts", "failure", "missing_fields"):
        assert name in failed, name
    for name in REQUIRED_FIELDS:
        assert name in failed["missing_fields"] or field_value(failed, name) not in (None, "", [])
    assert failed["plan_sha256"] is None
    assert failed["seed"] is None
    assert failed["data"]["source"] is None
    assert "plan_sha256" in failed["missing_fields"]
    assert "seed" in failed["missing_fields"]
    assert failed["failure"]["reason"] == "full scoring checks failed"
    assert "failure.reason" not in failed["missing_fields"]
    summary = json.loads((tmp_path / "index" / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["gaps"]["per_field"]["plan_sha256"] == 2
    assert summary["gaps"]["missing_archives_count"] == 0
    interrupted = entries["run3"]
    assert interrupted["failure"]["reason"] == "container unreachable"
    assert "ended_at" in interrupted["missing_fields"]
    assert "code.commit" in interrupted["missing_fields"]


def test_artifacts_and_metrics_hashes_resolve(tmp_path):
    root = fixture_tree(tmp_path)
    build_index(root, tmp_path / "index")
    run1 = next(entry for entry in read_entries(tmp_path / "index")
                if entry["run_id"] == "run-1")
    assert run1["metrics"]["sha256"] == sha256(METRICS_BYTES)
    assert run1["metrics"]["matches_manifest"] is True
    assert run1["artifact_count"] == 2
    assert all(row["exists"] for row in run1["artifacts"])
    assert run1["code"]["dirty"] is True
    assert run1["companions"]["metrics_json"] is True


def test_attempts_group_every_attempt_of_an_experiment(tmp_path):
    root = fixture_tree(tmp_path)
    build_index(root, tmp_path / "index")
    entries = [entry for entry in read_entries(tmp_path / "index")
               if entry["experiment_id"] == "exp-a"]
    assert [(entry["run_id"], entry["attempt_index"]) for entry in entries] == \
        [("run-1", 1), ("run-2", 2), ("run-4", 3)]
    assert all(entry["experiment_attempts"] == 3 for entry in entries)
    summary = json.loads((tmp_path / "index" / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    experiment = next(row for row in summary["experiments"] if row["experiment_id"] == "exp-a")
    assert experiment["attempts"] == 3
    assert experiment["statuses"] == {"completed": 2, "failed": 1}


def test_query_filters_by_experiment_status_run_and_started_window(tmp_path):
    root = fixture_tree(tmp_path)
    build_index(root, tmp_path / "index")
    entries = read_entries(tmp_path / "index")
    assert [entry["run_id"] for entry in query_records(entries, experiment_id="exp-b")] == ["run-3"]
    assert sorted(entry["run_id"] for entry in query_records(entries,
                                                             status=["failed", "interrupted"])) == \
        ["run-2", "run-3"]
    assert [entry["run_id"] for entry in query_records(entries, run_id="run-1")] == ["run-1"]
    assert sorted(entry["run_id"] for entry in query_records(
        entries, since="2026-01-02T00:00:00+00:00", until="2026-01-03T00:00:00+00:00")) == \
        ["run-2", "run-4"]
    assert [entry["run_id"] for entry in query_records(
        entries, since="2026-01-01T00:00:00+00:00",
        until="2026-01-01T00:00:00+00:00")] == []


def test_empty_fact_source_builds_an_empty_index(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    summary = build_index(root, tmp_path / "index")
    assert summary["entries"]["total"] == 0
    assert summary["status_counts"] == {}
    assert summary["gaps"]["missing_archives_count"] == 0
    assert read_entries(tmp_path / "index") == []
    connection = sqlite3.connect(tmp_path / "index" / SQLITE_FILENAME)
    try:
        assert connection.execute("select count(*) from runs").fetchone()[0] == 0
    finally:
        connection.close()


def test_sqlite_mirror_is_queryable(tmp_path):
    root = fixture_tree(tmp_path)
    build_index(root, tmp_path / "index")
    connection = sqlite3.connect(tmp_path / "index" / SQLITE_FILENAME)
    try:
        assert connection.execute("select count(*) from runs").fetchone()[0] == 5
        assert connection.execute(
            "select run_id from runs where status='failed'").fetchall() == [("run-2",)]
        assert connection.execute("select count(*) from gaps").fetchone()[0] > 0
        assert connection.execute(
            "select count(*) from runs where started_at >= ? and started_at < ?",
            ("2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00")).fetchone()[0] == 2
    finally:
        connection.close()


def test_cli_query_reads_the_built_index(tmp_path, monkeypatch, capsys):
    root = fixture_tree(tmp_path)
    build_index(root, tmp_path / "index")
    monkeypatch.setattr(sys, "argv", ["build_experiment_index.py", "--output-dir",
                                      str(tmp_path / "index"), "--query", "--status", "failed"])
    assert main() == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert [entry["run_id"] for entry in printed] == ["run-2"]


@pytest.mark.skipif(not (ROOT / "runs").is_dir(), reason="runs/ fact source not present")
def test_real_runs_index_matches_the_manifest_fact_source(tmp_path):
    runs_root = ROOT / "runs"
    manifests_before = discover_manifests(runs_root)
    assert manifests_before, "runs/ must keep at least one manifest as the fact source"
    samples = manifests_before[:3]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in samples}
    summary = build_index(runs_root, tmp_path / "index")
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in samples}
    assert before == after
    entries = read_entries(tmp_path / "index")
    manifests_after = discover_manifests(runs_root)
    indexed = [(ROOT / entry["manifest_path"]).resolve() for entry in entries]
    # One entry per manifest: no phantom rows, no duplicates, nothing dropped. runs/ is a live
    # tree (teammates append and clean up archives while tests run), so the 1:1 statement is
    # written as an invariant that collapses to strict equality whenever the tree stays quiet.
    assert len(indexed) == len(set(indexed)) == len(entries) == summary["entries"]["total"]
    assert set(indexed) <= {path.resolve() for path in manifests_after}
    stable = {path.resolve() for path in manifests_before} & {path.resolve() for path in manifests_after}
    assert stable <= set(indexed)
    # Status counters mirror each indexed manifest's own document, not a frozen population size.
    statuses: dict[str, int] = {}
    for entry in entries:
        document = json.loads((ROOT / entry["manifest_path"]).read_text(encoding="utf-8"))
        assert entry["status"] == document.get("status"), entry["index_key"]
        key = "null" if entry["status"] is None else str(entry["status"])
        statuses[key] = statuses.get(key, 0) + 1
    assert summary["status_counts"] == dict(sorted(statuses.items()))
    assert all(not entry["manifest_path"].startswith("runs/frozen/") for entry in entries)
    assert summary["snapshots"]["frozen_manifest_count"] == 17
    assert summary["snapshots"]["frozen_manifest_statuses"].get("running") == 1
    assert summary["snapshots"]["frozen_manifest_statuses"].get("failed") == 1
    assert summary["gaps"]["missing_archives_count"] == 2
    missing = {row["path"] for row in summary["gaps"]["missing_archives"]}
    assert missing == {"runs/training-duration-20261015", "runs/training-progression"}
    assert summary["gaps"]["manifestless_artifacts_count"] == 1
    derived = summary["gaps"]["manifestless_artifacts"][0]
    assert derived["path"].endswith("windows/jan2024/tlob_s29") and derived["exists"] is True
    assert summary["gaps"]["unindexed_total"] == 3
    assert summary["gaps"]["per_field"]["plan_sha256"] >= 1
    known = next(entry for entry in entries
                 if entry["experiment_id"] == "mlplob-v3-60d-csz-19800")
    assert known["seed"] == 11
    assert known["config_sha256"] and known["data"]["sha256"]
    assert known["metrics"]["sha256"] and known["artifact_count"] > 0
    failed = next(entry for entry in entries if entry["status"] == "failed")
    assert failed["failure"]["reason"]
    # A stored index is only a local cache: new runs legitimately make it stale until the next
    # rebuild, so it is comparable only while it claims the same population as the fact source.
    stored_summary = runs_root / "index" / SUMMARY_FILENAME
    if stored_summary.is_file():
        stored = json.loads(stored_summary.read_text(encoding="utf-8"))
        if stored["entries"]["total"] == summary["entries"]["total"]:
            assert stored["artifacts"]["jsonl"]["sha256"] == \
                summary["artifacts"]["jsonl"]["sha256"]
