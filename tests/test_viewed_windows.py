"""Behavior checks for the research-area registry (viewed windows and sealed-interval evidence)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_experiment_index import (
    INDEX_FILENAME,
    SEALED_INTERVAL,
    SQLITE_FILENAME,
    SUMMARY_FILENAME,
    VIEWED_WINDOWS_FILENAME,
    build_index,
    canonical_config_sha256,
    query_records,
    read_entries,
)

ROOT = Path(__file__).resolve().parents[1]
SEALED_FINAL = {"start": "2024-10-08T00:00:00", "end": "2025-01-01T00:00:00"}


def config_document(*, final=None, version="cpt_jyc_2026",
                    source="official_local_download") -> dict:
    return {
        "schema_version": 1,
        "evaluation_split": "validation",
        "splits": {
            "train": {"start": "2023-08-01T00:00:00", "end": "2023-11-01T00:00:00"},
            "validation": {"start": "2023-11-01T00:00:00", "end": "2023-11-15T00:00:00"},
            "final": final,
        },
        "data": {"version": version, "source": source,
                 "allowed_start": "2020-01-01", "allowed_end": "2025-01-01"},
    }


def daily_csv(*dates: str) -> str:
    return "date,rank_ic\n" + "".join(f"{date},0.1\n" for date in dates)


def write_run(root: Path, relative: str, *, run_id: str, experiment_id: str,
              config: dict | None = None, daily: str | None = None,
              metrics: dict | None = None, data_version: dict | None = None,
              extra: dict[str, bytes] | None = None) -> Path:
    run_dir = root / relative
    run_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if daily is not None:
        (run_dir / "daily.csv").write_text(daily, encoding="utf-8")
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for name, payload in (extra or {}).items():
        (run_dir / name).write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:10:00+00:00",
        "config_sha256": canonical_config_sha256(config) if config else None,
    }
    if data_version is not None:
        manifest["data_version"] = data_version
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def fixture_tree(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    write_run(root, "exp-a/exact-outside", run_id="run-outside", experiment_id="exp-a",
              config=config_document(final=SEALED_FINAL),
              daily=daily_csv("2023-11-01", "2023-11-02"), metrics={"day_count": 2})
    write_run(root, "exp-a/declared-only", run_id="run-declared", experiment_id="exp-a",
              config=config_document(final=SEALED_FINAL), metrics={"day_count": 5})
    write_run(root, "exp-b/sealed-hit", run_id="run-hit", experiment_id="exp-b",
              config=config_document(final=SEALED_FINAL),
              daily=daily_csv("2024-10-08", "2024-12-31", "2025-01-02"),
              metrics={"day_count": 3})
    write_run(root, "exp-b/partial", run_id="run-partial", experiment_id="exp-b",
              config=config_document(final=None),
              daily=daily_csv("2023-11-01", "2023-11-02"), metrics={"day_count": 5})
    write_run(root, "exp-c/unparseable", run_id="run-bad-dates", experiment_id="exp-c",
              config=config_document(final=None),
              daily="date,rank_ic\nnot-a-date,0.1\n2023-11-01,0.1\n", metrics={"day_count": 1})
    write_run(root, "exp-c/no-daily", run_id="run-no-daily", experiment_id="exp-c")
    return root


def build(root: Path, output: Path) -> tuple[list[dict], dict, dict]:
    summary = build_index(root, output)
    entries = read_entries(output)
    viewed = json.loads((output / VIEWED_WINDOWS_FILENAME).read_text(encoding="utf-8"))
    return entries, summary, viewed


def entry_by_run(entries: list[dict], run_id: str) -> dict:
    return next(entry for entry in entries if entry["run_id"] == run_id)


def test_declared_splits_and_data_version_are_backfilled(tmp_path):
    root = fixture_tree(tmp_path)
    entries, _, viewed = build(root, tmp_path / "index")
    entry = entry_by_run(entries, "run-outside")
    assert entry["splits"]["final"] == SEALED_FINAL
    assert entry["splits"]["train"] == {"start": "2023-08-01T00:00:00", "end": "2023-11-01T00:00:00"}
    assert entry["declared"]["config_snapshot"]["matches_manifest_config_sha256"] is True
    assert entry["splits"]["declared_source"]["path"].endswith("config.json")
    assert entry["data"]["version"] == "cpt_jyc_2026"
    assert entry["data"]["backfilled_from"] == "run_config_snapshot"
    assert entry["data"]["sha256"] is None
    record = next(row for row in viewed["runs"] if row["run_id"] == "run-outside")
    assert record["declared_final"] == SEALED_FINAL
    assert record["declares_sealed_final"] is True
    assert record["intersects_sealed"] is False
    assert viewed["declared_sealed_final_but_not_observed"]


def test_sealed_intersection_requires_daily_evidence_not_declaration(tmp_path):
    root = fixture_tree(tmp_path)
    entries, _, viewed = build(root, tmp_path / "index")
    declared_only = entry_by_run(entries, "run-declared")
    assert declared_only["declared"]["final_overlaps_sealed"] is True
    assert declared_only["evaluated"]["determination"] == "unknown"
    assert declared_only["evaluated"]["from"] is None
    assert declared_only["evaluated"]["sealed"]["intersects"] is None
    assert declared_only["evaluated"]["reason"] == "no_daily_artifact"
    hit = entry_by_run(entries, "run-hit")
    assert hit["evaluated"]["determination"] == "exact"
    assert hit["evaluated"]["sealed"]["intersects"] is True
    assert hit["evaluated"]["sealed"]["intersecting_dates"] == ["2024-10-08", "2024-12-31"]
    assert hit["evaluated"]["sealed"]["after_sealed_end"] == ["2025-01-02"]
    assert [row["run_id"] for row in viewed["intersects_sealed"]] == ["run-hit"]
    assert viewed["counts"]["intersects_sealed_true"] == 1
    assert "相交" in viewed["wording"]
    unknown = next(row for row in viewed["unknown"] if row["run_id"] == "run-declared")
    assert unknown["reason"] == "no_daily_artifact"


def test_partial_and_unknown_determinations_are_counted(tmp_path):
    root = fixture_tree(tmp_path)
    entries, _, viewed = build(root, tmp_path / "index")
    partial = entry_by_run(entries, "run-partial")
    assert partial["evaluated"]["determination"] == "partial"
    assert partial["evaluated"]["reason"] == "metrics_day_count_exceeds_daily_dates"
    assert partial["evaluated"]["from"] == "2023-11-01"
    bad = entry_by_run(entries, "run-bad-dates")
    assert bad["evaluated"]["determination"] == "partial"
    assert bad["evaluated"]["reason"] == "unparseable_daily_rows"
    assert bad["evaluated"]["unparseable_rows"] == 1
    no_daily = entry_by_run(entries, "run-no-daily")
    assert no_daily["evaluated"]["determination"] == "unknown"
    assert no_daily["evaluated"]["source_files"] == []
    assert viewed["counts"] == {
        "total": 6, "exact": 2, "partial": 2, "unknown": 2,
        "intersects_sealed_true": 1, "intersects_sealed_false": 3,
        "intersects_sealed_unknown": 2, "declared_sealed_final": 3,
        "declared_sealed_final_without_actual_intersection": 2,
        "declared_sealed_final_observed_false": 1,
        "declared_sealed_final_unknown": 1,
        "after_sealed_end": 1,
    }
    assert viewed["coverage"] == {
        "config_snapshots": 5,
        "config_snapshot_matched_manifest": 5,
        "config_snapshot_mismatched_manifest": 0,
        "config_snapshot_unlinked_manifest": 0,
        "daily_artifact_runs": 4,
        "daily_artifact_files": 4,
    }


def test_registry_and_index_are_byte_identical_on_rebuild(tmp_path):
    root = fixture_tree(tmp_path)
    first = build(root, tmp_path / "index-a")
    second = build(root, tmp_path / "index-b")
    for name in (INDEX_FILENAME, SQLITE_FILENAME, SUMMARY_FILENAME, VIEWED_WINDOWS_FILENAME):
        assert (tmp_path / "index-a" / name).read_bytes() == \
            (tmp_path / "index-b" / name).read_bytes(), name
    assert first[2] == second[2]


def test_evaluated_fields_are_queryable(tmp_path):
    root = fixture_tree(tmp_path)
    entries, _, viewed = build(root, tmp_path / "index")
    assert [row["run_id"] for row in query_records(entries, determination=["partial"])] == \
        ["run-partial", "run-bad-dates"]
    assert [row["run_id"] for row in query_records(entries, intersects_sealed="true")] == \
        ["run-hit"]
    assert len(query_records(entries, intersects_sealed="false")) == 3
    assert len(query_records(entries, intersects_sealed="unknown")) == 2
    connection = sqlite3.connect(tmp_path / "index" / SQLITE_FILENAME)
    try:
        assert connection.execute(
            "select run_id from runs where determination='partial' order by run_id").fetchall() == \
            [("run-bad-dates",), ("run-partial",)]
        assert connection.execute(
            "select count(*) from runs where intersects_sealed=1").fetchone()[0] == 1
        assert connection.execute(
            "select evaluated_from, evaluated_to from runs where run_id='run-hit'").fetchone() == \
            ("2024-10-08", "2025-01-02")
    finally:
        connection.close()
    assert viewed["sealed_interval"] == list(SEALED_INTERVAL)


@pytest.mark.skipif(not (ROOT / "runs").is_dir(), reason="runs/ fact source not present")
def test_real_registry_reports_determined_subset_and_unknown(tmp_path):
    summary = build_index(ROOT / "runs", tmp_path / "index")
    viewed = json.loads((tmp_path / "index" / VIEWED_WINDOWS_FILENAME).read_text(encoding="utf-8"))
    assert summary["entries"]["evaluated_determination"] == {"exact": 485, "partial": 0,
                                                            "unknown": 216}
    assert summary["entries"]["declared_splits_backfilled"] == 589
    assert viewed["counts"]["intersects_sealed_true"] == 0
    assert viewed["intersects_sealed"] == []
    assert viewed["counts"]["intersects_sealed_unknown"] == 216
    assert "在可确定子集内未发现读取封存区间" in viewed["wording"]
    assert "未验证" in viewed["wording"]
    assert viewed["counts"]["declared_sealed_final"] == 118
    assert viewed["counts"]["declared_sealed_final_without_actual_intersection"] == 118
    assert viewed["counts"]["declared_sealed_final_observed_false"] == 34
    assert viewed["counts"]["declared_sealed_final_unknown"] == 84
    assert viewed["counts"]["after_sealed_end"] == 180
    assert viewed["coverage"] == {
        "config_snapshots": 589,
        "config_snapshot_matched_manifest": 525,
        "config_snapshot_mismatched_manifest": 0,
        "config_snapshot_unlinked_manifest": 64,
        "daily_artifact_runs": 485,
        "daily_artifact_files": 489,
    }
    assert all(row["is_test_fixture"] for row in viewed["after_sealed_end"])
    entries = read_entries(tmp_path / "index")
    determined = [entry for entry in entries
                  if entry["evaluated"]["determination"] in {"exact", "partial"}]
    assert len(determined) == 485
    assert Counter(entry["evaluated"]["determination"] for entry in entries) == \
        Counter({"exact": 485, "unknown": 216})
    shipped = ROOT / "runs" / "index" / VIEWED_WINDOWS_FILENAME
    if shipped.is_file():
        assert hashlib.sha256(shipped.read_bytes()).hexdigest() == \
            summary["artifacts"]["viewed_windows"]["sha256"]
