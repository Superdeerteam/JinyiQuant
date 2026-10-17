"""Annual execution boundaries and resume behavior; synthetic inputs only."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jinyiquant.io import digest, file_sha256, read_json, write_json
from jinyiquant.tracking import Recorder, artifacts
from scripts import annual_baseline as annual


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    dates = pd.bdate_range("2021-12-01", "2022-12-30")
    pd.DataFrame({"date": dates, "instrument": "A"}).to_parquet(data / "cpt_jyc_2026_instruments.parquet")
    (data / "cpt_jyc_2026_stock_bar1m_instrument_mapping.csv").write_text("instrument_id,instrument\n1,A\n")
    for name in ("bars", "labels"):
        (data / name).mkdir()
        for month in ["202112", *annual.MONTHS]:
            # Preparation fingerprints these files; it must never decode answers.
            (data / name / f"{month}.parquet").write_bytes(b"unopened synthetic data")
    monkeypatch.setattr(annual, "capture_source", lambda root: {"files": {}, "snapshot": "test-only"})
    root = tmp_path / "run"
    plan = annual.prepare(root, data, max_rss_mb=40960, cache_dir=tmp_path / "cache")
    return root, plan


def test_prepare_preserves_recipe_and_independent_complete_grid(prepared):
    root, plan = prepared
    assert annual.load_plan(root) == plan
    assert [w["name"] for w in plan["windows"]] == annual.MONTHS
    assert len(plan["expected_grid"]) == len(plan["evaluation_dates"]) * 8
    assert plan["windows"][0]["last_training_date"] == "2021-12-31"
    for raw in plan["configs"].values():
        cfg = annual.validate_recipe(raw)
        assert cfg.model.max_steps == 1500 and cfg.model.batch_size == 32
        assert cfg.model.learning_rate == 1e-4 and cfg.data.lookback == 128
        assert list(cfg.data.sample_times) == annual.TIMES
        assert cfg.budget.max_rss_mb == 40960
        assert cfg.splits.final.start.year == 2024
    with pytest.raises(FileExistsError):
        annual.prepare(root, plan["data_root"])


@pytest.mark.parametrize("mutation", ["future", "duplicate", "old_ten", "missing_day", "missing_time", "reserved"])
def test_validate_window_rejects_noncausal_or_incomplete_windows(prepared, mutation):
    _, plan = prepared
    window = copy.deepcopy(plan["windows"][0])
    if mutation == "future":
        window["train_dates"][-1] = "2022-01-03"
    elif mutation == "duplicate":
        window["train_dates"][-1] = window["train_dates"][-2]
    elif mutation == "old_ten":
        before = [d for d in plan["trading_calendar"] if d < "2022-01-01"]
        window.update(train_dates=before[-11:-1], train_start=before[-11], last_training_date=before[-2])
    elif mutation == "missing_day":
        window["evaluation_dates"].pop()
        window["expected_timestamps"] = annual.timestamps(window["evaluation_dates"])
    elif mutation == "missing_time":
        window["expected_timestamps"].pop()
    else:
        window.update(name="202407", evaluation_start="2024-07-01", evaluation_end_exclusive="2024-08-01")
    with pytest.raises(ValueError):
        annual.validate_window(window, plan["trading_calendar"])


@pytest.mark.parametrize("changed", ["plan", "registry", "data"])
def test_frozen_plan_registry_and_data_reject_changes(prepared, changed):
    root, plan = prepared
    if changed == "plan":
        path = root / "plan.json"
    elif changed == "registry":
        path = root / "roles.json"
    else:
        path = next(iter(plan["source_inputs_sha256"]))
    with open(path, "ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="changed"):
        annual.load_plan(root)


def fake_fit(root, plan, month):
    path = root / "fit" / month
    path.mkdir(parents=True)
    window = next(w for w in plan["windows"] if w["name"] == month)
    (path / "model.pt").write_bytes(month.encode())
    for filename, value in {
        "modified_tlob_weights.json": {}, "config.json": plan["configs"][month],
        "training_identity.json": {"dates": window["train_dates"]},
        "training.json": {"steps": 1500}, "checks.json": {"passed": True},
    }.items():
        write_json(path / filename, value)
    write_json(path / "manifest.json", {"status": "completed", "exit_code": 0,
        "run_id": f"fit-{month}-{digest(str(root))[:12]}", "plan_sha256": file_sha256(root / "plan.json"),
        "config_sha256": plan["config_sha256"][month], "artifacts": artifacts(path)})
    return path


def test_weight_mapping_shares_january_and_never_borrows_later_training(prepared):
    root, plan = prepared
    january = fake_fit(root, plan, "202201")
    fake_fit(root, plan, "202202")
    first = annual.weight_sources(root, "202201", plan)
    assert first["fixed"] == first["monthly"]
    second = annual.weight_sources(root, "202202", plan)
    assert second["fixed"]["fit_month"] == "202201"
    assert second["monthly"]["fit_month"] == "202202"
    assert second["fixed"]["last_training_date"] == "2021-12-31"
    assert second["monthly"]["last_training_date"] == "2022-01-31"
    (january / "model.pt").write_bytes(b"changed weight")
    with pytest.raises(ValueError, match="artifacts differ"):
        annual.weight_sources(root, "202202", plan)


def test_resume_refuses_incomplete_or_changed_config(prepared):
    root, plan = prepared
    path = fake_fit(root, plan, "202201")
    manifest = read_json(path / "manifest.json")
    write_json(path / "manifest.json", manifest | {"status": "running"})
    with pytest.raises(ValueError, match="identity or artifacts"):
        annual.verify_completed(root, "202201", "fit", plan)
    write_json(path / "manifest.json", manifest)
    changed = copy.deepcopy(plan["configs"]["202201"])
    changed["model"]["max_steps"] = 1501
    write_json(path / "config.json", changed)
    with pytest.raises(ValueError, match="identity or artifacts"):
        annual.verify_completed(root, "202201", "fit", plan)


def test_first_month_inference_is_label_free_and_reuses_same_prediction(prepared, monkeypatch):
    from modified_tlob.adapter import TLOBAdapter

    root, plan = prepared
    cfg = annual.validate_recipe(plan["configs"]["202201"])
    out = root / "test_inference"
    out.mkdir()
    window = plan["windows"][0] | {"evaluation_dates": ["2022-01-03"]}
    source = {"path": str(root / "model.pt"), "sha256": "fake"}
    monkeypatch.setattr(annual, "weight_sources", lambda *args: {"fixed": source, "monthly": source})
    keys = pd.DataFrame({"date": pd.to_datetime(annual.timestamps(window["evaluation_dates"])), "instrument": "A"})
    calls = []
    model = SimpleNamespace(predict=lambda batch: calls.append("predict") or np.arange(len(keys), dtype=float))
    monkeypatch.setattr(TLOBAdapter, "load", lambda *args, **kwargs: model)
    monkeypatch.setattr(annual, "inference_keys", lambda *args: keys)
    history = keys.date.to_numpy(dtype="datetime64[ns]")[:, None]
    batch = SimpleNamespace(keys=keys, history_dates=history, effective_rows=np.ones((8, 1), dtype=bool),
                            coverage={"min_section_coverage": 1.})
    monkeypatch.setattr(annual, "build_batch", lambda *args: batch)
    def loader(*args, **kwargs):
        assert kwargs["labels"] is False
        return SimpleNamespace(bars=None, metadata={"files": {}}, cache_events=[], data_events=[])
    monkeypatch.setattr(annual, "guarded_data", loader)
    annual._infer(plan, window, cfg, root, out, Recorder(out, annual.EXP))
    assert calls == ["predict"]
    pd.testing.assert_frame_equal(pd.read_parquet(out / "fixed.parquet"), pd.read_parquet(out / "monthly.parquet"))
    coverage = pd.read_parquet(out / "history_coverage.parquet")
    assert coverage.timestamp.tolist() == annual.timestamps(window["evaluation_dates"])
    assert coverage.history_coverage.eq(1).all() and coverage.short_history_count.eq(0).all()


def test_watchdog_budget_failure_kills_tree_and_records_consistent_run_id(prepared, monkeypatch):
    root, plan = prepared
    process = SimpleNamespace(pid=10, returncode=None)
    process.poll = lambda: process.returncode
    monkeypatch.setattr(annual, "environment", lambda: {"determinism": {}})
    monkeypatch.setattr(annual.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(annual, "process_tree_usage", lambda pid: {"rss_mb": 50000.})
    killed = []
    def kill(value):
        killed.append(value.pid)
        value.returncode = -9
    monkeypatch.setattr(annual, "_kill_tree", kill)
    with pytest.raises(TimeoutError, match="budget exceeded"):
        annual.launch(root, plan, "202201", "fit", remaining=3600)
    assert killed == [10]
    out = root / "fit" / "202201"
    manifest = read_json(out / "manifest.json")
    assert manifest["status"] == "failed" and manifest["resources"]["peak_rss_mb"] == 50000
    import json
    events = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {event["run_id"] for event in events} == {manifest["run_id"]}
    assert any(event["event_type"] == "failed" for event in events)
