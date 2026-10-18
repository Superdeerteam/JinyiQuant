import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jinyiquant.data import build_batch, history_diagnostics
from jinyiquant.io import read_json, write_json
from scripts.diagnose_training_coverage import (
    exported_detail,
    quality_sections,
    quarter_jobs,
    resume_source,
    reusable_predictions,
)
from scripts.freeze_training_coverage import freeze_windows, purge_training

ROOT = Path(__file__).resolve().parents[1]


def test_source_label_end_controls_purge_even_at_equal_boundary():
    labels = pd.DataFrame({"date": pd.to_datetime(["2023-10-31 14:30"] * 3),
                           "instrument": ["A", "B", "C"], "target": [.01, .02, .03],
                           "label_end": pd.to_datetime(["2023-10-31 15:00", "2023-11-01 00:00", "2023-11-01 09:31"])})
    selected, audit = purge_training(labels, labels[["date", "instrument"]], ["2023-10-31"], "2023-11-01")
    assert selected.instrument.tolist() == ["A"]
    assert audit["boundary_purged"] == 2 and audit["actual_label_end_available"]


def test_purge_restricts_dates_universe_and_marks_inferred_lunch_clock():
    labels = pd.DataFrame({"date": pd.to_datetime(["2023-10-30 11:30", "2023-10-31 11:30", "2023-10-31 11:30"]),
                           "instrument": ["A", "A", "OUT"], "target": [.01, .02, .03]})
    universe = labels.loc[labels.instrument.eq("A"), ["date", "instrument"]]
    selected, audit = purge_training(labels, universe, ["2023-10-31"], "2023-11-01")
    assert len(selected) == 1 and selected.label_end.iloc[0] == pd.Timestamp("2023-10-31 13:30")
    assert not audit["actual_label_end_available"] and audit["outside_official_universe"] == 1
    assert audit["max_conservative_label_end"] == "2023-10-31T15:00:00"
    assert audit["boundary_proof"] == "official_no_overnight_upper_bound_1500"


def test_frozen_configs_preserve_original_cutoffs_and_reject_recipe_change():
    prior = read_json(ROOT / "configs/fixed_seed_ensemble_20261017_plan.json")
    original = prior["protocol"]["candidate_windows"]
    # A minimal calendar preserving all registered training/evaluation days plus earlier dates.
    dates = sorted({d for w in original for d in w["train_dates"] + w["evaluation_dates"]}
                   | {d.strftime("%Y-%m-%d") for w in original for d in pd.bdate_range(
                       end=pd.Timestamp(w["train_dates"][0]) - pd.Timedelta(days=1), periods=40)})
    windows = freeze_windows(prior, pd.to_datetime(dates))
    for old, window in zip(original, windows):
        assert window["arms"]["10"]["train_dates"] == old["train_dates"]
        for arm in window["arms"].values():
            assert arm["train_end_exclusive"] == old["train_end_exclusive"]
        assert len(window["arms"]["40"]["train_dates"]) == 40
    assert windows[0]["role"] == windows[1]["role"] == "research_used"
    invalid = copy.deepcopy(prior)
    invalid["configs"]["research_a"]["model"]["batch_size"] = 64
    with pytest.raises(ValueError, match="recipe"):
        freeze_windows(invalid, pd.to_datetime(dates))


def test_history_coverage_does_not_hide_incomplete_and_stale_windows():
    keys = pd.DataFrame({"date": pd.to_datetime(["2022-01-04 09:30"] * 2), "instrument": ["A", "B"]})
    bars = pd.DataFrame({"date": pd.to_datetime(["2021-12-31 15:00", "2022-01-04 09:30"]),
                         "instrument": ["A", "B"], "close": [1., np.nan]})
    raw = history_diagnostics(build_batch(bars, keys, {"feature_columns": ["close"], "lookback": 128}))
    section = quality_sections(raw).iloc[0]
    assert section.full_observed_window_fraction == 0
    assert section.short_128_count == 2 and section.no_effective_history_count == 1
    assert section.latest_effective_not_at_cutoff_count == 1
    assert section.latest_effective_age_undefined_count == 1
    assert section.latest_effective_age_minutes_max == 5430


def test_quarter_grid_is_prescribed_and_existing_exports_verify():
    index = read_json(ROOT / "modified_tlob/annual_baseline_2022_20261017/index.json")
    jobs = quarter_jobs(index)
    assert [(j["origin"], j["month"]) for j in jobs if j["reuse_policy"] is None] == [
        ("202204", "202205"), ("202204", "202206"), ("202207", "202208"),
        ("202207", "202209"), ("202210", "202211"), ("202210", "202212")]
    data = exported_detail()
    assert len(data) == 3872 and data.groupby("policy").timestamp.nunique().eq(1936).all()
    assert not data.duplicated(["policy", "timestamp"]).any()


def test_resume_reuses_only_a_complete_finite_declared_month(tmp_path):
    index = read_json(ROOT / "modified_tlob/annual_baseline_2022_20261017/index.json")
    window = next(w for w in index["fit_windows"] if w["name"] == "202205")
    days = window["evaluation_dates"][:2]
    grid = pd.DataFrame({"date": pd.to_datetime([f"{d} {t}" for d in days for t in index["sample_times"]]),
                         "instrument": "A", "score": 0.1})
    path = tmp_path / "quarterly_202205.parquet"
    grid.to_parquet(path, index=False)
    short = dict(window, evaluation_dates=days)
    assert len(reusable_predictions(path, short, index)) == 2 * 8
    with pytest.raises(ValueError, match="grid"):
        reusable_predictions(path, window, index)
    grid.assign(score=np.nan).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        reusable_predictions(path, short, index)


def test_resume_source_rejects_a_different_frozen_index(tmp_path):
    write_json(tmp_path / "manifest.json", {"kind": "quarter_origins", "index_sha256": "0" * 64})
    write_json(tmp_path / "jobs.json", [])
    with pytest.raises(ValueError, match="index"):
        resume_source(tmp_path, quarter_jobs(read_json(ROOT / "modified_tlob/annual_baseline_2022_20261017/index.json")))
