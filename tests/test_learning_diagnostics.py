"""Learning diagnostics must preserve time boundaries and detect broken controls."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from jinyiquant.io import read_json, write_json  # noqa: E402
from modified_tlob.diagnostics import (  # noqa: E402
    NegativeControl,
    load_plan,
    permute_training_targets,
    run_diagnostics,
    summarize_controls,
)
from scripts.analyze_learning_diagnostics import analyze  # noqa: E402
from scripts.tlob_fixture import create_fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_permutation_preserves_exact_timestamps_multisets_and_input():
    dates = pd.to_datetime(["2023-07-17 14:00", "2023-07-17 14:30", "2023-07-18 14:00"])
    keys = pd.DataFrame({"date": np.repeat(dates, 12), "instrument": list("ABCDEFGHIJKL") * 3})
    targets = np.arange(36, dtype=float)
    original = targets.copy()
    shuffled, mapping, sections = permute_training_targets(keys, targets, 17)
    assert np.array_equal(targets, original)
    assert np.array_equal(shuffled, permute_training_targets(keys, targets, 17)[0])
    assert not np.array_equal(shuffled, permute_training_targets(keys, targets, 29)[0])
    assert not np.array_equal(shuffled, targets)
    for _, indices in keys.groupby("date").indices.items():
        assert np.array_equal(np.sort(shuffled[indices]), np.sort(targets[indices]))
    reconstructed = mapping.merge(keys.assign(original=targets), left_on=["date", "source_instrument"],
                                   right_on=["date", "instrument"], validate="one_to_one")
    np.testing.assert_array_equal(reconstructed.training_target, reconstructed.original)
    assert len(sections) == 3


@pytest.mark.parametrize("case", ["duplicates", "nonfinite", "misaligned"])
def test_permutation_rejects_invalid_inputs(case):
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-07-17"] * 2), "instrument": ["A", "B"]})
    targets = np.array([1., 2.])
    if case == "duplicates":
        keys["instrument"] = "A"
    elif case == "nonfinite":
        targets[0] = np.nan
    else:
        targets = targets[:1]
    with pytest.raises(ValueError):
        permute_training_targets(keys, targets, 1)


def _control_rows():
    rows = []
    for period in ("july_18_21", "july_25_28"):
        for seed in (11, 29, 47):
            for day in range(4):
                rows.append({"period": period, "seed": seed, "date": str(day),
                    "rank_ic_correct": .3, "rank_ic_shuffled": .2, "rank_ic_initial": -.1,
                    "ic_correct": .2, "ic_shuffled": .1,
                    "long_short_return_correct": .002, "long_short_return_shuffled": .001})
    return pd.DataFrame(rows)


def test_repeated_positive_sham_is_flagged_but_missing_evidence_is_not_cleared():
    control = NegativeControl.model_validate(read_json(ROOT / "configs/learning_diagnostics.json")["negative_control"])
    rows = _control_rows()
    result = summarize_controls(rows, control)
    assert result["status"] == "investigate"
    assert result["repeated_positive_shuffled_advantage"]
    interval = result["periods"][0]["estimates"]["rank_ic_correct_minus_shuffled"]["interval"]
    np.testing.assert_allclose(interval, [.1, .1])
    rows.loc[0, "rank_ic_shuffled"] = np.nan
    result = summarize_controls(rows, control)
    assert result["status"] == "inconclusive"
    assert not result["repeated_positive_shuffled_advantage"]
    result = summarize_controls(_control_rows().query("seed != 47"), control)
    assert result["status"] == "inconclusive"


def test_neutral_sham_and_short_dates_cannot_establish_model_validity():
    control = NegativeControl.model_validate(read_json(ROOT / "configs/learning_diagnostics.json")["negative_control"])
    rows = _control_rows()
    rows["rank_ic_shuffled"] = 0.
    result = summarize_controls(rows, control)
    assert result["status"] == "no_repeated_positive_control_detected"
    assert "does not rule out leakage" in result["limits"]
    assert summarize_controls(rows.query("date == '0'"), control)["status"] == "inconclusive"


def _fixture_plan(tmp_path):
    base_path = create_fixture(tmp_path / "fixture")
    base = read_json(base_path)
    base["splits"] = {"train": {"start": "2020-01-02", "end": "2020-01-03"},
                      "validation": {"start": "2020-01-03", "end": "2020-01-07"},
                      "final": {"start": "2020-01-10", "end": "2020-01-13"}}
    write_json(base_path, base)
    plan = read_json(ROOT / "configs/learning_diagnostics.json")
    plan["base_config"] = str(base_path)
    plan["small_sample"].update(samples=3, epochs=2, record_every=1)
    plan["negative_control"].update(seeds=[11, 29], sample_times=["10:00"], bootstrap_samples=100, block_days=1,
        periods=[{"name": "first", "train": {"start": "2020-01-02", "end": "2020-01-03"},
                  "validation": {"start": "2020-01-03", "end": "2020-01-07"}},
                 {"name": "second", "train": {"start": "2020-01-07", "end": "2020-01-08"},
                  "validation": {"start": "2020-01-08", "end": "2020-01-10"}}])
    plan["budget"] = {"wall_seconds": 120, "max_rss_mb": 2048}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    return path


def test_final_interval_guard_rejects_research_overlap(tmp_path):
    path = _fixture_plan(tmp_path)
    plan = read_json(path)
    plan["negative_control"]["periods"][1]["validation"]["end"] = "2020-01-11"
    write_json(path, plan)
    with pytest.raises(ValueError):
        load_plan(path)


def test_diagnostic_cli_archives_both_experiments_and_does_not_require_memorization_pass(tmp_path):
    path = _fixture_plan(tmp_path)
    output, code = run_diagnostics(path, tmp_path / "diagnostics")
    assert code == 0, (output / "worker.log").read_text(encoding="utf-8")
    result = read_json(output / "metrics.json")
    assert result["execution_completed"] and result["final_views"] == 0 and not result["model_promoted"]
    assert len(read_json(output / "attempts.json")) == 8
    tiny = result["small_sample"]
    assert not tiny["learning_passed"]
    assert tiny["constant_mse"] > 0 and tiny["initial_mse"] > 0
    curve = read_json(Path(tiny["run"]) / "learning_curve.json")
    assert [row["steps"] for row in curve] == [0, 1, 2]
    assert all(value > 0 for value in tiny["parameter_delta_norms"].values())
    assert tiny["reproduction"]["passed"]
    assert tiny["reproduction"]["max_abs_prediction_difference"] <= 1e-6
    assert read_json(output / "checks.json")["passed"]
    assert read_json(output / "manifest.json")["status"] == "completed"
    assert len(pd.read_csv(output / "paired_daily.csv")) == 8
    audit = analyze(output, tmp_path / "independent_audit")
    assert audit["passed"] and len(audit["runs"]) == 8 and len(audit["pairs"]) == 4
    assert audit["findings"]["same_seed_reproduction_passed"]
    resumed, code = run_diagnostics(path, tmp_path / "resumed", resume_from=output)
    assert code == 0, (resumed / "worker.log").read_text(encoding="utf-8")
    assert read_json(resumed / "attempts.json") == read_json(output / "attempts.json")
    assert read_json(resumed / "metrics.json")["negative_control"] == result["negative_control"]
    with pytest.raises(FileExistsError):
        run_diagnostics(path, output)
    base_path = Path(read_json(path)["base_config"])
    changed = read_json(base_path)
    changed["splits"]["train"]["start"] = "2020-01-01"
    write_json(base_path, changed)
    with pytest.raises(ValueError, match="frozen base configuration"):
        run_diagnostics(path, tmp_path / "changed_base", resume_from=output)
