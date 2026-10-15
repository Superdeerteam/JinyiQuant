"""Causal postprocessing uses complete inference sections and preserves source archives."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from test_training_round import archive, reseal

from jinyiquant.io import file_sha256, read_json, write_json
from scripts.compare_signal_smoothing import compare, smooth_scores


def predictions():
    keys = pd.DataFrame([(pd.Timestamp(day + " " + time), stock)
                         for day in ("2023-07-18", "2023-07-19")
                         for time in ("09:30", "10:00", "14:30") for stock in "ABC"],
                        columns=["date", "instrument"])
    return keys.assign(score=np.random.default_rng(5).normal(size=len(keys)))


def test_future_append_and_changes_never_change_earlier_scores():
    frame = predictions()
    cutoff = pd.Timestamp("2023-07-18 10:00")
    past = frame.loc[frame.date <= cutoff]
    expected = smooth_scores(past)
    complete = smooth_scores(frame)
    pd.testing.assert_frame_equal(complete.loc[complete.date <= cutoff], expected)
    frame.loc[frame.date > cutoff, "score"] = np.arange((frame.date > cutoff).sum()) ** 2
    shuffled = smooth_scores(frame.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(shuffled.loc[shuffled.date <= cutoff], expected)


def test_daily_reset_stock_separation_and_new_stock_first_observation():
    frame = pd.DataFrame([
        ("2023-07-18 09:30", "A", -2.), ("2023-07-18 09:30", "B", 2.),
        ("2023-07-18 10:00", "A", 2.), ("2023-07-18 10:00", "B", -2.),
        ("2023-07-18 14:30", "A", -2.), ("2023-07-18 14:30", "B", 0.),
        ("2023-07-18 14:30", "C", 2.),
        ("2023-07-19 09:30", "A", -5.), ("2023-07-19 09:30", "B", 5.)],
        columns=["date", "instrument", "score"])
    result = smooth_scores(frame)
    np.testing.assert_allclose(result.score, [-2, 2, 0, 0, -1, 0, 2, -5, 5], atol=1e-15)


def test_alpha_one_identity_and_constant_sections_stay_finite():
    frame = predictions()
    frame.loc[frame.date == pd.Timestamp("2023-07-18 10:00"), "score"] = 7
    expected = frame.sort_values(["date", "instrument"], ignore_index=True)
    pd.testing.assert_frame_equal(smooth_scores(frame, alpha=1), expected)
    result = smooth_scores(frame)
    assert np.isfinite(result.score).all()
    assert result.loc[result.date == pd.Timestamp("2023-07-18 10:00"), "score"].eq(7).all()
    for alpha in (0, -1, 1.1, np.nan):
        with pytest.raises(ValueError):
            smooth_scores(frame, alpha=alpha)


def archived_plan(tmp_path):
    run = archive(tmp_path / "source", epochs=12)
    full = pd.read_parquet(run / "predictions.parquet")
    extra = full.loc[full.instrument == "A"].assign(instrument="UNLABELED")
    extra["score"] = np.tile([100., -100.], len(extra) // 2)
    full = pd.concat([full, extra], ignore_index=True)
    full.to_parquet(run / "predictions.parquet", index=False)
    reseal(run)
    plan = tmp_path / "plan.json"
    write_json(plan, {"schema_version": 1, "experiment_id": "smoothing-test",
                      "runs": [{"period": "first", "seed": 11, "run": "source"}],
                      "bootstrap_samples": 100, "bootstrap_seed": 4})
    return plan, run, full


def test_cli_uses_unlabeled_stocks_before_filtering_and_archives_both_arms(tmp_path):
    plan, run, full = archived_plan(tmp_path)
    before = {path.name: file_sha256(path) for path in run.iterdir()}
    output = tmp_path / "output"
    process = subprocess.run([sys.executable, "-m", "scripts.compare_signal_smoothing", str(plan),
                              "--output", str(output)], capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    result = read_json(output / "comparison.json")
    assert result["alpha"] == .5 and not result["model_promoted"] and result["final_views"] == 0
    assert result["conclusion"] == "insufficient_evidence"
    assert set(result["quality_guardrails"]) == {"rank_ic", "long_short_return"}
    assert result["inputs"][0]["manifest_sha256"] == before["manifest.json"]
    assert len(result["script_sha256"]) == 64
    candidate = pd.read_parquet(output / "run_000" / "candidate_predictions.parquet")
    pd.testing.assert_frame_equal(candidate, smooth_scores(full))
    selected = pd.read_parquet(output / "run_000" / "candidate_scored_predictions.parquet")
    pd.testing.assert_frame_equal(selected, candidate.loc[candidate.instrument != "UNLABELED"].reset_index(drop=True))
    incorrect = smooth_scores(pd.read_parquet(run / "scored_predictions.parquet"))
    assert not np.allclose(selected.score, incorrect.score)
    assert len(read_json(output / "metrics.json")) == 2
    assert set(pd.read_csv(output / "daily.csv").arm) == {"baseline", "candidate"}
    assert set(pd.read_csv(output / "by_timestamp.csv").arm) == {"baseline", "candidate"}
    assert read_json(output / "paired_intervals.json") == result["paired_intervals"]
    assert before == {path.name: file_sha256(path) for path in run.iterdir()}
    with pytest.raises(ValueError, match="new directory"):
        compare(plan, output)


@pytest.mark.parametrize("problem", ["final", "incomplete", "alpha_search", "output_inside_source", "score_changed"])
def test_invalid_inputs_fail_without_writing_output(tmp_path, problem):
    plan, run, _ = archived_plan(tmp_path)
    output = tmp_path / "rejected"
    if problem == "final":
        config = read_json(run / "config.json")
        config["evaluation_split"] = "final"
        write_json(run / "config.json", config)
        reseal(run)
    elif problem == "incomplete":
        manifest = read_json(run / "manifest.json")
        manifest["status"] = "running"
        write_json(run / "manifest.json", manifest)
    elif problem == "alpha_search":
        document = read_json(plan)
        document["alpha"] = .25
        write_json(plan, document)
    elif problem == "output_inside_source":
        output = run / "rejected"
    else:
        full = pd.read_parquet(run / "predictions.parquet")
        full.loc[0, "score"] += 1
        full.to_parquet(run / "predictions.parquet", index=False)
        reseal(run)
    with pytest.raises(ValueError):
        compare(plan, output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["train_split", "train_keys"])
def test_same_period_seeds_cannot_mix_training_windows_or_samples(tmp_path, problem):
    first = archive(tmp_path / "first", epochs=12, seed=11)
    second = archive(tmp_path / "second", epochs=12, seed=29)
    if problem == "train_split":
        config = read_json(second / "config.json")
        config["splits"]["train"]["start"] = "2023-07-10"
        write_json(second / "config.json", config)
    else:
        history = pd.read_parquet(second / "train_history.parquet")
        history.iloc[1:].to_parquet(second / "train_history.parquet", index=False)
    reseal(second)
    plan = tmp_path / "plan.json"
    write_json(plan, {"schema_version": 1, "experiment_id": "fixed-seed-training", "runs": [
        {"period": "first", "seed": seed, "run": str(path)} for seed, path in ((11, first), (29, second))]})
    with pytest.raises(ValueError, match="Training data, splits or keys changed across seeds"):
        compare(plan, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
