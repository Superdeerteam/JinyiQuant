"""Behavior checks for the fixed-ratio TLOB/MLPLOB replay comparison helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def comparison_module():
    spec = importlib.util.spec_from_file_location(
        "compare_blend_replay", ROOT / "scripts/compare_blend_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_frames():
    keys = pd.DataFrame({
        "date": pd.to_datetime(["2023-11-01 09:30"] * 4 + ["2023-11-01 10:00"] * 4),
        "instrument": ["A", "B", "C", "D"] * 2,
    })
    frames = {}
    for index, member in enumerate(comparison_module().MEMBERS):
        frames[member] = keys.assign(score=np.arange(8, dtype=float) + index)
    return keys, frames


def test_blend_arms_carry_declared_trunk_ratios():
    comparison = comparison_module()
    for arm, weights in comparison.ARMS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-12, arm
        tlob = sum(weight for member, weight in weights.items() if member.startswith("tlob"))
        mlp = sum(weight for member, weight in weights.items() if member.startswith("mlp"))
        assert abs(tlob + mlp - 1.0) < 1e-12
    assert abs(sum(comparison.ARMS["blend_75_25"][member] for member in comparison.TL0B) - 0.75) < 1e-12
    assert abs(sum(comparison.ARMS["blend_25_75"][member] for member in comparison.TL0B) - 0.25) < 1e-12
    assert abs(sum(comparison.ARMS["blend_50_50"].values()) - 1.0) < 1e-12


def test_arm_scores_equal_the_declared_weighted_member_mean():
    comparison = comparison_module()
    keys, frames = synthetic_frames()
    scores = comparison.arm_scores(frames, comparison.ARMS["blend_75_25"]).score.to_numpy()
    expected = np.zeros(len(keys))
    for member, weight in comparison.ARMS["blend_75_25"].items():
        expected += weight * frames[member].score.to_numpy()
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)


def test_section_correlations_are_plus_one_identical_and_minus_one_reversed():
    comparison = comparison_module()
    keys, frames = synthetic_frames()
    left = frames["tlob_s11"][["date", "instrument"]].assign(
        score_left=frames["tlob_s11"].score)
    right = left.rename(columns={"score_left": "score_right"})
    perfect = comparison.section_correlations(left, right)
    assert np.allclose(perfect.pearson, 1.0) and np.allclose(perfect.spearman, 1.0)
    assert np.allclose(perfect.top_overlap, 1.0)
    reversed_right = right.assign(score_right=-right.score_right)
    flipped = comparison.section_correlations(left, reversed_right)
    assert np.allclose(flipped.pearson, -1.0) and np.allclose(flipped.spearman, -1.0)


def test_block_interval_mean_matches_the_sample_mean():
    comparison = comparison_module()
    values = np.array([0.01, -0.02, 0.03, 0.0, 0.01, 0.02, -0.01, 0.02])
    interval = comparison.block_interval(values, block=3)
    assert abs(interval["mean"] - values.mean()) < 1e-12
    assert interval["ci_low"] <= interval["mean"] <= interval["ci_high"]
