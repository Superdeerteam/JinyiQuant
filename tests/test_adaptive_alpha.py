import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from jinyiquant.adaptive_alpha import (
    ControllerConfig,
    build_panel,
    churn_curve,
    control_path,
    operator_causality,
    smooth_scores,
    solve_alpha,
    weight_change,
)
from turnover_stability_change import smooth_scores as reference_smooth_scores


def synthetic(days: int = 6, sections: int = 4, stocks: int = 12, seed: int = 3):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        base = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        for section in range(sections):
            stamp = base + pd.Timedelta(minutes=30 * section)
            draw = rng.normal(size=stocks)
            for stock in range(stocks):
                rows.append((stamp, f"S{stock:02d}", float(draw[stock])))
    frame = pd.DataFrame(rows, columns=["date", "instrument", "score"])
    return frame[["date", "instrument"]].copy(), frame["score"].to_numpy()


def day_alphas(panel, value):
    return {pd.Timestamp(label): value for label in panel.day_labels}


def test_operator_matches_recorded_constant_alpha_reference():
    keys, scores = synthetic(days=5, sections=4, stocks=13)
    panel = build_panel(keys, scores)
    ours, diagnostics = smooth_scores(keys, scores, day_alphas(panel, 0.5))
    theirs, _ = reference_smooth_scores(keys, scores, 0.5)
    assert diagnostics["day_count"] == 5
    assert np.max(np.abs(ours - theirs)) <= 1e-12


def test_alpha_one_reproduces_raw_cross_sectional_ranks():
    keys, scores = synthetic(days=4, sections=3, stocks=11)
    panel = build_panel(keys, scores)
    smoothed, _ = smooth_scores(keys, scores, day_alphas(panel, 1.0))
    raw = pd.Series(scores).groupby(keys["date"].to_numpy()).rank().to_numpy()
    shaped = pd.Series(smoothed).groupby(keys["date"].to_numpy()).rank().to_numpy()
    assert np.array_equal(raw, shaped)


def test_alpha_path_is_causal_under_truncation():
    keys, scores = synthetic(days=14, sections=4, stocks=15, seed=8)
    panel = build_panel(keys, scores)
    config = ControllerConfig(target_turnover=1.0, window_days=5, min_history_days=3)
    curve = churn_curve(panel, config.alpha_grid, config.quantile)
    full = control_path(panel, curve, config)
    assert full.ready.sum() == 11
    for cut in (4, 7, 11):
        prefix = panel.slice_rows(int(panel.day_starts[cut]))
        partial = control_path(prefix, curve[:cut], config)
        assert np.array_equal(partial.alphas, full.alphas[:cut])
        assert np.array_equal(partial.realized_turnover, full.realized_turnover[:cut])
        assert np.nanmax(np.abs(partial.realized_intraday - full.realized_intraday[:cut])) <= 0.0


def test_controller_tracks_target_on_synthetic_history():
    keys, scores = synthetic(days=40, sections=4, stocks=15, seed=11)
    panel = build_panel(keys, scores)
    curve = churn_curve(panel, ControllerConfig(target_turnover=0.1).alpha_grid, 0.2)
    middle = float(np.mean([curve[:, 0].mean(), curve[:, -1].mean()]))
    config = ControllerConfig(target_turnover=middle, window_days=10, min_history_days=5)
    path = control_path(panel, curve, config)
    ready = path.ready
    assert np.all(path.alphas[:5] == 0.5)
    assert np.all((path.alphas[ready] > 0) & (path.alphas[ready] <= 1))
    assert set(path.solver_status[5:]) <= {"ok", "below_grid", "above_grid"}
    realized = path.realized_turnover[ready]
    assert abs(float(np.mean(realized)) - middle) < 0.15


def test_warmup_and_infeasible_targets_clamp():
    keys, scores = synthetic(days=12, sections=4, stocks=12, seed=4)
    panel = build_panel(keys, scores)
    grid = ControllerConfig(target_turnover=0.1).alpha_grid
    curve = np.tile(np.linspace(0.20, 1.00, len(grid)), (panel.day_count, 1))
    low = ControllerConfig(target_turnover=0.001, window_days=5, min_history_days=3)
    high = ControllerConfig(target_turnover=100.0, window_days=5, min_history_days=3)
    low_path, high_path = control_path(panel, curve, low), control_path(panel, curve, high)
    assert np.all(low_path.alphas[:3] == low.warmup_alpha)
    assert np.all(low_path.alphas[3:] == grid[0])
    assert np.all(high_path.alphas[3:] == 1.0)
    assert set(low_path.solver_status[3:]) == {"below_grid"}
    assert set(high_path.solver_status[3:]) == {"above_grid"}


def test_solve_alpha_interpolates_an_increasing_curve():
    alpha, status = solve_alpha(np.array([0.2, 0.6, 1.0]), (0.2, 0.5, 1.0), 0.4)
    assert status == "ok"
    assert alpha == pytest.approx(0.35)
    assert solve_alpha(np.array([0.2, 0.6, 1.0]), (0.2, 0.5, 1.0), 0.05) == (0.2, "below_grid")
    assert solve_alpha(np.array([0.2, 0.6, 1.0]), (0.2, 0.5, 1.0), 3.0) == (1.0, "above_grid")
    with pytest.raises(ValueError):
        solve_alpha(np.array([0.6, 0.4, 0.9]), (0.2, 0.5, 1.0), 0.5)


def test_weight_change_counts_a_full_membership_swap_as_two():
    a = pd.Series({"A": 1.0, "B": -1.0})
    b = pd.Series({"C": 1.0, "D": -1.0})
    assert weight_change(a, a) == 0.0
    assert weight_change(a, None) != weight_change(a, a)
    assert weight_change(a, b) == pytest.approx(2.0)


def test_operator_causality_passes_on_the_alpha_path():
    keys, scores = synthetic(days=10, sections=4, stocks=14, seed=6)
    panel = build_panel(keys, scores)
    config = ControllerConfig(target_turnover=1.0, window_days=4, min_history_days=2)
    curve = churn_curve(panel, config.alpha_grid, config.quantile)
    path = control_path(panel, curve, config)
    report = operator_causality(panel, path.alpha_lookup)
    assert report["passed"]
    assert report["truncated_recompute_max_abs_difference"] <= 1e-15


def test_invalid_inputs_are_rejected():
    keys, scores = synthetic(days=3, sections=2, stocks=8)
    with pytest.raises(ValueError):
        ControllerConfig(target_turnover=-1.0)
    with pytest.raises(ValueError):
        ControllerConfig(target_turnover=0.7, window_days=3, min_history_days=4)
    with pytest.raises(ValueError):
        ControllerConfig(target_turnover=0.7, alpha_grid=(0.5, 0.5))
    with pytest.raises(ValueError):
        ControllerConfig(target_turnover=0.7, alpha_grid=(0.5, 1.5))
    shuffled = keys.copy()
    shuffled["date"] = shuffled["date"].iloc[::-1].to_numpy()
    with pytest.raises(ValueError):
        build_panel(shuffled, scores)
    duplicated = pd.concat([keys, keys.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_panel(duplicated, np.concatenate([scores, scores[:1]]))
    bad_scores = scores.copy()
    bad_scores[0] = np.nan
    with pytest.raises(ValueError):
        build_panel(keys, bad_scores)
