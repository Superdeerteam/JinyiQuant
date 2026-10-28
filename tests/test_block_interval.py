"""Caliber checks for the house moving-block bootstrap used on daily rank_ic series.

`compare_blend_replay.block_interval` must describe the sampling variability of the pooled
daily mean.  Before 2026-10-28 it averaged a single 5-day block per replicate, so its half
width was sqrt(n / 5) too large; these tests fail if that regression comes back.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def comparison_module():
    spec = importlib.util.spec_from_file_location(
        "compare_blend_replay", ROOT / "scripts/compare_blend_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def half_width(interval):
    return (interval["ci_high"] - interval["ci_low"]) / 2.0


def ar1(phi, n, rng, sigma=1.0):
    values = np.empty(n)
    values[0] = rng.normal(0.0, sigma)
    root = sigma * np.sqrt(1.0 - phi * phi)
    for index in range(1, n):
        values[index] = phi * values[index - 1] + rng.normal(0.0, root)
    return values


def test_iid_half_width_matches_the_iid_standard_error():
    """n=1000 iid draws with 5-day blocks: half width ~1.96/sqrt(1000)=0.062, not ~0.88.

    The retired estimator (one block mean per replicate) returns ~0.85 on this input, so the
    upper bound below is exactly the assertion that catches that mutation.
    """
    module = comparison_module()
    values = np.random.default_rng(0).standard_normal(1000)
    interval = module.block_interval(values, block=5, draws=20000, seed=20261028)
    half = half_width(interval)
    assert 0.062 * 0.8 <= half <= 0.062 * 1.2, half
    retired = module.archived_single_block_interval(values, block=5, draws=20000,
                                                    seed=20261028)
    assert half_width(retired) > 0.5, half_width(retired)


def test_interval_mean_is_the_sample_mean_and_the_seed_is_deterministic():
    module = comparison_module()
    values = np.array([0.01, -0.02, 0.03, 0.0, 0.01, 0.02, -0.01, 0.02])
    first = module.block_interval(values, block=3, draws=4000, seed=20261028)
    second = module.block_interval(values, block=3, draws=4000, seed=20261028)
    assert first == second
    assert abs(first["mean"] - values.mean()) < 1e-15
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_width_grows_with_ar1_persistence():
    """A mean interval must widen with persistence: 1.0 / 1.27 / 2.06 at phi = 0 / 0.5 / 0.9.

    Theory: the mean of ceil(n/block) block means has sd sigma * sqrt(1 + 2 * sum((5-k)/5 * phi^k))
    / sqrt(n), i.e. only the block-level persistence inflates the interval.
    """
    module = comparison_module()
    widths = {}
    for phi in (0.0, 0.5, 0.9):
        rng = np.random.default_rng(20261028)
        series = ar1(phi, 203, rng)
        widths[phi] = half_width(module.block_interval(series, block=5, draws=4000,
                                                       seed=20261028))
    assert widths[0.0] < widths[0.5] < widths[0.9]
    for phi in (0.5, 0.9):
        theory = np.sqrt(1.0 + 2.0 * sum((5 - lag) / 5 * phi ** lag for lag in range(1, 5)))
        ratio = widths[phi] / widths[0.0]
        assert abs(ratio / theory - 1.0) <= 0.25, (phi, ratio, theory)


def test_width_shrinks_with_the_number_of_days():
    module = comparison_module()
    rng = np.random.default_rng(7)
    long_series = rng.standard_normal(1000)
    short_series = long_series[:250]
    long_half = half_width(module.block_interval(long_series, block=5, draws=4000,
                                                 seed=20261028))
    short_half = half_width(module.block_interval(short_series, block=5, draws=4000,
                                                  seed=20261028))
    assert abs(long_half / short_half - 0.5) <= 0.15


def test_retired_estimator_reproduces_the_pre_fix_formula():
    """The archived caliber is kept only for reproducibility; pin it against its own formula."""
    module = comparison_module()
    values = np.linspace(-0.01, 0.02, 61)
    interval = module.archived_single_block_interval(values, block=5, draws=2000, seed=20261028)
    rng = np.random.default_rng(20261028)
    starts = rng.integers(0, len(values), size=2000)
    indices = (starts[:, None] + np.arange(5)[None, :]) % len(values)
    means = values[indices].mean(axis=1)
    assert interval["ci_low"] == float(np.quantile(means, 0.025))
    assert interval["ci_high"] == float(np.quantile(means, 0.975))
