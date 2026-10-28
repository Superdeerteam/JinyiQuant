"""Behavioral checks for the dev10 post-processing study (transforms, sigma, bootstrap)."""

import numpy as np
import pandas as pd
import pytest

from jinyiquant.config import EvaluationConfig
from jinyiquant.evaluation import evaluate
from scripts import dev10_postprocess as post

CFG = EvaluationConfig(quantile=0.2, periods_per_year=252, min_cross_section=3,
                       constant_portfolios="undefined", single_side_rate=0.0)


def synthetic(n_instruments: int = 40, n_times: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-02 09:30", periods=n_times, freq="30min")
    rows = [{"date": time, "instrument": f"{index:06d}.SZ",
             "score": float(rng.normal()), "target": float(rng.normal())}
            for time in times for index in range(n_instruments)]
    frame = pd.DataFrame(rows)
    return frame[["date", "instrument", "score"]], frame[["date", "instrument", "target"]]


def daily_of(scores: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    _, daily, _ = evaluate(scores, targets, CFG, return_sections=True,
                           undefined_constant_portfolios=True)
    return daily


def test_centered_rank_is_selection_neutral():
    scores, targets = synthetic()
    raw = daily_of(scores, targets)
    ranked = daily_of(post.centered_rank(scores), targets)
    assert np.allclose(raw.long_short_return, ranked.long_short_return, atol=1e-12)
    assert np.allclose(raw.turnover, ranked.turnover, atol=1e-12)
    assert np.allclose(raw.rank_ic, ranked.rank_ic, atol=1e-12)


def test_winsorize_clips_at_declared_quantiles():
    scores, _ = synthetic(n_instruments=101)
    clipped = post.winsorize(scores)
    for _, section in clipped.groupby("date"):
        raw = scores.loc[section.index]
        assert section.score.min() >= raw.score.quantile(0.01) - 1e-12
        assert section.score.max() <= raw.score.quantile(0.99) + 1e-12
    assert (scores.score - clipped.score).abs().max() > 0.0


def test_sigma_uses_only_prior_days():
    days = pd.bdate_range("2024-01-01", periods=25)
    rows = [{"day": day, "instrument": "000001.SZ", "vwap_return": 0.01} for day in days]
    spike = days[-1]
    rows.append({"day": spike, "instrument": "000001.SZ", "vwap_return": 10.0})
    labels = pd.DataFrame(rows)
    sigma, filled, unfilled = post.sigma_panel(labels)
    column = sigma["000001.SZ"]
    # The spike day's own label must not enter its scale.
    assert column.loc[spike] == pytest.approx(0.01, rel=1e-12)
    # Day 21 uses exactly the squared returns of days 1..20.
    assert column.loc[days[20]] == pytest.approx(0.01, rel=1e-12)
    # Leading rows have fewer than ten prior observations; every row from day 11 on
    # carries a causal value (row-median fill, never a future-day median).
    assert sigma.loc[days[10]:].notna().all().all()
    assert unfilled == 10  # the ten leading all-NaN rows, dropped from the panel
    assert filled >= 10


def test_vol_scale_divides_by_sigma_and_keeps_keys():
    scores, _ = synthetic(n_instruments=10, seed=5)
    days = sorted(scores.date.dt.normalize().unique())
    instruments = sorted(scores.instrument.unique())
    sigma = pd.DataFrame([{"day": day, "instrument": instrument, "sigma": 0.02}
                          for day in days for instrument in instruments])
    scaled = post.volatility_scale(scores, sigma)
    assert scaled[["date", "instrument"]].equals(scores[["date", "instrument"]])
    assert np.allclose(scaled.score, scores.score / 0.02)


def test_compositions_follow_declared_order():
    scores, _ = synthetic(n_instruments=25, seed=7)
    days = sorted(scores.date.dt.normalize().unique())
    instruments = sorted(scores.instrument.unique())
    sigma = pd.DataFrame([{"day": day, "instrument": instrument, "sigma": 0.02}
                          for day in days for instrument in instruments])
    variants = post.variants_of(scores, sigma)
    assert set(variants) == set(post.TRANSFORMS)
    for frame in variants.values():
        assert frame[["date", "instrument"]].equals(scores[["date", "instrument"]])
    winsor = post.winsorize(variants["vol"])
    assert np.allclose(variants["vol_winsor"].score, winsor.score)
    # all3 = rank(winsor(vol(score))), the registered formula.
    assert np.allclose(variants["all3"].score, post.centered_rank(winsor).score)


def test_block_bootstrap_keeps_blocks_inside_months():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.0, 0.01, 12), rng.normal(0.02, 0.01, 20)])
    months = np.array(["2024-01"] * 12 + ["2024-02"] * 20)
    out = post.block_bootstrap(values, months, block=5, draws=400, seed=1, confidence=0.95)
    assert out["mean"] == pytest.approx(float(values.mean()))
    assert out["low"] <= out["mean"] <= out["high"]
    assert out["se"] > 0.0 and out["months"] == 2
    # Month sizes are held fixed, so a constant series has no bootstrap spread.
    constant = post.block_bootstrap(np.full(10, 0.02), np.array(["2024-01"] * 10),
                                    block=5, draws=200, seed=1, confidence=0.95)
    assert constant["mean"] == pytest.approx(0.02)
    assert constant["se"] == pytest.approx(0.0)


def test_sharpe_delta_interval_detects_paired_improvement():
    rng = np.random.default_rng(11)
    raw = rng.normal(0.0, 0.01, 40)
    cand = raw + 0.004
    months = np.array(["2024-01"] * 20 + ["2024-02"] * 20)
    out = post.sharpe_delta_interval(raw, cand, months, block=5, draws=300, seed=2, confidence=0.95)
    assert out["days"] == 40 and out["months"] == 2
    assert out["mean"] > 0.0 and out["low"] > 0.0
    assert out["se"] > 0.0 and out["high"] > out["low"]
