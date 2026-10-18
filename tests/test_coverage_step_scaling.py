"""Behavioral checks for the 40-day step-scaling round: frozen screen, pairing and trajectory."""

import json

import numpy as np
import pandas as pd
import pytest

from modified_tlob.ensemble import paired_block_intervals
from scripts.run_coverage_step_scaling import (
    CANDIDATE,
    CONTROL,
    REFERENCE,
    STEPS,
    analysis,
    assert_steps_are_the_only_recipe_change,
    screen_decision,
    step_effect,
    trajectory_report,
)

NOISE = 0.00013896761043335348
STATISTICS = {"bootstrap_samples": 200, "bootstrap_seed": 20261017, "block_days": [5, 2]}
GATES = {"rank_ic_delta_at_least": 0.0, "worst_day_delta_at_least": -NOISE}
ARMS = (CANDIDATE, REFERENCE, CONTROL)
WINDOW_DATES = {"research_a": list(pd.date_range("2023-11-01", periods=10)),
                "research_b": list(pd.date_range("2024-03-01", periods=10))}


def config(steps=STEPS[CONTROL], **model):
    return {"experiment_id": "old", "hypothesis": "old", "seed": 11,
            "budget": {"max_rss_mb": 9472.0, "wall_seconds": 900.0, "max_train_samples": 319632},
            "model": {"max_steps": steps, "learning_rate": 1e-4, "batch_size": 32, "epochs": 12, **model},
            "evaluation": {"quantile": 0.2}, "data": {"lookback": 128}}


def test_only_the_step_count_and_execution_caps_may_change():
    archived = config()
    candidate = config(steps=STEPS[CANDIDATE])
    candidate["budget"] = {"max_rss_mb": 9472.0, "wall_seconds": 1500.0, "max_train_samples": 319632}
    candidate["experiment_id"], candidate["hypothesis"] = "new", "new"
    assert assert_steps_are_the_only_recipe_change(archived, candidate, "research_a") is None
    # Any other recipe change is a different treatment and must be refused.
    for field, value in (("learning_rate", 2e-4), ("batch_size", 64), ("epochs", 1)):
        with pytest.raises(ValueError, match="only model.max_steps|unexpected recipe differences"):
            assert_steps_are_the_only_recipe_change(archived, config(steps=STEPS[CANDIDATE], **{field: value}), "a")
    with pytest.raises(ValueError, match="step change"):
        assert_steps_are_the_only_recipe_change(archived, config(steps=3000), "a")
    with pytest.raises(ValueError, match="unexpected recipe differences"):
        assert_steps_are_the_only_recipe_change(archived, {**config(steps=STEPS[CANDIDATE]), "data": {}}, "a")
    with pytest.raises(ValueError, match="unexpected budget fields"):
        assert_steps_are_the_only_recipe_change(
            archived, {**config(steps=STEPS[CANDIDATE]), "budget": {"wall_seconds": 1.0, "warmup": 2}}, "a")


def scored(*, gaps=(0.02, 0.02), worst_gap=0.0, seeds=(11,)):
    """Two synthetic windows; ``gaps`` is the per-window RankIC edge of the candidate."""
    windows = {}
    for name, gap in zip(("research_a", "research_b"), gaps):
        daily, metrics = {}, {}
        for arm in ARMS:
            for seed in seeds:
                base = {CANDIDATE: 0.03, REFERENCE: 0.0, CONTROL: 0.005}[arm]
                edge = gap if arm == CANDIDATE else 0.0
                long_short = np.full(10, 0.0002)
                if worst_gap and arm == CANDIDATE:
                    long_short[5] -= worst_gap
                daily[(arm, seed)] = pd.DataFrame(
                    {"rank_ic": np.full(10, base + edge), "long_short_return": long_short,
                     "turnover": np.full(10, -0.01)}, index=pd.Index(WINDOW_DATES[name], name="date"))
                metrics[(arm, seed)] = {"rank_ic_mean": float(base + edge)}
        windows[name] = {"daily": daily, "metrics": metrics}
    return windows


def plan_for(screen=True):
    plan = {"windows": [{"name": name} for name in WINDOW_DATES], "statistics": STATISTICS}
    if screen:
        plan["screen"] = {"gates": GATES, "rule": "rule"}
    return plan


def screen_of(windows):
    paired, intervals, summaries = analysis(plan_for(), windows, CANDIDATE, REFERENCE, (11,))
    return screen_decision(paired, summaries, intervals, plan_for())


def test_screen_passes_only_when_neither_window_degrades():
    result = screen_of(scored())
    assert result["result"] == "pass"
    assert all(result["checks"].values()) and result["seeds"] == [11]
    assert "continue to seeds 29/47" in result["next"]
    assert result["period_mean_deltas"]["research_b"] == pytest.approx(0.05)

    # research_b now sits below the retained reference while research_a stays ahead.
    result = screen_of(scored(gaps=(0.02, -0.04)))
    assert result["result"] == "reject"
    assert result["checks"]["research_a_rank_ic_delta_at_least"] is True
    assert result["checks"]["research_b_rank_ic_delta_at_least"] is False
    assert "no other step count is searched" in result["next"]


def test_worst_day_guardrail_alone_rejects_the_screen():
    result = screen_of(scored(worst_gap=2 * NOISE))
    assert result["result"] == "reject"
    assert result["checks"]["research_a_rank_ic_delta_at_least"] is True
    assert result["per_window"]["research_a"]["worst_day_delta"] == pytest.approx(-2 * NOISE)
    # A worst day inside the frozen tolerance must not fail the screen.
    assert screen_of(scored(worst_gap=0.9 * NOISE))["result"] == "pass"


def test_pairing_averages_seeds_before_days_and_reports_per_seed_deltas():
    windows = scored(seeds=(11, 47))
    for name in windows:
        frame = windows[name]["daily"][(CANDIDATE, 47)].copy()
        frame["rank_ic"] = frame.rank_ic + 0.01
        windows[name]["daily"][(CANDIDATE, 47)] = frame
    paired, intervals, summaries = analysis(plan_for(), windows, CANDIDATE, REFERENCE, (11, 47))
    # The paired unit is the day; two seeds per day, not twenty observations.
    assert len(paired) == 20 and paired.groupby("period").size().eq(10).all()
    assert [block["block_days"] for block in intervals] == [5, 2]
    per_seed = summaries[0]["per_seed_mean_delta"]
    assert per_seed["47"]["rank_ic"] == pytest.approx(per_seed["11"]["rank_ic"] + 0.01)
    window_a = paired[paired.period == "research_a"]
    assert window_a.rank_ic.mean() == pytest.approx((per_seed["11"]["rank_ic"] + per_seed["47"]["rank_ic"]) / 2)
    with pytest.raises(KeyError):
        analysis(plan_for(), windows, CANDIDATE, REFERENCE, (11, 29))


def test_step_effect_is_descriptive_and_never_carries_gates():
    windows = scored()
    paired, intervals, summaries = analysis(plan_for(False), windows, CANDIDATE, CONTROL, (11,))
    block = step_effect(paired, summaries, intervals)
    assert block["scope"].startswith("descriptive_mechanism_evidence")
    assert "checks" not in block and "result" not in block
    assert block["comparison"] == f"{CANDIDATE} minus {CONTROL}"
    assert set(block["period_mean_deltas"]) == {"research_a", "research_b"}
    assert block["mean_rank_ic_delta"] == pytest.approx(0.045)


def test_trajectory_equivalence_compares_the_step_1500_loss(tmp_path):
    plan = {"windows": []}
    losses = {("research_a", "11"): 0.00042, ("research_a", "29"): 0.0004,
              ("research_b", "11"): 0.00042, ("research_b", "29"): 0.0004}
    for name in ("research_a", "research_b"):
        trajectory = {}
        for seed in ("11", "29"):
            fitting = tmp_path / "fits" / f"{name}_st{STEPS[CANDIDATE]}_s{seed}"
            fitting.mkdir(parents=True)
            events = [{"stage": "training", "event_type": "progress", "context": {"step": 100, "loss": 9.0}},
                      {"stage": "training", "event_type": "progress",
                       "context": {"step": 1500, "loss": losses[(name, seed)]}}]
            (fitting / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events),
                                                  encoding="utf-8")
            trajectory[seed] = {"step": 1500, "observed_loss": losses[(name, seed)]}
        plan["windows"].append({"name": name, "trajectory": trajectory})
    rows, checks = trajectory_report(plan, tmp_path)
    assert len(rows) == 4 and all(row["matches"] for row in rows)
    assert checks[0]["name"] == "candidate_reproduces_the_control_trajectory_to_step_1500"
    assert checks[0]["passed"] is True
    # A refit that drifted from the control trajectory must be visible, not silently accepted.
    broken = tmp_path / "fits" / f"research_b_st{STEPS[CANDIDATE]}_s11"
    (broken / "events.jsonl").write_text(json.dumps(
        {"stage": "training", "event_type": "progress", "context": {"step": 1500, "loss": 0.0009}}),
        encoding="utf-8")
    rows, checks = trajectory_report(plan, tmp_path)
    assert checks[0]["passed"] is False and any(not row["matches"] for row in rows)


def test_paired_intervals_require_the_registered_block_lengths():
    # Guards the assumption the decision code makes about the two frozen block lengths.
    frame = pd.DataFrame({"period": ["research_a"] * 10, "date": WINDOW_DATES["research_a"],
                          "rank_ic": 0.01, "long_short_return": 0.0, "turnover": 0.0})
    assert paired_block_intervals(frame, block_days=5, samples=200)["block_days"] == 5
