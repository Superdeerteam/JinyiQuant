"""Research comparisons reject mismatched populations and preserve the day as the sampling unit."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jinyiquant.evaluation import evaluate
from jinyiquant.io import file_sha256, read_json, write_json
from scripts.compare_training_round import METRICS, _curve_finished, compare, paired_intervals, screen_gate


def paired_rows(seeds=(11, 29, 47)):
    rows = []
    for period in ("first", "second"):
        for seed in seeds:
            for day, delta in enumerate([-.1, .1, .2, .4]):
                row = {"period": period, "seed": seed, "date": str(day)}
                for metric in METRICS:
                    row.update({metric + "_baseline": .1, metric + "_candidate": .1 + delta})
                rows.append(row)
    return pd.DataFrame(rows)


def test_day_blocks_are_seed_replication_invariant_and_preserve_undefined_values():
    once = paired_intervals(paired_rows((11,)), bootstrap_samples=200, seed=4)
    repeated = paired_intervals(paired_rows(), bootstrap_samples=200, seed=4)
    for left, right in zip(once, repeated):
        assert right["mean"] == pytest.approx(.15)
        assert right["mean"] == pytest.approx(left["mean"])
        assert right["interval"] == pytest.approx(left["interval"])
        assert right["days"] == (8 if right["period"] == "all" else 4)
    rows = paired_rows()
    rows.loc[0, "rank_ic_candidate"] = np.nan
    missing = paired_intervals(rows, bootstrap_samples=200)
    assert next(r for r in missing if r["metric"] == "rank_ic" and r["period"] == "all")["status"] == "undefined"
    with pytest.raises(ValueError, match="Incomplete"):
        paired_intervals(rows.iloc[1:], bootstrap_samples=200)


def archive(path, *, epochs, seed=11, start="2023-07-18", prediction_key_mismatch=False,
            times=("10:00", "14:00")):
    path.mkdir()
    dates = pd.date_range(start, periods=4)
    keys = pd.DataFrame([(date + pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:])), instrument)
                         for date in dates for t in times for instrument in "ABCDE"],
                        columns=["date", "instrument"])
    targets = keys.assign(target=np.tile([-.02, -.01, 0, .01, .02], len(dates) * len(times)))
    predictions = keys.assign(score=targets.target * (1 if epochs == 12 else -1))
    config = {"seed": seed, "evaluation_split": "validation", "experiment_id": "test",
              "hypothesis": "test", "model": {"kind": "modified_tlob", "epochs": epochs},
              "data": {"version": "test", "sample_times": list(times)}, "budget": {"wall_seconds": 120},
              "splits": {"train": {"start": "2023-07-14", "end": "2023-07-18"},
                         "validation": {"start": str(dates[0]), "end": str(dates[-1] + pd.Timedelta(days=1))},
                         "final": {"start": "2023-08-03", "end": "2023-08-04"}},
              "evaluation": {"quantile": .2, "periods_per_year": 252, "min_cross_section": 3,
                             "constant_portfolios": "undefined"}}
    metrics, daily, _ = evaluate(predictions, targets, config["evaluation"], return_sections=True,
                                 undefined_constant_portfolios=True)
    for filename, value in (("config.json", config), ("data.json", {"version": "test"}),
                             ("metrics.json", metrics), ("checks.json", {"passed": True}),
                             ("learning_curve.json", [{"epoch": epochs, "steps": 1, "train_mse": .01,
                                                        "validation_mse": metrics["mse"]}])):
        write_json(path / filename, value)
    daily.to_csv(path / "daily.csv", index=False)
    predictions.to_parquet(path / "scored_predictions.parquet", index=False)
    if prediction_key_mismatch:
        predictions.loc[0, "instrument"] = "OTHER"
    predictions.to_parquet(path / "predictions.parquet", index=False)
    targets.to_parquet(path / "targets.parquet", index=False)
    keys.to_parquet(path / "train_history.parquet", index=False)
    inventory = [{"path": p.name, "sha256": file_sha256(p)} for p in path.iterdir()]
    write_json(path / "manifest.json", {"status": "completed", "artifacts": inventory,
        "code": {"files": {"modified_tlob/adapter.py": "test-same-source"}},
        "environment": {"python": "test", "dependencies": {"torch": "test"}},
        "resources": {"wall_seconds": 2, "peak_rss_mb": 50}})
    return path


def plan(tmp_path, *, mismatch=False, full=False, times=("10:00", "14:00")):
    pairs = []
    for period, start in (("first", "2023-07-18"), ("second", "2023-07-25")) if full else (("first", "2023-07-18"),):
        for seed in (11, 29, 47) if full else (11,):
            arms = {arm: str(archive(tmp_path / f"{period}_{seed}_{arm}", epochs=epochs, seed=seed, start=start,
                         prediction_key_mismatch=mismatch and arm == "candidate", times=times))
                    for arm, epochs in (("baseline", 2), ("candidate", 12))}
            pairs.append({"period": period, "seed": seed, **arms})
    value = {"schema_version": 1, "experiment_id": "test", "change": {"factor": "model.epochs",
             "baseline": 2, "candidate": 12, "hypothesis": "longer training"}, "pairs": pairs,
             "bootstrap_samples": 100, "bootstrap_seed": 4}
    path = tmp_path / "plan.json"
    write_json(path, value)
    return path


def test_actual_prediction_key_mismatch_is_rejected_without_touching_inputs(tmp_path):
    path = plan(tmp_path, mismatch=True)
    original = copy.deepcopy(read_json(path))
    with pytest.raises(ValueError, match="Paired keys/labels differ"):
        compare(path, tmp_path / "comparison")
    assert read_json(path) == original
    assert not (tmp_path / "comparison").exists()


@pytest.mark.parametrize("full, conclusion", [(False, "证据不足"), (True, "改善线索")])
def test_comparison_reports_partial_and_complete_grids_without_promoting(tmp_path, full, conclusion):
    path = plan(tmp_path, full=full)
    if full:
        value = read_json(path)
        value["screen_periods"] = ["first", "second"]
        write_json(path, value)
    result = compare(path, tmp_path / "comparison")
    assert result["checks"]["complete_two_period_three_seed_grid"] is full
    assert result["conclusion"] == conclusion and result["model_promoted"] is False
    assert result["summary"][1]["rank_ic_mean"] == pytest.approx(1)
    assert (tmp_path / "comparison" / "by_timestamp.csv").is_file()
    assert (tmp_path / "comparison" / "learning_curves.csv").is_file()
    delta = next(r for r in result["paired_intervals"] if r["period"] == "all" and r["metric"] == "rank_ic")
    assert delta["mean"] == pytest.approx(2)
    assert result["screen_gate"]["passed"] is full
    assert result["screen_gate"]["expected_periods_source"] == ("plan.screen_periods" if full else "plan.pairs")
    assert read_json(tmp_path / "comparison" / "comparison.json")["screen_gate"] == result["screen_gate"]


def screening_comparison():
    pairs = [{"period": period, "seed": seed} for period in ("first", "second") for seed in (11, 29)]
    return {"plan": {"pairs": pairs}, "checks": {
        "passed": True, "paired_keys_labels_data_and_single_factor": True,
        "complete_target_eligible_correlation_coverage": True},
        "summary": [{**pair, "arm": arm, "rank_ic_mean": .1, "long_short_return_mean": .001}
                    for pair in pairs for arm in ("baseline", "candidate")]}


def test_screen_gate_accepts_zero_changes_without_promoting_or_requiring_positive_ci():
    comparison = screening_comparison()
    comparison["paired_intervals"] = [{"interval": [-1, 1]}]
    original = copy.deepcopy(comparison)
    gate = screen_gate(comparison)
    assert comparison == original
    assert gate["passed"] and len(gate["rows"]) == 8
    assert all(row["delta"] == 0 for row in gate["rows"])
    assert gate["expected_periods_source"] == "plan.pairs"
    assert gate["risk_and_resource_review_required"]
    assert not gate["automatic_action"] and not gate["model_promoted"]


@pytest.mark.parametrize("field", ["rank_ic_mean", "long_short_return_mean"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1])
def test_screen_gate_rejects_undefined_or_negative_change_even_when_other_seed_improves(field, value):
    comparison = screening_comparison()
    comparison["summary"][1][field] = value
    comparison["summary"][3][field] = 100
    gate = screen_gate(comparison)
    assert not gate["passed"]
    assert not next(row for row in gate["rows"]
                    if row["period"] == "first" and row["seed"] == 11
                    and row["metric"] == field.removesuffix("_mean"))["passed"]


@pytest.mark.parametrize("bad", ["single_window", "missing_pair", "duplicate_pair", "missing_arm", "duplicate_arm",
                                  "wrong_expected_window", "duplicate_expected_window"])
def test_screen_gate_rejects_incomplete_or_ambiguous_declared_grid(bad):
    comparison = screening_comparison()
    if bad == "single_window":
        comparison["plan"]["pairs"] = [row for row in comparison["plan"]["pairs"] if row["period"] == "first"]
        comparison["summary"] = [row for row in comparison["summary"] if row["period"] == "first"]
    elif bad == "missing_pair":
        comparison["plan"]["pairs"].pop()
        comparison["summary"] = comparison["summary"][:-2]
    elif bad == "duplicate_pair":
        comparison["plan"]["pairs"].append(comparison["plan"]["pairs"][0])
    elif bad == "missing_arm":
        comparison["summary"].pop()
    elif bad == "duplicate_arm":
        comparison["summary"].append(comparison["summary"][0])
    else:
        comparison["plan"]["screen_periods"] = ["first", "other" if bad == "wrong_expected_window" else "first"]
    gate = screen_gate(comparison)
    assert not gate["passed"] and gate["failed_prerequisites"]


@pytest.mark.parametrize("check", ["passed", "paired_keys_labels_data_and_single_factor",
                                    "complete_target_eligible_correlation_coverage"])
def test_screen_gate_requires_passed_fairness_and_correlation_checks(check):
    comparison = screening_comparison()
    comparison["checks"][check] = False
    assert not screen_gate(comparison)["passed"]


def test_added_plan_is_allowed_but_changed_training_code_is_rejected(tmp_path):
    path = plan(tmp_path)
    candidate = tmp_path / "first_11_candidate" / "manifest.json"
    manifest = read_json(candidate)
    manifest["code"]["files"]["configs/new_plan.json"] = "new-plan-hash"
    write_json(candidate, manifest)
    assert compare(path, tmp_path / "allowed")["checks"]["passed"]
    manifest["code"]["files"]["modified_tlob/adapter.py"] = "changed-training-code"
    write_json(candidate, manifest)
    with pytest.raises(ValueError, match="core source"):
        compare(path, tmp_path / "rejected")


def test_update_cap_or_epoch_limit_completes_curve_but_partial_stop_does_not():
    config = {"model": {"epochs": 12, "max_steps": 1500}}
    for epoch, steps in ((1.25, 1500), (12, 1000)):
        _curve_finished([{"epoch": epoch, "steps": steps}], config)
    for epoch, steps in ((1.25, 1499), (13, 1500), (12, 1501)):
        with pytest.raises(ValueError, match="update cap or epoch limit"):
            _curve_finished([{"epoch": epoch, "steps": steps}], config)


def reseal(path):
    manifest = read_json(path / "manifest.json")
    manifest["artifacts"] = [{"path": p.name, "sha256": file_sha256(p)}
                             for p in path.iterdir() if p.is_file() and p.name != "manifest.json"]
    write_json(path / "manifest.json", manifest)


def expansion_plan(tmp_path, factor):
    times = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
    path = plan(tmp_path, full=factor == "splits.train.start", times=times)
    value = read_json(path)
    change = {"factor": factor, "baseline": {}, "candidate": {}, "hypothesis": "more training data"}
    if factor == "data.training_sample_times":
        change.update(baseline=["09:30", "14:30"], candidate=times)
    for pair in value["pairs"]:
        for arm in ("baseline", "candidate"):
            run = Path(pair[arm])
            config = read_json(run / "config.json")
            start = pd.Timestamp(config["splits"]["validation"]["start"]) - pd.Timedelta(
                days=8 if factor == "splits.train.start" and arm == "candidate" else 4)
            end = pd.Timestamp(config["splits"]["validation"]["start"])
            selected = change[arm] if factor == "data.training_sample_times" else times
            if factor == "splits.train.start":
                change[arm][pair["period"]] = str(start.date())
            config["model"].update(epochs=12, max_steps=1500)
            config["data"].update(training_sample_times=selected, history_days=10)
            config["splits"]["train"] = {"start": str(start), "end": str(end)}
            days = pd.bdate_range(start, end - pd.Timedelta(days=1))
            keys = pd.DataFrame([(day + pd.Timedelta(hours=int(t[:2]), minutes=int(t[3:])), stock)
                                 for day in days for t in selected for stock in "ABCDE"], columns=["date", "instrument"])
            keys.to_parquet(run / "train_history.parquet", index=False)
            rows = (len(days) + 4) * len(times) * 5
            data = {"source": "fixed", "version": "test", "data_sha256": "fixed-files",
                    "files": {"bars": [{"sha256": "same-monthly-file", "bytes": 1000}]},
                    "requested_start": str(start), "requested_end": config["splits"]["validation"]["end"],
                    "bars_start": str(start - pd.Timedelta(days=10)), "bar_rows": rows * 8,
                    "label_rows": rows, "missing_targets": 0,
                    "alignment": {"policy": "exclude", "labels_input_rows": rows, "labels_retained_rows": rows}}
            curve = read_json(run / "learning_curve.json")
            curve[-1].update(epoch=1.25, steps=1500)
            for name, document in (("config.json", config), ("data.json", data), ("learning_curve.json", curve)):
                write_json(run / name, document)
            reseal(run)
    value["change"] = change
    write_json(path, value)
    return path


@pytest.mark.parametrize("factor", ["data.training_sample_times", "splits.train.start"])
def test_training_expansion_preserves_full_eight_time_validation(tmp_path, factor):
    path = expansion_plan(tmp_path, factor)
    result = compare(path, tmp_path / "comparison")
    assert result["checks"]["passed"]
    assert result["summary"][1]["training_samples"] > result["summary"][0]["training_samples"]
    sections = pd.read_csv(tmp_path / "comparison" / "by_timestamp.csv")
    assert sections.groupby(["period", "seed", "arm"]).size().eq(32).all()
    assert all(row["steps"] == 1500 for row in result["summary"])


@pytest.mark.parametrize("bad", ["validation_times", "validation_split", "old_key_removed", "extra_old_scope_key", "file_hash"])
def test_expansion_rejects_undeclared_validation_population_or_data_changes(tmp_path, bad):
    path = expansion_plan(tmp_path, "data.training_sample_times")
    run = Path(read_json(path)["pairs"][0]["candidate"])
    if bad in {"validation_times", "validation_split"}:
        config = read_json(run / "config.json")
        if bad == "validation_times":
            config["data"]["sample_times"].append("13:00")
        else:
            config["splits"]["validation"]["end"] = "2023-07-23"
        write_json(run / "config.json", config)
    elif bad == "file_hash":
        data = read_json(run / "data.json")
        data["files"]["bars"][0]["sha256"] = "different-raw-data"
        write_json(run / "data.json", data)
    else:
        keys = pd.read_parquet(run / "train_history.parquet")
        if bad == "old_key_removed":
            keys = keys.iloc[1:]
        else:
            keys = pd.concat([keys, keys.iloc[[0]].assign(instrument="NEW")], ignore_index=True)
        keys.to_parquet(run / "train_history.parquet", index=False)
    reseal(run)
    with pytest.raises(ValueError):
        compare(path, tmp_path / "rejected")


def test_date_expansion_cannot_add_new_stocks_in_old_training_dates(tmp_path):
    path = expansion_plan(tmp_path, "splits.train.start")
    pair = read_json(path)["pairs"][0]
    run = Path(pair["candidate"])
    old = pd.read_parquet(Path(pair["baseline"]) / "train_history.parquet")
    keys = pd.read_parquet(run / "train_history.parquet")
    pd.concat([keys, old.iloc[[0]].assign(instrument="NEW")], ignore_index=True).to_parquet(
        run / "train_history.parquet", index=False)
    reseal(run)
    with pytest.raises(ValueError, match="outside the declared expansion"):
        compare(path, tmp_path / "rejected")
