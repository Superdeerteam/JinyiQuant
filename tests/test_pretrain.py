"""R14 behaviour checks: the mask task, the split guards, the trunk transfer and the budget.

What these hold fixed is exactly what the arm's claim rests on: pretraining only sees the
training split's non-label minutes, the masked span carries no information about itself, only
the trunk is transferred, and pretraining plus fine-tune train the control's total budget.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from jinyiquant.data import bar_arrays, lob_trade_columns  # noqa: E402
from jinyiquant.io import write_json  # noqa: E402
from jinyiquant.pool import training_pool  # noqa: E402
from jinyiquant.pretrain import (  # noqa: E402
    PretrainSpec,
    apply_checkpoint,
    assert_train_window,
    bar_window_guard,
    encoder_state,
    mask_windows,
    month_inventory,
    non_label_minutes,
    pretrain_keys,
    run_pretraining,
    save_checkpoint,
    score_head_keys,
    state_digest,
)
from modified_tlob.model import MLPLOB  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pretrain_finetune as orchestrator  # noqa: E402

LABEL_TIMES = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"]
FEATURES = list(lob_trade_columns())


# ---------------------------------------------------------------------------- the minute grid

def test_non_label_minutes_is_the_unsampled_allowed_grid():
    minutes = non_label_minutes(LABEL_TIMES)
    assert len(minutes) == 31
    assert minutes[0] == "09:35" and minutes[-1] == "14:25"
    assert "11:35" not in minutes and "12:00" not in minutes  # outside the allowed clock
    assert not set(minutes).intersection(LABEL_TIMES)
    clock = [int(value[:2]) * 60 + int(value[3:]) for value in minutes]
    assert all(value % 5 == 0 for value in clock)
    assert all(570 <= value <= 660 or value == 690 or 780 <= value <= 870 for value in clock)


def test_non_label_minutes_refuses_a_recipe_that_samples_everything():
    crowded = [f"{hour:02d}:{minute:02d}" for hour in range(9, 15)
               for minute in range(0, 60, 5)]
    crowded = [value for value in crowded
               if 570 <= int(value[:2]) * 60 + int(value[3:]) <= 660
               or int(value[:2]) * 60 + int(value[3:]) == 690
               or 780 <= int(value[:2]) * 60 + int(value[3:]) <= 870]
    assert len(crowded) >= 31
    with pytest.raises(ValueError):
        non_label_minutes(crowded)


# ------------------------------------------------------------------------------- mask windows

def test_mask_windows_replaces_one_contiguous_span_with_the_context_mean():
    generator = np.random.default_rng(11)
    values = generator.normal(size=(20, 12, 3)) * 3 + 100.0
    observed = np.ones_like(values, dtype=bool)
    masked = mask_windows(values, observed, mask_length=4, generator=np.random.default_rng(5))
    inputs, targets, weight = masked["inputs"], masked["targets"], masked["weight"]
    assert masked["masked_fraction"] == pytest.approx(4 / 12)
    reference = values.astype(np.float32)
    starts = set()
    for window in range(values.shape[0]):
        changed = (inputs[window] != reference[window]).any(axis=1)
        hidden = np.flatnonzero(changed)
        assert len(hidden) == 4
        assert hidden.max() - hidden.min() == 3  # contiguous
        starts.add(int(hidden.min()))
        visible = np.setdiff1d(np.arange(12), hidden)
        np.testing.assert_array_equal(inputs[window, visible], reference[window, visible])
        for feature in range(3):
            context = values[window, visible, feature]
            expected = context.mean()
            np.testing.assert_allclose(inputs[window, hidden, feature], expected, rtol=1e-5)
            standardised = targets[window, visible, feature]
            assert standardised.mean() == pytest.approx(0.0, abs=1e-4)
            assert standardised.std(ddof=1) == pytest.approx(1.0, rel=1e-3)
            assert weight[window, hidden, feature].all()
            assert not weight[window, visible, feature].any()
    assert len(starts) > 1  # the span is drawn per window, not fixed


class FixedStart:
    """Draw the masked span at bar 0 so the assertion can build the outlier on purpose."""

    def integers(self, low, high, size):
        return np.zeros(size, dtype=np.int64)


def test_mask_windows_clips_an_outlier_standardised_target():
    """A near-constant context must not turn a one-tick move into a target of thousands."""
    values = np.zeros((1, 12, 1), dtype=np.float32)
    values[0, :, 0] = 0.5 + 0.0011 * np.where(np.arange(12) % 2 == 0, 1.0, -1.0)
    values[0, :3, 0] = 5.0  # the masked span deviates far more than the context deviation
    masked = mask_windows(values, np.ones_like(values, dtype=bool), mask_length=3,
                          generator=FixedStart())
    context = values[0, 3:, 0]
    unclipped = abs(float(values[0, 0, 0]) - context.mean()) / context.std(ddof=1)
    assert unclipped > 100.0
    targets = masked["targets"][0, :3, 0]
    assert np.abs(targets).max() == pytest.approx(5.0)  # clipped, not ~1e3
    assert masked["weight"][0, :3, 0].all()


def test_pretraining_keeps_the_loss_on_the_context_scale():
    pool, sample_times = pool_fixture()
    network = small_network(seed=5)
    spec = PretrainSpec(steps=60, batch_size=4, learning_rate=1e-3, mask_length=4,
                        instruments_per_slot=2, seed=11, minutes=tuple(sample_times))
    metrics = run_pretraining(network, pool, spec, device="cpu", log_every_steps=20)
    assert np.isfinite(metrics["final_loss"])
    assert metrics["final_loss"] < 2.0 * metrics["first_loss"]
    assert all(np.isfinite(row["gradient_norm"]) for row in metrics["loss_curve"])


def test_mask_windows_drops_features_without_a_usable_context():
    values = np.zeros((4, 12, 2), dtype=np.float32)
    values[:, :, 0] = np.random.default_rng(3).normal(size=(4, 12))
    values[:, :, 1] = 5.0  # constant: no deviation, so no standardised target
    observed = np.ones_like(values, dtype=bool)
    masked = mask_windows(values, observed, mask_length=3, generator=np.random.default_rng(3))
    assert masked["weight"][..., 0].any()
    assert not masked["weight"][..., 1].any()


def test_mask_windows_refuses_a_batch_without_any_target():
    values = np.full((2, 12, 2), 7.0, dtype=np.float32)
    observed = np.ones_like(values, dtype=bool)
    with pytest.raises(ValueError):
        mask_windows(values, observed, mask_length=3, generator=np.random.default_rng(0))


def test_mask_windows_refuses_an_impossible_span():
    values = np.zeros((1, 8, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        mask_windows(values, np.ones_like(values, dtype=bool), mask_length=8,
                     generator=np.random.default_rng(0))


# ------------------------------------------------------------------------------------ guards

def test_bar_window_guard_refuses_a_read_that_passes_the_training_split():
    dates = pd.date_range("2023-10-25 09:30", periods=10, freq="min").to_numpy("datetime64[ns]")
    guard = bar_window_guard(dates, train_end="2023-11-01", validation_start="2023-11-01",
                             sealed_start="2024-10-08")
    assert guard["latest"].startswith("2023-10-25T09:39")
    hitting = pd.date_range("2023-11-01 09:30", periods=3, freq="min").to_numpy("datetime64[ns]")
    with pytest.raises(ValueError):
        bar_window_guard(hitting, train_end="2023-11-01", validation_start="2023-11-01")
    with pytest.raises(ValueError):
        bar_window_guard(hitting, train_end="2024-01-01", validation_start="2024-06-01",
                         sealed_start="2023-10-30")


def test_assert_train_window_refuses_a_key_outside_the_split():
    keys = pd.DataFrame({"date": pd.to_datetime(["2022-06-01 09:35", "2023-11-01 09:35"]),
                         "instrument": ["A", "A"]})
    window = assert_train_window(keys.iloc[:1], train_start="2022-01-04",
                                 train_end="2023-11-01")
    assert window["keys"] == 1
    with pytest.raises(ValueError):
        assert_train_window(keys, train_start="2022-01-04", train_end="2023-11-01")


def test_pretrain_keys_are_one_window_per_day_minute_instrument_inside_the_split():
    universe = pd.DataFrame({"date": pd.to_datetime(["2022-01-04"] * 3 + ["2022-01-05"] * 3),
                             "instrument": ["A", "B", "C"] * 2})
    keys = pretrain_keys(universe, train_start="2022-01-04", train_end="2023-11-01",
                         minutes=["09:35", "13:00"], per_slot=2, seed=11)
    assert len(keys) == 2 * 2 * 2
    assert keys["date"].min() >= pd.Timestamp("2022-01-04")
    assert keys["date"].max() < pd.Timestamp("2023-11-01")
    assert not keys.duplicated(["date", "instrument"]).any()
    assert {stamp.strftime("%H:%M") for stamp in keys["date"]} == {"09:35", "13:00"}


# ------------------------------------------------------------------------ trunk / head split

def small_network(seed: int = 0) -> MLPLOB:
    torch.manual_seed(seed)
    return MLPLOB(hidden_dim=8, num_layers=2, seq_size=8, num_features=len(FEATURES))


def test_the_transferred_part_is_exactly_the_trunk_without_the_score_head():
    network = small_network()
    head = set(score_head_keys(network))
    state = network.state_dict()
    assert head == {f"final_layers.{len(network.final_layers) - 1}.weight",
                    f"final_layers.{len(network.final_layers) - 1}.bias"}
    trunk = encoder_state(network)
    assert set(trunk) == set(state) - head
    assert "norm_layer.y1" in trunk and not any(key.startswith("final_layers.") and key in head
                                                for key in trunk)
    digest = state_digest(trunk)
    trunk[next(iter(trunk))] = trunk[next(iter(trunk))] + 1.0
    assert state_digest(trunk) != digest


def test_apply_checkpoint_moves_the_trunk_and_leaves_the_head_fresh(tmp_path):
    source = small_network(seed=1)
    decoder = torch.nn.Linear(source.representation_dim, 8 * len(FEATURES))
    spec = PretrainSpec(steps=1, batch_size=1, learning_rate=1e-3, mask_length=2,
                        instruments_per_slot=1, seed=11, minutes=("09:35",))
    path = tmp_path / "checkpoint.pt"
    record = save_checkpoint(path, source, decoder, spec, metadata={"guards": {}})
    target = small_network(seed=2)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    applied = apply_checkpoint(SimpleNamespace(network=target), path)
    assert applied["head_source"] == "fresh_reinitialised"
    assert applied["task"] == "masked_segment_reconstruction_v1"
    assert applied["encoder_keys"] == record["encoder_keys"] == len(encoder_state(source))
    head = set(score_head_keys(target))
    for key, value in target.state_dict().items():
        if key in head:
            assert torch.equal(value.cpu(), before[key])
        else:
            assert torch.equal(value.cpu(), source.state_dict()[key].cpu())
    payload = torch.load(path, weights_only=True)
    assert payload["decoder"] and payload["spec"]["mask_length"] == 2


def test_apply_checkpoint_refuses_a_partial_match(tmp_path):
    source = small_network(seed=1)
    decoder = torch.nn.Linear(source.representation_dim, 8 * len(FEATURES))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, source, decoder, PretrainSpec(
        steps=1, batch_size=1, learning_rate=1e-3, mask_length=2, instruments_per_slot=1,
        seed=11, minutes=("09:35",)), metadata={})
    payload = torch.load(path, weights_only=True)
    payload["encoder"].pop(sorted(payload["encoder"])[0])
    torch.save(payload, path)
    with pytest.raises(ValueError):
        apply_checkpoint(SimpleNamespace(network=small_network(seed=2)), path)


# --------------------------------------------------------- pretraining loop on a real pool

def pool_fixture(instruments: int = 3, minutes: int = 21, seed: int = 4):
    """Minute bars plus keys on three non-label columns, pooled exactly as the arm does."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2022-01-04 09:30", periods=minutes, freq="min")
    frames = []
    for name in [f"S{index:02d}" for index in range(instruments)]:
        values = rng.normal(size=(minutes, len(FEATURES))).cumsum(axis=0) + 1000.0
        frames.append(pd.DataFrame({"date": stamps, "instrument": name,
                                    **dict(zip(FEATURES, values.T))}))
    bars = pd.concat(frames, ignore_index=True)
    sample_times = ["09:35", "09:40", "09:45"]
    keys = pd.DataFrame([{"date": stamp, "instrument": f"S{number:02d}"}
                         for stamp in pd.to_datetime(["2022-01-04 " + value
                                                      for value in sample_times])
                         for number in range(instruments)])
    settings = SimpleNamespace(feature_columns=FEATURES, lookback=8, min_history=1,
                               missing_policy="ffill", max_ffill_steps=5,
                               sample_times=sample_times, price_scale=1.0, missing_sentinel=None,
                               history_days=1, allowed_start="2022-01-01",
                               allowed_end="2024-10-08", bars_path=None, labels_path=None,
                               instruments_path=None, instrument_map_path=None)
    arrays = bar_arrays(bars, FEATURES)
    return training_pool(None, keys, settings, streaming=True, arrays=arrays), sample_times


def test_run_pretraining_accounts_for_every_step_and_keeps_the_label_path_out():
    pool, sample_times = pool_fixture()
    network = small_network(seed=3)
    spec = PretrainSpec(steps=3, batch_size=2, learning_rate=1e-3, mask_length=2,
                        instruments_per_slot=1, seed=11, minutes=tuple(sample_times))
    metrics = run_pretraining(network, pool, spec, device="cpu", log_every_steps=1)
    assert metrics["steps"] == 3 and metrics["samples_seen"] == 6 and metrics["epochs"] == 1
    assert metrics["usable_windows"] == len(pool) == 9
    assert metrics["batches_per_epoch"] == 5
    assert np.isfinite(metrics["first_loss"]) and np.isfinite(metrics["final_loss"])
    assert [row["step"] for row in metrics["loss_curve"]] == [1, 2, 3]
    assert metrics["decoder_state"]["weight"].shape == (8 * len(FEATURES),
                                                       network.representation_dim)
    assert "target" not in str(metrics["spec"])  # the task never names a label
    with pytest.raises(ValueError):
        run_pretraining(network, pool, PretrainSpec(
            steps=1, batch_size=64, learning_rate=1e-3, mask_length=2, instruments_per_slot=1,
            seed=11, minutes=tuple(sample_times)), device="cpu")


def test_pretrain_spec_refuses_an_out_of_range_mask():
    base = dict(steps=10, batch_size=2, learning_rate=1e-3, instruments_per_slot=2, seed=11,
                minutes=["09:35"])
    with pytest.raises(ValueError):
        PretrainSpec.from_mapping({**base, "mask_length": 0})
    with pytest.raises(ValueError):
        PretrainSpec.from_mapping({**base, "mask_length": 128})
    with pytest.raises(ValueError):
        PretrainSpec.from_mapping({**base, "mask_length": 4, "minutes": ["09:35", "09:35"]})
    missing = dict(base, mask_length=4)
    missing.pop("seed")
    with pytest.raises(ValueError):
        PretrainSpec.from_mapping(missing)
    assert PretrainSpec.from_mapping({**base, "mask_length": 4}).as_mapping()["task"] == \
        "masked_segment_reconstruction_v1"


# ----------------------------------------------------------------------- plan and budget rules

CONTROL_STEPS = 145269
USABLE = 3521680


def plan_document(**overrides) -> dict:
    document = {"schema_version": "r14-pretrain-plan/1",
                "experiment_id": "r14-pretrain-441d-csz-seed11",
                "recipe_config": "configs/r14-pretrain-441d-csz-seed11.json",
                "recipe_sha256": "0" * 64,
                "budget_points": [{"name": "r14-eq132", "effective_epochs": 1.187997}],
                "lr_schedule": "constant",
                "exposure": {"min_coverage": 1.0, "effective_epochs_atol": 0.005},
                "pretrain": {"steps": 14527, "batch_size": 32, "learning_rate": 1e-3,
                             "mask_length": 32, "instruments_per_slot": 10, "seed": 11,
                             "minutes": non_label_minutes(LABEL_TIMES)},
                "control": {"experiment_id": "scaling-441d-seed11",
                            "path": "/mnt/work/runs/scaling-441d-seed11",
                            "budget_point": "scaling-1.32"},
                "budget_match": {"rule": "share", "pretrain_steps": 14527,
                                 "finetune_steps": 130742, "control_steps": CONTROL_STEPS}}
    document.update(overrides)
    return document


def test_the_plan_refuses_an_arm_it_cannot_verify(tmp_path):
    path = tmp_path / "plan.json"
    write_json(path, plan_document())
    assert orchestrator.pretrain_plan(path)["_path"].endswith("plan.json")
    two_points = plan_document(budget_points=[{"name": "a", "effective_epochs": 1.0},
                                             {"name": "b", "effective_epochs": 2.0}])
    write_json(path, two_points)
    with pytest.raises(ValueError):
        orchestrator.pretrain_plan(path)
    write_json(path, plan_document(schema_version="other/1"))
    with pytest.raises(ValueError):
        orchestrator.pretrain_plan(path)
    broken = plan_document()
    broken.pop("pretrain")
    write_json(path, broken)
    with pytest.raises(ValueError):
        orchestrator.pretrain_plan(path)


def test_the_matched_budget_is_the_control_budget_split_in_two():
    config = SimpleNamespace(model=SimpleNamespace(batch_size=32))
    control = {"steps": CONTROL_STEPS, "usable_samples": USABLE, "experiment_id": "c",
               "run_dir": "/x", "budget_point": "scaling-1.32", "effective_epochs": 1.32,
               "config": {}}
    spec = PretrainSpec.from_mapping(plan_document()["pretrain"])
    budget = orchestrator.matched_budget(plan_document(), config, control, spec)
    assert budget["finetune_steps"] == 130742
    assert budget["pretrain_steps"] + budget["finetune_steps"] == CONTROL_STEPS
    assert budget["matched"] is True
    assert budget["scored_point"]["steps"] == 130742
    tampered = plan_document()
    tampered["budget_match"] = {**tampered["budget_match"], "finetune_steps": 130741}
    with pytest.raises(ValueError):
        orchestrator.matched_budget(tampered, config, control, spec)
    short = plan_document(budget_points=[{"name": "x", "effective_epochs": 1.0}])
    with pytest.raises(ValueError):
        orchestrator.matched_budget(short, config, control, spec)


def test_control_record_needs_exactly_one_run_with_the_scored_point(tmp_path):
    root = tmp_path / "scaling-441d-seed11"
    for name, steps in (("runA", CONTROL_STEPS),):
        run = root / name
        run.mkdir(parents=True)
        write_json(run / "trajectory.json",
                   {"experiment_id": "scaling-441d-seed11",
                    "completed_points": [{"name": "scaling-1.32", "steps": steps,
                                          "effective_epochs": 1.32}]})
        write_json(run / "budget.json", {"usable_samples": USABLE})
        write_json(run / "config.json", {"seed": 11})
    plan = plan_document(control={"experiment_id": "scaling-441d-seed11", "path": root.as_posix(),
                                  "budget_point": "scaling-1.32"})
    record = orchestrator.control_record(plan)
    assert record["steps"] == CONTROL_STEPS and record["usable_samples"] == USABLE
    with pytest.raises(ValueError):
        orchestrator.control_record(plan_document(control={"experiment_id": "x",
                                                           "path": root.as_posix(),
                                                           "budget_point": "scaling-2.64"}))


def test_recipe_identity_allows_only_the_arm_identity_to_differ(tmp_path):
    control = {"experiment_id": "scaling-441d-seed11", "hypothesis": "h",
               "output_root": "/mnt/work/runs/scaling-441d-seed11", "seed": 11,
               "data": {"bars_path": "/root/data/bars", "cache_dir": "/root/cache/c",
                        "lookback": 128},
               "model": {"kind": "mlplob", "hidden_dim": 40},
               "splits": {"train": {"start": "2022-01-04", "end": "2023-11-01"}}}
    same = {**control, "experiment_id": "r14-pretrain-441d-csz-seed11", "hypothesis": "other",
            "output_root": "/mnt/work/runs/r14", "data": {**control["data"],
                                                          "bars_path": Path("/root/data/bars")}}
    config = SimpleNamespace(snapshot=lambda: same)
    identity = orchestrator.recipe_identity(plan_document(), config, {"config": control})
    assert set(identity["deviating_fields"]) == {"experiment_id", "hypothesis"}
    different = {**same, "model": {"kind": "mlplob", "hidden_dim": 48}}
    with pytest.raises(ValueError):
        orchestrator.recipe_identity(plan_document(),
                                     SimpleNamespace(snapshot=lambda: different),
                                     {"config": control})


def test_the_gate_needs_a_finished_rankhinge_record(tmp_path):
    gate = importlib.import_module("r14_gate")
    for seed in (11, 29, 47):
        run = tmp_path / f"screen10-rankhinge-seed{seed}" / "20261028T000000.000000Z_aaaaaaaaaaaa"
        run.mkdir(parents=True)
        (run / "trajectory.json").write_text('{"completed_points": [{"name": "scaling-0.66"}]}',
                                             encoding="utf-8")
    assert gate.rankhinge_unfinished(str(tmp_path)) == [11, 29, 47]
    for seed in (11, 29):
        run = next((tmp_path / f"screen10-rankhinge-seed{seed}").glob("*/trajectory.json"))
        run.write_text('{"completed_points": [{"name": "scaling-1.32"}]}', encoding="utf-8")
    assert gate.rankhinge_unfinished(str(tmp_path)) == [47]


def test_the_equivalence_check_reports_a_bitwise_difference(tmp_path):
    verifier = importlib.import_module("verify_mlplob_equivalence")
    left = tmp_path / "legacy.npy"
    right = tmp_path / "current.npy"
    np.save(left, np.array([0.5, -1.25, 3.0], dtype=np.float32))
    np.save(right, np.array([0.5, -1.25, 3.0], dtype=np.float32))
    same = verifier.compare_probes(left, right)
    assert same["bitwise_equal"] and same["max_abs_diff"] == 0.0 and same["elements"] == 3
    np.save(right, np.array([0.5, -1.25, 3.5], dtype=np.float32))
    differ = verifier.compare_probes(left, right)
    assert not differ["bitwise_equal"] and differ["max_abs_diff"] == pytest.approx(0.5)


def _comparison_frames(shift: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic date/instrument/score predictions and the targets they are scored against."""
    rng = np.random.default_rng(7)
    days = pd.date_range("2023-11-01", periods=9, freq="D")
    instruments = [f"SH{code:06d}" for code in range(600001, 600006)]
    targets = pd.DataFrame([{"date": day, "instrument": name} for day in days
                            for name in instruments])
    targets["target"] = rng.normal(size=len(targets))
    predictions = targets[["date", "instrument"]].copy()
    predictions["score"] = targets.target.to_numpy() + shift + rng.normal(
        scale=0.1, size=len(targets))
    return predictions, targets


def test_comparison_of_an_arm_against_itself_is_a_zero_delta():
    comparison = importlib.import_module("compare_r14_validation")
    evaluation = {"quantile": 0.2, "periods_per_year": 252, "min_cross_section": 3,
                  "label_horizon_minutes": 30}
    predictions, targets = _comparison_frames()
    arm = comparison.evaluate_arm(predictions, targets, evaluation)
    record = comparison.paired(arm, arm, block=5, draws=2000, seed=20261028)
    assert record["days"] == 9
    assert record["rank_ic"]["delta"] == pytest.approx(0.0)
    assert record["rank_ic"]["positive_days"] == 0
    assert record["rank_ic"]["interval"]["ci_low"] == pytest.approx(0.0)
    assert record["rank_ic"]["candidate_mean"] > 0.5


def test_comparison_refuses_arms_that_scored_different_keys():
    comparison = importlib.import_module("compare_r14_validation")
    left, _ = _comparison_frames()
    right, _ = _comparison_frames()
    comparison.assert_same_keys(left, right)
    with pytest.raises(ValueError):
        comparison.assert_same_keys(left.iloc[:-1], right)


def test_month_inventory_lists_the_months_and_their_digest():
    dates = np.array(["2022-01-04T09:30", "2022-02-01T10:00", "2023-11-01T09:30"],
                     dtype="datetime64[ns]")
    inventory = month_inventory(dates)
    assert inventory["months"] == ["2022-01", "2022-02", "2023-11"]
    assert inventory["first"] == "2022-01" and inventory["last"] == "2023-11"
    assert inventory["forbidden_from"] == "2024-01-01" and inventory["bars"] == 3
    expected = hashlib.sha256(chr(10).join(["2022-01", "2022-02", "2023-11"]).encode("utf-8"))
    assert inventory["sha256"] == expected.hexdigest()


def test_month_inventory_refuses_a_month_at_or_past_the_sealed_year():
    dates = np.array(["2023-11-01T09:30", "2024-01-02T09:30"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="2024-01"):
        month_inventory(dates)


def test_the_budget_record_spells_out_the_sample_asymmetry():
    module = importlib.import_module("run_pretrain_finetune")
    budget = {"rule": "pretrain + fine-tune == control", "control_steps": 145269,
              "pretrain_steps": 14527, "finetune_steps": 130742, "total_steps": 145269,
              "matched": True, "usable_samples": 3521680, "batch_size": 32,
              "control": {"experiment_id": "scaling-441d-seed11", "run_dir": "x",
                          "budget_point": "scaling-1.32", "steps": 145269,
                          "effective_epochs": 1.32},
              "scored_point": {"name": "r14-eq132"}}
    budget["budget_basis"] = module.budget_basis(budget)
    record = module.budget_record(budget, plan_sha256="a" * 64, metrics={"wall_seconds": 413.8})
    assert record["pretrain_steps"] + record["finetune_steps"] == record["control_steps"]
    assert "145269" in record["budget_basis"] and "4183744" in record["budget_basis"]
    assert "10% fewer" in record["interpretation"]
    assert "4183744" in record["interpretation"] and "4648608" in record["interpretation"]
    assert record["pretrain_wall_seconds"] == 413.8
