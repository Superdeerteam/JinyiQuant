"""The driver's ``--stream-bars`` switch must not change the trajectory it produces.

``scripts/run_training_trajectory.py`` reads the bar window in bounded chunks and trains on a
``StreamingPool`` when the switch is on, and materialises the window tensor when it is off. Both
runs here are the same recipe on the same synthetic corpus, one per pool, and the report of the
run must be identical: derived budget, exposure, per-step loss, checkpoint state and validation
scores. ``--stream-bars`` is off by default, so the materialised run is the one every existing
experiment took.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from jinyiquant.trajectory import CHECKPOINT_FILENAME, file_sha256  # noqa: E402
from modified_tlob.schema import lob_columns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs" / "mlplob_v3_132d_csz_19800.json"
FEATURES = list(lob_columns(3))
SAMPLE_TIMES = ("10:00", "10:30")
TRAIN_DAYS, VALIDATION_DAYS = 7, 1


def session(day) -> pd.DatetimeIndex:
    """The trading sessions of one day: 09:30-11:29 and 13:00-14:59, one bar per minute."""
    morning = pd.date_range(pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30), periods=120, freq="min")
    afternoon = pd.date_range(pd.Timestamp(day) + pd.Timedelta(hours=13), periods=120, freq="min")
    return morning.append(afternoon)


def corpus(root: Path) -> dict[str, Path]:
    """A monthly bar source, monthly labels, the point-in-time universe and the id map."""
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2020-01-06", periods=TRAIN_DAYS + VALIDATION_DAYS)
    validation_start = days[TRAIN_DAYS]
    instruments = [f"S{index:02d}" for index in range(4)]
    identifiers = {name: index + 1 for index, name in enumerate(instruments)}
    bars, labels, universe = [], [], []
    for day in days:
        stamps = session(day)
        for name in instruments:
            values = rng.integers(1000, 2000, size=(len(stamps), len(FEATURES)))
            values[rng.random(values.shape) < 0.02] = -1          # configured missing sentinel
            bars.append(pd.DataFrame({"date": stamps, "instrument_id": identifiers[name],
                                      **dict(zip(FEATURES, values.T))}))
        for time in SAMPLE_TIMES:
            stamp = pd.Timestamp(f"{day.date()} {time}")
            for name in instruments:
                universe.append({"date": stamp, "instrument": name})
                if day < validation_start:
                    labels.append({"date": stamp, "instrument": name,
                                   "target": float(rng.normal(scale=0.01)),
                                   "label_end": stamp + pd.Timedelta(minutes=30)})
    bars_dir, labels_dir = root / "bars", root / "labels"
    bars_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    frame = pd.concat(bars, ignore_index=True)
    for month, group in frame.groupby(frame["date"].dt.strftime("%Y-%m"), sort=True):
        # Record batches arrive shuffled inside a month, like the real archive.
        group.pipe(lambda part: pd.concat([part.iloc[index::3] for index in range(3)],
                                          ignore_index=True)).to_feather(bars_dir / f"{month}.0.feather")
    pd.DataFrame(labels).to_feather(labels_dir / "2020-01.0.feather")
    pd.DataFrame(universe).to_parquet(root / "universe.parquet", index=False)
    pd.DataFrame({"instrument": instruments,
                  "instrument_id": [identifiers[name] for name in instruments]}).to_csv(
        root / "map.csv", index=False)
    return {"bars": bars_dir, "labels": labels_dir,
            "instruments": root / "universe.parquet", "map": root / "map.csv"}


def recipe(root: Path, paths: dict[str, Path]) -> Path:
    """The registered MLPLOB v3 recipe with only the window, the paths and the budget relaxed."""
    document = json.loads(RECIPE.read_text(encoding="utf-8"))
    document["data"].update(bars_path=str(paths["bars"]), labels_path=str(paths["labels"]),
                            instruments_path=str(paths["instruments"]),
                            instrument_map_path=str(paths["map"]), cache_dir=None,
                            history_days=1, lookback=16,
                            # one event per step so the loss trace of both runs is comparable
                            sample_times=list(SAMPLE_TIMES),
                            training_sample_times=list(SAMPLE_TIMES))
    days = pd.bdate_range("2020-01-06", periods=TRAIN_DAYS + VALIDATION_DAYS)
    document.update(experiment_id="synthetic-streaming-driver",
                    hypothesis="the streaming pool must not change the trajectory",
                    output_root=str(root / "runs"))
    document["splits"] = {
        "train": {"start": f"{days[0].date()}T00:00:00", "end": f"{days[TRAIN_DAYS].date()}T00:00:00"},
        "validation": {"start": f"{days[TRAIN_DAYS].date()}T00:00:00",
                       "end": f"{(days[TRAIN_DAYS] + pd.Timedelta(days=1)).date()}T00:00:00"},
        "final": {"start": "2024-10-08T00:00:00", "end": "2025-01-01T00:00:00"}}
    document["model"].update(hidden_dim=8, num_layers=2, batch_size=4, epochs=4, device="cpu",
                             max_steps=19800, log_every_steps=1)
    path = root / "configs" / "recipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return path


def plan(root: Path, recipe_path: Path, *, name: str = "plan.json",
         points=((0.3, "p1"), (0.6, "p2"))) -> Path:
    document = {"schema_version": 1, "experiment_id": "synthetic-streaming-driver",
                "recipe_config": recipe_path.relative_to(root).as_posix(),
                "recipe_sha256": file_sha256(recipe_path),
                "budget_points": [{"name": point, "effective_epochs": epochs}
                                  for epochs, point in points],
                "lr_schedule": "constant",
                "exposure": {"min_coverage": 0.1, "effective_epochs_atol": 0.2}}
    path = root / name
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return path


def drive(root: Path, plan_path: Path, output: Path, monkeypatch, *, streaming: bool,
          resume_from: Path | None = None) -> Path:
    """Run the driver once through its command line and return the run directory it created."""
    from scripts import run_training_trajectory

    # Source capture archives the whole checkout; this test covers the trajectory wiring.
    monkeypatch.setattr(run_training_trajectory, "capture_source",
                        lambda run_dir: {"commit": "0" * 40, "content_sha256": "0" * 64,
                                         "snapshot": None, "files": {}})
    argv = ["run_training_trajectory.py", "--plan", str(plan_path), "--repo", str(root),
            "--output-root", str(output)]
    if streaming:
        argv.append("--stream-bars")
    if resume_from is not None:
        argv += ["--resume-from", str(resume_from)]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_training_trajectory.main() == 0
    runs = sorted(path for path in output.iterdir() if path.is_dir())
    assert len(runs) == 1, runs
    return runs[0]


def events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in
            (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_state(run_dir: Path, point: str) -> dict[str, bytes]:
    """Every byte a resumed update depends on: parameters, optimizer, RNG, sampler, exposure."""
    payload = torch.load(run_dir / "trajectory" / "points" / point / CHECKPOINT_FILENAME,
                         map_location="cpu", weights_only=True)
    flat = {f"network.{name}": value.contiguous().numpy().tobytes()
            for name, value in payload["network_state_dict"].items()}
    for index, entry in enumerate(payload["optimizer_state_dict"]["state"].values()):
        for key, value in entry.items():
            flat[f"optimizer.{index}.{key}"] = (value.contiguous().numpy().tobytes()
                                                if torch.is_tensor(value) else repr(value).encode())
    flat["generator_before_epoch"] = payload["generator_before_epoch"].numpy().tobytes()
    flat["exposure_counts"] = payload["exposure_counts"].numpy().tobytes()
    for name in ("position", "fingerprint", "rng"):
        flat[name] = json.dumps(payload[name], sort_keys=True, default=str).encode()
    return flat


def test_stream_bar_driver_reproduces_the_materialised_trajectory(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    root.mkdir()
    recipe_path = recipe(root, corpus(root))
    plan_path = plan(root, recipe_path)
    materialised = drive(root, plan_path, root / "runs" / "materialised", monkeypatch,
                         streaming=False)
    streamed = drive(root, plan_path, root / "runs" / "streamed", monkeypatch, streaming=True)

    # The switch is the pool, not the accounting: same usable pool, same derived steps.
    assert json.loads((materialised / "budget.json").read_text(encoding="utf-8")) \
        == json.loads((streamed / "budget.json").read_text(encoding="utf-8"))
    assert json.loads((materialised / "exposure.json").read_text(encoding="utf-8")) \
        == json.loads((streamed / "exposure.json").read_text(encoding="utf-8"))
    assert json.loads((materialised / "trajectory.json").read_text(encoding="utf-8")) \
        == json.loads((streamed / "trajectory.json").read_text(encoding="utf-8"))
    assert (materialised / "data.json").read_text(encoding="utf-8") != \
        (streamed / "data.json").read_text(encoding="utf-8")   # provenance: bounded read recorded

    # Same per-step losses, and only the streamed run says which pool it trained on.
    def losses(run_dir):
        return [row["context"]["loss"] for row in events(run_dir)
                if row["event_type"] == "progress"]

    assert losses(materialised) == losses(streamed) and len(losses(streamed)) == 8
    started = [[row for row in events(run_dir) if row["event_type"] == "started"][0]
               for run_dir in (materialised, streamed)]
    assert "streaming_pool" not in started[0]["context"]
    assert started[1]["context"]["streaming_pool"] is True

    # Same weights, same checkpoints, same validation scores at every budget point.
    for point in ("01-p1", "02-p2"):
        left, right = checkpoint_state(materialised, point), checkpoint_state(streamed, point)
        assert left.keys() == right.keys()
        assert [name for name in left if left[name] != right[name]] == []
    for name in ("p1", "p2"):
        assert file_sha256(materialised / f"modified_tlob_weights-{name}.json") \
            == file_sha256(streamed / f"modified_tlob_weights-{name}.json")
        left = pd.read_parquet(materialised / "budget_points" / f"{name}_predictions.parquet")
        right = pd.read_parquet(streamed / "budget_points" / f"{name}_predictions.parquet")
        pd.testing.assert_frame_equal(left, right)
        assert np.isfinite(left["score"].to_numpy()).all() and len(left) > 0


def test_resume_from_extends_the_trajectory_bit_for_bit(tmp_path, monkeypatch):
    """A plan may append budget points and continue from the shorter plan's last checkpoint.

    ``short`` trains the first two points; ``continued`` resumes from that run under a plan that
    appends the epoch-end point and one step past it; ``whole`` trains the extended plan from
    scratch. The continuation has to land on exactly the artifacts ``whole`` produced - that is
    the only thing that makes 927d@2.64 -> 3.96 cheap instead of a retrain.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    recipe_path = recipe(root, corpus(root))
    short_plan = plan(root, recipe_path, name="short.json")
    long_plan = plan(root, recipe_path, name="long.json",
                     points=((0.3, "p1"), (0.6, "p2"), (1.0, "p3"), (1.1, "p4")))
    short = drive(root, short_plan, root / "runs" / "short", monkeypatch, streaming=True)
    continued = drive(root, long_plan, root / "runs" / "continued", monkeypatch, streaming=True,
                      resume_from=short)
    whole = drive(root, long_plan, root / "runs" / "whole", monkeypatch, streaming=True)

    document = json.loads((continued / "trajectory.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in document["completed_points"]] == ["p3", "p4"]
    assert json.loads((continued / "manifest.json").read_text(encoding="utf-8"))[
        "resumed_from"]["steps"] == 8
    # Point numbering stays the plan's, so a skipped point's artifacts are absent, not renumbered.
    assert not (continued / "trajectory" / "points" / "01-p1").exists()
    for point in ("03-p3", "04-p4"):
        assert (continued / "trajectory" / "points" / point / "training_state.pt").is_file()
    for name, point in (("p3", "03-p3"), ("p4", "04-p4")):
        left, right = checkpoint_state(whole, point), checkpoint_state(continued, point)
        assert [field for field in left if left[field] != right[field]] == []
        for artifact in (f"model-{name}.pt", f"modified_tlob_weights-{name}.json"):
            assert file_sha256(whole / artifact) == file_sha256(continued / artifact), artifact
        pd.testing.assert_frame_equal(
            pd.read_parquet(whole / "budget_points" / f"{name}_predictions.parquet"),
            pd.read_parquet(continued / "budget_points" / f"{name}_predictions.parquet"))

    # A checkpoint written exactly at an epoch end takes the other restore branch: the sampler
    # generator is not rewound to ``generator_before_epoch`` because the next epoch redraws its
    # permutation from the saved state. One step of continuation has to be identical all the same.
    boundary = drive(root, long_plan, root / "runs" / "boundary", monkeypatch, streaming=True,
                     resume_from=whole / "trajectory" / "points" / "03-p3" / "training_state.pt")
    boundary_document = json.loads((boundary / "trajectory.json").read_text(encoding="utf-8"))
    assert [row["name"] for row in boundary_document["completed_points"]] == ["p4"]
    left, right = checkpoint_state(whole, "04-p4"), checkpoint_state(boundary, "04-p4")
    assert [field for field in left if left[field] != right[field]] == []
    for artifact in ("model-p4.pt", "modified_tlob_weights-p4.json"):
        assert file_sha256(whole / artifact) == file_sha256(boundary / artifact), artifact

    def trace(run_dir):
        return {row["context"]["step"]: row["context"]["loss"] for row in events(run_dir)
                if row["event_type"] == "progress"}

    whole_trace, continued_trace = trace(whole), trace(continued)
    resumed = [step for step in continued_trace if step > 8]
    assert resumed and len(resumed) > 4
    assert [continued_trace[step] for step in resumed] == [whole_trace[step] for step in resumed]


def test_resume_from_refuses_a_source_this_plan_does_not_continue(tmp_path, monkeypatch):
    """A resume that would make the archive describe a different trajectory must fail loudly."""
    from scripts import run_training_trajectory

    root = tmp_path / "corpus"
    root.mkdir()
    recipe_path = recipe(root, corpus(root))
    short = drive(root, plan(root, recipe_path, name="short.json"),
                  root / "runs" / "short", monkeypatch, streaming=True)
    # Same points, first one shifted: the source's trained steps are not the head of this plan.
    shifted = plan(root, recipe_path, name="shifted.json",
                   points=((0.35, "p1"), (0.6, "p2"), (1.1, "p3")))
    monkeypatch.setattr(run_training_trajectory, "capture_source",
                        lambda run_dir: {"commit": "0" * 40, "content_sha256": "0" * 64,
                                         "snapshot": None, "files": {}})
    monkeypatch.setattr(sys, "argv", ["run_training_trajectory.py", "--plan", str(shifted),
                                      "--repo", str(root), "--output-root",
                                      str(root / "runs" / "shifted"), "--stream-bars",
                                      "--resume-from", str(short)])
    with pytest.raises(ValueError, match="not the head of this plan's points"):
        run_training_trajectory.main()
    assert not (root / "runs" / "shifted").exists() or not any(
        (root / "runs" / "shifted").iterdir())
