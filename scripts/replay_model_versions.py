"""Replay frozen model weight sets on fixed development windows with saved preprocessing.

The three requested identities are the submitted July V0 package, V1 and V2-60d; the
October local V0 is replayed as an auxiliary control because it shares V1's training
dates and therefore isolates the label-scheme change from the training-window change.
Every version is scored on exactly the same keys, dates and labels; weights never
change inside a window. Metrics stay the local approximate convention of
``jinyiquant.evaluation`` (gross forward target returns, no fees or execution).
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.data import build_batch, load_data, select_targets  # noqa: E402
from jinyiquant.evaluation import evaluate  # noqa: E402
from jinyiquant.io import file_sha256, read_json, utc_now, write_json  # noqa: E402
from jinyiquant.tracking import Recorder, capture_source, finish, new_run_dir  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402

SUMMARY_METRICS = (
    "ic_mean", "rank_ic_mean", "rank_ic_ir", "long_return_mean", "short_return_mean",
    "long_short_return_mean", "long_short_sharpe", "turnover_mean", "min_daily_long_short_return",
)
DAILY_METRICS = ("ic", "rank_ic", "long_short_return", "long_return", "short_return", "turnover")
DEFAULT_PAIRS = (
    ("v1_oct_csz", "v0_july_submitted"), ("v2_60d_csz", "v0_july_submitted"),
    ("v1_oct_csz", "v0_local_oct"), ("v2_60d_csz", "v0_local_oct"),
    ("v1_oct_csz", "v2_60d_csz"),
)


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_plan(path: Path) -> tuple[dict, dict]:
    plan = read_json(path)
    base = path.resolve().parent
    data = dict(plan["data"])
    for name in ("bars_path", "labels_path", "instruments_path", "instrument_map_path", "cache_dir"):
        if data.get(name):
            data[name] = str(resolve(base, data[name]))
    for entry in plan["versions"]:
        entry["weights"] = str(resolve(base, entry["weights"]))
    for entry in plan.get("identity_candidates", []):
        entry["path"] = str(resolve(base, entry["path"]))
    return plan, data


def window_keys(bundle, config: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Independently generated prediction keys: point-in-time universe at approved clocks."""
    universe = bundle.universe
    if universe is None:
        raise ValueError("Replay requires the point-in-time instruments table")
    expected = universe.loc[(universe.date >= start) & (universe.date < end)]
    times = config["sample_times"]
    expected = expected.loc[expected.date.dt.strftime("%H:%M").isin(times)]
    split = {"start": start, "end": end}
    keys = select_targets(expected.assign(target=0.0), split, config["label_horizon_minutes"])[
        ["date", "instrument"]]
    if keys.empty:
        raise ValueError(f"No prediction keys in window {start}..{end}")
    return keys.sort_values(["date", "instrument"]).reset_index(drop=True)


def window_targets(bundle, config: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return select_targets(bundle.labels, {"start": start, "end": end},
                          config["label_horizon_minutes"], sample_times=config["sample_times"])


def predict_window(models: dict, bundle, config: dict, keys: pd.DataFrame, chunk_days: int):
    """Build identical batches once per day chunk and score every version on them."""
    days = np.sort(pd.unique(keys.date.dt.normalize()))
    scores = {name: [] for name in models}
    chunks, fallback_keys = [], []
    for offset in range(0, len(days), chunk_days):
        chunk_days_set = set(pd.to_datetime(days[offset:offset + chunk_days]))
        chunk_keys = keys.loc[keys.date.dt.normalize().isin(chunk_days_set)].reset_index(drop=True)
        started = time.monotonic()
        batch = build_batch(bundle.bars, chunk_keys, config)
        missing = batch.effective_rows.sum(axis=1) < int(config["min_history"])
        no_history = ~batch.effective_rows.any(axis=1)
        fallback_keys.append(batch.keys.loc[missing, ["date", "instrument"]].assign(
            no_history=no_history[missing]))
        for name, model in models.items():
            scores[name].append(model.predict(batch))
        chunks.append({"days": len(chunk_days_set), "keys": len(batch.keys),
                       "coverage": batch.coverage, "seconds": time.monotonic() - started})
        del batch
        gc.collect()
    result = {name: np.concatenate(values) for name, values in scores.items()}
    return result, chunks, pd.concat(fallback_keys, ignore_index=True)


def daily_counts(keys: pd.DataFrame, targets: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    day = keys.date.dt.normalize()
    expected = keys.assign(day=day).groupby("day").size()
    scored = targets.assign(day=targets.date.dt.normalize()).groupby("day").size()
    no_history = fallback.assign(day=fallback.date.dt.normalize()).groupby("day").size()
    rows = pd.DataFrame({"date": expected.index.strftime("%Y-%m-%d")})
    rows["expected_sample_count"] = expected.to_numpy()
    rows["scored_sample_count"] = rows.date.map(
        scored.rename(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"))).fillna(0).astype(int)
    rows["missing_answer_count"] = rows.expected_sample_count - rows.scored_sample_count
    rows["no_history_count"] = rows.date.map(
        no_history.rename(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"))).fillna(0).astype(int)
    return rows


def evaluate_version(prediction: np.ndarray, batch_keys: pd.DataFrame, targets: pd.DataFrame,
                     evaluation: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    predictions = batch_keys.assign(score=prediction)
    scored = predictions.merge(targets[["date", "instrument"]], on=["date", "instrument"],
                               how="inner", validate="one_to_one")
    if scored.empty:
        raise ValueError("No scored predictions after exact key alignment")
    metrics, daily, sections = evaluate(
        scored, targets, evaluation, return_sections=True,
        undefined_constant_portfolios=evaluation.get("constant_portfolios") == "undefined")
    return metrics, daily, sections


def by_time(sections: pd.DataFrame) -> pd.DataFrame:
    frame = sections.assign(clock=pd.to_datetime(sections.timestamp).dt.strftime("%H:%M"))
    grouped = frame.groupby("clock", sort=True)
    return pd.DataFrame({
        "sections": grouped.size(),
        "days": grouped.date.nunique(),
        "ic_mean": grouped.ic.mean(),
        "rank_ic_mean": grouped.rank_ic.mean(),
        "long_short_return_mean": grouped.long_short_return.mean(),
        "turnover_mean": grouped.turnover.mean(),
    }).reset_index()


def run(plan_path: Path, output_root: Path | None = None) -> Path:
    plan, config = load_plan(plan_path)
    root = output_root or resolve(plan_path.resolve().parent, plan["output_root"])
    run_dir = new_run_dir(root)
    recorder = Recorder(run_dir, plan["experiment_id"])
    recorder.event("run", "started", plan=str(plan_path), windows=len(plan["windows"]),
                   versions=len(plan["versions"]), label_input=False)
    write_json(run_dir / "plan.json", plan)
    started = time.monotonic()
    device = "cuda" if _cuda_available() else "cpu"
    models = {}
    for version in plan["versions"]:
        path = Path(version["weights"])
        digest = file_sha256(path)
        if version.get("expected_sha256") and digest != version["expected_sha256"]:
            raise ValueError(f"{version['name']}: weight checksum changed: {digest}")
        model = TLOBAdapter.load(path, device=device, recorder=recorder)
        saved = dict(model.preprocessing)
        for name in ("feature_columns", "lookback", "min_history", "history_days", "max_ffill_steps",
                     "missing_policy", "missing_sentinel", "price_scale", "sample_times"):
            if name in saved and saved[name] != config[name]:
                raise ValueError(f"{version['name']}: saved preprocessing {name} differs from replay config")
        model.config["batch_size"] = int(plan.get("predict_batch_size", 32))
        models[version["name"]] = model
        recorder.event("loading", "model_loaded", version=version["name"], sha256=digest,
                       device=str(model.device), training=False,
                       predict_batch_size=model.config["batch_size"])
    responses = []
    for window in plan["windows"]:
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        recorder.event("data", "loading", window=window["name"], start=str(start), end=str(end))
        bundle = load_data(config, start, end)
        keys = window_keys(bundle, config, start, end)
        targets = window_targets(bundle, config, start, end)
        if set(map(tuple, targets[["date", "instrument"]].to_numpy())) - set(
                map(tuple, keys[["date", "instrument"]].to_numpy())):
            raise ValueError("Selected targets lie outside the prediction universe")
        recorder.event("data", "validated", window=window["name"], keys=len(keys),
                       targets=len(targets), bars=len(bundle.bars))
        scores, chunks, fallback = predict_window(models, bundle, config, keys,
                                                  int(plan.get("chunk_days", 11)))
        counts = daily_counts(keys, targets, fallback)
        write_json(run_dir / "windows" / window["name"] / "coverage.json",
                   {"keys": len(keys), "targets": len(targets), "chunks": chunks,
                    "no_history_keys": int(len(fallback)),
                    "no_history_ratio": float(len(fallback) / len(keys))})
        keys.to_parquet(run_dir / "windows" / window["name"] / "keys.parquet", index=False)
        targets.to_parquet(run_dir / "windows" / window["name"] / "targets.parquet", index=False)
        for version in plan["versions"]:
            name = version["name"]
            metrics, daily, sections = evaluate_version(scores[name], keys, targets, plan["evaluation"])
            merged_daily = daily.merge(counts, on="date", how="left", validate="one_to_one")
            metrics["coverage"] = {"prediction_samples": len(keys), "scored_samples": len(targets),
                                   "missing_answer_samples": len(keys) - len(targets),
                                   "no_history_samples": int(len(fallback)),
                                   "no_history_ratio": float(len(fallback) / len(keys))}
            out = run_dir / "windows" / window["name"] / name
            out.mkdir(parents=True, exist_ok=True)
            keys.assign(score=scores[name]).to_parquet(out / "predictions.parquet", index=False)
            merged_daily.to_csv(out / "daily.csv", index=False)
            sections.to_csv(out / "by_timestamp.csv", index=False)
            by_time(sections).to_csv(out / "by_time.csv", index=False)
            write_json(out / "metrics.json", metrics)
            responses.append({"window": window["name"], "version": name, **metrics})
            recorder.event("evaluation", "completed", window=window["name"], version=name,
                           rank_ic_mean=metrics["rank_ic_mean"], ic_mean=metrics["ic_mean"],
                           long_short_return_mean=metrics["long_short_return_mean"],
                           turnover_mean=metrics["turnover_mean"], scored_samples=len(targets))
        del bundle, keys, targets, scores, fallback
        gc.collect()
    write_json(run_dir / "summary_metrics.json", {"plan": plan_path.name, "responses": responses,
                                                  "device": device, "created_at": utc_now()})
    manifest = {"schema_version": 1, "run_id": run_dir.name, "experiment_id": plan["experiment_id"],
                "status": "completed", "started_at": utc_now(), "device": device,
                "weights": {entry["name"]: {"path": entry["weights"], "sha256": file_sha256(Path(entry["weights"]))}
                            for entry in plan["versions"]}}
    manifest["source"] = capture_source(run_dir)
    write_json(run_dir / "manifest.json", manifest)
    recorder.event("run", "completed", elapsed_seconds=time.monotonic() - started)
    finish(run_dir, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
    return run_dir


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _block_delta(frame: pd.DataFrame, column: str, block: int, draws: int, seed: int) -> dict:
    values = frame[column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"days": int(len(values)), "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(draws,))
    offsets = np.arange(block)[None, :]
    indices = (starts[:, None] + offsets) % len(values)
    means = values[indices].mean(axis=1)
    return {"days": int(len(values)), "mean": float(values.mean()),
            "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def pair_list(plan: dict) -> list[tuple[str, str]]:
    """Plan-defined comparison pairs, defaulting to the frozen 20261023 set."""
    pairs = [tuple(pair) for pair in plan.get("pairs", DEFAULT_PAIRS)]
    names = {version["name"] for version in plan["versions"]}
    for pair in pairs:
        if len(pair) != 2 or not set(pair) <= names:
            raise ValueError(f"Invalid comparison pair {pair}; versions are {sorted(names)}")
    return pairs


def report(plan_path: Path, run_dir: Path) -> Path:
    plan, _ = load_plan(plan_path)
    versions = [version["name"] for version in plan["versions"]]
    labels = {version["name"]: version["label"] for version in plan["versions"]}
    daily_rows, time_rows, summary_rows = [], [], []
    for window in plan["windows"]:
        name = window["name"]
        coverage = read_json(run_dir / "windows" / name / "coverage.json")
        for version in versions:
            out = run_dir / "windows" / name / version
            daily = pd.read_csv(out / "daily.csv")
            daily.insert(0, "version_label", labels[version])
            daily.insert(0, "version", version)
            daily.insert(0, "window", name)
            daily_rows.append(daily)
            time = pd.read_csv(out / "by_time.csv").rename(columns={"clock": "time"})
            time.insert(0, "version", version)
            time.insert(0, "window", name)
            time_rows.append(time)
            metrics = read_json(out / "metrics.json")
            ic_series = daily["ic"].dropna()
            ic_ir = (float(ic_series.mean() / ic_series.std(ddof=1)) if len(ic_series) > 1
                     and ic_series.std(ddof=1) > 0 else None)
            summary_rows.append({
                "window": name, "version": version, "label": labels[version],
                "ic_ir": ic_ir,
                **{metric: metrics[metric] for metric in SUMMARY_METRICS},
                "day_count": metrics["day_count"], "cross_section_count": metrics["cross_section_count"],
                "sample_count": metrics["sample_count"],
                "no_history_samples": coverage["no_history_keys"],
                "no_history_ratio": coverage["no_history_ratio"],
            })
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    daily_all = pd.concat(daily_rows, ignore_index=True)
    daily_all.to_csv(report_dir / "daily_all.csv", index=False)
    by_time_all = pd.concat(time_rows, ignore_index=True)
    by_time_all.to_csv(report_dir / "eight_time_points.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "summary.csv", index=False)
    # Paired daily differences on identical keys; pairs come from the plan.
    wide = daily_all.pivot_table(index=["window", "date"], columns="version",
                                 values=["ic", "rank_ic", "long_short_return", "turnover"])
    pairs = pair_list(plan)
    delta_rows, paired_rows = [], []
    for window in plan["windows"]:
        name = window["name"]
        for left, right in pairs:
            for metric in ("ic", "rank_ic", "long_short_return", "turnover"):
                delta = (wide[(metric, left)] - wide[(metric, right)]).loc[name].dropna()
                for date, value in delta.items():
                    delta_rows.append({"window": name, "pair": f"{left}-minus-{right}",
                                       "metric": metric, "date": date, "delta": value})
                block = _block_delta(delta.rename("delta").reset_index(), "delta", 5,
                                     int(plan.get("bootstrap_draws", 20000)), int(plan["seed"]))
                paired_rows.append({"window": name, "pair": f"{left}-minus-{right}", "metric": metric,
                                    "days": block["days"], "mean_delta": block["mean"],
                                    "block5_ci_low": block["ci_low"], "block5_ci_high": block["ci_high"],
                                    "positive_days": int((delta > 0).sum()),
                                    "negative_days": int((delta < 0).sum()),
                                    "max_abs_date": str(delta.abs().idxmax()) if len(delta) else None})
    pd.DataFrame(delta_rows).to_csv(report_dir / "daily_deltas.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(report_dir / "paired_summary.csv", index=False)
    return report_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("plan", type=Path)
    execute.add_argument("--output-root", type=Path)
    build = sub.add_parser("report")
    build.add_argument("plan", type=Path)
    build.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run(args.plan, args.output_root))
    else:
        print(report(args.plan, args.run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
