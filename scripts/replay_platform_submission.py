"""Replay the seven-file platform packages on the same windows as the local replay.

Each package runs in its own copied directory and interpreter (Python 3.11, Torch 2.5
CPU, the pinned platform environment). The DAI source is served from the downloaded
official tables in platform units, so ``platform_price_scale=1`` is exercised exactly
as on the platform. The output is a key-by-key comparison with the local replay plus
the history-recovery and mean-fallback counts reported by the package itself.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinyiquant.evaluation import EVALUATION_VERSION  # noqa: E402
from jinyiquant.io import read_json, write_json  # noqa: E402


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def packages(plan: dict, base: Path) -> list[dict]:
    packages = []
    for entry in plan.get("platform_packages", []):
        packages.append({"version": entry["version"], "dir": str(resolve(base, entry["path"]))})
    return packages


def run(plan_path: Path, local_run: Path, output_root: Path, python: Path, chosen_windows, chosen_versions):
    plan = read_json(plan_path)
    base = plan_path.resolve().parent
    data = dict(plan["data"])
    for name in ("bars_path", "instruments_path", "instrument_map_path"):
        data[name] = str(resolve(base, data[name]))
    run_dir = output_root.resolve() / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for window in plan["windows"]:
        if chosen_windows and window["name"] not in chosen_windows:
            continue
        for package in packages(plan, base):
            if chosen_versions and package["version"] not in chosen_versions:
                continue
            target = run_dir / "windows" / window["name"] / package["version"]
            if target.exists():
                continue
            work = target / "package"
            work.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(package["dir"], work)
            environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            environment["PYTHONIOENCODING"] = "utf-8"
            command = [str(python), str(ROOT / "scripts/platform_replay_worker.py"), str(work),
                       window["start"], window["end"], data["bars_path"], data["instruments_path"],
                       data["instrument_map_path"], str(target), "cpu"]
            started = pd.Timestamp.now(tz="UTC")
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                       text=True, encoding="utf-8", errors="replace")
            (target / "worker.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                summaries.append({"window": window["name"], "version": package["version"],
                                  "status": "failed", "returncode": completed.returncode})
                continue
            summary = read_json(target / "worker.json")
            summary.update({"window": window["name"], "version": package["version"], "status": "completed",
                            "started_utc": started.isoformat()})
            summaries.append(summary)
    write_json(run_dir / "platform_summary.json", {"runs": summaries})
    return run_dir


def compare(local_run: Path, platform_run: Path) -> Path:
    rows, daily_rows = [], []
    for window_dir in sorted((platform_run / "windows").iterdir()):
        window = window_dir.name
        for version_dir in sorted(window_dir.iterdir()):
            version = version_dir.name
            worker = read_json(version_dir / "worker.json")
            platform = pd.read_parquet(version_dir / "predictions.parquet").rename(columns={"factor": "platform"})
            local_file = local_run / "windows" / window / version / "predictions.parquet"
            if not local_file.is_file():
                month = platform.date.min().strftime("%b%Y").lower()
                local_file = local_run / "windows" / month / version / "predictions.parquet"
            if not local_file.is_file():
                rows.append({"window": window, "version": version, "status": "local_missing"})
                continue
            local = pd.read_parquet(local_file).rename(columns={"score": "local"})
            local = local.loc[local.date.between(platform.date.min(), platform.date.max())]
            merged = local.merge(platform, on=["date", "instrument"], how="outer", indicator=True, validate="one_to_one")
            both = merged.loc[merged._merge.eq("both")].copy()
            both["difference"] = both.platform - both.local
            both["day"] = both.date.dt.strftime("%Y-%m-%d")
            per_day = both.groupby("day").agg(difference=("difference", lambda values: float(values.abs().max())),
                                              rows=("difference", "size")).reset_index()
            for entry in per_day.to_dict("records"):
                daily_rows.append({"window": window, "version": version, **entry})
            rows.append({
                "window": window, "version": version, "status": "compared",
                "local_rows": len(local), "platform_rows": len(platform), "matched_rows": len(both),
                "missing_here": int((merged._merge == "left_only").sum()),
                "missing_there": int((merged._merge == "right_only").sum()),
                "max_abs_difference": float(both.difference.abs().max()) if len(both) else None,
                "mean_abs_difference": float(both.difference.abs().mean()) if len(both) else None,
                "identical": bool((both.difference == 0).all()) if len(both) else False,
                "fallback_samples": (worker.get("history_handling") or {}).get("fallback_samples"),
                "recovered_samples": (worker.get("history_handling") or {}).get("recovered_samples"),
                "backfill_queries": worker.get("backfill_queries"),
                "elapsed_seconds": worker.get("elapsed_seconds"),
                "platform_rows_per_section": worker.get("rows_per_instrument", {}).get("mean"),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(platform_run / "platform_vs_local.csv", index=False, encoding="utf-8")
    if daily_rows:
        pd.DataFrame(daily_rows).to_csv(platform_run / "platform_vs_local_daily.csv", index=False, encoding="utf-8")
    return platform_run


def consolidate(platform_runs: list[Path], output: Path) -> Path:
    frames = []
    for run_dir in platform_runs:
        file = run_dir / "platform_vs_local.csv"
        if not file.is_file():
            continue
        frame = pd.read_csv(file)
        frame.insert(0, "platform_run", run_dir.name)
        frames.append(frame)
    if not frames:
        raise ValueError("No compared platform runs found")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8")
    record = {
        "schema_version": 1,
        "evaluation_version": EVALUATION_VERSION,
        "experiment_id": "model-version-platform-replay-20261023",
        "scope": ("Seven-file packages executed under Python 3.11 / Torch 2.5.0 CPU with the official "
                  "downloaded tables served as DAI. Probes cover the first two or three trading days of "
                  "each window; full-month neural replay uses the local pipeline with the same saved "
                  "preprocessing."),
        "platform_price_scale": 1,
        "local_price_scale": 100,
        "results": combined.to_dict("records"),
        "checks": {
            "prediction_keys_exact": bool(combined[["missing_here", "missing_there"]].fillna(-1).eq(0).all().all()),
            "max_absolute_difference_below_1e-6": bool(combined.max_abs_difference.max() < 1e-6),
            "no_recovered_samples_in_probes": bool(combined.recovered_samples.fillna(0).eq(0).all()),
            "fallback_equals_local_no_history": bool((combined.fallback_samples.fillna(0).astype(int)
                                                      <= combined.local_rows).all()),
        },
        "caveats": [
            "One November universe name has no rows in the downloaded bar table at all; it receives the "
            "saved training mean in both pipelines (176 of 176000 November samples). The platform bar "
            "table may contain it, so its real platform prediction is not reproduced locally.",
            "Probe comparisons are limited to the opening days; mid-window suspensions longer than the "
            "ten-day warm-up were not observed because local fallback counts stay at the same single "
            "name every November day and zero in January and March.",
            "The platform score itself (including the user-reported first public 0.18503) cannot be "
            "verified offline."
        ],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("plan", type=Path)
    execute.add_argument("--local-run", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, default=ROOT / "runs/platform-replay-20261023")
    execute.add_argument("--python", type=Path, default=ROOT / ".venv-tlob/Scripts/python.exe")
    execute.add_argument("--windows", nargs="*")
    execute.add_argument("--versions", nargs="*")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--local-run", type=Path, required=True)
    compare_parser.add_argument("--platform-run", type=Path, required=True)
    merge = sub.add_parser("consolidate")
    merge.add_argument("--runs", nargs="+", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        print(run(args.plan, args.local_run, args.output_root, args.python, args.windows, args.versions))
    elif args.command == "compare":
        print(compare(args.local_run, args.platform_run))
    else:
        print(consolidate(args.runs, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
