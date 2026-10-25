"""Operational checks for the cloud sweep: local run watch and artifact fetch.

The sweep trains on the rented GPU container and writes run directories under a
remote ``runs/`` root. Only the small evaluation artifacts are pulled locally; the
24 MB weight files are fetched explicitly. ``local`` is the once-per-20-minutes
watch: it reports run directories that appeared since the previous call and the
best validation RankIC among them (local approximation, not the official score).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SUMMARY = ("rank_ic_mean", "long_short_return_mean", "turnover_mean", "min_daily_long_short_return")
ARTIFACTS = ("config.json", "config.sha256", "data.json", "coverage.json", "selection.json",
             "checks.json", "manifest.json", "events.jsonl", "metrics.json", "daily.csv",
             "predictions.parquet", "targets.parquet", "train_history.parquet", "device.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan_local(root: Path) -> list[dict]:
    runs = []
    for metrics in sorted(root.glob("*/*/metrics.json")):
        run_dir = metrics.parent
        try:
            document = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entry = {"run_dir": run_dir.as_posix(), "modified_utc":
                 datetime.fromtimestamp(metrics.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        for key in SUMMARY:
            value = document.get(key)
            entry[key] = float(value) if isinstance(value, (int, float)) else None
        config = run_dir / "config.json"
        if config.exists():
            try:
                entry["experiment_id"] = json.loads(config.read_text(encoding="utf-8")).get("experiment_id")
            except (OSError, ValueError):
                pass
        runs.append(entry)
    return runs


def local(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    seen = json.loads(state_path.read_text(encoding="utf-8")).get("seen", []) if state_path.exists() else []
    runs = scan_local(Path(args.root))
    new = [run for run in runs if run["run_dir"] not in seen]
    ranked = [run for run in runs if run["rank_ic_mean"] is not None]
    best = max(ranked, key=lambda run: run["rank_ic_mean"]) if ranked else None
    report = {"checked_at_utc": utc_now(), "runs": len(runs), "new": new, "best_by_validation_rank_ic": best,
              "latest_modified": max(runs, key=lambda run: run["modified_utc"]) if runs else None}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"checked_at_utc": report["checked_at_utc"],
                                      "seen": [run["run_dir"] for run in runs]}, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def ssh_command() -> list[str]:
    command = os.environ.get("JYQ_CLOUD_SSH")
    if not command:
        raise SystemExit("JYQ_CLOUD_SSH is not set, e.g. 'ssh -i ~/.ssh/key user@host -p 26305'")
    return shlex.split(command)


def fetch(args: argparse.Namespace) -> int:
    remote_root = os.environ.get("JYQ_CLOUD_RUNS", "/root/v5/repo/runs")
    destination = Path(args.dest)
    destination.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        files = " ".join(shlex.quote(f"{run}/{name}") for name in ARTIFACTS)
        remote = f"cd {shlex.quote(remote_root)} && tar -cf - $(for f in {files}; do [ -e \"$f\" ] && echo \"$f\"; done)"
        tar = subprocess.run(ssh_command() + [remote], stdout=subprocess.PIPE, check=True)
        subprocess.run(["tar", "-xf", "-", "-C", str(destination)], input=tar.stdout, check=True)
        print(f"fetched {run} -> {destination}")
    return 0


def remote(args: argparse.Namespace) -> int:
    log = os.environ.get("JYQ_CLOUD_LOG", "/root/v5/logs/arms.log")
    command = (f"tail -n {args.lines} {shlex.quote(log)}; echo '===ACTIVE'; "
               "ps -eo args | grep 'jinyiquant run' | grep -v grep || true; echo '===RUNS'; "
               f"ls -1 {shlex.quote(os.environ.get('JYQ_CLOUD_RUNS', '/root/v5/repo/runs'))}")
    subprocess.run(ssh_command() + [command], check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("local", help="report run directories that appeared since the previous check")
    watch.add_argument("--root", default="runs")
    watch.add_argument("--state", default="scratch/cloud_model_watch.json")
    watch.set_defaults(func=local)
    pull = sub.add_parser("fetch", help="pull the small artifact subset of remote run directories")
    pull.add_argument("runs", nargs="+", help="run paths relative to the remote runs root")
    pull.add_argument("--dest", default="runs")
    pull.set_defaults(func=fetch)
    view = sub.add_parser("remote", help="tail the cloud arm log and list active runs")
    view.add_argument("--lines", type=int, default=20)
    view.set_defaults(func=remote)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
