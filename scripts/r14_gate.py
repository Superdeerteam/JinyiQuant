"""Is the box ready for the R14 arm? Exit 0 only when the hard dependency is really satisfied.

The R14 arm must not preempt anything on the training host. "Ready" therefore means all of:

* no ``run_training_trajectory.py`` process is alive (the R1 rankhinge seeds and every arm the
  team queued behind them have exited — a single ``ps`` snapshot during a restart gap is exactly
  the reading that fooled an earlier version of the watcher);
* each R1 rankhinge run is terminal -- it recorded the scored point in its ``trajectory.json``
  (finished) or its manifest carries a failed status with a reason (it died and produced no
  weights).  A run that merely stopped with neither record is *not* terminal;
* some GPU has room (memory used below the limit and utilisation below the limit), because the
  arm needs a device and running next to two saturated cards only slows every arm down.

    python scripts/r14_gate.py            # print the state, exit 0 when ready
    python scripts/r14_gate.py --quiet
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_RUNS = "/mnt/work/runs"
RANKHINGE_SEEDS = (11, 29, 47)
POINT_SUFFIX = "1.32"


def training_processes() -> int:
    completed = subprocess.run(["pgrep", "-fc", "scripts/run_training_trajectory.py"],
                               capture_output=True, text=True)
    return int(completed.stdout.strip() or 0)


def seed_states(runs: str = DEFAULT_RUNS) -> dict[int, str]:
    """``finished``, ``failed`` or ``unfinished`` for each R1 rankhinge seed."""
    states = {}
    for seed in RANKHINGE_SEEDS:
        root = Path(runs) / f"screen10-rankhinge-seed{seed}"
        state = "unfinished"
        for path in sorted(root.glob("*/trajectory.json")):
            points = json.loads(path.read_text(encoding="utf-8")).get("completed_points") or []
            if any(POINT_SUFFIX in str(point.get("name", "")) for point in points):
                state = "finished"
        if state == "unfinished":
            for path in sorted(root.glob("*/manifest.json")):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") == "failed" and manifest.get("error"):
                    state = "failed"
        states[seed] = state
    return states


def rankhinge_unfinished(runs: str = DEFAULT_RUNS) -> list[int]:
    """Seeds whose rankhinge run is neither finished nor recorded as failed."""
    return [seed for seed, state in seed_states(runs).items() if state == "unfinished"]


def free_gpus(*, memory_mb: int = 12000, utilisation: int = 50) -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if completed.returncode != 0:
        return []
    free = []
    for line in completed.stdout.strip().splitlines():
        index, memory, util = (part.strip() for part in line.split(","))
        if int(memory) < memory_mb and int(util) < utilisation:
            free.append(int(index))
    return free


def state(*, memory_mb: int = 12000, utilisation: int = 50) -> dict:
    busy = training_processes()
    unfinished = rankhinge_unfinished()
    free = free_gpus(memory_mb=memory_mb, utilisation=utilisation)
    return {"ready": busy == 0 and not unfinished and bool(free),
            "training_processes": busy,
            "rankhinge_unfinished_seeds": unfinished,
            "rankhinge_states": seed_states(),
            "free_gpus": free}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--memory-mb", type=int, default=12000)
    parser.add_argument("--utilisation", type=int, default=50)
    options = parser.parse_args(argv)
    record = state(memory_mb=options.memory_mb, utilisation=options.utilisation)
    if not options.quiet:
        print(json.dumps(record, sort_keys=True))
    return 0 if record["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
