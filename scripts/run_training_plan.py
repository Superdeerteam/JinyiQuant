"""Run a frozen list of existing pipeline configs sequentially, retaining every attempt."""

import argparse
import time
from pathlib import Path

from jinyiquant.config import RunConfig, load_config
from jinyiquant.io import read_json, utc_now, write_json
from jinyiquant.runner import run
from jinyiquant.tracking import Recorder, capture_source, environment, finish, new_run_dir


def execute(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    entries = plan["runs"]
    if not entries or len({entry["name"] for entry in entries}) != len(entries):
        raise ValueError("Run names must be unique and nonempty")
    configs = [load_config(plan_path.parent / entry["config"]) for entry in entries]
    if any(config.evaluation_split != "validation" for config in configs):
        raise ValueError("Training research never opens the final split")
    if any(not entry["name"].replace("-", "").replace("_", "").isalnum() for entry in entries):
        raise ValueError("Run names must be simple identifiers")
    output = new_run_dir((plan_path.parent / plan["output_root"]).resolve())
    rec = Recorder(output, plan["experiment_id"])
    started = time.monotonic()
    manifest = {
        "schema_version": 1, "run_id": output.name, "experiment_id": plan["experiment_id"],
        "started_at": utc_now(), "status": "running", "final_views": 0,
    }
    write_json(output / "manifest.json", manifest)
    attempts = []
    rec.event("run", "started", planned_runs=len(entries), budget=plan["total_wall_seconds"])
    print(str(output), flush=True)
    try:
        write_json(output / "plan.json", plan)
        manifest.update(code=capture_source(output), environment=environment())
        write_json(output / "manifest.json", manifest)
        for entry, config in zip(entries, configs):
            remaining = plan["total_wall_seconds"] - (time.monotonic() - started)
            if remaining < config.budget.wall_seconds:
                raise TimeoutError("Remaining stage budget cannot cover the next declared run cap")
            raw = config.snapshot()
            raw["output_root"] = str(output / entry["name"])
            config = RunConfig.model_validate(raw)
            rec.event("training", "cell_started", name=entry["name"], config=config.snapshot())
            path, code = run(config)
            attempts.append({**entry, "run": str(path), "exit_code": code})
            write_json(output / "attempts.json", attempts)
            rec.event("training", "cell_ended", **attempts[-1])
            print(f"{entry['name']}: exit={code} run={path}", flush=True)
            if code:
                raise RuntimeError(f"Cell {entry['name']} failed with exit code {code}; see {path}")
        write_json(output / "metrics.json", {"execution_completed": True, "final_views": 0,
                                             "completed_runs": len(attempts), "model_promoted": False})
        finish(output, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
        return 0
    except BaseException as exc:
        rec.failure("training_plan", exc)
        finish(output, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", 1,
               resources={"wall_seconds": time.monotonic() - started}, error={"reason": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    raise SystemExit(execute(parser.parse_args().plan))
