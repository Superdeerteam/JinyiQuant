"""Cloud job manifest for one continuous training trajectory.

The manifest answers, before anything is rented or uploaded: which code snapshot runs, which
data files must be on the remote host, which commands produce the checkpointed trajectory and
which artifacts have to come back. It is derived from a pre-registered plan and the fixed
recipe configuration; nothing here trains, uploads or evaluates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from jinyiquant.config import load_config
from jinyiquant.trajectory import derive_budget_steps, load_plan, plan_digest

MANIFEST_SCHEMA_VERSION = 1
DATA_FIELDS = ("bars_path", "labels_path", "instruments_path", "instrument_map_path")
CHECKPOINT_ARTIFACTS = ("trajectory/training_state.pt", "trajectory/training_state.json",
                        "trajectory.json", "exposure.json")
RUN_ARTIFACTS = ("manifest.json", "config.json", "config.sha256", "data.json", "events.jsonl")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_snapshot(repo: Path) -> dict:
    """The exact commit plus whether the working tree is dirty (an unpinned run is refused)."""
    def run(*arguments: str) -> str:
        return subprocess.run(["git", *arguments], cwd=repo, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True).stdout

    commit = run("rev-parse", "HEAD").strip()
    status = run("status", "--porcelain").rstrip()
    diff = run("diff", "HEAD").encode("utf-8")
    return {"commit": commit, "dirty": bool(status),
            "dirty_paths": [line[3:].strip() for line in status.splitlines() if line.strip()],
            "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "note": "run only a clean snapshot: an unpinned working tree cannot be reproduced"}


def data_inventory(data, *, digest: bool = True) -> list[dict]:
    """Every file the fixed recipe reads, one row per upload shard, with size and digest.

    A directory in the recipe is a shard set (the monthly bar/label files); a single file is
    one shard. ``digest=False`` records sizes only, which keeps the launch checklist cheap.
    """
    rows = []
    for field in DATA_FIELDS:
        value = getattr(data, field)
        if value is None:
            continue
        path = Path(value)
        files = sorted(shard for shard in path.iterdir() if shard.is_file()) if path.is_dir() else [path]
        for shard in files:
            present = shard.is_file()
            rows.append({"field": field, "path": shard.as_posix(), "shard": shard.name,
                         "exists": present,
                         "bytes": shard.stat().st_size if present else None,
                         "sha256": sha256_file(shard) if present and digest else None})
    return rows


def build_manifest(plan_path: Path, repo: Path, *, digest: bool = True) -> dict:
    plan_path, repo = Path(plan_path).resolve(), Path(repo).resolve()
    plan = load_plan(plan_path)
    recipe_path = (repo / plan["recipe_config"]).resolve()
    config = load_config(recipe_path)
    reference = plan.get("derived_reference", {})
    samples, batch_size = reference.get("recipe_training_samples"), config.model.batch_size
    budget = (derive_budget_steps(plan["budget_points"], int(samples), int(batch_size))
              if samples else plan["budget_points"])
    run_root = Path(plan["cloud"].get("remote_runs_root", "/root/v5/repo/runs")) / plan["experiment_id"]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "cloud_trajectory_job",
        "experiment_id": plan["experiment_id"],
        "plan": {"path": plan_path.as_posix(), "sha256": plan_digest(plan),
                 "budget_points": plan["budget_points"], "lr_schedule": plan["lr_schedule"]},
        "recipe": {"path": recipe_path.relative_to(repo).as_posix(),
                   "sha256": sha256_file(recipe_path),
                   "budget_only_overrides": plan.get("budget_only_overrides", {})},
        "code": git_snapshot(repo),
        "data": {"files": data_inventory(config.data, digest=digest),
                 "bundle": "the fixed recipe's own data paths",
                 "note": "upload the listed files only; the shard reader interface is pinned by t2"},
        "budget": {"derived_from": "plan effective epochs x recipe training samples / batch size"
                   if samples else "plan points (no reference pool size declared)",
                   "batch_size": batch_size,
                   "points": [{**point, "steps": point.get("steps")} for point in budget]},
        "commands": {
            "upload": "rsync -a <data files> <remote>:/root/v5/数据/",
            "run": f"cd <remote-repo> && python scripts/run_training_trajectory.py "
                   f"--plan {plan_path.as_posix()} --output-root {run_root.as_posix()}",
            "fetch": f"python scripts/cloud_runs.py fetch {run_root.name}/<run_id> --dest runs",
        },
        "retrieve": {"checkpoints": [f"{run_root.name}/<run_id>/{name}" for name in CHECKPOINT_ARTIFACTS],
                     "run_artifacts": [f"{run_root.name}/<run_id>/{name}" for name in RUN_ARTIFACTS],
                     "budget_points": [f"{run_root.name}/<run_id>/budget_points/*.json",
                                       f"{run_root.name}/<run_id>/budget_points/*.parquet"]},
        "resources": {"gpu_class": plan["cloud"].get("gpu_class", "24GB platform caliber"),
                      "memory_cap_applies_to": "inference/submission path only"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, default=Path("configs/training_trajectory_20261027_plan.json"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_manifest(args.plan, args.repo)
    output = args.output or (args.plan.parent / f"{args.plan.stem}_cloud_job.json")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"cloud job manifest -> {output}")
    print(f"code commit={manifest['code']['commit']} dirty={manifest['code']['dirty']}")
    files = manifest["data"]["files"]
    total = sum(row["bytes"] or 0 for row in files)
    print(f"budget points={len(manifest['budget']['points'])} data shards={len(files)} "
          f"({total / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
