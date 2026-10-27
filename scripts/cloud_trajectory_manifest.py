"""Cloud job manifest for one continuous training trajectory.

The manifest answers, before anything is rented or uploaded: which code snapshot runs, which
data files must be on the remote host, which commands produce the checkpointed trajectory and
which artifacts have to come back. It is derived from a pre-registered plan and the fixed
recipe configuration; nothing here trains, uploads or evaluates.

The recipe names the cloud data root (``/root/v5/数据``). ``--data-root`` points the same
shard inventory at a local copy, so the upload list is real sizes and checksums instead of four
absent paths. Only shards that overlap the run's own read window are listed: the training pool
reads bars from ``train.start - history_days`` and labels from ``train.start``, both up to the
evaluation split end. ``--archive`` writes ``git archive HEAD`` (the exact clean snapshot that
is uploaded) and records its checksum next to the commit and the local dirty paths, so an
unpinned working tree cannot silently enter the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from jinyiquant.config import load_config
from jinyiquant.data import _month_overlaps
from jinyiquant.trajectory import derive_budget_steps, load_plan, plan_digest

MANIFEST_SCHEMA_VERSION = 1
DATA_FIELDS = ("bars_path", "labels_path", "instruments_path", "instrument_map_path")
CHECKPOINT_ARTIFACTS = ("trajectory/training_state.pt", "trajectory/training_state.json",
                        "trajectory.json", "exposure.json")
RUN_ARTIFACTS = ("manifest.json", "config.json", "config.sha256", "data.json", "events.jsonl")
READABLE_SUFFIXES = {".parquet", ".feather", ".arrow", ".csv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repo, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True).stdout


def git_snapshot(repo: Path) -> dict:
    """The exact commit plus whether the working tree is dirty (an unpinned run is refused)."""
    commit = _git(repo, "rev-parse", "HEAD").strip()
    status = _git(repo, "status", "--porcelain").rstrip()
    diff = _git(repo, "diff", "HEAD").encode("utf-8")
    return {"commit": commit, "dirty": bool(status),
            "dirty_paths": [line[3:].strip() for line in status.splitlines() if line.strip()],
            "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "note": "run only a clean snapshot: an unpinned working tree cannot be reproduced"}


def git_archive(repo: Path, output: Path) -> dict:
    """Write the deployable clean snapshot and fingerprint exactly what is uploaded."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=tar", "-o", output.as_posix(), "HEAD"],
                   cwd=repo, check=True)
    return {"path": output.as_posix(), "sha256": sha256_file(output), "bytes": output.stat().st_size,
            "ref": "HEAD", "commit": _git(repo, "rev-parse", "HEAD").strip(),
            "note": "clean snapshot of HEAD; working-tree changes are excluded from the run"}


def read_windows(config) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """The data read window the driver actually opens, per source role."""
    split = getattr(config.splits, config.evaluation_split)
    history = int(getattr(config.data, "history_days", 0) or 0)
    start, end = pd.Timestamp(config.splits.train.start), pd.Timestamp(split.end)
    return {"bars": (start - pd.Timedelta(days=history), end), "labels": (start, end)}


def _selected(shard: Path, window: tuple[pd.Timestamp, pd.Timestamp] | None) -> bool:
    """Mirror jinyiquant.data._read: supported suffix plus the read window's month overlap."""
    if shard.suffix.lower() not in READABLE_SUFFIXES:
        return False
    return _month_overlaps(shard.name, *(window if window is not None else (None, None)))


def local_copy(declared: str, local_root: Path) -> Path:
    """Map a recipe path onto the local copy without changing the canonical remote path."""
    path = Path(declared)
    if path.exists():
        return path
    parts = path.parts
    if "数据" in parts:
        return Path(local_root) / Path(*parts[parts.index("数据") + 1:])
    return Path(local_root) / path.name


def data_inventory(data, *, digest: bool = True, local_root: Path | None = None,
                   windows: dict | None = None) -> list[dict]:
    """Every shard the fixed recipe reads inside the run window, with size and digest."""
    rows = []
    windows = windows or {}
    for field in DATA_FIELDS:
        declared = getattr(data, field)
        if declared is None:
            continue
        window = windows.get(field.removesuffix("_path"))
        source = Path(declared) if local_root is None else local_copy(declared, Path(local_root))
        shards = sorted(item for item in source.iterdir() if item.is_file()) if source.is_dir() else [source]
        for shard in shards:
            present = shard.is_file()
            if present and window is not None and not _selected(shard, window):
                continue
            remote = (Path(declared) if not source.is_dir() else Path(declared) / shard.name).as_posix()
            row = {"field": field, "shard": shard.name, "path": remote,
                   "exists": present,
                   "bytes": shard.stat().st_size if present else None,
                   "sha256": sha256_file(shard) if present and digest else None}
            if local_root is not None:
                row.update(remote_path=remote, local_path=shard.as_posix())
            rows.append(row)
    return rows


def build_manifest(plan_path: Path, repo: Path, *, digest: bool = True,
                   data_root: Path | None = None, archive: Path | None = None) -> dict:
    plan_path, repo = Path(plan_path).resolve(), Path(repo).resolve()
    plan = load_plan(plan_path)
    recipe_path = (repo / plan["recipe_config"]).resolve()
    config = load_config(recipe_path)
    reference = plan.get("derived_reference", {})
    samples, batch_size = reference.get("recipe_training_samples"), config.model.batch_size
    budget = (derive_budget_steps(plan["budget_points"], int(samples), int(batch_size))
              if samples else plan["budget_points"])
    run_root = Path(plan["cloud"].get("remote_runs_root", "/root/v5/repo/runs")) / plan["experiment_id"]
    windows = read_windows(config)
    files = data_inventory(config.data, digest=digest, local_root=data_root, windows=windows)
    code = git_snapshot(repo)
    if archive is not None:
        code["deployment"] = git_archive(repo, Path(archive))
    total = sum(row["bytes"] or 0 for row in files)
    data = {"files": files,
            "selection": {role: [str(start.date()), str(end.date())] for role, (start, end) in windows.items()},
            "totals": {"shards": len(files), "bytes": total, "gib": round(total / 2**30, 3)},
            "bundle": "the fixed recipe's own data paths",
            "note": "upload exactly the listed shards; bars include the history warm-up, labels start at train.start"}
    if data_root is not None:
        data["local_root"] = Path(data_root).resolve().as_posix()
        data["remote_root"] = "/root/v5/数据"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "cloud_trajectory_job",
        "experiment_id": plan["experiment_id"],
        "plan": {"path": plan_path.as_posix(), "sha256": plan_digest(plan),
                 "budget_points": plan["budget_points"], "lr_schedule": plan["lr_schedule"]},
        "recipe": {"path": recipe_path.relative_to(repo).as_posix(),
                   "sha256": sha256_file(recipe_path),
                   "budget_only_overrides": plan.get("budget_only_overrides", {})},
        "code": code,
        "data": data,
        "budget": {"derived_from": "plan effective epochs x recipe training samples / batch size"
                   if samples else "plan points (no reference pool size declared)",
                   "batch_size": batch_size,
                   "points": [{**point, "steps": point.get("steps")} for point in budget]},
        "commands": {
            "upload": "tar -C <local-root> -cf - <listed shards> | "
                      "ssh <host> 'mkdir -p /root/v5/数据 && tar -xf - -C /root/v5/数据'",
            "run": f"cd <remote-repo> && PYTHONPATH=src:. <python> scripts/run_training_trajectory.py "
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
    parser.add_argument("--data-root", type=Path, default=None,
                        help="local copy of the recipe's data root, for real sizes and digests")
    parser.add_argument("--archive", type=Path, default=None,
                        help="write git archive HEAD here and record its checksum")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_manifest(args.plan, args.repo, data_root=args.data_root, archive=args.archive)
    output = args.output or (args.plan.parent / f"{args.plan.stem}_cloud_job.json")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"cloud job manifest -> {output}")
    print(f"code commit={manifest['code']['commit']} dirty={manifest['code']['dirty']}")
    if "deployment" in manifest["code"]:
        print(f"deploy archive={manifest['code']['deployment']['sha256'][:16]} "
              f"bytes={manifest['code']['deployment']['bytes']}")
    print(f"budget points={len(manifest['budget']['points'])} "
          f"data shards={manifest['data']['totals']['shards']} "
          f"({manifest['data']['totals']['gib']:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
