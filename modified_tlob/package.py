"""Build an explicit offline upload payload from a successfully checked training run."""

import argparse
import shutil
from pathlib import Path

from jinyiquant.checks import verify_artifacts
from jinyiquant.io import file_sha256, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
TLOB_FILES = ("__init__.py", "model.py", "normalization.py", "schema.py", "device.py", "adapter.py",
              "submission.py", "train.py", "infer.py", "provenance.json", "LICENSE")
JINYI_FILES = ("__init__.py", "__main__.py", "checks.py", "cli.py", "comparison.py", "config.py", "data.py",
               "demo.py", "evaluation.py", "experiments.py", "io.py", "models.py", "preparation.py", "protocol.py",
               "runner.py", "tracking.py")


def build_package(run_dir, destination):
    run_dir, destination = Path(run_dir), Path(destination)
    cfg = read_json(run_dir / "config.json")
    if cfg["model"]["kind"] != "modified_tlob" or cfg["evaluation_split"] != "validation":
        raise ValueError("Package requires modified_tlob with development validation")
    if read_json(run_dir / "manifest.json")["status"] != "completed" or not verify_artifacts(run_dir)["passed"]:
        raise ValueError("Package requires a completed untampered run")
    if not read_json(run_dir / "checks.json")["passed"]:
        raise ValueError("Training run failed acceptance")
    # Ship the archived source that produced the weights, rejecting changed code.
    code = read_json(run_dir / "manifest.json")["code"]["files"]
    files = [(ROOT / "modified_tlob" / name, Path("modified_tlob") / name) for name in TLOB_FILES]
    files += [(ROOT / "src" / "jinyiquant" / name, Path("jinyiquant") / name) for name in JINYI_FILES]
    files += [(ROOT / "modified_tlob" / "inference.ipynb", Path("inference.ipynb")),
              (ROOT / "modified_tlob" / "requirements-cpu.lock.txt", Path("requirements-cpu.lock.txt")),
              (ROOT / "modified_tlob" / "requirements-cuda.lock.txt", Path("requirements-cuda.lock.txt"))]
    for source, _ in files:
        relative = source.relative_to(ROOT).as_posix()
        if relative not in code or file_sha256(source) != code[relative]:
            raise ValueError(f"Source changed since training: {relative}; rerun before packaging")
    destination.mkdir(parents=True, exist_ok=False)
    for source, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(run_dir / "modified_tlob_weights.json", destination / "modified_tlob_weights.json")
    # Freeze splits and settings, replacing local paths with portable generated-data names.
    cfg["output_root"] = "training_runs"
    cfg["data"].update(bars_path="bars.parquet", labels_path="labels.parquet", instruments_path="instruments.parquet",
                       instrument_map_path=None, cache_dir=None)
    write_json(destination / "training_config.json", cfg)
    write_json(destination / "submission_config.json", {
        "schema_version": 1, "device": "auto", "platform_price_scale": 1, "wall_seconds": 10800,
        "min_coverage": cfg["acceptance"]["min_coverage"],
        "weights_sha256": file_sha256(destination / "modified_tlob_weights.json")})
    inventory = {p.relative_to(destination).as_posix(): file_sha256(p) for p in destination.rglob("*") if p.is_file()}
    write_json(destination / "package_manifest.json", {"schema_version": 1, "files": inventory,
               "training_run_id": run_dir.name, "platform_acceptance": "not_verified",
               "scope": "synthetic engineering only" if cfg["data"]["source"].startswith("synthetic") else "development"})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(build_package(args.run_dir, args.destination))


if __name__ == "__main__":
    main()
