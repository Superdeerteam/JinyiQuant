"""Build a checked training payload or refresh only an existing package's inference entry."""

import argparse
import ast
import shutil
import stat
from pathlib import Path, PurePosixPath

from jinyiquant.checks import verify_artifacts
from jinyiquant.io import file_sha256, read_json, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
TLOB_FILES = ("__init__.py", "model.py", "normalization.py", "schema.py", "device.py", "adapter.py",
              "submission.py", "train.py", "infer.py", "provenance.json", "LICENSE")
JINYI_FILES = ("__init__.py", "__main__.py", "checks.py", "cli.py", "comparison.py", "config.py", "data.py",
               "demo.py", "evaluation.py", "experiments.py", "io.py", "models.py", "preparation.py", "protocol.py",
               "runner.py", "tracking.py")
INFERENCE_ENTRY = "modified_tlob/submission.py"
RUNTIME_DIRECTORIES = {"inference_runs", "training_runs"}


def _is_link(path):
    # Windows junctions/reparse points need the same protection as POSIX symlinks.
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _runtime_path(relative):
    return relative.parts[0] in RUNTIME_DIRECTORIES or "__pycache__" in relative.parts[:-1]


def _check_inference_dependencies(source, inventory, package_dir):
    """Check direct local imports without executing the old package."""
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.level == 1:
            module = "modified_tlob/" + node.module.replace(".", "/") + ".py"
        elif node.level == 0 and node.module.startswith("jinyiquant."):
            module = node.module.replace(".", "/") + ".py"
        else:
            continue
        if module not in inventory:
            raise ValueError(f"Refreshed inference requires a missing package dependency: {module}")
        if module == "modified_tlob/device.py" and any(n.name == "device_resources" for n in node.names):
            tree = ast.parse((package_dir / module).read_text(encoding="utf-8"))
            if not any(isinstance(n, ast.FunctionDef) and n.name == "device_resources" for n in tree.body):
                raise ValueError("Existing device.py does not provide device_resources")


def refresh_inference_package(package_dir, destination, *, recover_missing_history=False):
    """Refresh inference and optionally its explicit history policy, without retraining.

    Unlisted inference_runs/, training_runs/ and __pycache__/ outputs are ignored
    and never copied. All upload files must be listed and match their old hashes.
    """
    package_dir, destination = Path(package_dir), Path(destination)
    if not package_dir.is_dir() or _is_link(package_dir):
        raise ValueError("Existing package must be a real directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination.resolve().is_relative_to(package_dir.resolve()):
        raise ValueError("Refreshed package must be outside the existing package")
    manifest_path = package_dir / "package_manifest.json"
    # Inspect links before reading any manifest or traversing listed source paths.
    observed = set()
    for path in package_dir.rglob("*"):
        if _is_link(path):
            raise ValueError(f"Package links are not allowed: {path.relative_to(package_dir)}")
        relative = path.relative_to(package_dir)
        if path.is_file() and not _runtime_path(relative):
            observed.add(relative.as_posix())
    manifest = read_json(manifest_path)
    inventory = manifest.get("files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Package manifest requires a nonempty file inventory")
    for name, expected in inventory.items():
        relative = PurePosixPath(name)
        if (not name or "\\" in name or ":" in name or relative.is_absolute()
                or any(part in {".", ".."} for part in relative.parts)
                or relative.as_posix() != name or name == "package_manifest.json"
                or _runtime_path(relative)):
            raise ValueError(f"Unsafe package inventory path: {name}")
        if name not in observed or not isinstance(expected, str) or file_sha256(package_dir / name) != expected:
            raise ValueError(f"Package file missing or checksum mismatch: {name}")
    unexpected = observed - set(inventory) - {"package_manifest.json"}
    if unexpected:
        raise ValueError(f"Unlisted package files: {sorted(unexpected)}")
    required = {INFERENCE_ENTRY, "submission_config.json", "modified_tlob_weights.json"}
    if not required.issubset(inventory):
        raise ValueError(f"Package is missing required files: {sorted(required - set(inventory))}")
    contract = read_json(package_dir / "submission_config.json")
    if contract.get("weights_sha256") != inventory["modified_tlob_weights.json"]:
        raise ValueError("Submission weight checksum mismatch")
    source = ROOT / INFERENCE_ENTRY
    if _is_link(source):
        raise ValueError("Refreshed inference source cannot be a link")
    _check_inference_dependencies(source, inventory, package_dir)
    old_manifest_hash = file_sha256(manifest_path)
    new_entry_hash = file_sha256(source)
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in inventory.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source if name == INFERENCE_ENTRY else package_dir / name, target)
        if file_sha256(target) != (new_entry_hash if name == INFERENCE_ENTRY else expected):
            raise ValueError(f"File changed while refreshing package: {name}")
    manifest["files"] = {**inventory, INFERENCE_ENTRY: new_entry_hash}
    if recover_missing_history:
        contract["history_policy"] = {"mode": "recover_then_mean", "max_backfill_days": 366,
                                      "instrument_batch_size": 50}
        write_json(destination / "submission_config.json", contract)
        manifest["files"]["submission_config.json"] = file_sha256(destination / "submission_config.json")
    manifest["platform_acceptance"] = "not_verified"
    manifest["inference_update"] = {
        "schema_version": 1, "updated_at": utc_now(),
        "source": "current_repository_working_tree", "path": INFERENCE_ENTRY,
        "previous_package_manifest_sha256": old_manifest_hash,
        "previous_submission_sha256": inventory[INFERENCE_ENTRY],
        "submission_sha256": new_entry_hash,
        "scope": ("Inference entry and explicit history policy updated; weights, model and preprocessing "
                  "are byte-identical." if recover_missing_history else
                  "Only inference entry replaced; all other upload files are byte-identical.") +
                 " Training provenance remains unchanged; the new inference entry is not the training snapshot.",
    }
    if recover_missing_history:
        manifest["inference_update"].update(history_policy=contract["history_policy"],
            previous_submission_config_sha256=inventory["submission_config.json"],
            submission_config_sha256=manifest["files"]["submission_config.json"])
    write_json(destination / "package_manifest.json", manifest)
    return destination


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
    parser.add_argument("--refresh-inference", action="store_true",
                        help="Treat run_dir as an existing package; replace only submission.py without retraining")
    parser.add_argument("--recover-missing-history", action="store_true",
                        help="Explicitly enable bounded historical backfill and saved-mean fallback")
    args = parser.parse_args()
    if args.recover_missing_history and not args.refresh_inference:
        parser.error("--recover-missing-history requires --refresh-inference")
    if args.refresh_inference:
        print(refresh_inference_package(args.run_dir, args.destination,
                                        recover_missing_history=args.recover_missing_history))
    else:
        print(build_package(args.run_dir, args.destination))


if __name__ == "__main__":
    main()
