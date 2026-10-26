"""Flatten checked inference code and shard large text weights below the upload limit."""

import argparse
import ast
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from jinyiquant.io import file_sha256, utc_now, write_json

UPLOAD_FILES = {
    "modified_tlob_weights.json", "provenance.json", "requirements.txt", "submission_config.json",
    "modified_tlob_flat.py", "jinyiquant_flat.py", "inference.ipynb",
}
UPLOAD_FILE_LIMIT = 50_000_000
DEFAULT_WEIGHT_FILE_BYTES = 49_000_000
SHARDED_BLEND_KIND = "modified_tlob_sharded_blend"
JINYI_MODULES = ("io", "data", "tracking")
TLOB_MODULES = ("schema", "device", "normalization", "model", "adapter", "submission")

FLAT_CAPTURE = '''
def capture_source(run_dir):
    names = ("jinyiquant_flat.py", "modified_tlob_flat.py", "inference.ipynb",
             "requirements.txt", "provenance.json", "submission_config.json")
    paths = [PROJECT_ROOT / name for name in names]
    inventory = {p.name: file_sha256(p) for p in paths}
    with zipfile.ZipFile(run_dir / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)
    provenance = read_json(PROJECT_ROOT / "provenance.json")
    return {"commit": provenance["flat_package"]["source_commit"],
            "working_tree_status": None, "snapshot": "source.zip",
            "snapshot_sha256": file_sha256(run_dir / "source.zip"),
            "content_sha256": digest(inventory), "files": inventory,
            "layout": "flat_files_with_optional_weight_shards"}
'''

SCHEMA_FORWARD = '''
def lob_trade_columns():
    # Resolve only when called, after both flat modules have finished importing.
    from modified_tlob_flat import lob_trade_columns as columns
    return columns()
'''


class FlatImports(ast.NodeTransformer):
    def __init__(self, package, module):
        self.package, self.module = package, module

    def visit_ImportFrom(self, node):
        if node.module == "__future__":
            return None
        if self.package == "jinyiquant":
            if node.level and node.module == "io":
                return None
            if node.module == "modified_tlob.schema":
                if [alias.name for alias in node.names] != ["lob_trade_columns"]:
                    raise ValueError("Unexpected cross-package schema dependency")
                return ast.parse(SCHEMA_FORWARD).body[0]
            if node.module == "modified_tlob.device":
                node.module = "modified_tlob_flat"
        elif self.package == "modified_tlob":
            if node.level:
                if node.module not in TLOB_MODULES:
                    raise ValueError(f"Unexpected internal model dependency: {node.module}")
                return None
            if node.module and node.module.startswith("jinyiquant."):
                if node.module.split(".")[1] not in JINYI_MODULES:
                    raise ValueError(f"Unexpected inference dependency: {node.module}")
                node.module = "jinyiquant_flat"
        return node

    def visit_Assign(self, node):
        if self.module == "tracking" and any(
                isinstance(target, ast.Name) and target.id == "PROJECT_ROOT" for target in node.targets):
            return ast.parse("PROJECT_ROOT = Path(__file__).resolve().parent").body[0]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if self.module == "tracking" and node.name == "capture_source":
            return ast.parse(FLAT_CAPTURE).body[0]
        return self.generic_visit(node)


def merged_source(package_dir, package, modules, license_text=""):
    sections, definitions = [], set()
    for module in modules:
        path = package_dir / package / f"{module}.py"
        source = path.read_text(encoding="utf-8")
        if module == "submission":
            source = source.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parent")
        tree = FlatImports(package, module).visit(ast.parse(source))
        ast.fix_missing_locations(tree)
        for node in tree.body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)]
            for name in names:
                if name in definitions:
                    raise ValueError(f"Flat module name collision: {package}.{name}")
                definitions.add(name)
        sections.append(f"# Source: {package}/{module}.py\n" + ast.unparse(tree))
    header = f'"""Generated flat platform inference module: {package}.\n{license_text}"""\n'
    result = header + "from __future__ import annotations\n\n" + "\n\n".join(sections) + "\n"
    compile(result, package + "_flat.py", "exec")
    return result


def write_upload_weights(source, destination, *, max_file_bytes=DEFAULT_WEIGHT_FILE_BYTES):
    """Preserve small checkpoints; split a large blend into lossless member JSON files."""
    if type(max_file_bytes) is not int or not 0 < max_file_bytes < UPLOAD_FILE_LIMIT:
        raise ValueError("Weight file target must be positive and below 50,000,000 bytes")
    source, destination = Path(source), Path(destination)
    target = destination / "modified_tlob_weights.json"
    source_sha = file_sha256(source)
    if source.stat().st_size < max_file_bytes:
        shutil.copyfile(source, target)
        return {"modified_tlob_weights.json"}, {"encoding": "original_json", "source_sha256": source_sha,
                                                "weights_sha256": source_sha, "values_unchanged": True}
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    entries = payload.get("members")
    if (payload.get("kind") != "modified_tlob_blend" or payload.get("schema_version") != 4
            or not isinstance(entries, list) or len(entries) < 2):
        raise ValueError("Oversized text checkpoint must be a schema-4 blend for member sharding")
    manifest = {"schema_version": 1, "kind": SHARDED_BLEND_KIND, "source_weights_sha256": source_sha,
                "blend": payload.get("blend", {}), "members": []}
    files = {target.name}
    for index, entry in enumerate(entries):
        filename = f"modified_tlob_weights_{index:03d}.json"
        shard = destination / filename
        with shard.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(entry["payload"], stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        size = shard.stat().st_size
        if size >= max_file_bytes:
            raise ValueError(f"Member {entry['name']} requires {size} bytes; exceeds shard target {max_file_bytes}")
        manifest["members"].append({"name": entry["name"], "weight": entry["weight"], "file": filename,
                                    "bytes": size, "sha256": file_sha256(shard),
                                    "source_member_sha256": entry.get("sha256")})
        files.add(filename)
    write_json(target, manifest)
    if target.stat().st_size >= max_file_bytes:
        raise ValueError("Sharded checkpoint manifest exceeds the weight file target")
    return files, {"encoding": "member_json_shards", "source_sha256": source_sha,
                   "weights_sha256": file_sha256(target), "values_unchanged": True,
                   "member_count": len(entries), "file_target_bytes": max_file_bytes}


def build_flat_package(package_dir, destination, *, max_weight_file_bytes=DEFAULT_WEIGHT_FILE_BYTES):
    package_dir, destination = Path(package_dir).resolve(), Path(destination).resolve()
    archive_path = destination.with_suffix(".zip")
    if destination.exists() or archive_path.exists():
        raise FileExistsError("Use a new output directory and archive")
    manifest_path = package_dir / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = package_dir / name
        if not path.resolve().is_relative_to(package_dir) or file_sha256(path) != expected:
            raise ValueError(f"Package checksum mismatch: {name}")
    contract = json.loads((package_dir / "submission_config.json").read_text(encoding="utf-8"))
    if contract["weights_sha256"] != manifest["files"]["modified_tlob_weights.json"]:
        raise ValueError("Submission weight checksum mismatch")
    license_text = (package_dir / "modified_tlob/LICENSE").read_text(encoding="utf-8")
    sources = {
        "jinyiquant_flat.py": merged_source(package_dir, "jinyiquant", JINYI_MODULES),
        "modified_tlob_flat.py": merged_source(package_dir, "modified_tlob", TLOB_MODULES, license_text),
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, text in sources.items():
        (destination / name).write_text(text, encoding="utf-8", newline="\n")
    weight_files, weight_storage = write_upload_weights(
        package_dir / "modified_tlob_weights.json", destination, max_file_bytes=max_weight_file_bytes)
    if weight_storage["encoding"] == "member_json_shards":
        if SHARDED_BLEND_KIND not in sources["modified_tlob_flat.py"]:
            raise ValueError("Source package adapter does not support sharded weights; refresh its inference source")
        contract["weights_sha256"] = weight_storage["weights_sha256"]
        write_json(destination / "submission_config.json", contract)
    else:
        shutil.copyfile(package_dir / "submission_config.json", destination / "submission_config.json")
    notebook = json.loads((package_dir / "inference.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = [line.replace("from modified_tlob.submission import", "from modified_tlob_flat import")
                              for line in cell["source"]]
            cell["outputs"], cell["execution_count"] = [], None
    write_json(destination / "inference.ipynb", notebook)
    # Use one public Torch version so an installed CPU or CUDA local build can
    # satisfy the same requirement. Torch resolves its own OS/GPU dependencies.
    requirements = ["# Python 3.11. BigQuant provides proprietary dai; do not install an unrelated PyPI package.",
                    "# Existing torch 2.5.0 CPU/CUDA local builds both satisfy torch==2.5.0."]
    for line in (package_dir / "requirements-cpu.lock.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "--")):
            requirements.append("torch==2.5.0" if line.startswith("torch==") else line)
    (destination / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    provenance = json.loads((package_dir / "modified_tlob/provenance.json").read_text(encoding="utf-8"))
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance["flat_package"] = {
        "schema_version": 1, "created_at": utc_now(), "source_commit": code_commit,
        "source_package_manifest_sha256": file_sha256(manifest_path),
        "training_run_id": manifest["training_run_id"],
        "weights_sha256": contract["weights_sha256"],
        "weights_changed": weight_storage["encoding"] != "original_json", "weight_storage": weight_storage,
        "history_policy": contract.get("history_policy", {"mode": "strict"}),
        "scope": "Frozen-weight platform inference; original training project remains separate.",
        "source_modules": {f"{package}/{module}.py": manifest["files"][f"{package}/{module}.py"]
                           for package, modules in (("jinyiquant", JINYI_MODULES), ("modified_tlob", TLOB_MODULES))
                           for module in modules},
        "layout_changes": ["Merge inference dependencies with imports adjusted to the two flat files.",
                           "Resolve artifact paths beside flat files; capture their source in runtime archives.",
                           "Preserve daily streaming, model arithmetic, preprocessing and inference ordering."],
        "license_included_in": "modified_tlob_flat.py module docstring",
        "platform_acceptance": "not_verified",
        "payload_sha256": {path.name: file_sha256(path) for path in sorted(destination.iterdir())},
    }
    write_json(destination / "provenance.json", provenance)
    upload_files = UPLOAD_FILES | weight_files
    assert {p.name for p in destination.iterdir()} == upload_files
    assert file_sha256(destination / "modified_tlob_weights.json") == contract["weights_sha256"]
    oversized = {path.name: path.stat().st_size for path in destination.iterdir()
                 if path.stat().st_size >= UPLOAD_FILE_LIMIT}
    if oversized:
        raise ValueError(f"Upload files must be below 50,000,000 bytes: {oversized}")
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(upload_files):
            archive.write(destination / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and set(archive.namelist()) == upload_files
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-weight-file-bytes", type=int, default=DEFAULT_WEIGHT_FILE_BYTES)
    args = parser.parse_args()
    print(build_flat_package(args.package_dir, args.destination,
                             max_weight_file_bytes=args.max_weight_file_bytes))


if __name__ == "__main__":
    main()
