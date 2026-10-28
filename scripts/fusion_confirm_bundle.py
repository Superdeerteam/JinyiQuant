"""装配"确认读数包"（confirmation bundle）：把读数所需输入复制进一个只读目录并做哈希清单。

为什么要这么干：jun2024/sep2024 每个方向只读一次，输入不能在读数期间变动；
包一旦封好，读数只用包里的字节，manifest 把每个文件的 sha256 钉住。

包结构：
  <bundle>/manifest.json                         清单（见下）
  <bundle>/inputs/<window>/trunk/<version>/predictions.parquet
  <bundle>/inputs/<window>/tree/<version>/predictions.parquet
  <bundle>/inputs/<window>/tilt/<version>/predictions.parquet
  <bundle>/inputs/<window>/labels/{keys,targets}.parquet

manifest 不含 registration 的 sha（registration 反过来钉 manifest sha，避免循环依赖）；
manifest 记录 windows / versions / weights / gate_v2 与每个文件的 sha256 + 字节数。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["date", "instrument"]
ROLES = ("trunk", "tree", "tilt", "labels")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--windows", required=True, help="逗号分隔，例如 may2024")
    parser.add_argument("--trunk-run", required=True)
    parser.add_argument("--trunk-versions", required=True, help="逗号分隔")
    parser.add_argument("--tree-run", required=True)
    parser.add_argument("--tree-version", required=True)
    parser.add_argument("--tilt-run", required=True)
    parser.add_argument("--tilt-version", required=True)
    parser.add_argument("--weight-a", type=float, required=True)
    parser.add_argument("--weight-b", type=float, required=True)
    parser.add_argument("--gate-v2", default="scripts/fusion_increment_gate_v2.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = ROOT / args.out_dir
    if out.exists():
        raise SystemExit(f"bundle already exists, refusing to overwrite: {out}")
    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    trunk_versions = [v.strip() for v in args.trunk_versions.split(",") if v.strip()]
    sources = {
        "trunk": (ROOT / args.trunk_run, trunk_versions),
        "tree": (ROOT / args.tree_run, [args.tree_version]),
        "tilt": (ROOT / args.tilt_run, [args.tilt_version]),
    }
    files: dict[str, dict] = {}
    (out / "inputs").mkdir(parents=True)

    def copy_one(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        rel = dest.relative_to(out).as_posix()
        files[rel] = {"sha256": sha256_file(dest), "bytes": dest.stat().st_size}

    for window in windows:
        for role, (run_dir, versions) in sources.items():
            for version in versions:
                src = run_dir / "windows" / window / version / "predictions.parquet"
                if not src.is_file():
                    raise SystemExit(f"missing input: {src}")
                copy_one(src, out / "inputs" / window / role / version / "predictions.parquet")
        for label in ("keys", "targets"):
            src = run_dir / "windows" / window / f"{label}.parquet"
            if not src.is_file() and label == "keys":
                # 对齐视图故意不写 keys.parquet（与闸门一致：用 targets 键集回退）
                target_path = run_dir / "windows" / window / "targets.parquet"
                if not target_path.is_file():
                    raise SystemExit(f"missing labels: {src}")
                dest = out / "inputs" / window / "labels" / "keys.parquet"
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.read_parquet(target_path)[KEYS].to_parquet(dest, index=False)
                rel = dest.relative_to(out).as_posix()
                files[rel] = {"sha256": sha256_file(dest), "bytes": dest.stat().st_size}
                continue
            if not src.is_file():
                raise SystemExit(f"missing labels: {src}")
            copy_one(src, out / "inputs" / window / "labels" / f"{label}.parquet")

    manifest = {
        "schema": "fusion-confirm-bundle/1",
        "bundle_id": out.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose_zh": "确认窗读数（每方向一次）的只读输入包；读数只使用包内字节。",
        "dry_run": bool(args.dry_run),
        "windows": windows,
        "versions": {"trunk": trunk_versions, "tree": [args.tree_version],
                     "tilt": [args.tilt_version]},
        "weights": {"a": args.weight_a, "b": args.weight_b},
        "source_run_dirs": {role: run_dir.relative_to(ROOT).as_posix()
                            for role, (run_dir, _) in sources.items()},
        "gate_v2": {"path": args.gate_v2, "sha256": sha256_file(ROOT / args.gate_v2)},
        "labels_origin": run_dir.relative_to(ROOT).as_posix(),
        "files": files,
        "files_count": len(files),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    os.chmod(manifest_path, stat.S_IREAD)
    for rel in files:
        os.chmod(out / rel, stat.S_IREAD)
    print(json.dumps({"bundle": out.relative_to(ROOT).as_posix(),
                      "manifest_sha256": sha256_file(manifest_path),
                      "files_count": len(files), "windows": windows,
                      "dry_run": bool(args.dry_run), "read_only": True},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
