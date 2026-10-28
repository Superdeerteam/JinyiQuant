"""确认窗读数（第二次、也是最后一次）——只读四个数，单次运行，不再改代码。

输入白名单：只读注册文件（v2）与已封好的只读 bundle（`fusion_confirm_bundle.py` 产物）。
输出白名单：池化 Δrank_ic、>0 交易日数、池化 Δturnover、池化 Δnet5；
外加 read_id 与输入 sha。不打印逐窗、不打印逐时点、不打印其它指标——
这是提前约定的口径，避免"读完再解释"。

用法：
  python scripts/fusion_confirm_read.py --registration <v2.json> --bundle <dir> [--dry-run]
dry-run 使用注册文件里的 dry_run_ledger（或 --ledger 覆盖），绝不写正式账本。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["date", "instrument"]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_scores(bundle: Path, window: str, role: str, versions: list[str],
                  keys: pd.DataFrame, gate_v2) -> np.ndarray:
    parts = []
    for version in versions:
        path = bundle / "inputs" / window / role / version / "predictions.parquet"
        frame = pd.read_parquet(path)[KEYS + ["score"]]
        merged = (frame.merge(keys, on=KEYS, how="inner", validate="one_to_one")
                  .sort_values(KEYS).reset_index(drop=True))
        if len(merged) != len(keys) or not merged[KEYS].equals(keys):
            raise SystemExit(f"{path}: prediction coverage mismatch")
        parts.append(merged["score"].to_numpy(dtype=float))
    blended = parts[0] if len(parts) == 1 else np.mean(np.vstack(parts), axis=0)
    return gate_v2.causal_ewma(keys, blended, gate_v2.EWMA_ALPHA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    registration = json.loads((ROOT / args.registration).read_text(encoding="utf-8"))
    registration_sha = sha256_file(ROOT / args.registration)
    bundle = ROOT / args.bundle
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha = sha256_file(bundle / "manifest.json")

    gate_path = registration["gate_script"]["path"]
    gate_sha = sha256_file(ROOT / gate_path)
    if gate_sha != registration["gate_script_sha256"]:
        raise SystemExit("gate script does not match the registration")
    if gate_sha != manifest["gate_v2"]["sha256"]:
        raise SystemExit("bundle was assembled against a different gate script")
    if manifest_sha != registration["bundle"]["manifest_sha256"]:
        raise SystemExit("bundle manifest does not match the registration")
    if manifest["weights"] != registration["candidate"]["weights"]:
        raise SystemExit("bundle weights differ from the registered weights")
    if manifest["windows"] != [w["name"] for w in registration["confirmation_windows"]]:
        raise SystemExit("bundle windows differ from the registered windows")
    for rel, meta in manifest["files"].items():
        path = bundle / rel
        if sha256_file(path) != meta["sha256"]:
            raise SystemExit(f"bundle file changed since sealing: {rel}")
    if manifest["dry_run"] and not args.dry_run:
        raise SystemExit("this bundle is marked dry_run=true")

    gate_v2 = _load("fusion_gate_v2", gate_path)
    base_role = registration["candidate"].get("base", "trunk")
    weights = registration["candidate"]["weights"]
    a, b = float(weights["a"]), float(weights["b"])
    trunk_versions = manifest["versions"]["trunk"]

    deltas = {"rank_ic": [], "turnover": [], "net_return_5bp": []}
    deltas_sub = {m: [] for m in deltas}
    for window_spec in registration["confirmation_windows"]:
        window = window_spec["name"]
        keys = pd.read_parquet(bundle / "inputs" / window / "labels" / "keys.parquet")[KEYS]
        targets = pd.read_parquet(bundle / "inputs" / window / "labels" / "targets.parquet")
        base_z = bundle_scores(bundle, window, base_role,
                               trunk_versions if base_role == "trunk"
                               else manifest["versions"]["submission"], keys, gate_v2)
        trunk_z = bundle_scores(bundle, window, "trunk", trunk_versions, keys, gate_v2)
        tree_z = bundle_scores(bundle, window, "tree", manifest["versions"]["tree"],
                               keys, gate_v2)
        tilt_z = bundle_scores(bundle, window, "tilt", manifest["versions"]["tilt"],
                               keys, gate_v2)
        tilt_delta = tilt_z - base_z
        scores = base_z + a * tree_z + b * tilt_delta
        combo = gate_v2.daily_metrics(keys, targets, scores)
        trunk = gate_v2.daily_metrics(keys, targets, trunk_z)
        sub = gate_v2.daily_metrics(keys, targets, base_z)
        for metric in deltas:
            value = combo[metric].to_numpy() - trunk[metric].to_numpy()
            deltas[metric].extend(value[np.isfinite(value)])
            value_sub = combo[metric].to_numpy() - sub[metric].to_numpy()
            deltas_sub[metric].extend(value_sub[np.isfinite(value_sub)])

    rank_ic = np.asarray(deltas["rank_ic"], dtype=float)
    turnover = np.asarray(deltas["turnover"], dtype=float)
    net5 = np.asarray(deltas["net_return_5bp"], dtype=float)
    rule = registration["confirmation_rule"]
    pooled_mean = float(rank_ic.mean())
    positive_days = int((rank_ic > 0).sum())
    decision = ("passes" if pooled_mean > float(rule.get("min_paired_mean", 0.0))
                and positive_days >= int(rule["min_positive_days"]) else "fails")

    read_id = f"confirm-{registration['experiment_id']}-{manifest_sha[:10]}"
    ledger_path = ROOT / (args.ledger or registration["ledger"]["path"])
    if not args.dry_run:
        if ledger_path.is_file():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("mode") == "confirm":
                    if json.loads(line).get("route") == registration["candidate"]["label"]:
                        raise SystemExit("this direction already used its confirmation read")
    entry = {
        "schema": "fusion-confirm-read/1",
        "read_id": read_id,
        "utc": datetime.now(timezone.utc).isoformat(),
        "kind": "confirmation",
        "mode": "confirm",
        "route": registration["candidate"]["label"],
        "dry_run": bool(args.dry_run),
        "gate_sha256": gate_sha,
        "registration_sha256": registration_sha,
        "bundle_manifest_sha256": manifest_sha,
        "weights": {"a": a, "b": b},
        "base": base_role,
        "windows": manifest["windows"],
        "days": int(len(rank_ic)),
        "delta_rank_ic_pooled": pooled_mean,
        "positive_days": positive_days,
        "delta_turnover_pooled": float(turnover.mean()),
        "delta_net5_5bp_pooled": float(net5.mean()),
        "delta_rank_ic_vs_base_pooled": float(
            np.mean(deltas_sub["rank_ic"]) if deltas_sub["rank_ic"] else float("nan")),
        "decision": decision,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    out = ROOT / (args.json_out or f"runs/fusion-gate/{read_id}.json")
    out.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"read_id: {read_id}")
    print(f"windows: {','.join(manifest['windows'])} ({len(rank_ic)} trading days, "
          f"base={base_role}, a={a}, b={b})")
    print(f"delta_rank_ic_pooled: {pooled_mean:+.6f}")
    print(f"positive_days: {positive_days}/{len(rank_ic)}")
    print(f"delta_turnover_pooled: {float(turnover.mean()):+.6f}")
    print(f"delta_net5_5bp_pooled: {float(net5.mean()):+.3e}")
    print(f"decision: {decision}")
    print(f"inputs: gate_v2={gate_sha[:16]} bundle={manifest_sha[:16]} "
          f"registration={registration_sha[:16]}")
    print(f"ledger: {ledger_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
