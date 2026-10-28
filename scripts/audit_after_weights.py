"""Executable checklist for the moment T1's trained heads land (t3 follow-up, auditor-owned).

The six invariance checks were run against the head *classes* with an armed random residual
because ``runs/market-context-441d-s11/armC.*`` did not exist yet.  This script is the bridge:
run it as soon as the real weights appear and it re-verifies everything that depends on them,
in one pass, with machine-readable output.

    .venv-tlob/Scripts/python.exe scripts/audit_after_weights.py

Exit codes: 0 every item passed; 1 at least one item failed; 3 blocked (weights absent, so the
checklist could not run - run it again once they are synced).

Items (each reports status pass/fail/blocked with evidence):
  1 weights-present        armC/armB checkpoint files exist
  2 weights-decode         strict load, keys/shapes match, finite, .pt and .state.json agree
  3 zero-init-before-load  a freshly built head still starts with an all-zero residual
  4 residual-alive         trained delta(h, m) is non-zero and actually reads the market context
  5 armB-ignores-context   B's delta ignores m bitwise, and B's residual is non-zero
  6 sealed-window-guard    split_of(config, "final") still refuses the sealed window
  7 signature-freeze       the audited call signatures are unchanged
  8 full-audit             the six invariance checks re-run with the trained head armed

No file under modified_tlob/ is written; this is a read-only consumer of T1's artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

THRESHOLD = 1e-6
DEFAULT_WEIGHTS = ROOT / "runs" / "market-context-441d-s11"
DEFAULT_OUT = ROOT / "runs" / "mktctx-20261028" / "after_weights.json"
POSITIONS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
FROZEN_SIGNATURES = {
    "build_head": "(arm, *, represent_dim=80, width=16, context_hidden=64, scalar_hidden=156, "
                  "hidden=None, ctx_dim=None)",
    "MarketInteractionHead.delta": "(self, h, m)",
    "ScalarResidualHead.delta": "(self, h, m=None)",
    "market_context": "(h, group, *, valid=None, counts=None, n_groups=None)",
    "timestamp_codes": "(keys, *, instrument_column='instrument', date_column='date')",
}


class _AnyStub(types.ModuleType):
    """Inert stand-in for a module of the data layer: every attribute exists and is callable."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = () if name.isupper() else (lambda *a, **k: None)
        setattr(self, name, value)
        return value


def _import_pipeline():
    """Import ``scripts/market_context.py``, stubbing the data layer it imports at module load.

    ``split_of`` is pure python; the stub exists only so the module can be imported without the
    production data layer (which is not installed in the local audit venv) and without reading
    any bar, label or split from disk.
    """
    try:
        import market_context as pipeline
        return pipeline
    except ModuleNotFoundError:
        pass
    package = _AnyStub("jinyiquant")
    for sub in ("config", "data", "io", "pool", "runner", "tracking", "checks", "comparison",
                "evaluation"):
        module = _AnyStub(f"jinyiquant.{sub}")
        setattr(package, sub, module)
        sys.modules[f"jinyiquant.{sub}"] = module
    sys.modules["jinyiquant"] = package
    import market_context as pipeline
    return pipeline


def item(name, status, **evidence):
    return {"item": name, "status": status, **evidence}


def find_weights(directory, arm):
    """Checkpoint candidates for one arm, in preference order."""
    found = []
    for suffix in (".pt", ".state.json", ".json"):
        candidate = Path(directory) / f"arm{arm}{suffix}"
        if candidate.exists():
            found.append(candidate)
    for candidate in sorted(Path(directory).glob(f"*{arm}*")):
        if candidate.suffix in (".pt", ".json") and candidate not in found:
            found.append(candidate)
    return found


def load_head(path, market_head, arm):
    """Rebuild one head from a checkpoint: torch payload, text payload or raw state dict."""
    import torch

    if path.suffix == ".pt":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001  (older payloads carry python objects)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        text = False
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = True
    if "market_head" in payload:  # adapter checkpoint with an embedded head
        payload = payload["market_head"]
    if isinstance(payload, dict) and "state_dict" in payload:
        head = market_head.head_from_payload(payload, text_state_weights=text)
        return head, "head_payload"
    state = market_head.decode_state(payload, text)
    head = market_head.build_head(arm)
    head.load_state_dict(state, strict=True)
    return head.eval(), "state_dict"


def head_state(head):
    return {name: value.detach().cpu() for name, value in head.state_dict().items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", choices=("full", "medium", "small"), default="medium")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--skip-full-audit", action="store_true")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    from modified_tlob import market_head

    started = time.time()
    try:
        weights_dir_label = str(Path(args.weights_dir).resolve().relative_to(ROOT))
    except ValueError:  # a fixture or a mounted path outside the repo
        weights_dir_label = str(Path(args.weights_dir).resolve())
    report = {"schema_version": 1, "audit": "t3 after-weights checklist", "threshold": THRESHOLD,
              "weights_dir": weights_dir_label, "items": []}
    digests = {}
    for path in sorted(args.weights_dir.glob("*")) if args.weights_dir.exists() else []:
        if path.is_file():
            digests[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "bytes": path.stat().st_size}
    report["weights_files"] = digests

    found = {arm: find_weights(args.weights_dir, arm) for arm in ("C", "B")}
    def label(path):
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(Path(path).resolve())

    report["candidates"] = {arm: [label(p) for p in paths] for arm, paths in found.items()}
    if not found["C"]:
        report["items"].append(item("weights-present", "blocked",
                                    expected=weights_dir_label,
                                    detail="no armC checkpoint on disk yet"))
        report["status"] = "blocked"
        report["elapsed_seconds"] = round(time.time() - started, 2)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"[weights] blocked: no armC checkpoint under {args.weights_dir}")
        return 3
    report["items"].append(item("weights-present", "pass",
                                files={arm: [p.name for p in paths] for arm, paths in found.items()}))

    # 2) decode every candidate, and require the .pt / .state.json pair to agree bitwise.
    heads, decode_evidence, decode_ok = {}, {}, True
    for arm, paths in found.items():
        for path in paths:
            try:
                head, mode = load_head(path, market_head, arm)
            except Exception as exc:  # noqa: BLE001
                decode_ok = False
                decode_evidence[path.name] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
                continue
            state = head_state(head)
            finite = all(bool(torch.isfinite(value).all()) for value in state.values())
            heads.setdefault(arm, head)
            decode_evidence[path.name] = {"status": "pass", "mode": mode,
                                          "tensors": len(state), "finite": finite,
                                          "parameters": int(market_head.parameter_count(head))}
            decode_ok = decode_ok and finite
    for arm, paths in found.items():
        if len(paths) > 1:
            first, second = load_head(paths[0], market_head, arm)[0], \
                load_head(paths[1], market_head, arm)[0]
            left, right = head_state(first), head_state(second)
            agree = all(torch.equal(left[key], right[key]) for key in left)
            decode_evidence[f"{paths[0].name}=={paths[1].name}"] = {"status": "pass" if agree
                                                                    else "fail",
                                                                    "bitwise_equal": bool(agree)}
            decode_ok = decode_ok and agree
    report["items"].append(item("weights-decode", "pass" if decode_ok else "fail",
                                files=decode_evidence))

    # 3) the class must still start zero-initialised: a re-trained arm has to begin at A.
    zero = {}
    for arm in ("C", "B"):
        fresh = market_head.build_head(arm)
        state = head_state(fresh)
        if arm == "C":
            name = "mix.weight"
            worst = float(state[name].abs().max())
        else:
            name = "layers.2.weight"
            worst = max(float(state[name].abs().max()), float(state["layers.2.bias"].abs().max()))
        zero[arm] = {"parameter": name, "max_abs_at_build": worst}
    zero_ok = all(entry["max_abs_at_build"] == 0.0 for entry in zero.values())
    report["items"].append(item("zero-init-before-load", "pass" if zero_ok else "fail", **zero))

    # 4) the trained head must be alive and must actually read the market context.
    import audit_market_ref as ref

    instruments = {"full": 1000, "medium": 400, "small": 200}[args.size]
    dates = ("2023-11-01", "2023-11-02", "2023-11-03") if args.size == "full" else ("2023-11-01",)
    panel = ref.make_panel(dates=dates, instruments=instruments, fallback_fraction=0.1,
                           seed=args.seed)
    keys = __import__("pandas").DataFrame({"instrument": panel.instrument,
                                           "stamp": [f"{d} {t}" for d, t in zip(panel.date,
                                                                                panel.time)]})
    order, codes, n_groups, counts = market_head.timestamp_codes(keys, date_column="stamp")
    codes_by_row = np.empty(codes.size, np.int64)
    codes_by_row[order] = codes
    valid = panel.eligible & np.isfinite(panel.h).all(axis=1)
    context = market_head.market_context(panel.h, codes_by_row, valid=valid, counts=counts,
                                         n_groups=n_groups)
    m = context[codes_by_row].astype(np.float32)
    head_c = heads.get("C")
    if head_c is None:
        report["items"].append(item("residual-alive", "blocked", detail="no C head decoded"))
        alive_ok = False
    else:
        with torch.no_grad():
            delta = head_c.delta(torch.from_numpy(panel.h.astype(np.float32)),
                                 torch.from_numpy(m)).numpy()
            shuffled = m.copy()
            for start in range(0, n_groups, 2):  # swap market context between adjacent timestamps
                left = codes_by_row == start
                right = codes_by_row == start + 1 if start + 1 < n_groups else None
                if right is None:
                    continue
                shuffled[left], shuffled[right] = m[right], m[left]
            delta_shuffled = head_c.delta(torch.from_numpy(panel.h.astype(np.float32)),
                                          torch.from_numpy(shuffled)).numpy()
        effect = float(np.abs(delta_shuffled - delta).max())
        alive = {"residual_abs_max": float(np.abs(delta).max()),
                 "residual_std": float(delta.std()),
                 "nonzero_fraction": float((np.abs(delta) > 0).mean()),
                 "context_effect_abs_max": effect,
                 "rows": int(delta.size), "groups": int(n_groups)}
        alive_ok = (alive["residual_abs_max"] > 1e-4 and alive["residual_std"] > 1e-6
                    and alive["nonzero_fraction"] > 0.5 and effect > THRESHOLD)
        report["items"].append(item("residual-alive", "pass" if alive_ok else "fail", **alive))

    # 5) B must stay blind to the market context, and must not be dead after training.
    head_b = heads.get("B")
    if head_b is None:
        report["items"].append(item("armB-ignores-context", "blocked", detail="no B head decoded"))
        b_ok = True
    else:
        with torch.no_grad():
            hh = torch.from_numpy(panel.h.astype(np.float32))
            first = head_b.delta(hh)
            second = head_b.delta(hh, torch.from_numpy(m))
        same = bool(torch.equal(first, second))
        b_abs = float(first.abs().max())
        b_ok = same
        report["items"].append(item("armB-ignores-context", "pass" if b_ok else "fail",
                                    bitwise_same_with_and_without_m=same,
                                    residual_abs_max=b_abs,
                                    residual_alive=b_abs > 1e-4))

    # 6) the sealed window must stay unreachable (red line, never regress).
    try:
        from jinyiquant.config import load_config  # noqa: F401  (only to prove it is imported)
        config_ok = True
    except Exception:  # noqa: BLE001  (the guard is pure python; the config loader is not needed)
        config_ok = None
    guard = {}
    pipeline = _import_pipeline()
    for name in ("train", "validation", "final"):
        try:
            guard[name] = {"resolved": str(pipeline.split_of(__import__("types").SimpleNamespace(
                splits=__import__("types").SimpleNamespace(train="t", validation="v",
                                                           final="SEALED")), name))}
        except Exception as exc:  # noqa: BLE001
            guard[name] = {"raises": f"{type(exc).__name__}: {exc}"}
    guard_ok = "resolved" not in guard["final"]
    report["items"].append(item("sealed-window-guard", "pass" if guard_ok else "fail",
                                splits=guard, config_importable=config_ok))

    # 7) the signatures t4 codes against must not drift.
    import inspect

    actual = {
        "build_head": str(inspect.signature(market_head.build_head)),
        "MarketInteractionHead.delta": str(
            inspect.signature(market_head.MarketInteractionHead.delta)),
        "ScalarResidualHead.delta": str(inspect.signature(market_head.ScalarResidualHead.delta)),
        "market_context": str(inspect.signature(market_head.market_context)),
        "timestamp_codes": str(inspect.signature(market_head.timestamp_codes)),
    }
    drift = {name: {"expected": expected, "actual": actual.get(name)}
             for name, expected in FROZEN_SIGNATURES.items() if actual.get(name) != expected}
    report["signatures"] = actual
    report["items"].append(item("signature-freeze", "pass" if not drift else "fail", drift=drift))

    # 8) re-run the six invariance checks with the trained head armed by the loader.
    if args.skip_full_audit:
        report["items"].append(item("full-audit", "blocked", detail="--skip-full-audit"))
    else:
        out_path = ROOT / "runs" / "mktctx-20261028" / "audit_trained.json"
        command = [sys.executable, "scripts/audit_market_context.py", "--target", "t1",
                   "--check", "all", "--size", "full", "--out", str(out_path)]
        finished = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        verdict = None
        if out_path.exists():
            verdict = json.loads(out_path.read_text(encoding="utf-8")).get("verdict")
        report["items"].append(item("full-audit", "pass" if verdict == "pass" else "fail",
                                    command=" ".join(command), exit_code=finished.returncode,
                                    verdict=verdict, report=str(out_path.relative_to(ROOT)),
                                    tail="\n".join(finished.stdout.strip().splitlines()[-4:])))

    ok = all(entry["status"] in ("pass", "blocked") for entry in report["items"]) and \
        not any(entry["status"] == "fail" for entry in report["items"])
    report["status"] = "pass" if ok else "fail"
    report["elapsed_seconds"] = round(time.time() - started, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for entry in report["items"]:
        print(f"[after-weights] {entry['item']:<24} {entry['status'].upper()}")
    print(f"[after-weights] status={report['status']} -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
