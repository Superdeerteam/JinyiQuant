"""Adversarial probe of the new cross-section gate (``MIN_CROSS_SECTION``), auditor-owned.

``TLOBAdapter._market_residual`` now refuses a batch whose smallest same-timestamp group is
below ``MIN_CROSS_SECTION`` (64 rows), unless ``adapter.allow_cross_section_fragment`` is set.
Two failure modes matter and this script measures both:

* the gate leaks: a 32-row fragment is summarised into ``m_t`` without complaint, so a chunked
  replay silently feeds the head a market context it was never trained on;
* the gate over-fires: it rejects a normal replay cross-section (1000 instruments per timestamp),
  which would block the whole t4 replay.

Cases (all through the real ``TLOBAdapter._market_residual``):

  1 floor_50          keys with a 50-row timestamp must raise ValueError
  2 boundary_63/_64   63 rows raise, 64 rows pass (the threshold is inclusive)
  3 fragment_allowed  same 50-row batch with ``allow_cross_section_fragment=True`` must not raise
  4 fragment_damage   the fragment's residual vs the same rows inside the full 1000-row group
  5 real_replay_keys  the actual 80000-row replay key table (1000 per timestamp) must pass
  6 arm_b             arm B is gated identically (the check sits before the arm branch)
  7 no_head           with no market head the gate is never reached (static evidence)

Read-only: no file under ``modified_tlob/`` is written, no bar/label data is read.

    .venv-tlob/Scripts/python.exe scripts/audit_cross_section_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DEFAULT_OUT = ROOT / "runs" / "mktctx-20261028" / "cross_section_gate.json"
THRESHOLD = 1e-6


class _AnyStub(types.ModuleType):
    """Inert stand-in for one module of the data layer (the adapter imports it at load)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = () if name.isupper() else (lambda *a, **k: None)
        setattr(self, name, value)
        return value


def _import_adapter():
    try:
        from modified_tlob import adapter
        return adapter
    except ModuleNotFoundError:
        pass
    package = _AnyStub("jinyiquant")
    for sub in ("config", "data", "io", "pool", "runner", "tracking", "checks", "comparison",
                "evaluation"):
        module = _AnyStub(f"jinyiquant.{sub}")
        setattr(package, sub, module)
        sys.modules[f"jinyiquant.{sub}"] = module
    sys.modules["jinyiquant"] = package
    from modified_tlob import adapter
    return adapter


def armed_head(market_head, arm, *, seed=20261028, scale=0.02):
    """A head whose residual is non-zero, so a fragment's damage is observable at all."""
    import torch

    head = market_head.build_head(arm)
    with torch.no_grad():
        if arm == "C":
            generator = torch.Generator().manual_seed(seed)
            head.mix.weight.copy_(torch.randn(head.mix.weight.shape, generator=generator) * scale)
        else:
            last = head.layers[-1]
            generator = torch.Generator().manual_seed(seed)
            last.weight.copy_(torch.randn(last.weight.shape, generator=generator) * scale)
            last.bias.copy_(torch.randn(last.bias.shape, generator=generator) * scale)
    return head.eval()


def bare_adapter(adapter_module, market_head, *, arm="C", allow=False):
    """An adapter instance that only carries what ``_market_residual`` reads."""
    instance = adapter_module.TLOBAdapter.__new__(adapter_module.TLOBAdapter)
    instance.market_head = armed_head(market_head, arm)
    instance.device = "cpu"
    instance.allow_cross_section_fragment = allow
    return instance


def panel(*group_sizes, seed=5, width=80):
    """Synthetic hidden states plus keys for the requested per-timestamp row counts."""
    rng = np.random.default_rng(seed)
    sizes = list(group_sizes)
    hidden = rng.standard_normal((sum(sizes), width)).astype(np.float32)
    hidden = np.maximum(hidden, -0.17)  # GELU output range of the frozen trunk
    instruments, stamps = [], []
    for index, size in enumerate(sizes):
        for row in range(size):
            instruments.append(f"S{index:02d}{row:05d}")
            stamps.append(f"2023-11-01 {9 + index // 2:02d}:{30 if index % 2 == 0 else 0:02d}"
                          if index < 2 else f"2023-11-01 {10 + index:02d}:00")
    keys = pd.DataFrame({"instrument": instruments, "date": stamps})
    missing = np.zeros(len(hidden), dtype=bool)
    return hidden, keys, missing


def probe(adapter_module, market_head, hidden, keys, missing, *, allow=False, arm="C"):
    adapter = bare_adapter(adapter_module, market_head, arm=arm, allow=allow)
    try:
        residual = adapter._market_residual(hidden, keys, missing)
        return {"raised": False, "max_abs": float(np.abs(residual).max()),
                "finite": bool(np.isfinite(residual).all()), "rows": int(residual.size),
                "residual": residual}
    except Exception as exc:  # noqa: BLE001
        return {"raised": True, "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keys", type=Path, default=Path("D:/tmp/mktctx_r8keys.parquet"),
                        help="real replay key table (date, instrument); optional")
    parser.add_argument("--rows", type=int, default=80000, help="rows of the real key table to use")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    adapter_module = _import_adapter()
    from modified_tlob import market_head

    started = time.time()
    report = {"schema_version": 1, "audit": "cross-section gate (MIN_CROSS_SECTION)",
              "threshold": THRESHOLD, "min_cross_section": market_head.MIN_CROSS_SECTION,
              "cases": {}, "failures": []}

    def record(name, ok, **evidence):
        report["cases"][name] = {"passed": bool(ok), **evidence}
        if not ok:
            report["failures"].append(name)
        print(f"[gate] {name:<22} {'PASS' if ok else 'FAIL'}")

    # 1) a 50-row timestamp is a fragment: it must be refused with an actionable message.
    hidden, keys, missing = panel(96, 96, 50)
    below = probe(adapter_module, market_head, hidden, keys, missing)
    message_ok = below.get("raised") and "allow_cross_section_fragment" in below["error"] \
        and "50 rows" in below["error"]
    record("floor_50", below.get("raised", False) and message_ok,
           raised=below.get("raised"), error=below.get("error"), group_sizes=[96, 96, 50])

    # 2) the threshold itself: 63 rows refused, 64 rows accepted.
    hidden63, keys63, missing63 = panel(96, 63)
    hidden64, keys64, missing64 = panel(96, 64)
    at63 = probe(adapter_module, market_head, hidden63, keys63, missing63)
    at64 = probe(adapter_module, market_head, hidden64, keys64, missing64)
    record("boundary_63", at63.get("raised", False), raised=at63.get("raised"),
           error=at63.get("error"))
    record("boundary_64", (not at64.get("raised", True)) and at64.get("finite", False),
           raised=at64.get("raised"), max_abs=at64.get("max_abs"), rows=at64.get("rows"))

    # 3) the escape hatch: the same 50-row batch passes when the caller asks for the damage.
    allowed = probe(adapter_module, market_head, hidden, keys, missing, allow=True)
    record("fragment_allowed", (not allowed.get("raised", True)) and allowed.get("finite", False),
           raised=allowed.get("raised"), max_abs=allowed.get("max_abs"), rows=allowed.get("rows"))

    # 4) how wrong is the fragment? Same 50 rows, summary over 50 rows vs over the whole group.
    width = hidden.shape[1]
    rng = np.random.default_rng(11)
    whole_hidden = (rng.standard_normal((1000, width)) * 0.8 + 0.6).astype(np.float32)
    whole_hidden = np.maximum(whole_hidden, -0.17)  # GELU output range of the frozen trunk
    whole_keys = pd.DataFrame({"instrument": [f"W{i:05d}" for i in range(1000)],
                               "date": ["2023-11-01 10:00"] * 1000})
    whole_missing = np.zeros(1000, dtype=bool)
    whole = probe(adapter_module, market_head, whole_hidden, whole_keys, whole_missing)
    fragment_rows = np.arange(100, 150)
    fragment = probe(adapter_module, market_head, whole_hidden[fragment_rows],
                     whole_keys.iloc[fragment_rows].reset_index(drop=True),
                     whole_missing[fragment_rows], allow=True)
    damage = None
    if whole.get("residual") is not None and fragment.get("residual") is not None:
        damage = float(np.abs(whole["residual"][fragment_rows] - fragment["residual"]).max())
    record("fragment_damage", damage is not None and damage > THRESHOLD,
           whole_group_rows=1000, fragment_rows=50, rows_compared=50,
           max_abs_residual_difference=damage,
           note="the same 50 rows score differently when m_t is summarised over 50 rows instead "
                "of the 1000-row cross-section: this is what the gate protects")

    # 5) the real replay key table must pass untouched: 1000 rows per timestamp, 80 timestamps.
    if args.keys.exists():
        real = pd.read_parquet(args.keys).iloc[:args.rows].reset_index(drop=True)
        sizes = real.groupby("date").size()
        rng = np.random.default_rng(17)
        real_hidden = np.maximum(rng.standard_normal((len(real), 80)).astype(np.float32) * 0.8
                                 + 0.6, -0.17)
        passed = probe(adapter_module, market_head, real_hidden, real, np.zeros(len(real), np.bool_))
        record("real_replay_keys", (not passed.get("raised", True)) and passed.get("finite", False),
               raised=passed.get("raised"), keys=str(args.keys), rows=int(len(real)), groups=int(sizes.size),
               min_group=int(sizes.min()), max_group=int(sizes.max()),
               max_abs=passed.get("max_abs"), error=passed.get("error"))
    else:
        record("real_replay_keys", False, error=f"missing key table {args.keys}")

    # 6) arm B is gated identically: the guard sits before the arm branch.
    b_below = probe(adapter_module, market_head, hidden, keys, missing, arm="B")
    b_allowed = probe(adapter_module, market_head, hidden, keys, missing, allow=True, arm="B")
    record("arm_b", b_below.get("raised", False) and not b_allowed.get("raised", True),
           refused_without_flag=b_below.get("raised"), passed_with_flag=not b_allowed.get("raised", True),
           max_abs_with_flag=b_allowed.get("max_abs"))

    # 7) with no market head the residual is never computed, so the gate cannot fire.
    source = Path(adapter_module.__file__).read_text(encoding="utf-8")
    guard_line = next((number + 1 for number, line in enumerate(source.splitlines())
                       if "if hidden is None else raw + self._market_residual" in line), None)
    record("no_head", guard_line is not None, evidence_line=guard_line,
           note="scores = raw when the adapter carries no market head (line above); the gate lives "
                "only in _market_residual")

    report["verdict"] = "pass" if not report["failures"] else "fail"
    report["elapsed_seconds"] = round(time.time() - started, 2)
    report["files"] = {name: {"sha256": __import__("hashlib").sha256(
        (ROOT / name).read_bytes()).hexdigest(), "bytes": (ROOT / name).stat().st_size}
        for name in ("modified_tlob/adapter.py", "modified_tlob/market_head.py")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    print(f"[gate] verdict={report['verdict']} -> {args.out}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
