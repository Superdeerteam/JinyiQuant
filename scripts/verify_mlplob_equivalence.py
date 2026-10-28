"""Prove this repo's MLPLOB reproduces the legacy branch under the control's own weights.

The R14 arm runs on a copy of the training repository whose ``modified_tlob/model.py`` adds the
representation interface (``representation_dim``, ``represent``, ``forward_with_representation``,
``_representation``) so the trunk can be pretrained without the scalar head. That is only
acceptable if the *forward pass is unchanged*: the arm has to stay comparable to the supervised
control it is budget-matched against. This check loads the control run's exported weights into
both implementations, runs one fixed input through both, and reports the difference.

    python scripts/verify_mlplob_equivalence.py \
        --legacy-repo /mnt/work/repo_stream2 --current-repo /mnt/work/repo_r14 \
        --weights /mnt/work/runs/scaling-441d-seed11/<stamp>/modified_tlob_weights-scaling-1.32.json \
        --output /mnt/work/logs/r14-compat.json

Run as a module from the orchestrator (``verify(...)``) or standalone; the two implementations are
imported in separate interpreters because they share the ``modified_tlob`` module name.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA = "mlplob-equivalence/1"
DEFAULT_SEED = 0
DEFAULT_SHAPE = (8, 128, 12)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(repo: str, weights: str, out: str, seed: int, shape: tuple[int, ...]) -> int:
    """Child entry point: one forward pass under ``repo``'s implementation."""
    sys.path.insert(0, repo)
    import torch

    from modified_tlob.adapter import TLOBAdapter

    adapter = TLOBAdapter.load(weights, device="cpu")
    torch.manual_seed(seed)
    x = torch.randn(*shape)
    with torch.no_grad():
        y = adapter.network(x)
    np.save(out, y.detach().numpy())
    print(json.dumps({"class": type(adapter.network).__name__, "shape": list(y.shape)}))
    return 0


def compare_probes(left: str | Path, right: str | Path) -> dict:
    """Compare two saved forward passes."""
    a = np.load(left)
    b = np.load(right)
    if a.shape != b.shape:
        raise ValueError(f"Probe shapes differ: {a.shape} vs {b.shape}")
    return {"bitwise_equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()),
            "elements": int(a.size),
            "sum": float(a.astype(np.float64).sum())}


def verify(legacy_repo: str, current_repo: str, weights_path: str, *,
           seed: int = DEFAULT_SEED, shape: tuple[int, ...] = DEFAULT_SHAPE,
           timeout: int = 900) -> dict:
    """Run the probe under both repositories and describe the difference."""
    weights = Path(weights_path)
    if not weights.exists():
        raise ValueError(f"Control weights {weights} are missing")
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        outputs = []
        for repo in (legacy_repo, current_repo):
            out = Path(tmp) / f"{Path(repo).name}.npy"
            completed = subprocess.run(
                [sys.executable, __file__, "--probe", repo, str(weights), str(out),
                 "--seed", str(seed), "--shape", ",".join(str(v) for v in shape)],
                check=False, capture_output=True, text=True, timeout=timeout)
            if completed.returncode != 0:
                raise ValueError(f"Probe under {repo} failed: {completed.stderr.strip()[-2000:]}")
            outputs.append(out)
        comparison = compare_probes(outputs[0], outputs[1])
    record = {"schema_version": SCHEMA,
              "legacy_repo": str(legacy_repo), "current_repo": str(current_repo),
              "weights": str(weights), "weights_sha256": _sha256(weights),
              "model_sha256": {str(repo): _sha256(Path(repo) / "modified_tlob" / "model.py")
                               for repo in (legacy_repo, current_repo)},
              "input": {"seed": seed, "shape": list(shape)},
              "seconds": round(time.time() - started, 3), **comparison}
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", nargs=3, metavar=("REPO", "WEIGHTS", "OUT"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--legacy-repo", default="/mnt/work/repo_stream2")
    parser.add_argument("--current-repo", default="/mnt/work/repo_r14")
    parser.add_argument("--weights")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shape", default=",".join(str(v) for v in DEFAULT_SHAPE))
    options = parser.parse_args(argv)
    shape = tuple(int(v) for v in options.shape.split(","))
    if options.probe:
        repo, weights, out = options.probe
        return _probe(repo, weights, out, options.seed, shape)
    if not options.weights:
        parser.error("--weights is required unless --probe is used")
    record = verify(options.legacy_repo, options.current_repo, options.weights,
                    seed=options.seed, shape=shape)
    if options.output:
        Path(options.output).write_text(json.dumps(record, indent=1, sort_keys=True) + chr(10),
                                        encoding="utf-8")
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0 if record["bitwise_equal"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
