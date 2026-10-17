"""Fail closed before a cloud training run when CUDA is unusable."""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def probe():
    started = time.monotonic()
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "status": "failed",
        "training_started": False,
        "labels_read": False,
        "dependencies": {},
    }
    for name in ("torch", "numpy", "pandas", "pyarrow", "pydantic", "psutil", "einops"):
        try:
            result["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["dependencies"][name] = None
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        result[name] = path.read_text().strip() if path.exists() else None
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        result["nvidia_smi"] = {"exit_code": smi.returncode, "stdout": smi.stdout.strip(), "stderr": smi.stderr.strip()}
        driver = ctypes.CDLL("libcuda.so.1")
        code = driver.cuInit(0)
        error = ctypes.c_char_p()
        driver.cuGetErrorString(code, ctypes.byref(error))
        result["driver_initialization"] = {"code": code, "message": error.value.decode() if error.value else None}
        import torch

        result["torch_cuda_version"] = torch.version.cuda
        result["torch_cuda_available"] = torch.cuda.is_available()
        if code != 0 or not result["torch_cuda_available"]:
            raise RuntimeError("CUDA driver initialization failed; training must not fall back to CPU")
        tensor = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8).requires_grad_()
        loss = tensor.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        result["cuda_forward_backward_finite"] = bool(torch.isfinite(loss).item() and torch.isfinite(tensor.grad).all().item())
        if not result["cuda_forward_backward_finite"]:
            raise RuntimeError("CUDA arithmetic returned nonfinite values")
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "error": result.get("error")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
