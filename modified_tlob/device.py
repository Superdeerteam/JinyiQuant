"""Device preflight only; runtime OOM/errors propagate to the run recorder."""

import os

import torch


def _configure_cuda_workspace():
    """PyTorch 2.5 requires one of these values before the first cuBLAS call."""
    name = "CUBLAS_WORKSPACE_CONFIG"
    value = os.environ.get(name)
    if value is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(f"Set {name} before CUDA initialization; restart in a fresh process")
        value = ":4096:8"
        os.environ[name] = value
    if value not in {":4096:8", ":16:8"}:
        raise ValueError(f"{name} must be ':4096:8' or ':16:8'; got {value!r}")
    return value


def _cuda_preflight():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable (CPU build, missing GPU or driver)")
    x = torch.ones((2, 2), device="cuda", requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize()


def select_device(requested="auto", recorder=None):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto/cpu/cuda")
    selected, reason = "cpu", "explicit_cpu"
    if requested != "cpu":
        _configure_cuda_workspace()
        try:
            _cuda_preflight()
        except (RuntimeError, AssertionError, OSError) as exc:
            if requested == "cuda":
                raise RuntimeError(f"Explicit CUDA preflight failed: {exc}") from exc
            reason = f"cuda_preflight_failed: {type(exc).__name__}: {exc}"
        else:
            selected, reason = "cuda", "cuda_compute_preflight_passed"
    if recorder:
        recorder.event("device", "selected", requested=requested, selected=selected, reason=reason,
                       torch_version=torch.__version__, cuda_build=torch.version.cuda,
                       cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       resources=device_resources(torch.device(selected)))
    return torch.device(selected)


def device_resources(device):
    import psutil
    p = psutil.Process()
    return {"rss_mb": p.memory_info().rss / 2**20, "cpu_seconds": sum(p.cpu_times()[:2]),
            "gpu_memory_mb": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "device": str(device)}
