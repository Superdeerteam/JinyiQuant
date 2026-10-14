"""Device preflight only; runtime OOM/errors propagate to the run recorder."""

import torch


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
                       resources=device_resources(torch.device(selected)))
    return torch.device(selected)


def device_resources(device):
    import psutil
    p = psutil.Process()
    return {"rss_mb": p.memory_info().rss / 2**20, "cpu_seconds": sum(p.cpu_times()[:2]),
            "gpu_memory_mb": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "device": str(device)}
