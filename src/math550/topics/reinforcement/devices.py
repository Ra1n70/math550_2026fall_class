"""Device selection shared by the reinforcement-learning teaching code."""

from __future__ import annotations

import torch


SUPPORTED_TORCH_DEVICES = ("auto", "cuda", "mps", "cpu")


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve CUDA, Apple Metal, or CPU without silently changing requests."""

    requested = str(requested).lower()
    if requested not in SUPPORTED_TORCH_DEVICES:
        choices = ", ".join(SUPPORTED_TORCH_DEVICES)
        raise ValueError(f"device must be one of: {choices}")
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)
