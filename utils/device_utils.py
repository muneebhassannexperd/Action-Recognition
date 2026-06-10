"""Resolve inference device (CUDA vs CPU) from user preference and torch availability."""

from __future__ import annotations

import warnings

import torch


def resolve_device(requested: str | None = None) -> str:
    req = (requested or "cuda").strip().lower()

    if req == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        if req in ("cuda", "gpu"):
            return "cuda"
        if req.startswith("cuda"):
            return req
        if req.isdigit():
            return req

    if req in ("cuda", "gpu") or req.startswith("cuda"):
        warnings.warn(
            "CUDA requested but not available. Using CPU.",
            stacklevel=2,
        )

    return "cpu"


__all__ = ["resolve_device"]
