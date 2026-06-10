"""Action recognition backend interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ActionRecognizer(ABC):
    """Common interface for skeleton-based action recognizers."""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights onto the target device."""

    @abstractmethod
    def predict(self, seq: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Run inference on a skeleton sequence.

        Returns:
            List of dicts with keys: class_id, label, confidence
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Human-readable model name for logging."""


def create_action_recognizer(
    model_name: str,
    device: str = "cuda",
    *,
    ctrgcn_weights_path: str | None = None,
    posec3d_weights_path: str | None = None,
    heatmap_mode: str | None = None,
) -> ActionRecognizer:
    """
    Factory for action recognition backends.

    Args:
        model_name: ``ctrgcn`` or ``posec3d``
    """
    name = model_name.lower().strip()
    if name in ("ctrgcn", "ctr-gcn"):
        from recognizers.ctrgcn_recognizer import CTRGCNRecognizer

        return CTRGCNRecognizer(weights_path=ctrgcn_weights_path, device=device)
    if name in ("posec3d", "pose-c3d"):
        from recognizers.posec3d_recognizer import PoseC3DRecognizer

        return PoseC3DRecognizer(
            weights_path=posec3d_weights_path,
            device=device,
            heatmap_mode=heatmap_mode,
        )
    raise ValueError(f"Unknown action model: {model_name!r}. Choose 'ctrgcn' or 'posec3d'.")


__all__ = ["ActionRecognizer", "create_action_recognizer"]
