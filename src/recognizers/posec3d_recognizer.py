"""PoseC3D action recognizer (raw COCO-17 keypoint input)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from action_recognizer import ActionRecognizer
from models.posec3d import PoseC3DRecognizer as PoseC3DModel
from posec3d_preprocessing import preprocess_coco_sequence
from utils.config import (
    NUM_CLASSES,
    POSEC3D_HEATMAP_MODE,
    TOP_K,
    load_label_map,
    resolve_posec3d_weights,
)
from utils.device_utils import resolve_device


class PoseC3DRecognizer(ActionRecognizer):
    """NTU120 PoseC3D SlowOnly R50 on COCO-17 keypoint heatmaps (no NTU-25 conversion)."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cuda",
        top_k: int = TOP_K,
        labels: dict[int, str] | None = None,
        heatmap_mode: str | None = None,
    ) -> None:
        self.weights_path = str(resolve_posec3d_weights(weights_path))
        self.device = resolve_device(device)
        self.top_k = top_k
        self.labels = labels or load_label_map()
        self.heatmap_mode = (heatmap_mode or POSEC3D_HEATMAP_MODE).lower()
        self.model: PoseC3DModel | None = None

    def load_model(self) -> None:
        self.model = PoseC3DModel(num_classes=NUM_CLASSES, in_channels=17)
        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def get_model_name(self) -> str:
        return "PoseC3D"

    def predict(self, seq: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        img_shape = kwargs.get("img_shape")
        if img_shape is None:
            raise ValueError("PoseC3D requires img_shape=(H, W) in predict() kwargs")

        x = preprocess_coco_sequence(
            seq,
            img_shape=tuple(img_shape),
            heatmap_mode=self.heatmap_mode,
        ).to(self.device)

        mode = kwargs.get("inference_mode", "single" if seq.ndim == 4 and seq.shape[0] == 1 else "pair")
        m = seq.shape[0] if seq.ndim == 4 else 1
        print(f"Action Model: {self.get_model_name()} ({mode}, M={m})")
        print(f"Input tensor shape: {tuple(x.shape)}")
        print(f"Heatmap mode: {self.heatmap_mode}")

        t0 = time.perf_counter()
        with torch.no_grad():
            probs = self.model(x)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        print(f"Inference latency: {latency_ms:.1f} ms")

        k = min(self.top_k, NUM_CLASSES)
        values, indices = torch.topk(probs, k)
        predictions = []
        for score, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
            class_id = idx + 1
            predictions.append({
                "class_id": class_id,
                "label": self.labels.get(class_id, f"class_{class_id}"),
                "confidence": round(float(score), 4),
            })

        self._print_top_predictions(predictions)
        return predictions

    @staticmethod
    def _print_top_predictions(predictions: list[dict[str, Any]]) -> None:
        print(f"Top {len(predictions)} predictions:")
        for rank, pred in enumerate(predictions, start=1):
            print(
                f"  {rank}. #{pred['class_id']:3d} {pred['label']:<40} "
                f"({pred['confidence']:.4f})"
            )
