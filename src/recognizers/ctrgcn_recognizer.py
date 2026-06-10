"""CTR-GCN action recognizer (NTU-25 skeleton input)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from action_recognizer import ActionRecognizer
from ctrgcn_inference import preprocess_sequence
from models.ctrgcn import Model
from utils.config import (
    CLIP_LEN,
    IN_CHANNELS,
    NUM_CLASSES,
    NUM_JOINTS,
    NUM_PERSONS,
    TOP_K,
    load_label_map,
    resolve_ctrgcn_weights,
)
from utils.device_utils import resolve_device


class CTRGCNRecognizer(ActionRecognizer):
    """NTU120 CTR-GCN joint model on NTU-25 skeleton sequences."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cuda",
        top_k: int = TOP_K,
        labels: dict[int, str] | None = None,
    ) -> None:
        self.weights_path = str(resolve_ctrgcn_weights(weights_path))
        self.device = resolve_device(device)
        self.top_k = top_k
        self.labels = labels or load_label_map()
        self.model: Model | None = None

    def load_model(self) -> None:
        self.model = Model(
            num_class=NUM_CLASSES,
            num_point=NUM_JOINTS,
            num_person=NUM_PERSONS,
            in_channels=IN_CHANNELS,
            graph_args={"labeling_mode": "spatial"},
        )
        state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def get_model_name(self) -> str:
        return "CTR-GCN"

    def predict(self, seq: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        x = preprocess_sequence(seq, clip_len=CLIP_LEN).to(self.device)
        print(f"Action Model: {self.get_model_name()}")
        print(f"Input tensor shape: {tuple(x.shape)}")

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
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
