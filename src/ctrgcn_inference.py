"""CTR-GCN NTU120 joint-model inference and skeleton preprocessing."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utils.class_mapping import TARGET_NTU_IDS
from utils.device_utils import resolve_device

from models.ctrgcn import Model

EPS = 1e-4


def pre_normalize_2d(skeleton: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Normalize 2D skeleton (M, T, V, C) with x,y,confidence channels.

    Adapted from pyskl PreNormalize2D (mode='auto').
    """
    keypoint = skeleton.astype(np.float32).copy()
    m, t, v, c = keypoint.shape
    assert c == 3

    mask = keypoint[..., 2] > threshold
    maskout = ~mask

    for person in range(m):
        for frame in range(t):
            vis = mask[person, frame]
            if np.any(vis):
                xs = keypoint[person, frame, vis, 0]
                ys = keypoint[person, frame, vis, 1]
                x_max, x_min = float(np.max(xs)), float(np.min(xs))
                y_max, y_min = float(np.max(ys)), float(np.min(ys))
            else:
                x_max = x_min = y_max = y_min = 0.0

            if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                keypoint[person, frame, :, 0] = (
                    (keypoint[person, frame, :, 0] - (x_max + x_min) / 2)
                    / (x_max - x_min)
                    * 2
                )
                keypoint[person, frame, :, 1] = (
                    (keypoint[person, frame, :, 1] - (y_max + y_min) / 2)
                    / (y_max - y_min)
                    * 2
                )

            keypoint[person, frame, maskout[person, frame], 0] = 0
            keypoint[person, frame, maskout[person, frame], 1] = 0

    keypoint[..., 2] = 0.0
    return keypoint


def uniform_sample(skeleton: np.ndarray, clip_len: int = CLIP_LEN) -> np.ndarray:
    """Resample (M, T, V, C) along time to clip_len frames (pyskl UniformSample)."""
    m, t, v, c = skeleton.shape
    if t == clip_len:
        return skeleton
    if t < 1:
        return np.zeros((m, clip_len, v, c), dtype=np.float32)

    if t < clip_len:
        idx = np.arange(clip_len) % t
    else:
        idx = np.linspace(0, t - 1, clip_len).astype(np.int64)

    return skeleton[:, idx].copy()


def format_gcn_input(skeleton: np.ndarray, num_person: int = NUM_PERSONS) -> np.ndarray:
    """Pad/select persons -> (M, T, V, C)."""
    m, t, v, c = skeleton.shape
    if m < num_person:
        pad = np.zeros((num_person - m, t, v, c), dtype=skeleton.dtype)
        skeleton = np.concatenate([skeleton, pad], axis=0)
    elif m > num_person:
        skeleton = skeleton[:num_person]
    return skeleton


def preprocess_sequence(
    seq: np.ndarray,
    clip_len: int = CLIP_LEN,
    num_persons: int | None = None,
) -> torch.Tensor:
    """
    Convert skeleton sequence to model tensor (1, C, T, V, M).

    Accepts:
        (T, V, 3)  — single person
        (M, T, V, 3) — one or two persons (M=1 single-track, M=2 pair)
    """
    if seq.ndim == 3:
        if seq.shape[1] != NUM_JOINTS:
            raise ValueError(f"Expected (T, {NUM_JOINTS}, 3), got {seq.shape}")
        skeleton = seq[np.newaxis, ...]
    elif seq.ndim == 4:
        m, t, v, c = seq.shape
        if v != NUM_JOINTS or c != 3:
            raise ValueError(f"Expected (M, T, {NUM_JOINTS}, 3), got {seq.shape}")
        if m > NUM_PERSONS:
            skeleton = seq[:NUM_PERSONS]
        else:
            skeleton = seq
    else:
        raise ValueError(f"Expected 3D or 4D skeleton array, got shape {seq.shape}")

    persons = num_persons if num_persons is not None else skeleton.shape[0]
    persons = min(max(persons, 1), NUM_PERSONS)

    skeleton = pre_normalize_2d(skeleton.astype(np.float32))
    skeleton = uniform_sample(skeleton, clip_len)
    skeleton = format_gcn_input(skeleton, persons)

    tensor = torch.from_numpy(skeleton).float().unsqueeze(0)
    tensor = tensor.permute(0, 4, 2, 3, 1)
    return tensor


class CTRGCNInference:
    """Backward-compatible wrapper around :class:`CTRGCNRecognizer`."""

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cuda",
        top_k: int = TOP_K,
        labels: dict[int, str] | None = None,
    ) -> None:
        from recognizers.ctrgcn_recognizer import CTRGCNRecognizer

        self._recognizer = CTRGCNRecognizer(
            weights_path=weights_path,
            device=device,
            top_k=top_k,
            labels=labels,
        )
        self.weights_path = self._recognizer.weights_path
        self.device = self._recognizer.device
        self.top_k = top_k
        self.labels = self._recognizer.labels
        self.model = None

    def load_model(self) -> None:
        self._recognizer.load_model()
        self.model = self._recognizer.model

    def predict(self, seq: np.ndarray) -> list[dict[str, Any]]:
        return self._recognizer.predict(seq)

    @staticmethod
    def filter_violence(
        predictions: list[dict[str, Any]],
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Deprecated: use utils.class_mapping.filter_target_events instead."""
        return [
            p for p in predictions
            if p["class_id"] in TARGET_NTU_IDS and p["confidence"] >= min_confidence
        ]


__all__ = ["CTRGCNInference", "preprocess_sequence", "pre_normalize_2d"]
