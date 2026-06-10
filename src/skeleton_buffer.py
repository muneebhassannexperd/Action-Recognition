"""Rolling per-track skeleton buffers (NTU-25 or raw COCO-17)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

import numpy as np

from joint_mapper import JointMapper

VALID_WINDOW_SIZES = (30, 48, 60, 90, 100, 120)
COCO_NUM_JOINTS = 17


def _yolo_to_coco_frame(yolo_keypoints: list[list[float]]) -> np.ndarray:
    """Convert one frame of YOLO keypoints to (17, 3) x, y, confidence."""
    frame = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    for i, pt in enumerate(yolo_keypoints[:COCO_NUM_JOINTS]):
        if len(pt) >= 2:
            frame[i, 0] = float(pt[0])
            frame[i, 1] = float(pt[1])
        if len(pt) >= 3:
            frame[i, 2] = float(pt[2])
    return frame


class SkeletonBufferBase(ABC):
    """Common rolling-window buffer API."""

    VALID_WINDOW_SIZES = VALID_WINDOW_SIZES

    def __init__(self, window_size: int) -> None:
        if window_size not in self.VALID_WINDOW_SIZES:
            raise ValueError(
                f"window_size must be one of {self.VALID_WINDOW_SIZES}, got {window_size}"
            )
        self.window_size = window_size
        self._buffers: dict[int, Deque[np.ndarray]] = {}

    @abstractmethod
    def _map_frame(self, yolo_keypoints: list[list[float]]) -> np.ndarray:
        """Convert YOLO keypoints to one stored frame."""

    def add(self, track_id: int, yolo_keypoints: list[list[float]]) -> None:
        frame = self._map_frame(yolo_keypoints)
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)
        self._buffers[track_id].append(frame)

    def is_ready(self, track_id: int, min_frames: int | None = None) -> bool:
        min_frames = min_frames or self.window_size
        buf = self._buffers.get(track_id)
        return buf is not None and len(buf) >= min_frames

    def get_sequence(self, track_id: int) -> np.ndarray | None:
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        return np.stack(list(buf), axis=0)

    def is_pair_ready(self, track_a: int, track_b: int, min_frames: int | None = None) -> bool:
        return self.is_ready(track_a, min_frames) and self.is_ready(track_b, min_frames)

    def get_pair_sequence(self, track_a: int, track_b: int) -> np.ndarray | None:
        seq_a = self.get_sequence(track_a)
        seq_b = self.get_sequence(track_b)
        if seq_a is None or seq_b is None:
            return None

        t = min(len(seq_a), len(seq_b))
        if t < self.window_size:
            return None

        seq_a = seq_a[-self.window_size :]
        seq_b = seq_b[-self.window_size :]
        return np.stack([seq_a, seq_b], axis=0)

    def frame_count(self, track_id: int) -> int:
        buf = self._buffers.get(track_id)
        return len(buf) if buf else 0

    def remove(self, track_id: int) -> None:
        self._buffers.pop(track_id, None)

    def track_ids(self) -> list[int]:
        return list(self._buffers.keys())

    def set_window_size(self, window_size: int) -> None:
        if window_size not in self.VALID_WINDOW_SIZES:
            raise ValueError(
                f"window_size must be one of {self.VALID_WINDOW_SIZES}, got {window_size}"
            )
        self.window_size = window_size
        for tid, buf in list(self._buffers.items()):
            frames = list(buf)
            self._buffers[tid] = deque(frames[-window_size:], maxlen=window_size)


class NTUSkeletonBuffer(SkeletonBufferBase):
    """CTR-GCN path: YOLO COCO-17 -> JointMapper -> NTU-25."""

    def __init__(self, window_size: int = 60, mapper: JointMapper | None = None) -> None:
        super().__init__(window_size)
        self.mapper = mapper or JointMapper()

    def _map_frame(self, yolo_keypoints: list[list[float]]) -> np.ndarray:
        return self.mapper.map_frame(yolo_keypoints)


class COCOSkeletonBuffer(SkeletonBufferBase):
    """PoseC3D path: store raw YOLO COCO-17 keypoints (no NTU conversion)."""

    def _map_frame(self, yolo_keypoints: list[list[float]]) -> np.ndarray:
        return _yolo_to_coco_frame(yolo_keypoints)


def create_skeleton_buffer(action_model: str, window_size: int) -> SkeletonBufferBase:
    """Create the skeleton buffer matching the selected action recognizer."""
    name = action_model.lower().strip()
    if name in ("ctrgcn", "ctr-gcn"):
        return NTUSkeletonBuffer(window_size=window_size)
    if name in ("posec3d", "pose-c3d"):
        return COCOSkeletonBuffer(window_size=window_size)
    raise ValueError(f"Unknown action model: {action_model!r}")


# Backward-compatible alias
SkeletonBuffer = NTUSkeletonBuffer

__all__ = [
    "SkeletonBuffer",
    "SkeletonBufferBase",
    "NTUSkeletonBuffer",
    "COCOSkeletonBuffer",
    "create_skeleton_buffer",
    "VALID_WINDOW_SIZES",
]
