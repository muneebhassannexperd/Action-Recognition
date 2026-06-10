"""YOLO Pose detection and Ultralytics ByteTrack / BoT-SORT tracking."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import POSE_CONF_THRESHOLD
from utils.device_utils import resolve_device

_VALID_TRACKERS = ("bytetrack.yaml", "botsort.yaml")


class PoseDetector:
    """
    YOLO Pose wrapper with optional Ultralytics multi-object tracking.

    ``track()`` returns stable IDs via ByteTrack or BoT-SORT.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = POSE_CONF_THRESHOLD,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = model_path
        self.device = resolve_device(device)
        self.conf = conf
        self.tracker = tracker if tracker in _VALID_TRACKERS else "bytetrack.yaml"
        self.model: YOLO | None = None

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO Pose checkpoint not found: {self.model_path}")
        self.model = YOLO(self.model_path)

    def _parse_results(self, results) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        for r in results:
            boxes = r.boxes
            kpts = getattr(r, "keypoints", None)
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().item())
                track_id = None
                if box.id is not None:
                    track_id = int(box.id.cpu().item())

                keypoints: list[list[float]] = []
                if kpts is not None and len(kpts) > i:
                    keypoints = kpts[i].data[0].cpu().numpy().tolist()

                entry: dict[str, Any] = {
                    "bbox": xyxy,
                    "confidence": conf,
                    "keypoints": keypoints,
                }
                if track_id is not None:
                    entry["track_id"] = track_id
                tracks.append(entry)
        return tracks

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Per-frame pose detection without tracking."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        results = self.model(frame, device=self.device, verbose=False, conf=self.conf)
        return self._parse_results(results)

    def track(self, frame: np.ndarray, persist: bool = True) -> list[dict[str, Any]]:
        """
        Pose + tracking via Ultralytics (ByteTrack or BoT-SORT).

        Returns tracks with ``track_id`` on each detection.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model.track(
            frame,
            device=self.device,
            verbose=False,
            conf=self.conf,
            tracker=self.tracker,
            persist=persist,
        )
        tracks = self._parse_results(results)
        return [t for t in tracks if "track_id" in t]

    def reset_tracker(self) -> None:
        """Reset tracker state between videos."""
        if self.model is not None:
            self.model.predictor = None  # type: ignore[attr-defined]


__all__ = ["PoseDetector"]
