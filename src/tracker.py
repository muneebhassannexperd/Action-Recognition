"""
Ultralytics ByteTrack / BoT-SORT tracker (via YOLO ``model.track``).

The legacy IoU tracker has been removed. Use ``PoseDetector.track()``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pose_detector import PoseDetector


class UltralyticsTracker:
    """Thin wrapper delegating to ``PoseDetector.track``."""

    def __init__(self, pose_detector: PoseDetector) -> None:
        self.pose_detector = pose_detector

    def update(self, frame: np.ndarray) -> list[dict[str, Any]]:
        return self.pose_detector.track(frame, persist=True)

    def reset(self) -> None:
        self.pose_detector.reset_tracker()


# Backward-compatible alias
ByteTracker = UltralyticsTracker

__all__ = ["UltralyticsTracker", "ByteTracker"]
