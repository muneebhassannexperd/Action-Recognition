"""Detect when two tracked persons are spatially close for N consecutive frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class InteractionCandidate:
    track_a: int
    track_b: int
    start_frame: int
    last_frame: int
    consecutive_frames: int


class InteractionDetector:
    """
    Track pairs of people that remain within ``distance_threshold`` pixels
    for at least ``min_consecutive_frames`` frames.
    """

    def __init__(
        self,
        distance_threshold: float = 150.0,
        min_consecutive_frames: int = 15,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self._active_pairs: dict[tuple[int, int], InteractionCandidate] = {}

    @staticmethod
    def _center(bbox: list[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def _pair_key(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    def update(self, tracks: list[dict[str, Any]], frame_idx: int) -> list[dict[str, Any]]:
        """
        Update pair state for the current frame.

        Returns newly activated pairs (first frame reaching min_consecutive_frames).
        """
        new_candidates: list[dict[str, Any]] = []

        if len(tracks) < 2:
            self._active_pairs.clear()
            return new_candidates

        centers = {t["track_id"]: self._center(t["bbox"]) for t in tracks}
        track_ids = list(centers.keys())
        seen_keys: set[tuple[int, int]] = set()

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                a, b = track_ids[i], track_ids[j]
                key = self._pair_key(a, b)
                dist = float(np.hypot(centers[a][0] - centers[b][0], centers[a][1] - centers[b][1]))
                seen_keys.add(key)

                if dist <= self.distance_threshold:
                    if key not in self._active_pairs:
                        self._active_pairs[key] = InteractionCandidate(
                            track_a=key[0],
                            track_b=key[1],
                            start_frame=frame_idx,
                            last_frame=frame_idx,
                            consecutive_frames=1,
                        )
                    else:
                        cand = self._active_pairs[key]
                        cand.last_frame = frame_idx
                        cand.consecutive_frames += 1
                        if cand.consecutive_frames == self.min_consecutive_frames:
                            new_candidates.append({
                                "track_a": cand.track_a,
                                "track_b": cand.track_b,
                                "start_frame": cand.start_frame,
                                "frame": frame_idx,
                            })
                elif key in self._active_pairs:
                    del self._active_pairs[key]

        for key in list(self._active_pairs):
            if key not in seen_keys:
                del self._active_pairs[key]

        return new_candidates

    def active_pairs(self) -> list[tuple[int, int]]:
        """Pairs currently close for at least min_consecutive_frames."""
        return [
            (c.track_a, c.track_b)
            for c in self._active_pairs.values()
            if c.consecutive_frames >= self.min_consecutive_frames
        ]


__all__ = ["InteractionDetector", "InteractionCandidate"]
