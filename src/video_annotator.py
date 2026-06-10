"""Draw track boxes and mapped target-class labels on video frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from utils.class_mapping import DEFAULT_TARGET_CLASS, TARGET_COLORS


@dataclass
class TrackActionLabel:
    """Latest mapped action prediction for a single track."""

    target_class: str
    target_class_id: int
    ntu_class_id: int
    ntu_label: str
    confidence: float
    updated_frame: int

    @property
    def is_normal(self) -> bool:
        return self.target_class == DEFAULT_TARGET_CLASS


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create annotated video writer: {output_path}")
    return writer


def _color_for_target(target_class: str) -> tuple[int, int, int]:
    return TARGET_COLORS.get(target_class, TARGET_COLORS[DEFAULT_TARGET_CLASS])


def _boxes_for_pair(
    tracks: list[dict[str, Any]],
    track_a: int,
    track_b: int,
) -> tuple[list[int], list[int]] | None:
    by_id = {t["track_id"]: [int(v) for v in t["bbox"]] for t in tracks}
    if track_a not in by_id or track_b not in by_id:
        return None
    return by_id[track_a], by_id[track_b]


def _box_center(bbox: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def annotate_frame(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    track_labels: dict[int, TrackActionLabel],
    frame_idx: int,
    *,
    active_pairs: list[tuple[int, int]] | None = None,
    show_skeleton: bool = False,
) -> np.ndarray:
    """Draw person boxes with target class name and confidence."""
    out = frame.copy()
    interacting_ids: set[int] = set()
    for a, b in active_pairs or []:
        interacting_ids.add(a)
        interacting_ids.add(b)
        boxes = _boxes_for_pair(tracks, a, b)
        if boxes is not None:
            c1 = _box_center(boxes[0])
            c2 = _box_center(boxes[1])
            cv2.line(out, c1, c2, (0, 200, 255), 2, cv2.LINE_AA)

    for track in tracks:
        tid = track["track_id"]
        bbox = [int(v) for v in track["bbox"]]
        x1, y1, x2, y2 = bbox

        action = track_labels.get(tid)
        if action is not None:
            color = _color_for_target(action.target_class)
            thickness = 2 if action.is_normal else 3
            label = (
                f"ID{tid} | #{action.target_class_id} {action.target_class} "
                f"({action.confidence:.2f})"
            )
        elif tid in interacting_ids:
            color = (0, 200, 255)
            thickness = 2
            label = f"ID{tid} | interaction buffering..."
        else:
            color = (140, 140, 140)
            thickness = 2
            label = f"ID{tid} | waiting for pair..."

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        _draw_label_box(out, label, x1, max(0, y1 - 6), color)

        if show_skeleton:
            kpts = track.get("keypoints") or []
            for kp in kpts:
                if len(kp) >= 3 and kp[2] < 0.3:
                    continue
                cx, cy = int(kp[0]), int(kp[1])
                cv2.circle(out, (cx, cy), 3, color, -1, cv2.LINE_AA)

    cv2.putText(
        out,
        f"frame {frame_idx}",
        (10, out.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def _draw_label_box(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    y_top = max(0, y - th - pad * 2)
    x2 = min(img.shape[1] - 1, x + tw + pad * 2)
    cv2.rectangle(img, (x, y_top), (x2, y_top + th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (x + pad, y_top + th + pad),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


__all__ = ["TrackActionLabel", "annotate_frame", "create_video_writer"]
