"""Load client YOLO pose keypoints (JSON / JSONL) for action recognition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class KeypointsStreamMeta:
    """Video-like metadata carried with a keypoints file."""

    fps: float
    width: int
    height: int
    source: str
    frame_count: int


@dataclass(frozen=True)
class KeypointsFrame:
    """One frame of tracked detections from the client."""

    frame_idx: int
    timestamp: float
    tracks: list[dict[str, Any]]


def _normalize_bbox(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values [x1,y1,x2,y2], got {bbox!r}")
    return [float(v) for v in bbox]


def _normalize_keypoints(keypoints: list[Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for pt in keypoints:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise ValueError(f"each keypoint must be [x, y] or [x, y, conf], got {pt!r}")
        x, y = float(pt[0]), float(pt[1])
        conf = float(pt[2]) if len(pt) >= 3 else 1.0
        out.append([x, y, conf])
    return out


def normalize_track(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize one client detection to the internal track dict.

    Required: ``track_id``, ``bbox``, ``keypoints`` (COCO-17, YOLO order).
    Optional: ``confidence`` (person detection score).
    """
    if "track_id" not in raw:
        raise ValueError("track missing required field: track_id")
    track_id = int(raw["track_id"])
    if track_id < 0:
        raise ValueError(f"invalid track_id: {track_id}")

    if "bbox" not in raw:
        raise ValueError(f"track {track_id} missing required field: bbox")
    if "keypoints" not in raw:
        raise ValueError(f"track {track_id} missing required field: keypoints")

    bbox = _normalize_bbox(raw["bbox"])
    keypoints = _normalize_keypoints(raw["keypoints"])
    confidence = float(raw.get("confidence", 1.0))

    return {
        "track_id": track_id,
        "bbox": bbox,
        "keypoints": keypoints,
        "confidence": confidence,
    }


def normalize_frame_record(
    raw: dict[str, Any],
    *,
    default_fps: float,
    default_width: int,
    default_height: int,
) -> KeypointsFrame:
    """Parse one frame object from JSON / JSONL."""
    if "frame" not in raw and "frame_idx" not in raw:
        raise ValueError("frame record missing 'frame' or 'frame_idx'")
    frame_idx = int(raw.get("frame", raw.get("frame_idx")))

    fps = float(raw.get("fps", default_fps))
    if "timestamp" in raw:
        timestamp = float(raw["timestamp"])
    elif "pts_timestamp" in raw:
        timestamp = float(raw["pts_timestamp"])
    else:
        timestamp = frame_idx / max(fps, 1e-6)

    tracks_raw = raw.get("tracks") or raw.get("detections") or []
    if not isinstance(tracks_raw, list):
        raise ValueError(f"frame {frame_idx}: tracks must be a list")

    tracks = [normalize_track(t) for t in tracks_raw]
    return KeypointsFrame(frame_idx=frame_idx, timestamp=timestamp, tracks=tracks)


def _load_raw_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "frames" in data:
            return list(data["frames"])
        if "frame" in data or "frame_idx" in data:
            return [data]
    raise ValueError(
        f"Unsupported keypoints JSON in {path}. "
        "Use JSONL (one frame per line), a JSON array of frames, or {{\"frames\": [...]}}."
    )


def load_keypoints_stream(
    path: str | Path,
    *,
    default_fps: float = 30.0,
    default_width: int = 1920,
    default_height: int = 1080,
) -> tuple[Iterator[KeypointsFrame], KeypointsStreamMeta]:
    """
    Load a client keypoints file.

    Per-frame schema::

        {
          "frame": 99,
          "timestamp": 3.3,
          "fps": 30.0,
          "width": 1920,
          "height": 1080,
          "tracks": [
            {
              "track_id": 1,
              "bbox": [x1, y1, x2, y2],
              "confidence": 0.92,
              "keypoints": [[x, y, conf], ...]   // 17 COCO joints, YOLO order
            }
          ]
        }

    Top-level ``fps`` / ``width`` / ``height`` may appear on the first frame only.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Keypoints file not found: {src}")

    records = _load_raw_records(src)
    if not records:
        meta = KeypointsStreamMeta(
            fps=default_fps,
            width=default_width,
            height=default_height,
            source=src.name,
            frame_count=0,
        )
        return iter(()), meta

    file_fps = float(records[0].get("fps", default_fps))
    file_width = int(records[0].get("width", default_width))
    file_height = int(records[0].get("height", default_height))

    frames = [
        normalize_frame_record(
            rec,
            default_fps=file_fps,
            default_width=file_width,
            default_height=file_height,
        )
        for rec in records
    ]
    frames.sort(key=lambda f: f.frame_idx)

    meta = KeypointsStreamMeta(
        fps=file_fps,
        width=file_width,
        height=file_height,
        source=src.name,
        frame_count=len(frames),
    )
    return iter(frames), meta


__all__ = [
    "KeypointsFrame",
    "KeypointsStreamMeta",
    "load_keypoints_stream",
    "normalize_frame_record",
    "normalize_track",
]
