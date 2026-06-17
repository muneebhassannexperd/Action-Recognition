"""Adapt MS-1 ``ai_detection`` Redis messages to the internal track format
expected by :meth:`VideoProcessor._process_frame`.

Kenneth's ``ai_detection`` schema
---------------------------------

.. code-block:: json

    {
      "type": "ai_detection",
      "camera_id": 3,
      "organization_id": 1,
      "frame_id": "cam3_f1042",
      "frame_sequence": 1042,
      "pts_timestamp": 34.733,
      "frame_shape": [1080, 1920, 3],
      "persons": [
        {
          "track_id": 7,
          "bounding_box": [x1, y1, x2, y2],
          "confidence": 0.91,
          "pose_keypoints": [[x, y, conf], ...]
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EXPECTED_KEYPOINTS = 17


@dataclass(frozen=True)
class ParsedFrame:
    """Result of parsing one ``ai_detection`` message."""

    frame_idx: int
    timestamp: float
    pts_timestamp: float
    camera_id: int
    organization_id: int
    frame_shape: tuple[int, int]
    tracks: list[dict[str, Any]]


class AdapterError(Exception):
    """Raised when an ``ai_detection`` message cannot be parsed."""


def _normalize_person(person: dict[str, Any]) -> dict[str, Any]:
    """Convert one ``persons[]`` entry to the internal track dict."""
    track_id = int(person["track_id"])
    if track_id < 0:
        raise AdapterError(f"invalid track_id: {track_id}")

    bbox = person.get("bounding_box")
    if bbox is None or len(bbox) != 4:
        raise AdapterError(f"track {track_id}: bounding_box must be [x1,y1,x2,y2]")
    bbox = [float(v) for v in bbox]

    raw_kpts = person.get("pose_keypoints")
    if not raw_kpts:
        raise AdapterError(f"track {track_id}: pose_keypoints is empty or missing")
    if len(raw_kpts) != EXPECTED_KEYPOINTS:
        raise AdapterError(
            f"track {track_id}: expected {EXPECTED_KEYPOINTS} keypoints, "
            f"got {len(raw_kpts)}"
        )

    keypoints: list[list[float]] = []
    for pt in raw_kpts:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            raise AdapterError(f"track {track_id}: bad keypoint {pt!r}")
        x, y = float(pt[0]), float(pt[1])
        conf = float(pt[2]) if len(pt) >= 3 else 1.0
        keypoints.append([x, y, conf])

    confidence = float(person.get("confidence", 1.0))

    return {
        "track_id": track_id,
        "bbox": bbox,
        "keypoints": keypoints,
        "confidence": confidence,
    }


def parse_ai_detection(raw: str | bytes | dict[str, Any]) -> ParsedFrame:
    """Parse a single ``ai_detection`` Redis message into a :class:`ParsedFrame`.

    Parameters
    ----------
    raw:
        Either the raw JSON string / bytes from Redis, or an already-decoded dict.

    Returns
    -------
    ParsedFrame
        Ready to feed into ``VideoProcessor._process_frame()``.
    """
    if isinstance(raw, (str, bytes)):
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"invalid JSON: {exc}") from exc
    else:
        msg = raw

    msg_type = msg.get("type", "")
    if msg_type != "ai_detection":
        raise AdapterError(f"unexpected message type: {msg_type!r}")

    try:
        frame_idx = int(msg["frame_sequence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(f"missing or invalid frame_sequence: {exc}") from exc

    try:
        pts_timestamp = float(msg["pts_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(f"missing or invalid pts_timestamp: {exc}") from exc

    camera_id = int(msg.get("camera_id", 0))
    organization_id = int(msg.get("organization_id", 0))

    shape_raw = msg.get("frame_shape", [1080, 1920, 3])
    if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) < 2:
        raise AdapterError(f"invalid frame_shape: {shape_raw!r}")
    frame_shape = (int(shape_raw[0]), int(shape_raw[1]))

    persons = msg.get("persons") or []
    tracks: list[dict[str, Any]] = []
    for person in persons:
        try:
            tracks.append(_normalize_person(person))
        except AdapterError as exc:
            logger.warning("Skipping person in frame %d: %s", frame_idx, exc)

    return ParsedFrame(
        frame_idx=frame_idx,
        timestamp=pts_timestamp,
        pts_timestamp=pts_timestamp,
        camera_id=camera_id,
        organization_id=organization_id,
        frame_shape=frame_shape,
        tracks=tracks,
    )


__all__ = ["AdapterError", "ParsedFrame", "parse_ai_detection"]
