"""Serialize pipeline events to client ``behavior_cues`` JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

BEHAVIOR_CUES_TYPE = "behavior_cues"
BEHAVIOR_CUES_FAMILY = "aggressive_interaction"


def frame_id(camera_id: int, frame: int) -> str:
    return f"cam{camera_id}_f{frame}"


def event_to_behavior_cue(
    event: dict[str, Any],
    *,
    camera_id: int,
    organization_id: int,
    fps: float,
    window_size: int,
    action_model: str,
    keypoint_model: str,
    module: str,
    module_version: str,
) -> dict[str, Any]:
    """Map one internal pipeline event to a behavior_cues delivery record."""
    mode = event["mode"]
    timestamp = round(float(event["timestamp"]), 3)
    frame = int(event["frame"])

    if mode == "pair":
        track_id = int(event["track_a"])
        detector_signals: dict[str, Any] = {
            "action_recognition": {
                "mode": "pair",
                "track_b": int(event["track_b"]),
                "ntu_class_id": int(event["ntu_class_id"]),
                "ntu_label": event["ntu_label"],
                "action_model": action_model,
            }
        }
    else:
        track_id = int(event["track_id"])
        detector_signals = {
            "action_recognition": {
                "mode": "single",
                "ntu_class_id": int(event["ntu_class_id"]),
                "ntu_label": event["ntu_label"],
                "action_model": action_model,
            }
        }

    window_seconds = round(window_size / max(float(fps), 1e-6), 3)

    return {
        "type": BEHAVIOR_CUES_TYPE,
        "family": BEHAVIOR_CUES_FAMILY,
        "camera_id": camera_id,
        "organization_id": organization_id,
        "track_id": track_id,
        "timestamp": timestamp,
        "pts_timestamp": timestamp,
        "frame_id": frame_id(camera_id, frame),
        "cues": [
            {
                "code": event["target_class"],
                "confidence": round(float(event["confidence"]), 4),
            }
        ],
        "context_hints": [],
        "metadata": {
            "module": module,
            "module_version": module_version,
            "window_seconds": window_seconds,
            "keypoint_model": keypoint_model,
            "detector_signals": detector_signals,
        },
    }


def events_to_behavior_cues(
    events: list[dict[str, Any]],
    *,
    camera_id: int,
    organization_id: int,
    fps: float,
    window_size: int,
    action_model: str,
    keypoint_model: str,
    module: str,
    module_version: str,
) -> list[dict[str, Any]]:
    return [
        event_to_behavior_cue(
            event,
            camera_id=camera_id,
            organization_id=organization_id,
            fps=fps,
            window_size=window_size,
            action_model=action_model,
            keypoint_model=keypoint_model,
            module=module,
            module_version=module_version,
        )
        for event in events
    ]


def write_behavior_cues_jsonl(
    path: str | Path,
    records: list[dict[str, Any]],
    *,
    stream: TextIO | None = None,
) -> None:
    """Write one JSON object per line (JSONL)."""
    if stream is not None:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        write_behavior_cues_jsonl(path, records, stream=f)


__all__ = [
    "BEHAVIOR_CUES_FAMILY",
    "BEHAVIOR_CUES_TYPE",
    "event_to_behavior_cue",
    "events_to_behavior_cues",
    "frame_id",
    "write_behavior_cues_jsonl",
]
