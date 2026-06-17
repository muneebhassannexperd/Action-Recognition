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
    source_service: str = "nex_posec3d",
    pts_timestamp_override: float | None = None,
) -> dict[str, Any]:
    """Map one internal pipeline event to a behavior_cues delivery record."""
    timestamp = round(float(event["timestamp"]), 3)
    pts_ts = pts_timestamp_override if pts_timestamp_override is not None else timestamp
    frame = int(event["frame"])
    confidence = round(float(event["confidence"]), 4)

    if event["mode"] == "pair":
        track_id = int(event["track_a"])
    else:
        track_id = int(event["track_id"])

    return {
        "type": BEHAVIOR_CUES_TYPE,
        "family": BEHAVIOR_CUES_FAMILY,
        "camera_id": camera_id,
        "organization_id": organization_id,
        "track_id": track_id,
        "timestamp": timestamp,
        "pts_timestamp": pts_ts,
        "frame_id": frame_id(camera_id, frame),
        "alert_triggered": True,
        "confidence": confidence,
        "cues": [
            {
                "code": event["target_class"],
                "confidence": confidence,
            }
        ],
        "context_hints": [],
        "metadata": {
            "family": BEHAVIOR_CUES_FAMILY,
            "source_service": source_service,
            "detection_timestamp": timestamp,
            "pts_timestamp": pts_ts,
        },
    }


def events_to_behavior_cues(
    events: list[dict[str, Any]],
    *,
    camera_id: int,
    organization_id: int,
    source_service: str = "nex_posec3d",
) -> list[dict[str, Any]]:
    return [
        event_to_behavior_cue(
            event,
            camera_id=camera_id,
            organization_id=organization_id,
            source_service=source_service,
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
