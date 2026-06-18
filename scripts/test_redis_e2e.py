"""End-to-end smoke test for the multi-camera PoseC3D Redis service.

Publishes synthetic ai_detection frames for multiple cameras, then listens
for behavior_cues on the output channels.

Usage (from host, while docker compose is running):

    pip install redis
    python scripts/test_redis_e2e.py

Or with custom settings:

    python scripts/test_redis_e2e.py --org 1 --device edge-node-01 --cameras 4,5 --frames 60
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time

import redis


def build_keypoints_pair(frame_idx: int, cam_offset: int = 0) -> list[dict]:
    """Generate two persons with COCO-17 keypoints simulating close interaction."""
    t = frame_idx * 0.1
    jitter = lambda: random.uniform(-3, 3)

    base_a_x, base_a_y = 460.0 + cam_offset, 400.0
    base_b_x, base_b_y = 520.0 + cam_offset, 400.0

    offset_a = math.sin(t) * 15
    offset_b = math.cos(t) * 15

    def make_coco17(cx: float, cy: float, offset: float) -> list[list[float]]:
        o = offset
        return [
            [cx + jitter(),       cy - 190 + o, 0.98],
            [cx + 5 + jitter(),   cy - 200 + o, 0.97],
            [cx - 5 + jitter(),   cy - 198 + o, 0.96],
            [cx + 15 + jitter(),  cy - 195 + o, 0.94],
            [cx - 15 + jitter(),  cy - 197 + o, 0.93],
            [cx + 40 + jitter(),  cy - 155 + o, 0.92],
            [cx - 40 + jitter(),  cy - 150 + o, 0.91],
            [cx + 55 + jitter(),  cy - 90 + o,  0.90],
            [cx - 55 + jitter(),  cy - 85 + o,  0.89],
            [cx + 60 + o,         cy - 35 + jitter(), 0.88],
            [cx - 60 - o,         cy - 30 + jitter(), 0.87],
            [cx + 20 + jitter(),  cy + 20 + o,  0.92],
            [cx - 20 + jitter(),  cy + 22 + o,  0.91],
            [cx + 25 + jitter(),  cy + 120 + o, 0.90],
            [cx - 25 + jitter(),  cy + 125 + o, 0.89],
            [cx + 30 + jitter(),  cy + 210 + o, 0.88],
            [cx - 30 + jitter(),  cy + 215 + o, 0.87],
        ]

    person_a = {
        "track_id": 7,
        "confidence": 0.91,
        "bounding_box": [base_a_x - 60, base_a_y - 210, base_a_x + 70, base_a_y + 220],
        "pose_keypoints": make_coco17(base_a_x, base_a_y, offset_a),
    }
    person_b = {
        "track_id": 12,
        "confidence": 0.87,
        "bounding_box": [base_b_x - 60, base_b_y - 210, base_b_x + 70, base_b_y + 220],
        "pose_keypoints": make_coco17(base_b_x, base_b_y, offset_b),
    }
    return [person_a, person_b]


def make_ai_detection(
    frame_idx: int,
    org_id: int,
    camera_id: int,
    fps: float = 30.0,
) -> dict:
    ts = frame_idx / fps
    return {
        "type": "ai_detection",
        "camera_id": camera_id,
        "organization_id": org_id,
        "timestamp": round(ts, 4),
        "pts_timestamp": round(ts, 4),
        "frame_id": frame_idx,
        "frame_sequence": frame_idx,
        "frame_shape": [1080, 1920, 3],
        "persons": build_keypoints_pair(frame_idx, cam_offset=camera_id * 10),
    }


def subscriber_thread(
    r: redis.Redis,
    pattern: str,
    received: list,
    stop_event: threading.Event,
) -> None:
    ps = r.pubsub()
    ps.psubscribe(pattern)
    while not stop_event.is_set():
        msg = ps.get_message(ignore_subscribe_messages=True, timeout=0.5)
        if msg and msg["type"] == "pmessage":
            cue = json.loads(msg["data"])
            received.append(cue)
            code = cue["cues"][0]["code"]
            conf = cue["confidence"]
            family = cue["family"]
            track = cue["track_id"]
            channel = msg.get("channel", "")
            print(
                f"  >> CUE on {channel}: {code} (family={family}, "
                f"track={track}, conf={conf:.4f})"
            )
    ps.punsubscribe()
    ps.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E smoke test for multi-camera PoseC3D service")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--org", type=int, default=1, help="organization_id")
    parser.add_argument("--device", default="edge-node-01", help="device_id")
    parser.add_argument("--cameras", default="4,5", help="Comma-separated camera IDs")
    parser.add_argument("--frames", type=int, default=60, help="Frames per camera")
    parser.add_argument("--fps", type=float, default=30.0, help="Simulated FPS")
    parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames (seconds)")
    args = parser.parse_args()

    camera_ids = [int(c.strip()) for c in args.cameras.split(",")]

    sub_pattern = f"org:{args.org}:device:{args.device}:behavior_cues:*"

    print(f"Redis:     {args.host}:{args.port}")
    print(f"Cameras:   {camera_ids}")
    print(f"Listening: {sub_pattern}")
    print(f"Frames:    {args.frames} per camera @ {args.delay:.3f}s delay")
    print()

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    r.ping()
    print("Redis connected\n")

    received: list[dict] = []
    stop = threading.Event()
    listener = threading.Thread(
        target=subscriber_thread,
        args=(
            redis.Redis(host=args.host, port=args.port, decode_responses=True),
            sub_pattern, received, stop,
        ),
        daemon=True,
    )
    listener.start()
    time.sleep(0.5)

    for cam_id in camera_ids:
        input_channel = f"org:{args.org}:device:{args.device}:base_detection:{cam_id}"
        print(f"Publishing {args.frames} frames to {input_channel}...")
        for i in range(args.frames):
            msg = make_ai_detection(i, args.org, cam_id, args.fps)
            r.publish(input_channel, json.dumps(msg))
            if (i + 1) % 10 == 0:
                print(f"  [cam:{cam_id}] Published frame {i + 1}/{args.frames}")
            time.sleep(args.delay)
        print()

    print(f"All frames published for {len(camera_ids)} camera(s).")
    print("Waiting 10s for remaining cues...\n")
    time.sleep(10)

    stop.set()
    listener.join(timeout=3)

    print("=" * 60)
    print(f"RESULTS: {len(received)} cue(s) received across all cameras")
    print("=" * 60)

    if received:
        for i, cue in enumerate(received, 1):
            print(f"\n--- Cue {i} ---")
            print(json.dumps(cue, indent=2))
    else:
        print("\nNo cues emitted.")
        print("This may be normal if:")
        print("  - Buffer not full yet (WINDOW_SIZE frames needed per track)")
        print("  - Inference stride not reached")
        print("  - Confidence below EVENT_MIN_CONFIDENCE")
        print("  - Motion gate filtered the prediction")
        print(f"\nTry increasing --frames (e.g. --frames 120)")


if __name__ == "__main__":
    main()
