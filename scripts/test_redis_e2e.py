"""End-to-end smoke test for the PoseC3D Redis service.

Publishes synthetic ai_detection frames to Redis on the subscribe channel,
then listens for behavior_cues on the publish channel.

Usage (from host, while docker compose is running):

    pip install redis
    python scripts/test_redis_e2e.py

Or with custom settings:

    python scripts/test_redis_e2e.py --org 1 --camera 4 --frames 60 --host localhost
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


def build_keypoints_pair(frame_idx: int) -> list[dict]:
    """Generate two persons with COCO-17 keypoints simulating close interaction.

    The keypoints oscillate to simulate motion (not static poses).
    """
    t = frame_idx * 0.1
    jitter = lambda: random.uniform(-3, 3)

    base_a_x, base_a_y = 460.0, 400.0
    base_b_x, base_b_y = 520.0, 400.0

    offset_a = math.sin(t) * 15
    offset_b = math.cos(t) * 15

    def make_coco17(cx: float, cy: float, offset: float) -> list[list[float]]:
        o = offset
        return [
            [cx + jitter(),       cy - 190 + o, 0.98],   # 0  nose
            [cx + 5 + jitter(),   cy - 200 + o, 0.97],   # 1  left_eye
            [cx - 5 + jitter(),   cy - 198 + o, 0.96],   # 2  right_eye
            [cx + 15 + jitter(),  cy - 195 + o, 0.94],   # 3  left_ear
            [cx - 15 + jitter(),  cy - 197 + o, 0.93],   # 4  right_ear
            [cx + 40 + jitter(),  cy - 155 + o, 0.92],   # 5  left_shoulder
            [cx - 40 + jitter(),  cy - 150 + o, 0.91],   # 6  right_shoulder
            [cx + 55 + jitter(),  cy - 90 + o,  0.90],   # 7  left_elbow
            [cx - 55 + jitter(),  cy - 85 + o,  0.89],   # 8  right_elbow
            [cx + 60 + o,         cy - 35 + jitter(), 0.88],   # 9  left_wrist (reaching)
            [cx - 60 - o,         cy - 30 + jitter(), 0.87],   # 10 right_wrist
            [cx + 20 + jitter(),  cy + 20 + o,  0.92],   # 11 left_hip
            [cx - 20 + jitter(),  cy + 22 + o,  0.91],   # 12 right_hip
            [cx + 25 + jitter(),  cy + 120 + o, 0.90],   # 13 left_knee
            [cx - 25 + jitter(),  cy + 125 + o, 0.89],   # 14 right_knee
            [cx + 30 + jitter(),  cy + 210 + o, 0.88],   # 15 left_ankle
            [cx - 30 + jitter(),  cy + 215 + o, 0.87],   # 16 right_ankle
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
        "persons": build_keypoints_pair(frame_idx),
    }


def subscriber_thread(
    r: redis.Redis,
    channel: str,
    received: list,
    stop_event: threading.Event,
) -> None:
    ps = r.pubsub()
    ps.subscribe(channel)
    while not stop_event.is_set():
        msg = ps.get_message(ignore_subscribe_messages=True, timeout=0.5)
        if msg and msg["type"] == "message":
            cue = json.loads(msg["data"])
            received.append(cue)
            code = cue["cues"][0]["code"]
            conf = cue["confidence"]
            family = cue["family"]
            track = cue["track_id"]
            print(
                f"  >> CUE RECEIVED: {code} (family={family}, "
                f"track={track}, conf={conf:.4f})"
            )
    ps.unsubscribe()
    ps.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E smoke test for PoseC3D Redis service")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--org", type=int, default=1, help="organization_id")
    parser.add_argument("--camera", type=int, default=4, help="camera_id")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to publish")
    parser.add_argument("--fps", type=float, default=30.0, help="Simulated FPS")
    parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames (seconds)")
    args = parser.parse_args()

    sub_channel = f"org:{args.org}:base_detection:{args.camera}"
    pub_channel = f"org:{args.org}:behavior_cues:aggressive_interaction:{args.camera}"

    print(f"Redis:     {args.host}:{args.port}")
    print(f"Input:     {sub_channel}")
    print(f"Listening: {pub_channel}")
    print(f"Frames:    {args.frames} @ {args.delay:.3f}s delay")
    print()

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    r.ping()
    print("Redis connected\n")

    received: list[dict] = []
    stop = threading.Event()
    listener = threading.Thread(
        target=subscriber_thread,
        args=(redis.Redis(host=args.host, port=args.port, decode_responses=True),
              pub_channel, received, stop),
        daemon=True,
    )
    listener.start()

    time.sleep(0.5)

    print(f"Publishing {args.frames} frames to {sub_channel}...")
    for i in range(args.frames):
        msg = make_ai_detection(i, args.org, args.camera, args.fps)
        r.publish(sub_channel, json.dumps(msg))
        if (i + 1) % 10 == 0:
            print(f"  Published frame {i + 1}/{args.frames}")
        time.sleep(args.delay)

    print(f"\nAll {args.frames} frames published.")
    print("Waiting 10s for remaining cues...\n")
    time.sleep(10)

    stop.set()
    listener.join(timeout=3)

    print("=" * 60)
    print(f"RESULTS: {len(received)} cue(s) received")
    print("=" * 60)

    if received:
        for i, cue in enumerate(received, 1):
            print(f"\n--- Cue {i} ---")
            print(json.dumps(cue, indent=2))
    else:
        print("\nNo cues emitted.")
        print("This may be normal if:")
        print(f"  - Buffer not full yet (WINDOW_SIZE frames needed per track)")
        print(f"  - Inference stride not reached")
        print(f"  - Confidence below EVENT_MIN_CONFIDENCE")
        print(f"  - Motion gate filtered the prediction")
        print(f"\nTry increasing --frames (e.g. --frames 120)")


if __name__ == "__main__":
    main()
