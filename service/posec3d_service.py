"""Live PoseC3D cue-detection service.

Subscribes to MS-1 ``ai_detection`` messages on Redis, runs the existing
action-recognition pipeline frame-by-frame, and publishes ``behavior_cues``
back to Redis using Kenneth's channel schema.

Channel schema
--------------
Subscribe (cloud):  ``org:{ORGANIZATION_ID}:base_detection:{CAMERA_ID}``
Subscribe (edge):   ``org:{ORGANIZATION_ID}:device:{DEVICE_ID}:base_detection:{CAMERA_ID}``
Publish:            ``org:{ORGANIZATION_ID}:behavior_cues:{family}:{CAMERA_ID}``

Environment variables
---------------------
REDIS_HOST          Redis hostname              (default: ``localhost``)
REDIS_PORT          Redis port                  (default: ``6379``)
REDIS_PASSWORD      Redis password              (default: empty)
ORGANIZATION_ID     Organization id             (required)
CAMERA_ID           Camera id                   (required)
DEVICE_ID           Edge device id              (default: empty = cloud mode)
DEVICE              ``cuda`` or ``cpu``         (default: ``cuda``)
ACTION_MODEL        ``posec3d`` or ``ctrgcn``   (default: ``posec3d``)
POSEC3D_WEIGHTS     Override checkpoint path    (default: auto-resolved)
WINDOW_SIZE         Skeleton buffer length      (default: ``100``)
INFERENCE_STRIDE    Frames between inferences   (default: ``15``)
LOG_LEVEL           Python log level            (default: ``INFO``)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from typing import Any

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SERVICE_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import redis

from video_processor import VideoProcessor
from utils.behavior_cues import event_to_behavior_cue, family_for_cue
from service.redis_adapter import AdapterError, ParsedFrame, parse_ai_detection

logger = logging.getLogger("posec3d_service")


# ---------------------------------------------------------------------------
# Channel builders (Kenneth's routing contract)
# ---------------------------------------------------------------------------

def build_subscribe_channel(
    organization_id: int,
    camera_id: int,
    device_id: str | None = None,
) -> str:
    """Build the Redis channel to subscribe for ``ai_detection`` messages.

    Cloud: ``org:{organization_id}:base_detection:{camera_id}``
    Edge:  ``org:{organization_id}:device:{device_id}:base_detection:{camera_id}``
    """
    if device_id:
        return f"org:{organization_id}:device:{device_id}:base_detection:{camera_id}"
    return f"org:{organization_id}:base_detection:{camera_id}"


def build_publish_channel(
    organization_id: int,
    family: str,
    camera_id: int,
) -> str:
    """Build the Redis channel to publish ``behavior_cues``.

    ``org:{organization_id}:behavior_cues:{family}:{camera_id}``
    """
    return f"org:{organization_id}:behavior_cues:{family}:{camera_id}"


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_required(name: str) -> str:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        raise RuntimeError(f"Required environment variable {name} is not set")
    return val


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PoseC3DService:
    """Long-running service that bridges Redis and VideoProcessor."""

    def __init__(self) -> None:
        self._running = False

        self.redis_host = _env("REDIS_HOST", "localhost")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_password = _env("REDIS_PASSWORD", "") or None

        self.organization_id = int(_env_required("ORGANIZATION_ID"))
        self.camera_id = int(_env_required("CAMERA_ID"))
        self.device_id = _env("DEVICE_ID", "") or None

        self.subscribe_channel = build_subscribe_channel(
            self.organization_id,
            self.camera_id,
            self.device_id,
        )

        device = _env("DEVICE", "cuda")
        action_model = _env("ACTION_MODEL", "posec3d")
        posec3d_weights = _env("POSEC3D_WEIGHTS", "") or None
        window_size = _env_int("WINDOW_SIZE", 100)
        inference_stride = _env_int("INFERENCE_STRIDE", 15)
        event_min_confidence = _env_float("EVENT_MIN_CONFIDENCE", 0.12)
        interaction_distance = _env_float("INTERACTION_DISTANCE", 300.0)
        interaction_frames = _env_int("INTERACTION_FRAMES", 15)

        logger.info("Initializing VideoProcessor (model=%s, device=%s)", action_model, device)
        self.processor = VideoProcessor(
            action_model=action_model,
            posec3d_weights_path=posec3d_weights,
            device=device,
            window_size=window_size,
            inference_stride=inference_stride,
            event_min_confidence=event_min_confidence,
            interaction_distance=interaction_distance,
            interaction_frames=interaction_frames,
            input_mode="keypoints",
        )

        self._events: list[dict[str, Any]] = []
        self._raw_predictions: list[dict[str, Any]] = []
        self._seen_event_keys: set[tuple] = set()
        self._pair_diag: Any = None
        self._motion_diag: Any = None
        self._predict_kwargs: dict[str, Any] = {}
        self._prev_event_count = 0

    def _init_run_state(self) -> None:
        """Replicate the initialization that ``process_keypoints()`` does."""
        (
            self._events,
            self._raw_predictions,
            self._seen_event_keys,
            self._pair_diag,
            self._motion_diag,
        ) = self.processor._reset_run_state()
        self._prev_event_count = 0

    def _load_model(self) -> None:
        logger.info("Loading PoseC3D model...")
        self.processor.recognizer.load_model()
        logger.info("Model loaded: %s", self.processor.recognizer.get_model_name())

    def _connect_redis(self) -> redis.Redis:
        logger.info(
            "Connecting to Redis %s:%d (sub=%s)",
            self.redis_host,
            self.redis_port,
            self.subscribe_channel,
        )
        r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=True,
        )
        r.ping()
        logger.info("Redis connected")
        return r

    def _handle_frame(
        self,
        frame: ParsedFrame,
        pub: redis.Redis,
    ) -> None:
        """Run one frame through the pipeline and publish any new events."""
        self.processor._frame_shape = frame.frame_shape
        self._predict_kwargs["img_shape"] = frame.frame_shape

        self.processor._process_frame(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            tracks=frame.tracks,
            events=self._events,
            raw_predictions=self._raw_predictions,
            seen_event_keys=self._seen_event_keys,
            pair_validation_diag=self._pair_diag,
            motion_gate_diag=self._motion_diag,
            predict_kwargs=self._predict_kwargs,
        )

        new_events = self._events[self._prev_event_count:]
        self._prev_event_count = len(self._events)

        for event in new_events:
            cue = event_to_behavior_cue(
                event,
                camera_id=self.camera_id,
                organization_id=self.organization_id,
                pts_timestamp_override=frame.pts_timestamp,
            )
            family = cue["family"]
            channel = build_publish_channel(
                self.organization_id, family, self.camera_id,
            )
            payload = json.dumps(cue, separators=(",", ":"))
            pub.publish(channel, payload)
            logger.info(
                "Published %s -> %s (track=%s, conf=%.4f, frame=%d)",
                cue["cues"][0]["code"],
                channel,
                cue["track_id"],
                cue["confidence"],
                frame.frame_idx,
            )

    def run(self) -> None:
        """Main service loop -- blocks until SIGINT / SIGTERM."""
        self._load_model()
        self._init_run_state()

        conn = self._connect_redis()
        pubsub = conn.pubsub()
        pubsub.subscribe(self.subscribe_channel)
        logger.info("Subscribed to '%s' -- waiting for messages", self.subscribe_channel)

        self._running = True

        def _shutdown(sig: int, _frame: Any) -> None:
            logger.info("Received signal %d, shutting down...", sig)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        frames_processed = 0
        errors = 0

        try:
            while self._running:
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    continue

                if msg["type"] != "message":
                    continue

                try:
                    frame = parse_ai_detection(msg["data"])
                except AdapterError as exc:
                    errors += 1
                    logger.warning("Adapter error (#%d): %s", errors, exc)
                    continue

                try:
                    self._handle_frame(frame, conn)
                    frames_processed += 1
                    if frames_processed % 500 == 0:
                        logger.info(
                            "Processed %d frames (%d events, %d errors)",
                            frames_processed,
                            len(self._events),
                            errors,
                        )
                except Exception:
                    errors += 1
                    logger.exception("Error processing frame %d", frame.frame_idx)

        finally:
            pubsub.unsubscribe()
            pubsub.close()
            conn.close()
            logger.info(
                "Service stopped. Frames=%d, events=%d, errors=%d",
                frames_processed,
                len(self._events),
                errors,
            )


def main() -> None:
    log_level = _env("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    svc = PoseC3DService()
    svc.run()


if __name__ == "__main__":
    main()
