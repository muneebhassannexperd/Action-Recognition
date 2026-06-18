"""Multi-camera PoseC3D cue-detection service.

One container per edge node.  On startup the service fetches assigned cameras
from MS-3, loads the PoseC3D model **once**, and creates isolated per-camera
processing state.  New cameras can be hot-added / removed at runtime via
the ``device:{DEVICE_ID}:commands`` Redis channel.

Channel schema
--------------
Subscribe:  ``org:{ORG}:device:{DEV}:base_detection:*``   (pattern)
Publish:    ``org:{ORG}:device:{DEV}:behavior_cues:{family}:{cam}``
Commands:   ``device:{DEV}:commands``

Environment variables
---------------------
REDIS_HOST          Redis hostname              (default: ``localhost``)
REDIS_PORT          Redis port                  (default: ``6379``)
REDIS_PASSWORD      Redis password              (default: empty)
ORGANIZATION_ID     Organization id             (required)
DEVICE_ID           Edge device id              (required)
MS3_URL             MS-3 API base URL           (required, e.g. ``http://ms3:8000``)
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
from dataclasses import dataclass, field
from typing import Any

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SERVICE_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import redis
import requests

from video_processor import VideoProcessor
from utils.behavior_cues import event_to_behavior_cue, family_for_cue
from service.redis_adapter import AdapterError, ParsedFrame, parse_ai_detection

logger = logging.getLogger("posec3d_service")


# ---------------------------------------------------------------------------
# Channel builders (Kenneth's routing contract)
# ---------------------------------------------------------------------------

def build_subscribe_pattern(
    organization_id: int,
    device_id: str,
) -> str:
    """Wildcard pattern for ``psubscribe``."""
    return f"org:{organization_id}:device:{device_id}:base_detection:*"


def build_publish_channel(
    organization_id: int,
    device_id: str,
    family: str,
    camera_id: int,
) -> str:
    return f"org:{organization_id}:device:{device_id}:behavior_cues:{family}:{camera_id}"


def build_command_channel(device_id: str) -> str:
    return f"device:{device_id}:commands"


def extract_camera_id_from_channel(channel: str) -> int:
    """Extract the trailing camera_id from ``...base_detection:{camera_id}``."""
    return int(channel.rsplit(":", 1)[-1])


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
# MS-3 camera discovery
# ---------------------------------------------------------------------------

def fetch_cameras_from_ms3(ms3_url: str, device_id: str) -> list[int]:
    """GET /api/edge/devices/{device_id}/cameras -> list of camera ids."""
    url = f"{ms3_url.rstrip('/')}/api/edge/devices/{device_id}/cameras"
    logger.info("Fetching cameras from MS-3: %s", url)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    camera_ids = resp.json()
    if not isinstance(camera_ids, list):
        raise RuntimeError(f"MS-3 returned unexpected type: {type(camera_ids)}")
    camera_ids = [int(c) for c in camera_ids]
    logger.info("MS-3 returned %d camera(s): %s", len(camera_ids), camera_ids)
    return camera_ids


# ---------------------------------------------------------------------------
# Per-camera state
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    """Isolated processing state for a single camera stream."""

    camera_id: int
    processor: VideoProcessor
    events: list[dict[str, Any]] = field(default_factory=list)
    raw_predictions: list[dict[str, Any]] = field(default_factory=list)
    seen_event_keys: set[tuple] = field(default_factory=set)
    pair_diag: Any = None
    motion_diag: Any = None
    predict_kwargs: dict[str, Any] = field(default_factory=dict)
    prev_event_count: int = 0
    frames_processed: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PoseC3DService:
    """Multi-camera orchestrator.  Loads model once, manages per-camera state."""

    def __init__(self) -> None:
        self._running = False

        self.redis_host = _env("REDIS_HOST", "localhost")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_password = _env("REDIS_PASSWORD", "") or None

        self.organization_id = int(_env_required("ORGANIZATION_ID"))
        self.device_id = _env_required("DEVICE_ID")
        self.ms3_url = _env("MS3_URL", "") or None
        self._fallback_camera_ids = _env("CAMERA_IDS", "") or None

        self._device = _env("DEVICE", "cuda")
        self._action_model = _env("ACTION_MODEL", "posec3d")
        self._posec3d_weights = _env("POSEC3D_WEIGHTS", "") or None
        self._window_size = _env_int("WINDOW_SIZE", 100)
        self._inference_stride = _env_int("INFERENCE_STRIDE", 15)
        self._event_min_confidence = _env_float("EVENT_MIN_CONFIDENCE", 0.12)
        self._interaction_distance = _env_float("INTERACTION_DISTANCE", 300.0)
        self._interaction_frames = _env_int("INTERACTION_FRAMES", 15)

        self.subscribe_pattern = build_subscribe_pattern(
            self.organization_id, self.device_id,
        )
        self.command_channel = build_command_channel(self.device_id)

        self._cameras: dict[int, CameraState] = {}
        self._shared_recognizer: Any = None

    # -- camera lifecycle ---------------------------------------------------

    def _create_processor(self) -> VideoProcessor:
        """Create a fresh VideoProcessor.

        If a shared recognizer already exists, swap it in so the heavy model
        weights are loaded only once.
        """
        proc = VideoProcessor(
            action_model=self._action_model,
            posec3d_weights_path=self._posec3d_weights,
            device=self._device,
            window_size=self._window_size,
            inference_stride=self._inference_stride,
            event_min_confidence=self._event_min_confidence,
            interaction_distance=self._interaction_distance,
            interaction_frames=self._interaction_frames,
            input_mode="keypoints",
        )
        if self._shared_recognizer is not None:
            proc.recognizer = self._shared_recognizer
        return proc

    def add_camera(self, camera_id: int) -> None:
        if camera_id in self._cameras:
            logger.info("Camera %d already active, skipping", camera_id)
            return

        logger.info("Adding camera %d", camera_id)
        proc = self._create_processor()

        if self._shared_recognizer is None:
            self._shared_recognizer = proc.recognizer

        state = CameraState(camera_id=camera_id, processor=proc)
        (
            state.events,
            state.raw_predictions,
            state.seen_event_keys,
            state.pair_diag,
            state.motion_diag,
        ) = proc._reset_run_state()
        self._cameras[camera_id] = state
        logger.info("Camera %d active (total cameras: %d)", camera_id, len(self._cameras))

    def remove_camera(self, camera_id: int) -> None:
        state = self._cameras.pop(camera_id, None)
        if state is None:
            logger.warning("remove_camera: camera %d not found", camera_id)
            return
        logger.info(
            "Removed camera %d (frames=%d, events=%d, errors=%d). Active cameras: %d",
            camera_id,
            state.frames_processed,
            len(state.events),
            state.errors,
            len(self._cameras),
        )

    # -- model loading ------------------------------------------------------

    def _load_model(self) -> None:
        if self._shared_recognizer is None:
            dummy = self._create_processor()
            self._shared_recognizer = dummy.recognizer

        logger.info("Loading PoseC3D model...")
        self._shared_recognizer.load_model()
        logger.info("Model loaded: %s", self._shared_recognizer.get_model_name())

    # -- redis --------------------------------------------------------------

    def _connect_redis(self) -> redis.Redis:
        logger.info("Connecting to Redis %s:%d", self.redis_host, self.redis_port)
        r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=True,
        )
        r.ping()
        logger.info("Redis connected")
        return r

    # -- frame handling -----------------------------------------------------

    def _handle_frame(self, frame: ParsedFrame, pub: redis.Redis) -> None:
        camera_id = frame.camera_id

        state = self._cameras.get(camera_id)
        if state is None:
            return

        state.processor._frame_shape = frame.frame_shape
        state.predict_kwargs["img_shape"] = frame.frame_shape

        state.processor._process_frame(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            tracks=frame.tracks,
            events=state.events,
            raw_predictions=state.raw_predictions,
            seen_event_keys=state.seen_event_keys,
            pair_validation_diag=state.pair_diag,
            motion_gate_diag=state.motion_diag,
            predict_kwargs=state.predict_kwargs,
        )

        new_events = state.events[state.prev_event_count:]
        state.prev_event_count = len(state.events)
        state.frames_processed += 1

        for event in new_events:
            cue = event_to_behavior_cue(
                event,
                camera_id=camera_id,
                organization_id=self.organization_id,
                pts_timestamp_override=frame.pts_timestamp,
            )
            family = cue["family"]
            channel = build_publish_channel(
                self.organization_id, self.device_id, family, camera_id,
            )
            payload = json.dumps(cue, separators=(",", ":"))
            pub.publish(channel, payload)
            logger.info(
                "[cam:%d] Published %s -> %s (track=%s, conf=%.4f, frame=%d)",
                camera_id,
                cue["cues"][0]["code"],
                channel,
                cue["track_id"],
                cue["confidence"],
                frame.frame_idx,
            )

        if state.frames_processed % 500 == 0:
            logger.info(
                "[cam:%d] Processed %d frames (%d events, %d errors)",
                camera_id,
                state.frames_processed,
                len(state.events),
                state.errors,
            )

    # -- command handling ---------------------------------------------------

    def _handle_command(self, raw_data: str) -> None:
        """Handle add_camera / remove_camera commands."""
        try:
            cmd = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        except json.JSONDecodeError:
            logger.warning("Invalid command JSON: %s", raw_data[:200])
            return

        action = cmd.get("action", "").lower()
        cam_id = cmd.get("camera_id")

        if cam_id is None:
            logger.warning("Command missing camera_id: %s", cmd)
            return

        cam_id = int(cam_id)

        if action == "add_camera":
            self.add_camera(cam_id)
        elif action == "remove_camera":
            self.remove_camera(cam_id)
        else:
            logger.warning("Unknown command action: %s", action)

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        """Main service loop -- blocks until SIGINT / SIGTERM."""
        self._load_model()

        if self.ms3_url:
            camera_ids = fetch_cameras_from_ms3(self.ms3_url, self.device_id)
        elif self._fallback_camera_ids:
            camera_ids = [int(c.strip()) for c in self._fallback_camera_ids.split(",")]
            logger.info("Using CAMERA_IDS fallback: %s", camera_ids)
        else:
            camera_ids = []

        for cam_id in camera_ids:
            self.add_camera(cam_id)

        if not self._cameras:
            logger.warning("No cameras assigned -- waiting for add_camera commands")

        conn = self._connect_redis()
        pubsub = conn.pubsub()

        pubsub.psubscribe(self.subscribe_pattern)
        logger.info("Subscribed to pattern '%s'", self.subscribe_pattern)

        pubsub.subscribe(self.command_channel)
        logger.info("Subscribed to command channel '%s'", self.command_channel)

        self._running = True

        def _shutdown(sig: int, _frame: Any) -> None:
            logger.info("Received signal %d, shutting down...", sig)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        total_frames = 0
        total_errors = 0

        try:
            while self._running:
                msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg is None:
                    continue

                msg_type = msg["type"]

                if msg_type == "pmessage":
                    channel = msg.get("channel", "")
                    try:
                        frame = parse_ai_detection(msg["data"])
                    except AdapterError as exc:
                        total_errors += 1
                        logger.warning("Adapter error (#%d): %s", total_errors, exc)
                        continue
                    try:
                        self._handle_frame(frame, conn)
                        total_frames += 1
                    except Exception:
                        total_errors += 1
                        logger.exception("Error processing frame")

                elif msg_type == "message" and msg.get("channel") == self.command_channel:
                    self._handle_command(msg["data"])

        finally:
            pubsub.punsubscribe()
            pubsub.unsubscribe()
            pubsub.close()
            conn.close()

            cam_summary = ", ".join(
                f"cam:{s.camera_id}={s.frames_processed}f/{len(s.events)}e"
                for s in self._cameras.values()
            )
            logger.info(
                "Service stopped. Total frames=%d, errors=%d. Per-camera: [%s]",
                total_frames,
                total_errors,
                cam_summary,
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
