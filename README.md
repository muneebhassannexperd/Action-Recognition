# PoseC3D Action Recognition Service

Inference-only CCTV pipeline for aggressive-interaction cue detection on NTU120 actions, with project-specific class mapping (`push_shove`, `swing_attempt`, `grapple_clinch`, `stumble_recover`, `aggressive_posture`).

Runs as a **multi-camera Docker service** on edge nodes, consuming pose keypoints from MS-1 via Redis and publishing `behavior_cues` back to Redis.

## Architecture

```
MS-1 (YOLO Pose + Tracking)
        │
        ▼  Redis Pub/Sub
┌─────────────────────────────────────────┐
│  PoseC3D Cue Service (this container)   │
│                                         │
│  ┌─ Camera 4 state ──────────────────┐  │
│  │  InteractionDetector              │  │
│  │  SkeletonBuffer                   │  │
│  │  Events / Track state             │  │
│  └───────────────────────────────────┘  │
│  ┌─ Camera 5 state ──────────────────┐  │
│  │  InteractionDetector              │  │
│  │  SkeletonBuffer                   │  │
│  │  Events / Track state             │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌─ Shared PoseC3D Model (loaded 1x) ┐  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
        │
        ▼  Redis Pub/Sub
MS-11 (Behavior Cues Consumer)
```

**Key design:** One container per edge node (not per camera). The PoseC3D model is loaded once and shared across all assigned cameras. Each camera has fully isolated tracking/buffer state.

## Redis Channel Schema

| Direction | Channel |
|-----------|---------|
| Subscribe (input) | `org:{ORG}:device:{DEV}:base_detection:*` (wildcard) |
| Publish (output) | `org:{ORG}:device:{DEV}:behavior_cues:{family}:{camera_id}` |
| Commands | `device:{DEV}:commands` (`add_camera` / `remove_camera`) |

## Camera Discovery

On startup the service fetches assigned cameras from MS-3:

```
GET {MS3_URL}/api/edge/devices/{DEVICE_ID}/cameras  →  [4, 5, 9]
```

At runtime, cameras can be hot-added or removed via the command channel:

```json
{"action": "add_camera", "camera_id": 7}
{"action": "remove_camera", "camera_id": 4}
```

## Quick Start (Docker)

### Build and run locally

```bash
cp .env.example .env
# Edit .env: set DEVICE=cpu, MS3_URL= (empty), CAMERA_IDS=4,5
docker compose up --build
```

### Production deployment (edge with GPU)

```bash
cp .env.example .env
# Edit .env: set ORGANIZATION_ID, DEVICE_ID, MS3_URL, REDIS_HOST
docker compose -f docker-compose.prod.yml up -d
```

### From exported image

```bash
docker load -i posec3d-cue-service.tar
cp .env.example .env
# Edit .env
docker compose -f docker-compose.prod.yml up -d
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ORGANIZATION_ID` | Yes | — | Organization ID for channel routing |
| `DEVICE_ID` | Yes | — | Edge device ID |
| `MS3_URL` | Prod | — | MS-3 API base URL for camera discovery |
| `CAMERA_IDS` | Dev | — | Fallback comma-separated camera IDs (when `MS3_URL` is empty) |
| `REDIS_HOST` | Yes | `localhost` | Redis hostname |
| `REDIS_PORT` | No | `6379` | Redis port |
| `REDIS_PASSWORD` | No | — | Redis password |
| `DEVICE` | No | `cuda` | `cuda` or `cpu` |
| `ACTION_MODEL` | No | `posec3d` | `posec3d` or `ctrgcn` |
| `WINDOW_SIZE` | No | `100` | Skeleton buffer length (frames) |
| `INFERENCE_STRIDE` | No | `15` | Frames between inferences |
| `EVENT_MIN_CONFIDENCE` | No | `0.12` | Min confidence to emit a cue |
| `INTERACTION_DISTANCE` | No | `300.0` | Max pixel distance for pair gating |
| `INTERACTION_FRAMES` | No | `15` | Consecutive close frames before pair inference |
| `LOG_LEVEL` | No | `INFO` | Python log level |

## Input Schema (`ai_detection`)

Messages published by MS-1 on the subscribe channel:

```json
{
  "type": "ai_detection",
  "camera_id": 4,
  "organization_id": 1,
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
```

- `pose_keypoints`: 17 COCO joints in YOLO order
- `frame_shape`: `[height, width, channels]`

## Output Schema (`behavior_cues`)

Published to `org:{ORG}:device:{DEV}:behavior_cues:{family}:{camera_id}`:

```json
{
  "type": "behavior_cues",
  "camera_id": 4,
  "organization_id": 1,
  "track_id": 7,
  "family": "aggressive_interaction",
  "alert_triggered": true,
  "confidence": 0.85,
  "cues": [
    {
      "code": "grapple_clinch",
      "confidence": 0.85
    }
  ],
  "metadata": {
    "family": "aggressive_interaction",
    "source_service": "posec3d_cue_service",
    "detection_timestamp": "2026-06-18T10:37:44.123Z",
    "pts_timestamp": 34.733
  }
}
```

## Detection Cues

| Cue | Inference Path | Family |
|-----|----------------|--------|
| `push_shove` | pair (M=2) | `aggressive_interaction` |
| `swing_attempt` | pair (M=2) | `aggressive_interaction` |
| `grapple_clinch` | pair (M=2) | `aggressive_interaction` |
| `aggressive_posture` | pair (M=2) | `aggressive_interaction` |
| `stumble_recover` | single (M=1) | `aggressive_interaction` |

## Pipeline Flow

```
ai_detection message
    │
    ▼
redis_adapter.py → ParsedFrame
    │
    ▼
VideoProcessor._process_frame()
    ├─ InteractionDetector (pair proximity)
    ├─ SkeletonBuffer (temporal windowing)
    ├─ PairValidator (geometry checks)
    ├─ MotionGate (energy filter)
    ├─ PoseC3D inference (pair M=2 or single M=1)
    └─ Class mapping (NTU120 → cue codes)
    │
    ▼
behavior_cues.py → JSON
    │
    ▼
Redis publish
```

## CLI Mode (Offline Testing)

The pipeline also supports offline CLI usage for local development and testing:

### Video mode

```bash
python src/main.py \
  --video Test-Videos/Grapple-Clinch/grapple_high_7.mp4 \
  --device cpu
```

Outputs: `outputs/<stem>.json` (debug) + `outputs/<stem>_annotated.mp4`

### Keypoints mode

```bash
python src/main.py \
  --input-mode keypoints \
  --keypoints outputs/grapple_high_7_keypoints.jsonl \
  --device cpu
```

Outputs: `outputs/<stem>_cues.jsonl`

## Project Structure

```
service/
  posec3d_service.py      Multi-camera Redis orchestrator
  redis_adapter.py        ai_detection message parser
  __init__.py
src/
  main.py                 CLI entry point
  video_processor.py      Pipeline orchestration
  keypoints_input.py      Keypoints JSONL loader
  pose_detector.py        YOLO pose + tracking (video mode)
  interaction_detector.py Pair proximity gating
  pair_validator.py       Geometric pair validation
  motion_energy.py        Motion-energy gate
  skeleton_buffer.py      Rolling skeleton windows
  joint_mapper.py         COCO-17 → NTU-25 mapping (CTR-GCN)
  recognizers/            PoseC3D and CTR-GCN backends
utils/
  config.py               Thresholds, model paths, defaults
  class_mapping.py        NTU120 → project cue codes
  behavior_cues.py        behavior_cues JSON serializer
scripts/
  download_models.sh      Fetch pretrained weights
  export_keypoints.py     Video → keypoints JSONL simulator
  test_redis_e2e.py       Multi-camera Redis smoke test
data/                     NTU120 label file
pretrained_model/         Checkpoints (gitignored, downloaded during build)
yolo_models/              YOLO pose weights (gitignored)
```

## Model Weights

Weights are downloaded automatically during `docker build`. For local development:

```bash
bash scripts/download_models.sh
```

| Model | Path |
|-------|------|
| PoseC3D NTU120 XSub (default) | `pretrained_model/posec3d/joint.pth` |
| CTR-GCN NTU120 joint | `pretrained_model/CTRGCN_NTU120_CSub_joint_84.9/runs-58-57072.pt` |
| YOLO pose (video mode only) | `yolo_models/yolo11l-pose.pt` or `yolo_models/best.pt` |

## License

Model weights are subject to their original licenses (OpenMMLab/pyskl, CTR-GCN, Ultralytics). Application code is provided as-is for internal use.
