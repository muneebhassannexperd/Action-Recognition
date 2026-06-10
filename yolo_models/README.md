# YOLO pose weights

Place YOLO pose model files here. They are **not** committed to git.

| File | Purpose |
|------|---------|
| `yolo11l-pose.pt` | Default fallback (Ultralytics pretrained) |
| `best.pt` | Custom fine-tuned weights (used if present) |

Ultralytics will auto-download `yolo11l-pose.pt` on first run if the file is missing. To prefetch:

```bash
mkdir -p yolo_models
curl -L -o yolo_models/yolo11l-pose.pt \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt"
```

Or copy your own `best.pt` into this folder.
