# Vision Assist PoC

Single-page FastAPI app that streams webcam frames to a backend, runs YOLOv8s + Depth Anything v2 small, and overlays object labels with distance estimates.

## Setup

1. Create and activate a virtual environment.
2. Install PyTorch for your system (CPU or GPU) from https://pytorch.org/get-started/locally/.
3. Install project dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and click "Start camera".

## Calibration

Distances are computed as:

$$
\text{distance}_m = \text{depth\_value} \times \text{depth\_scale}
$$

Set `DEPTH_SCALE` via environment variable or POST JSON to the calibration endpoint:

```bash
curl -X POST http://localhost:8000/api/calibration \
  -H "Content-Type: application/json" \
  -d "{\"depth_scale\": 1.0}"
```

You can derive `depth_scale` by measuring a known distance and comparing it with a depth value reported in the WebSocket response.

## Configuration

Environment variables:

- `YOLO_WEIGHTS` (default `yolov8s.pt`)
- `DEPTH_MODEL_ID` (default `depth-anything/Depth-Anything-V2-Small-hf`)
- `YOLO_CONF` (default `0.25`)
- `MAX_DET` (default `20`)
- `DEPTH_SCALE` (default `1.0`)
- `DEVICE` (optional, e.g. `cuda`, `cpu`, `cuda:0`)
