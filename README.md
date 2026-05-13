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
\text{depth\_scale} \over \text{depth\_value}
$$

## Configuration

Environment variables:

- `YOLO_WEIGHTS` (default `best_YOLO.pt`)
- `DEPTH_MODEL_ID` (default `depth-anything/Depth-Anything-V2-Small-hf`)
- `YOLO_CONF` (default `0.5`)
- `MAX_DET` (default `20`)
- `DEPTH_SCALE` (default `1.0`)
- `DEVICE` (optional, e.g. `cuda`, `cpu`, `cuda:0`)
