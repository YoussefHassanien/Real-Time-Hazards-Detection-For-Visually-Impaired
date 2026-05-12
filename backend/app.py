import os
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .inference import InferenceService

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s.pt")
DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL_ID",
                           "depth-anything/Depth-Anything-V2-Small-hf")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
MAX_DET = int(os.getenv("MAX_DET", "20"))
DEPTH_SCALE = float(os.getenv("DEPTH_SCALE", "0.4"))
DEVICE = os.getenv("DEVICE")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = InferenceService(
        yolo_weights=YOLO_WEIGHTS,
        depth_model_id=DEPTH_MODEL_ID,
        yolo_conf=YOLO_CONF,
        max_det=MAX_DET,
        device=DEVICE,
    )
    yield


app = FastAPI(title="Vision Assist PoC", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

STATE: Dict[str, Any] = {"depth_scale": DEPTH_SCALE}


@app.get("/")
def index() -> FileResponse:
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    service: InferenceService = app.state.service
    frame_id = 0

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is None:
                continue

            frame_id += 1
            result = service.predict_from_bytes(
                message["bytes"],
                depth_scale=STATE["depth_scale"],
            )
            result["frame_id"] = frame_id
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc))


@app.exception_handler(FileNotFoundError)
def file_not_found_handler(_, __) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": "file_not_found"})


if __name__ == "__main__":
    working_directory = os.getcwd()
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[working_directory],
        reload_includes=["*.py", "*.js", "*.html", "*.css"],
    )
