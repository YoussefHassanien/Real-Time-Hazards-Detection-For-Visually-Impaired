import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class InferenceService:

    def __init__(
        self,
        yolo_weights: str,
        depth_model_id: str,
        yolo_conf: float = 0.5,
        max_det: int = 20,
        device: Optional[str] = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.yolo_device = 0 if self.device.type == "cuda" else "cpu"
        self.yolo_conf = yolo_conf
        self.max_det = max_det

        self.yolo = YOLO(yolo_weights)
        self.depth_processor = AutoImageProcessor.from_pretrained(
            depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            depth_model_id)
        self.depth_model.to(self.device)
        self.depth_model.eval()

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def decode_jpeg(self, data: bytes) -> Optional[np.ndarray]:
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return frame

    def predict_from_bytes(self, data: bytes,
                           depth_scale: float) -> Dict[str, Any]:
        frame = self.decode_jpeg(data)
        if frame is None:
            return {"error": "decode_failed"}
        return self.predict(frame, depth_scale)

    def predict(self, frame_bgr: np.ndarray,
                depth_scale: float) -> Dict[str, Any]:
        start_time = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        depth_map = self._predict_depth(frame_bgr)
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        yolo_results = self.yolo.predict(
            source=frame_bgr,
            conf=self.yolo_conf,
            iou=0.7,
            max_det=self.max_det,
            device=self.yolo_device,
            verbose=False,
        )
        result = yolo_results[0]

        detections: List[Dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            clss = result.boxes.cls.detach().cpu().numpy().astype(int)

            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = self._clip_box(box, width, height)
                depth_value = self._median_depth(depth_map, (x1, y1, x2, y2))
                distance_m = self._depth_to_distance(depth_value, depth_scale)
                label = result.names.get(int(cls_id), str(int(cls_id)))

                detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "box": [x1, y1, x2, y2],
                    "depth_value": depth_value,
                    "distance_m": distance_m,
                })

        processing_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "width": width,
            "height": height,
            "processing_ms": round(processing_ms, 2),
            "detections": detections,
        }

    def _predict_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.depth_processor(images=frame_rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        if predicted_depth.dim() == 3:
            predicted_depth = predicted_depth.unsqueeze(1)
        elif predicted_depth.dim() != 4:
            raise RuntimeError("Unexpected depth output shape")

        height, width = frame_bgr.shape[:2]
        depth = torch.nn.functional.interpolate(
            predicted_depth,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        depth = depth.squeeze().detach().cpu().numpy()
        return depth

    def _clip_box(self, box: np.ndarray, width: int,
                  height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(width - 1, round(x1))))
        y1 = int(max(0, min(height - 1, round(y1))))
        x2 = int(max(0, min(width - 1, round(x2))))
        y2 = int(max(0, min(height - 1, round(y2))))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return x1, y1, x2, y2

    def _median_depth(self, depth_map: np.ndarray, box: Tuple[int, int, int,
                                                              int]) -> float:
        x1, y1, x2, y2 = box
        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.median(roi))

    def _depth_to_distance(self, depth_value: float, depth_scale: float) -> Optional[float]:
        if depth_value <= 0.0:
            return None
        return round(depth_scale / depth_value, 3)
