"""Vehicle detection module using YOLOv8 when available."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

from config import Config


class VehicleDetector:
    """Detect vehicles in frames and return class labels."""

    ALLOWED_VEHICLE_LABELS = {
        "car",
        "motorcycle",
        "motorbike",
        "bike",
        "bus",
        "truck",
        "van",
        "suv",
        "jeep",
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model = None
        self.model_path = model_path or Config.VEHICLE_MODEL_PATH
        if YOLO is not None and self.model_path:
            try:
                self.model = YOLO(self.model_path)
                if hasattr(self.model, "names"):
                    print("[VehicleDetector] loaded model names:", self.model.names)
            except Exception as exc:
                print("[VehicleDetector] failed to load model:", exc)
                self.model = None

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Return vehicle detections for a frame."""
        if self.model is not None:
            try:
                results = self.model(frame, stream=False, conf=0.35)[0]
                detections = []
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = str(results.names.get(cls, "vehicle")).lower()
                    if label not in self.ALLOWED_VEHICLE_LABELS:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                    })
                return detections
            except Exception:
                pass

        # Fallback heuristic for offline environments.
        gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 2000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                "label": "car",
                "confidence": 0.6,
                "bbox": [x, y, x + w, y + h],
            })
        return detections
