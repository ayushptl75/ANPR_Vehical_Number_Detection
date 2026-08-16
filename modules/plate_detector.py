"""Plate detection helpers for locating number plate regions."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config
from modules.utils import get_logger

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


class PlateDetector:
    """Detect number plate regions with dedicated model loader and strict geometry filtering."""

    def __init__(self, model_path: str | None = None) -> None:
        self.logger = get_logger("plate_detector")
        self.model = None
        self.is_dedicated = False
        self.model_status = "unloaded"
        
        raw_path = model_path if model_path is not None else Config.PLATE_MODEL_PATH
        self.model_path = self._resolve_model_path(raw_path)
        self._init_model()

    def _init_model(self) -> None:
        """Validate and load the specified YOLO model file."""
        if YOLO is None:
            self.model_status = "ultralytics_missing"
            self.logger.warning("[PlateDetector] ultralytics package not available.")
            return

        if not self.model_path:
            self.model_status = "no_path"
            self.logger.info("[PlateDetector] No plate model path configured. Using fallback detection.")
            return

        target_path = Path(self.model_path)
        if not target_path.exists():
            self.model_status = "file_not_found"
            self.logger.warning(
                "[PlateDetector] Dedicated plate model file NOT found at '%s'. "
                "Operating in OpenCV contour fallback mode. "
                "To use a dedicated detector, place your trained weights file (e.g. license_plate_detector.pt) "
                "in the 'models/' directory and update PLATE_MODEL_PATH in .env.",
                self.model_path,
            )
            return

        try:
            self.model = YOLO(str(target_path))
            model_name = target_path.name.lower()
            
            # Check if this model is generic COCO (yolov8n.pt / yolov8s.pt) vs dedicated plate weights
            is_generic_coco = False
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                names_values = [str(v).lower() for v in self.model.names.values()]
                if "person" in names_values and "car" in names_values and "plate" not in names_values and "license_plate" not in names_values:
                    is_generic_coco = True

            if is_generic_coco or model_name in {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}:
                self.is_dedicated = False
                self.model_status = "generic_coco_loaded"
                self.logger.info(
                    "[PlateDetector] Loaded '%s' (Generic COCO model, not a dedicated plate model). "
                    "Plate detection will use geometric & contour filters.",
                    target_path.name,
                )
            else:
                self.is_dedicated = True
                self.model_status = "dedicated_loaded"
                self.logger.info("[PlateDetector] Dedicated license plate YOLO model loaded successfully from '%s'.", self.model_path)
        except Exception as exc:
            self.model = None
            self.is_dedicated = False
            self.model_status = "load_error"
            self.logger.error("[PlateDetector] Failed to load YOLO model from '%s': %s", self.model_path, exc)

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Detect plate bounding boxes in a given frame or crop."""
        image = np.array(frame)
        if image.size == 0:
            return []

        detections: list[dict[str, Any]] = []
        conf_thresh = Config.PLATE_CONFIDENCE_THRESHOLD if self.is_dedicated else 0.50

        if self.model is not None:
            try:
                results = self.model(image, stream=False, conf=conf_thresh)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    bbox = [x1, y1, x2, y2]
                    if self._is_valid_bbox(bbox, image.shape, conf, is_dedicated=self.is_dedicated):
                        detections.append({"bbox": bbox, "confidence": round(conf, 2), "label": "plate"})
            except Exception as exc:
                self.logger.warning("[PlateDetector] Model inference error: %s", exc)

        # Fallback to OpenCV contour detection if YOLO produced no valid candidates
        if not detections:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, x + w, y + h]
                if self._is_valid_bbox(bbox, image.shape, 0.70, is_dedicated=False):
                    detections.append({"bbox": bbox, "confidence": 0.70, "label": "plate"})

        if not detections:
            return []

        selected = self._select_best_detection(detections, image.shape)
        return [selected] if selected else []

    def _resolve_model_path(self, model_path: str | None, search_roots: list[Path] | None = None) -> str | None:
        if not model_path:
            return None

        raw_path = str(model_path).strip()
        if not raw_path:
            return None

        candidates: list[Path] = []
        if Path(raw_path).is_absolute():
            candidates.append(Path(raw_path))
        else:
            candidates.extend([
                Path(raw_path),
                BASE_DIR / raw_path,
                BASE_DIR / "models" / raw_path,
                BASE_DIR / "weights" / raw_path,
            ])

        if search_roots:
            for root in search_roots:
                candidates.append(root / raw_path)
                candidates.append(root / "models" / raw_path)

        if Path(raw_path).name != raw_path:
            basename = Path(raw_path).name
            for root in [BASE_DIR, BASE_DIR / "models", BASE_DIR / "weights", *(search_roots or [])]:
                if root.exists():
                    candidates.append(root / basename)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return raw_path

    def _select_best_detection(self, detections: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> dict[str, Any] | None:
        filtered: list[dict[str, Any]] = []
        for detection in detections:
            bbox = detection.get("bbox")
            conf = float(detection.get("confidence", 0.0))
            if not self._is_valid_bbox(bbox, image_shape, conf, is_dedicated=self.is_dedicated):
                continue
            filtered.append(detection)

        if not filtered:
            return None

        filtered = self._apply_nms(filtered, image_shape)
        filtered.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                self._aspect_score(item.get("bbox", [])),
                self._bbox_area(item.get("bbox", [])),
            ),
            reverse=True,
        )
        return filtered[0]

    def _apply_nms(self, detections: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        ordered = sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        while ordered:
            current = ordered.pop(0)
            kept.append(current)
            current_bbox = current.get("bbox", [])
            ordered = [
                item
                for item in ordered
                if self._iou(current_bbox, item.get("bbox", [])) < 0.35
            ]
        return kept

    def _is_valid_bbox(self, bbox: list[int] | None, image_shape: tuple[int, int, int], conf: float, is_dedicated: bool = False) -> bool:
        if not bbox or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]

        # 1. Frame boundary validation
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return False

        # Reject full-frame background boxes (e.g. entire vehicle crop being misidentified as plate)
        width_px = x2 - x1
        height_px = y2 - y1
        if width_px >= 0.95 * width and height_px >= 0.95 * height:
            return False

        # 2. Minimum dimension constraints (min width 60px, min height 18px)
        if width_px < 60 or height_px < 18:
            return False

        # 3. Reasonable aspect ratio filtering (2.0 to 6.0)
        aspect_ratio = width_px / float(max(1, height_px))
        if not (2.0 <= aspect_ratio <= 6.0):
            return False

        # 4. Confidence validation
        min_conf = Config.PLATE_CONFIDENCE_THRESHOLD if is_dedicated else 0.50
        if conf < min_conf:
            return False

        return True

    def _iou(self, a: list[int], b: list[int]) -> float:
        if not a or not b or len(a) != 4 or len(b) != 4:
            return 0.0
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / float(union) if union else 0.0

    def _aspect_score(self, bbox: list[int]) -> float:
        if not bbox or len(bbox) != 4:
            return 0.0
        width_px = max(1, bbox[2] - bbox[0])
        height_px = max(1, bbox[3] - bbox[1])
        ratio = width_px / float(height_px)
        return 1.0 - abs(ratio - 4.0) / 4.0

    def _bbox_area(self, bbox: list[int]) -> float:
        return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
