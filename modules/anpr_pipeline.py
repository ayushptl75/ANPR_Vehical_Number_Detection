"""End-to-end ANPR pipeline for vehicle, plate, OCR, and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.llm import LLMNormalizer
from modules.ocr import OCRReader
from modules.plate_detector import PlateDetector
from modules.preprocessing import PlatePreprocessor
from modules.validator import PlateValidator


class ANPRPipeline:
    """Coordinate vehicle detection, plate detection, OCR, and validation."""

    def __init__(self, vehicle_detector: Any, plate_detector: Any | None = None, ocr_reader: Any | None = None, llm_reader: Any | None = None) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector or PlateDetector()
        self.ocr_reader = ocr_reader or OCRReader()
        self.llm_reader = llm_reader or LLMNormalizer()
        self.validator = PlateValidator()
        self.preprocessor = PlatePreprocessor()
        self.output_dir = Path("static/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame: Any, camera_state: str = "") -> dict[str, Any]:
        image = np.array(frame)
        if image.size == 0:
            return {"status": "FAILED", "reason": "No image provided"}

        vehicles = self.vehicle_detector.detect(image)
        if not vehicles:
            return {"status": "FAILED", "reason": "No valid vehicle detected"}

        vehicle = max(vehicles, key=lambda item: float(item.get("confidence", 0.0)))
        vehicle_bbox = vehicle.get("bbox") or [0, 0, image.shape[1], image.shape[0]]
        vehicle_crop = image[vehicle_bbox[1]:vehicle_bbox[3], vehicle_bbox[0]:vehicle_bbox[2]]
        self._save_image(image, "original.jpg")
        self._save_image(vehicle_crop, "vehicle_detected.jpg")

        print("[DEBUG] Vehicle Confidence:", vehicle.get("confidence"))
        print("[DEBUG] Vehicle Coordinates:", vehicle_bbox)

        plate_candidates = self.plate_detector.detect(vehicle_crop)
        if not plate_candidates:
            return {"status": "FAILED", "reason": "License plate not detected"}

        selected_plate = None
        for candidate in plate_candidates:
            bbox = candidate.get("bbox") or []
            conf = float(candidate.get("confidence", 0.0))
            if conf < 0.7:
                continue
            if not self._is_within_vehicle(bbox, vehicle_bbox):
                continue
            if not self._meets_geometry(bbox):
                continue
            selected_plate = candidate
            break

        if selected_plate is None:
            return {"status": "FAILED", "reason": "License plate not detected"}

        x1, y1, x2, y2 = [int(v) for v in selected_plate["bbox"]]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(vehicle_crop.shape[1], x2)
        y2 = min(vehicle_crop.shape[0], y2)
        pad = 5
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(vehicle_crop.shape[1], x2 + pad)
        y2 = min(vehicle_crop.shape[0], y2 + pad)
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        self._save_image(plate_crop, "cropped_plate.jpg")

        print("[DEBUG] Plate Confidence:", selected_plate.get("confidence"))
        print("[DEBUG] Plate Coordinates:", [x1, y1, x2, y2])
        print("[DEBUG] Plate Width:", x2 - x1)
        print("[DEBUG] Plate Height:", y2 - y1)
        print("[DEBUG] Plate Aspect Ratio:", (x2 - x1) / float(max(1, y2 - y1)))

        if plate_crop.size == 0:
            return {"status": "FAILED", "reason": "License plate not detected"}

        enhanced = self.preprocessor.preprocess(plate_crop)
        self._save_image(enhanced, "enhanced_plate.jpg")
        ocr_result = self.ocr_reader.read(enhanced)
        ocr_text = ocr_result.get("text", "")
        ocr_confidence = float(ocr_result.get("confidence", 0.0))
        print("[DEBUG] OCR Confidence:", ocr_confidence)
        print("[DEBUG] OCR Text:", ocr_text)
        if ocr_confidence < 0.7:
            return {"status": "FAILED", "reason": "Low OCR confidence"}

        normalized_text, llm_ok = self.llm_reader.normalize(ocr_text)
        valid, final_text = self.validator.validate(normalized_text if llm_ok else ocr_text)
        if not valid:
            return {"status": "FAILED", "reason": "License plate not detected"}

        self._save_image(image.copy(), "ocr_result.jpg")
        print("[DEBUG] Final Plate:", final_text)
        return {
            "status": "SUCCESS",
            "plate": final_text,
            "state": camera_state or "Unknown",
            "vehicle_type": vehicle.get("label", "vehicle"),
            "ocr_confidence": round(ocr_confidence * 100, 2),
            "ocr_text": ocr_text,
            "bbox": [x1, y1, x2, y2],
        }

    def _save_image(self, image: Any, filename: str) -> None:
        output_path = self.output_dir / filename
        if image is None:
            return
        img = np.array(image)
        if img.size == 0:
            return
        if len(img.shape) == 2:
            cv2.imwrite(str(output_path), img)
        else:
            cv2.imwrite(str(output_path), img)

    def _is_within_vehicle(self, bbox: list[int], vehicle_bbox: list[int]) -> bool:
        x1, y1, x2, y2 = bbox
        vx1, vy1, vx2, vy2 = vehicle_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2

    def _meets_geometry(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width < 60 or height < 20:
            return False
        return 2.0 <= (width / float(max(1, height))) <= 6.0
