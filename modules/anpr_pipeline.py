"""End-to-end unified ANPR detection pipeline for vehicle detection, plate detection, OCR, and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config
from modules.geometry import add_controlled_padding, clamp_bbox, crop_to_global_bbox, is_bbox_center_inside, validate_plate_crop_dimensions
from modules.llm import LLMNormalizer
from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.preprocessing import PlatePreprocessor
from modules.utils import get_logger, is_valid_indian_plate, validate_indian_plate_with_details
from modules.validator import PlateValidator


class ANPRPipeline:
    """Unified core ANPR detection pipeline shared across Image Upload, Video, and Stream Processing."""

    def __init__(self, vehicle_detector: Any, plate_detector: Any | None = None, ocr_reader: Any | None = None, llm_reader: Any | None = None) -> None:
        self.logger = get_logger("pipeline")
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector or PlateDetector()
        self.ocr_reader = ocr_reader or OCRReader()
        self.llm_reader = llm_reader or LLMNormalizer()
        self.validator = PlateValidator()
        self.preprocessor = PlatePreprocessor()
        self.output_dir = Path("static/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, frame: Any, camera_state: str = "") -> dict[str, Any]:
        """Execute the complete unified ANPR pipeline against a single image."""
        if frame is None:
            return {
                "status": "FAILED",
                "reason": "No image provided",
                "confidence": 0.0,
                "ocr_confidence": 0.0,
                "selected_plate_confidence": 0.0,
                "plate_number": "NOT_DETECTED",
            }

        image = np.array(frame)
        if image is None or image.size == 0:
            return {
                "status": "FAILED",
                "reason": "Empty image provided",
                "confidence": 0.0,
                "ocr_confidence": 0.0,
                "selected_plate_confidence": 0.0,
                "plate_number": "NOT_DETECTED",
            }

        annotated = image.copy()
        self._save_image(image, "original.jpg")

        # Step 1: Vehicle Detection
        vehicles = self.vehicle_detector.detect(image)
        if not vehicles:
            return {
                "status": "FAILED",
                "reason": "No valid vehicle detected",
                "annotated_frame": annotated,
                "confidence": 0.0,
                "ocr_confidence": 0.0,
                "selected_plate_confidence": 0.0,
                "plate_number": "NOT_DETECTED",
            }

        vehicle = max(vehicles, key=lambda item: float(item.get("confidence", 0.0)))
        vehicle_bbox = clamp_bbox(vehicle.get("bbox") or [0, 0, image.shape[1], image.shape[0]], image.shape)
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vehicle_label = str(vehicle.get("label", "vehicle")).upper()
        vehicle_conf = float(vehicle.get("confidence", 0.0))

        vehicle_crop = image[vy1:vy2, vx1:vx2] if (vx2 > vx1 and vy2 > vy1) else image
        self._save_image(vehicle_crop, "vehicle_detected.jpg")

        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
        cv2.putText(annotated, vehicle_label, (vx1, max(vy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Step 2: Dedicated Plate Detection on Vehicle Crop
        plate_candidates = self.plate_detector.detect(vehicle_crop)
        if not plate_candidates:
            return {
                "status": "FAILED",
                "reason": "License plate not detected",
                "vehicle": vehicle,
                "vehicle_crop": vehicle_crop,
                "annotated_frame": annotated,
                "confidence": 0.0,
                "ocr_confidence": 0.0,
                "selected_plate_confidence": 0.0,
                "plate_number": "NOT_DETECTED",
            }

        selected_plate = None
        for candidate in plate_candidates:
            bbox = candidate.get("bbox") or []
            conf = float(candidate.get("confidence", 0.0))
            if conf < Config.PLATE_CONFIDENCE_THRESHOLD:
                continue
            if not self._is_within_vehicle(bbox, vehicle_bbox, image.shape):
                continue
            if not self._meets_geometry(bbox):
                continue
            selected_plate = candidate
            break

        if selected_plate is None:
            selected_plate = plate_candidates[0]

        local_bbox = selected_plate["bbox"]
        plate_det_conf = float(selected_plate.get("confidence", 0.0))

        # Step 3 & 4: Controlled Padding & Coordinate Transformation
        padded_local_bbox = add_controlled_padding(local_bbox, vehicle_crop.shape, pad_percent=0.05, min_pad_px=4)
        px1, py1, px2, py2 = padded_local_bbox
        plate_crop = vehicle_crop[py1:py2, px1:px2]

        global_plate_bbox = crop_to_global_bbox(local_bbox, vx1, vy1, image.shape)
        gx1, gy1, gx2, gy2 = global_plate_bbox

        self._save_image(plate_crop, "cropped_plate.jpg")

        # Step 5: Crop Dimension Validation (Reject poor-quality/tiny plate crops)
        if not validate_plate_crop_dimensions(
            plate_crop,
            min_width=Config.PLATE_MIN_WIDTH,
            min_height=Config.PLATE_MIN_HEIGHT,
            min_aspect=Config.PLATE_MIN_ASPECT_RATIO,
            max_aspect=Config.PLATE_MAX_ASPECT_RATIO,
        ):
            return {
                "status": "FAILED",
                "reason": "Invalid or poor-quality plate crop dimensions",
                "vehicle": vehicle,
                "vehicle_crop": vehicle_crop,
                "plate_crop": plate_crop,
                "annotated_frame": annotated,
                "confidence": 0.0,
                "ocr_confidence": 0.0,
                "selected_plate_confidence": plate_det_conf,
                "plate_number": "NOT_DETECTED",
            }

        # Step 6: Step-by-Step Preprocessing & Multi-Variant OCR
        enhanced = self.preprocessor.preprocess(plate_crop)
        self._save_image(enhanced, "enhanced_plate.jpg")

        ocr_result = self.ocr_reader.read(plate_crop)
        raw_ocr_text = ocr_result.get("raw_text") or ocr_result.get("text") or ""
        ocr_confidence = float(ocr_result.get("ocr_confidence") or ocr_result.get("confidence") or 0.0)
        final_plate_text = str(ocr_result.get("plate_number") or ocr_result.get("text") or "")
        val_status = str(ocr_result.get("validation_status") or "INVALID_FORMAT")
        variant_used = str(ocr_result.get("processing_variant") or ocr_result.get("variant") or "standard")

        plate_label = final_plate_text or "PLATE"
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
        cv2.putText(annotated, plate_label, (gx1, max(gy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self._save_image(annotated, "ocr_result.jpg")

        is_valid = bool(ocr_result.get("valid", is_valid_indian_plate(final_plate_text)))
        if ocr_confidence < 0.70:
            return {
                "status": "FAILED",
                "reason": "Low OCR confidence",
                "plate": final_plate_text,
                "plate_number": final_plate_text,
                "ocr_confidence": round(ocr_confidence, 2),
                "confidence": round(ocr_confidence, 2),
                "annotated_frame": annotated,
                "vehicle": vehicle,
                "vehicle_crop": vehicle_crop,
                "plate_crop": plate_crop,
            }

        return {
            "status": "SUCCESS",
            "plate": final_plate_text,
            "plate_number": final_plate_text,
            "ocr_text": raw_ocr_text,
            "ocr_confidence": round(ocr_confidence, 2),
            "confidence": round(ocr_confidence, 2),
            "selected_plate_confidence": round(plate_det_conf, 2),
            "detected_vehicle_type": vehicle_label,
            "detected_vehicle_confidence": round(vehicle_conf, 2),
            "state": camera_state or "Unknown",
            "vehicle": vehicle,
            "bbox": global_plate_bbox,
            "global_plate_bbox": global_plate_bbox,
            "local_plate_bbox": local_bbox,
            "validation_status": val_status,
            "processing_variant": variant_used,
            "plate_valid": is_valid,
            "original_image": image,
            "annotated_frame": annotated,
            "vehicle_crop": vehicle_crop,
            "plate_crop": plate_crop,
            "enhanced_plate": enhanced,
        }

    def process_frame(self, frame: Any, camera_state: str = "") -> dict[str, Any]:
        """Wrapper for process_image for backwards compatibility."""
        return self.process_image(frame, camera_state=camera_state)

    def _save_image(self, image: Any, filename: str) -> None:
        output_path = self.output_dir / filename
        if image is None:
            return
        img = np.array(image)
        if img.size == 0:
            return
        cv2.imwrite(str(output_path), img)

    def _is_within_vehicle(self, local_plate_bbox: list[int], vehicle_bbox: list[int], frame_shape: tuple[int, int]) -> bool:
        global_plate_bbox = crop_to_global_bbox(local_plate_bbox, vehicle_bbox[0], vehicle_bbox[1], frame_shape)
        return is_bbox_center_inside(global_plate_bbox, vehicle_bbox)

    def _meets_geometry(self, bbox: list[int]) -> bool:
        if len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width < Config.PLATE_MIN_WIDTH or height < Config.PLATE_MIN_HEIGHT:
            return False
        aspect = width / float(max(1, height))
        return Config.PLATE_MIN_ASPECT_RATIO <= aspect <= Config.PLATE_MAX_ASPECT_RATIO
