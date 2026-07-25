"""Video/image processing pipeline for uploads and camera feed processing."""
from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import Config
from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.utils import (
    clean_plate_text,
    get_logger,
    get_safe_filename,
    is_valid_indian_plate,
    positionally_correct_plate_text,
)
from modules.vehicle_detector import VehicleDetector


class VideoProcessor:
    """Run the detection pipeline over images and video files."""

    def __init__(self, vehicle_detector: VehicleDetector, plate_detector: PlateDetector, ocr_reader: OCRReader) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        self.logger = get_logger("video")
        self.processed_dir = Path(Config.PROCESSED_FOLDER)
        self.vehicle_images_dir = Path(Config.VEHICLE_IMAGES_FOLDER)
        self.plate_images_dir = Path(Config.PLATE_IMAGES_FOLDER)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vehicle_images_dir.mkdir(parents=True, exist_ok=True)
        self.plate_images_dir.mkdir(parents=True, exist_ok=True)

    def process_static_image(self, image_path: str) -> dict[str, Any]:
        """Process a single image and return OCR details."""
        image = cv2.imread(image_path)
        if image is None:
            return {"plate_number": "", "vehicle_type": "Unknown", "confidence": 0.0}

        # Detect vehicles first.
        vehicles = self.vehicle_detector.detect(image)
        selected_vehicle = vehicles[0] if vehicles else None
        vehicle_bbox = selected_vehicle["bbox"] if selected_vehicle else [0, 0, image.shape[1], image.shape[0]]
        vehicle_type = selected_vehicle["label"] if selected_vehicle else "Unknown"
        vehicle_confidence = float(selected_vehicle["confidence"]) if selected_vehicle else 0.0

        vehicle_crop = image[vehicle_bbox[1]:vehicle_bbox[3], vehicle_bbox[0]:vehicle_bbox[2]]
        print("[DEBUG] Vehicle bbox:", vehicle_bbox)
        print("[DEBUG] Vehicle image size:", vehicle_crop.shape)

        plate_detections = self.plate_detector.detect(vehicle_crop)
        valid_plate_detections = []
        for plate in plate_detections:
            if float(plate.get("confidence", 0.0)) < 0.75:
                continue
            if self._validate_plate_bbox(plate["bbox"], vehicle_crop.shape):
                valid_plate_detections.append(plate)

        valid_plate_detections = self._apply_nms(valid_plate_detections)
        if valid_plate_detections:
            valid_plate_detections.sort(key=lambda item: (item["confidence"], self._aspect_score(item["bbox"]), self._bbox_area(item["bbox"])), reverse=True)
            selected_plate = valid_plate_detections[0]
        else:
            selected_plate = None

        plate_crop = None
        plate_bbox = None
        if selected_plate is not None:
            plate_bbox_local = self._refine_plate_bbox(selected_plate["bbox"], vehicle_crop.shape)
            plate_crop = vehicle_crop[plate_bbox_local[1]:plate_bbox_local[3], plate_bbox_local[0]:plate_bbox_local[2]]
            if plate_crop.size == 0 or plate_crop.shape[0] < 25 or plate_crop.shape[1] < 80:
                print("[DEBUG] Plate crop rejected due to invalid size", plate_crop.shape if plate_crop is not None else None)
                plate_crop = None
            else:
                plate_bbox = [
                    vehicle_bbox[0] + plate_bbox_local[0],
                    vehicle_bbox[1] + plate_bbox_local[1],
                    vehicle_bbox[0] + plate_bbox_local[2],
                    vehicle_bbox[1] + plate_bbox_local[3],
                ]

        ocr_candidates = []
        if plate_crop is not None:
            preprocess_variants = self._generate_plate_variants(plate_crop)
            for variant in preprocess_variants:
                ocr_output = self.ocr_reader.read(variant["image"])
                raw_text = ocr_output.get("raw_text", "")
                normalized = clean_plate_text(raw_text)
                corrected = positionally_correct_plate_text(normalized)
                ocr_confidence = float(ocr_output.get("confidence", 0.0))
                is_valid = is_valid_indian_plate(corrected)
                score = ocr_confidence + (1.0 if is_valid else 0.0)
                print("[DEBUG] OCR variant", variant["name"], "raw:", raw_text, "normalized:", corrected, "conf:", ocr_confidence, "valid:", is_valid)
                ocr_candidates.append({
                    "raw_text": raw_text,
                    "normalized_text": corrected,
                    "confidence": ocr_confidence,
                    "valid": is_valid,
                    "variant_name": variant["name"],
                    "score": score,
                })

        selected_plate_text = ""
        selected_plate_confidence = 0.0
        selected_plate_variant = None
        for candidate in sorted(ocr_candidates, key=lambda item: (item["valid"], item["score"]), reverse=True):
            if candidate["valid"]:
                selected_plate_text = candidate["normalized_text"]
                selected_plate_confidence = candidate["confidence"]
                selected_plate_variant = candidate["variant_name"]
                break

        plate_number = selected_plate_text

        full_plate_bbox = plate_bbox if plate_bbox is not None else [0, 0, image.shape[1], image.shape[0]]

        annotated = image.copy()
        try:
            x1, y1, x2, y2 = vehicle_bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"Vehicle: {vehicle_type}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception:
            pass
        try:
            px1, py1, px2, py2 = full_plate_bbox
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)
            label = f"License Plate: {int(selected_plate['confidence'] * 100)}%" if selected_plate else "License Plate"
            cv2.putText(annotated, label, (px1, max(py1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception:
            pass

        vehicle_img_path = self._save_image(vehicle_crop, "vehicle")
        plate_img_path = self._save_image(plate_crop if plate_crop is not None else image, "plate")
        if plate_crop is not None:
            self._save_image(plate_crop, "cropped_plate")
        if Config.DEBUG_MODE:
            self._save_debug_image(image, "original.jpg")
            if vehicle_crop is not None:
                self._save_debug_image(vehicle_crop, "vehicle.jpg")
            if plate_bbox is not None:
                plate_box_image = image.copy()
                x1, y1, x2, y2 = [int(v) for v in plate_bbox]
                cv2.rectangle(plate_box_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                self._save_debug_image(plate_box_image, "plate_box.jpg")
            if plate_crop is not None:
                self._save_debug_image(plate_crop, "cropped_plate.jpg")
                enhanced_plate = self._enhance_plate_for_debug(plate_crop)
                self._save_debug_image(enhanced_plate, "enhanced_plate.jpg")
        if plate_crop is not None:
            static_output_dir = Path("static/output")
            static_output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(static_output_dir / "cropped_plate.jpg"), plate_crop)
        processed_name = get_safe_filename(f"processed_{Path(image_path).stem}.jpg")
        processed_path = str(self.processed_dir / processed_name)
        cv2.imwrite(processed_path, annotated)

        output = {
            "plate_number": plate_number,
            "detected_vehicle_type": vehicle_type,
            "detected_vehicle_confidence": vehicle_confidence,
            "confidence": selected_plate_confidence,
            "vehicle_image_path": vehicle_img_path,
            "plate_image_path": plate_img_path,
            "processed_image_path": processed_path,
            "original_image_path": image_path,
            "detected_at": datetime.utcnow().isoformat(),
            "plate_detections": plate_detections,
            "ocr_candidates": ocr_candidates,
            "selected_plate_bbox": full_plate_bbox,
            "selected_plate_confidence": selected_plate["confidence"] if selected_plate else 0.0,
            "plate_selected": bool(plate_number),
        }
        return output

    def _refine_plate_bbox(self, bbox: list[int], image_shape: tuple[int, int, int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)
        shrink_x = int(round(0.03 * width_px))
        shrink_y = int(round(0.03 * height_px))
        shrink_x = max(2, min(shrink_x, 8))
        shrink_y = max(2, min(shrink_y, 8))
        height, width = image_shape[:2]
        x1 = max(0, x1 + shrink_x)
        y1 = max(0, y1 + shrink_y)
        x2 = min(width, x2 - shrink_x)
        y2 = min(height, y2 - shrink_y)
        if x2 <= x1:
            x2 = min(width, x1 + max(10, width_px // 2))
        if y2 <= y1:
            y2 = min(height, y1 + max(10, height_px // 2))
        return [x1, y1, x2, y2]

    def _pad_plate_bbox(self, bbox: list[int], image_shape: tuple[int, int, int]) -> list[int]:
        x1, y1, x2, y2 = bbox
        pad = 2
        height, width = image_shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        return [x1, y1, x2, y2]

    def _aspect_score(self, bbox: list[int]) -> float:
        if not bbox or len(bbox) != 4:
            return 0.0
        width_px = max(1, bbox[2] - bbox[0])
        height_px = max(1, bbox[3] - bbox[1])
        ratio = width_px / float(height_px)
        return 1.0 - abs(ratio - 4.0) / 4.0

    def _bbox_area(self, bbox: list[int]) -> float:
        if not bbox or len(bbox) != 4:
            return 0.0
        return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    def _apply_nms(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not detections:
            return []
        ordered = sorted(detections, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        kept: list[dict[str, Any]] = []
        while ordered:
            current = ordered.pop(0)
            kept.append(current)
            current_bbox = current.get("bbox", [])
            ordered = [item for item in ordered if self._iou(current_bbox, item.get("bbox", [])) < 0.35]
        return kept

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

    def _validate_plate_bbox(self, bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
        if len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]
        if x2 <= x1 or y2 <= y1:
            return False
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
        if aspect_ratio < 2.5 or aspect_ratio > 5.5:
            return False
        if (x2 - x1) < 80 or (y2 - y1) < 25:
            return False
        return True

    def _bbox_center_inside(self, bbox: list[int], container_bbox: list[int]) -> bool:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return container_bbox[0] <= cx <= container_bbox[2] and container_bbox[1] <= cy <= container_bbox[3]

    def _generate_plate_variants(self, plate_crop: np.ndarray) -> list[dict[str, Any]]:
        variants: list[dict[str, Any]] = []
        base = plate_crop.copy()

        variants.append({"name": "original", "image": base})

        upscaled = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append({"name": "upscaled", "image": upscaled})

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        variants.append({"name": "grayscale", "image": gray})

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append({"name": "clahe", "image": clahe_img})

        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        variants.append({"name": "bilateral", "image": bilateral})

        adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
        variants.append({"name": "adaptive_threshold", "image": adaptive})

        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append({"name": "otsu_threshold", "image": otsu})

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        variants.append({"name": "sharpened", "image": sharpened})

        return variants

    def _save_debug_image(self, image: np.ndarray, filename: str) -> None:
        debug_dir = Path(Config.DEBUG_FOLDER)
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / filename
        cv2.imwrite(str(path), image)

    def _enhance_plate_for_debug(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
        if gray.shape[1] < 480:
            scale = 480 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif gray.shape[1] > 480:
            scale = 480 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return gray

    def process_video(self, video_path: str, analyze_interval: int = 30) -> dict[str, Any]:
        """Process a video file frame by frame, annotate detections and return output path plus detected plates.

        analyze_interval: number of frames between running full detection/OCR to reduce work.
        Returns dict with keys: output_path and detections (list).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to read video")

        output_path = str(self.processed_dir / f"processed_{Path(video_path).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        detections: list[dict[str, Any]] = []
        seen_plates: set[str] = set()

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1

            # Draw visual detections for output
            self._draw_detections(frame)

            # Occasionally run OCR to detect plates (reduces processing load)
            if frame_count % max(1, analyze_interval) == 0:
                vehicles = self.vehicle_detector.detect(frame)
                vehicle_bbox = vehicles[0]["bbox"] if vehicles else None
                if vehicle_bbox is None:
                    vehicle_bbox = [0, 0, frame.shape[1], frame.shape[0]]

                vehicle_crop = frame[vehicle_bbox[1]:vehicle_bbox[3], vehicle_bbox[0]:vehicle_bbox[2]]
                plates = self.plate_detector.detect(vehicle_crop)
                plate_bbox = plates[0]["bbox"] if plates else None
                if plate_bbox is None:
                    plate_bbox = [0, 0, vehicle_crop.shape[1], vehicle_crop.shape[0]]

                plate_crop = vehicle_crop[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                ocr = self.ocr_reader.read(plate_crop)
                plate_text = ocr.get("text", "")
                confidence = ocr.get("confidence", 0.0)

                if plate_text:
                    # avoid duplicate entries in the same processed video
                    if plate_text not in seen_plates:
                        seen_plates.add(plate_text)
                        vehicle_type = vehicles[0]["label"] if vehicles else "Vehicle"
                        vehicle_img_path = self._save_image(vehicle_crop, "vehicle")
                        plate_img_path = self._save_image(plate_crop, "plate")
                        detections.append(
                            {
                                "frame": frame_count,
                                "plate": plate_text,
                                "confidence": confidence,
                                "vehicle_type": vehicle_type,
                                "vehicle_image": vehicle_img_path,
                                "plate_image": plate_img_path,
                            }
                        )

            writer.write(frame)

        cap.release()
        writer.release()
        return {"output_path": output_path, "detections": detections}

    def _draw_detections(self, frame: np.ndarray) -> None:
        """Draw simple bounding boxes and labels on the frame."""
        vehicles = self.vehicle_detector.detect(frame)
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, vehicle["label"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        plates = self.plate_detector.detect(frame)
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Plate", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _save_image(self, image: np.ndarray, kind: str) -> str:
        """Save an image crop to disk and return the path."""
        filename = get_safe_filename(f"{kind}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png")
        path = self.vehicle_images_dir if kind == "vehicle" else self.plate_images_dir
        destination = path / filename
        cv2.imwrite(str(destination), image)
        return str(destination)
