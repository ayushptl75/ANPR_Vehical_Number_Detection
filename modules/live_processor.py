"""Live camera processing loop and helper methods for real-time detection."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np

from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.vehicle_detector import VehicleDetector


class LiveProcessor:
    """Coordinate a simple live detection loop with optional camera frames."""

    def __init__(self, vehicle_detector: VehicleDetector, plate_detector: PlateDetector, ocr_reader: OCRReader) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader

    def process_frame(
        self,
        frame: Any,
        fps: float = 0.0,
        camera_name: str = "Camera",
        detection_time: str | None = None,
    ) -> dict[str, Any]:
        """Run the complete detection pipeline against a single frame."""
        vehicles = self.vehicle_detector.detect(frame)
        vehicle = vehicles[0] if vehicles else None
        plate = None
        vehicle_image = None
        plate_image = None
        plate_confidence = 0.0
        annotated_frame = frame.copy() if isinstance(frame, np.ndarray) else None

        if vehicle:
            bbox = vehicle["bbox"]
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            if x2 > x1 and y2 > y1:
                vehicle_image = frame[y1:y2, x1:x2]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    vehicle["label"].upper(),
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                crop = vehicle_image
                detections = self.plate_detector.detect(crop)
                if detections:
                    plate_bbox = detections[0]["bbox"]
                    px1, py1, px2, py2 = plate_bbox
                    ch, cw = crop.shape[:2]
                    px1 = max(0, min(px1, cw - 1))
                    px2 = max(px1 + 1, min(px2, cw))
                    py1 = max(0, min(py1, ch - 1))
                    py2 = max(py1 + 1, min(py2, ch))
                    plate_image = crop[py1:py2, px1:px2]
                    if plate_image.size != 0:
                        pad = 5
                        px1 = max(0, px1 - pad)
                        py1 = max(0, py1 - pad)
                        px2 = min(cw, px2 + pad)
                        py2 = min(ch, py2 + pad)
                        plate_image = crop[py1:py2, px1:px2]
                        processed_plate = self.ocr_reader.preprocess(plate_image)
                        ocr_result = self.ocr_reader.read(processed_plate)
                        plate = {
                            "text": ocr_result.get("text", ""),
                            "confidence": ocr_result.get("confidence", 0.0),
                            "bbox": [px1, py1, px2, py2],
                        }
                        plate_confidence = ocr_result.get("confidence", 0.0)
                        plate_label = plate["text"] or "PLATE"
                        cv2.rectangle(
                            annotated_frame,
                            (x1 + px1, y1 + py1),
                            (x1 + px2, y1 + py2),
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            annotated_frame,
                            plate_label,
                            (x1 + px1, max(y1 + py1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )
                        confidence_text = f"OCR: {round(plate_confidence * 100)}%"
                        cv2.putText(
                            annotated_frame,
                            confidence_text,
                            (x1 + px1, min(y1 + py2 + 25, annotated_frame.shape[0] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2,
                        )

        if annotated_frame is not None:
            timestamp = detection_time or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            header_text = f"FPS: {fps:.1f}   Camera: {camera_name}"
            footer_text = timestamp
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                header_text,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0] - 35), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                footer_text,
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return {
            "vehicle": vehicle,
            "plate": plate,
            "vehicle_image": vehicle_image,
            "plate_image": plate_image,
            "plate_confidence": plate_confidence,
            "annotated_frame": annotated_frame,
        }
