"""Live camera processing loop and helper methods for real-time detection."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np

from modules.camera_stream import FrameSampler
from modules.geometry import add_controlled_padding, clamp_bbox, crop_to_global_bbox
from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.tracker import VehiclePlateTracker
from modules.vehicle_detector import VehicleDetector


class LiveProcessor:
    """Coordinate a live detection loop with original camera frame coordinate mapping, multi-frame tracking, and frame sampling."""

    def __init__(
        self,
        vehicle_detector: VehicleDetector,
        plate_detector: PlateDetector,
        ocr_reader: OCRReader,
        tracker: VehiclePlateTracker | None = None,
        skip_interval: int | None = None,
    ) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        self.tracker = tracker or VehiclePlateTracker()
        self.sampler = FrameSampler(skip_interval=skip_interval)
        
        self.last_result: dict[str, Any] | None = None

    def process_frame(
        self,
        frame: Any,
        fps: float = 0.0,
        camera_name: str = "Camera",
        detection_time: str | None = None,
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Run the complete detection pipeline against a single camera frame."""
        frame_arr = np.array(frame)
        if frame_arr.size == 0:
            return {
                "vehicle": None,
                "plate": None,
                "vehicle_image": None,
                "plate_image": None,
                "plate_confidence": 0.0,
                "annotated_frame": None,
                "tracking_info": None,
                "latency_ms": latency_ms,
            }

        should_run_detection = self.sampler.should_sample()

        # If skipping frame for performance, reuse cached tracking overlay on new frame
        if not should_run_detection and self.last_result is not None:
            res = dict(self.last_result)
            annotated = frame_arr.copy()
            global_plate_bbox = res.get("plate", {}).get("bbox") if res.get("plate") else None
            tracking_info = res.get("tracking_info") or {}
            active_plate_text = tracking_info.get("plate_number") or ""
            active_conf = tracking_info.get("confidence") or 0.0

            if global_plate_bbox and annotated is not None:
                gx1, gy1, gx2, gy2 = global_plate_bbox
                plate_label = active_plate_text or "PLATE"
                status_tag = " [FINAL]" if tracking_info.get("is_finalized") else ""
                display_label = f"{plate_label}{status_tag}"

                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0) if tracking_info.get("is_finalized") else (0, 255, 255), 2)
                cv2.putText(annotated, display_label, (gx1, max(gy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            self._draw_overlay(annotated, fps, latency_ms, camera_name, detection_time)
            res["annotated_frame"] = annotated
            res["latency_ms"] = latency_ms
            return res

        vehicles = self.vehicle_detector.detect(frame_arr)
        vehicle = vehicles[0] if vehicles else None
        plate = None
        vehicle_image = None
        plate_image = None
        plate_confidence = 0.0
        ocr_result = {}
        annotated_frame = frame_arr.copy()
        global_plate_bbox = None

        if vehicle:
            v_bbox = clamp_bbox(vehicle["bbox"], frame_arr.shape)
            vx1, vy1, vx2, vy2 = v_bbox
            vehicle["bbox"] = v_bbox

            if vx2 > vx1 and vy2 > vy1:
                vehicle_image = frame_arr[vy1:vy2, vx1:vx2]
                cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    vehicle["label"].upper(),
                    (vx1, max(vy1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Plate detection on vehicle crop
                crop = vehicle_image
                detections = self.plate_detector.detect(crop)
                if detections:
                    raw_plate_bbox = detections[0]["bbox"]
                    
                    # Convert to global coordinates on full camera frame
                    global_plate_bbox = crop_to_global_bbox(raw_plate_bbox, vx1, vy1, frame_arr.shape)
                    gx1, gy1, gx2, gy2 = global_plate_bbox

                    # Extract plate crop with controlled padding for OCR
                    padded_local = add_controlled_padding(raw_plate_bbox, crop.shape, pad_percent=0.05, min_pad_px=4)
                    px1, py1, px2, py2 = padded_local
                    plate_image = crop[py1:py2, px1:px2]

                    if plate_image.size != 0:
                        ocr_result = self.ocr_reader.read(plate_image)
                        plate_confidence = float(ocr_result.get("ocr_confidence") or ocr_result.get("confidence") or 0.0)
                        plate_text = str(ocr_result.get("plate_number") or ocr_result.get("text") or "")
                        
                        plate = {
                            "text": plate_text,
                            "confidence": plate_confidence,
                            "bbox": global_plate_bbox,
                            "local_bbox": raw_plate_bbox,
                        }

        # Multi-frame tracker update
        v_bbox_list = vehicle["bbox"] if vehicle else None
        v_label = vehicle.get("label", "vehicle") if vehicle else "vehicle"
        
        tracking_info = self.tracker.process_detection(
            vehicle_bbox=v_bbox_list,
            global_plate_bbox=global_plate_bbox,
            vehicle_type=v_label,
            ocr_result=ocr_result,
        )

        active_plate_text = tracking_info["plate_number"] or (plate.get("text") if plate else "")
        active_conf = tracking_info["confidence"] or plate_confidence

        if global_plate_bbox and annotated_frame is not None:
            gx1, gy1, gx2, gy2 = global_plate_bbox
            plate_label = active_plate_text or "PLATE"
            status_tag = " [FINAL]" if tracking_info["is_finalized"] else ""
            display_label = f"{plate_label}{status_tag}"

            cv2.rectangle(
                annotated_frame,
                (gx1, gy1),
                (gx2, gy2),
                (0, 255, 255) if not tracking_info["is_finalized"] else (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                display_label,
                (gx1, max(gy1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            confidence_text = f"Conf: {round(active_conf * 100)}% ({tracking_info['observation_count']} obs)"
            cv2.putText(
                annotated_frame,
                confidence_text,
                (gx1, min(gy2 + 25, annotated_frame.shape[0] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        self._draw_overlay(annotated_frame, fps, latency_ms, camera_name, detection_time)

        if plate is None and active_plate_text:
            plate = {"text": active_plate_text, "confidence": active_conf, "bbox": global_plate_bbox or [0, 0, 0, 0]}
        elif plate:
            plate["text"] = active_plate_text
            plate["confidence"] = active_conf

        res = {
            "vehicle": vehicle,
            "plate": plate,
            "vehicle_image": vehicle_image,
            "plate_image": plate_image,
            "plate_confidence": active_conf,
            "annotated_frame": annotated_frame,
            "tracking_info": tracking_info,
            "latency_ms": latency_ms,
        }
        self.last_result = res
        return res

    def _draw_overlay(self, annotated_frame: np.ndarray, fps: float, latency_ms: float, camera_name: str, detection_time: str | None) -> None:
        if annotated_frame is None or annotated_frame.size == 0:
            return
        timestamp = detection_time or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        header_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Cam: {camera_name}"
        footer_text = timestamp
        
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(annotated_frame, header_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0] - 35), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(annotated_frame, footer_text, (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
