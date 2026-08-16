"""Video processing pipeline using unified ANPR core detection and temporal multi-frame tracking."""
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
from modules.anpr_pipeline import ANPRPipeline
from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.tracker import VehiclePlateTracker
from modules.utils import (
    clean_plate_text,
    get_logger,
    get_safe_filename,
    is_valid_indian_plate,
    positionally_correct_plate_text,
)
from modules.vehicle_detector import VehicleDetector


class VideoProcessor:
    """Run the unified ANPR detection and temporal tracking pipeline over static images and video files."""

    def __init__(self, vehicle_detector: VehicleDetector, plate_detector: PlateDetector, ocr_reader: OCRReader) -> None:
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_reader = ocr_reader
        self.pipeline = ANPRPipeline(vehicle_detector, plate_detector, ocr_reader)
        self.logger = get_logger("video")
        
        self.processed_dir = Path(Config.PROCESSED_FOLDER)
        self.vehicle_images_dir = Path(Config.VEHICLE_IMAGES_FOLDER)
        self.plate_images_dir = Path(Config.PLATE_IMAGES_FOLDER)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vehicle_images_dir.mkdir(parents=True, exist_ok=True)
        self.plate_images_dir.mkdir(parents=True, exist_ok=True)

    def process_static_image(self, image_path: str) -> dict[str, Any]:
        """Process a single image using the unified core ANPR pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            return {"plate_number": "", "vehicle_type": "Unknown", "confidence": 0.0}

        res = self.pipeline.process_image(image)
        plate_number = res.get("plate_number") or ""
        vehicle_type = res.get("detected_vehicle_type") or "Unknown"
        ocr_conf = float(res.get("ocr_confidence") or 0.0)

        vehicle_crop = res.get("vehicle_crop")
        plate_crop = res.get("plate_crop")
        annotated = res.get("annotated_frame") if res.get("annotated_frame") is not None else image

        vehicle_img_path = self._save_image(vehicle_crop, "vehicle") if vehicle_crop is not None else ""
        plate_img_path = self._save_image(plate_crop, "plate") if plate_crop is not None else ""

        processed_name = get_safe_filename(f"processed_{Path(image_path).stem}.jpg")
        processed_path = str(self.processed_dir / processed_name)
        cv2.imwrite(processed_path, annotated)

        return {
            "plate_number": plate_number,
            "detected_vehicle_type": vehicle_type,
            "detected_vehicle_confidence": res.get("detected_vehicle_confidence", 0.0),
            "confidence": ocr_conf,
            "vehicle_image_path": vehicle_img_path,
            "plate_image_path": plate_img_path,
            "processed_image_path": processed_path,
            "original_image_path": image_path,
            "detected_at": datetime.utcnow().isoformat(),
            "selected_plate_bbox": res.get("global_plate_bbox") or [0, 0, image.shape[1], image.shape[0]],
            "selected_plate_confidence": res.get("selected_plate_confidence", 0.0),
            "plate_selected": bool(plate_number),
            "plate_valid": res.get("plate_valid", False),
            "validation_status": res.get("validation_status", "INVALID_FORMAT"),
            "processing_variant": res.get("processing_variant", "standard"),
        }

    def process_video(self, video_path: str, analyze_interval: int | None = None) -> dict[str, Any]:
        """Process a video file using unified core detection and multi-frame temporal tracking.

        Reduces workload via configurable frame sampling while tracking vehicle/plate bounding box trajectories.
        Suppresses duplicate log spam by confirming final plates once per vehicle track.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to read video file")

        sample_skip = analyze_interval if analyze_interval is not None else Config.VIDEO_SAMPLE_INTERVAL
        output_name = get_safe_filename(f"processed_{Path(video_path).stem}.mp4")
        output_path = str(self.processed_dir / output_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = VehiclePlateTracker(
            min_observations=Config.TRACKER_MIN_OBSERVATIONS,
            min_confidence=Config.TRACKER_CONFIDENCE_THRESHOLD,
            timeout_seconds=Config.TRACKER_TIMEOUT_SECONDS,
        )

        frame_count = 0
        detections: list[dict[str, Any]] = []
        finalized_tracks: set[str] = set()

        last_known_bbox = None
        last_known_plate_text = ""

        while True:
            success, frame = cap.read()
            if not success or frame is None or frame.size == 0:
                break

            frame_count += 1
            annotated = frame.copy()
            now_sec = frame_count / float(max(1.0, fps))

            # Sample 1 frame every `sample_skip` frames
            if (frame_count % sample_skip) == 0:
                res = self.pipeline.process_image(frame)
                
                v_bbox = res.get("vehicle", {}).get("bbox") if res.get("vehicle") else None
                g_plate_bbox = res.get("global_plate_bbox")
                v_type = str(res.get("detected_vehicle_type") or "vehicle")
                
                ocr_payload = {
                    "plate_number": res.get("plate_number") or "",
                    "ocr_confidence": float(res.get("ocr_confidence") or 0.0),
                    "valid": bool(res.get("plate_valid", False)),
                    "validation_status": str(res.get("validation_status") or ""),
                }

                tracking_info = tracker.process_detection(
                    vehicle_bbox=v_bbox,
                    global_plate_bbox=g_plate_bbox,
                    vehicle_type=v_type,
                    ocr_result=ocr_payload,
                    now=now_sec,
                )

                track_id = tracking_info.get("track_id")
                confirmed_plate = tracking_info.get("plate_number") or ""
                conf = tracking_info.get("confidence") or 0.0
                is_finalized = bool(tracking_info.get("is_finalized"))
                is_newly_finalized = bool(tracking_info.get("is_newly_finalized"))

                last_known_plate_text = confirmed_plate
                if g_plate_bbox:
                    last_known_bbox = g_plate_bbox

                # Log final plate ONLY ONCE per vehicle track
                if is_newly_finalized or (is_finalized and track_id and track_id not in finalized_tracks):
                    if track_id:
                        finalized_tracks.add(track_id)

                    vehicle_crop = res.get("vehicle_crop")
                    plate_crop = res.get("plate_crop")
                    v_img_path = self._save_image(vehicle_crop, "vehicle") if vehicle_crop is not None else ""
                    p_img_path = self._save_image(plate_crop, "plate") if plate_crop is not None else ""

                    detections.append({
                        "frame": frame_count,
                        "track_id": track_id,
                        "plate": confirmed_plate,
                        "confidence": conf,
                        "vehicle_type": v_type,
                        "vehicle_image": v_img_path,
                        "plate_image": p_img_path,
                        "is_finalized": True,
                        "timestamp_sec": round(now_sec, 2),
                    })

            # Draw trajectory annotations on output frame
            if last_known_bbox:
                gx1, gy1, gx2, gy2 = last_known_bbox
                label = f"{last_known_plate_text} [FINAL]" if last_known_plate_text else "PLATE"
                cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (gx1, max(gy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            writer.write(annotated)

        cap.release()
        writer.release()

        return {
            "output_path": output_path,
            "detections": detections,
            "total_frames": frame_count,
            "finalized_count": len(detections),
        }

    def _save_image(self, image: Any, kind: str) -> str:
        """Save an image crop to disk and return the path."""
        if image is None:
            return ""
        img_arr = np.array(image)
        if img_arr.size == 0:
            return ""
        filename = get_safe_filename(f"{kind}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.png")
        path = self.vehicle_images_dir if kind == "vehicle" else self.plate_images_dir
        destination = path / filename
        cv2.imwrite(str(destination), img_arr)
        return str(destination)
