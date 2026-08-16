"""Multi-frame vehicle/plate tracker with confidence-weighted voting and temporal state management."""
from __future__ import annotations

import time
from typing import Any, Sequence

from config import Config
from modules.geometry import clamp_bbox
from modules.utils import is_valid_indian_plate


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
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
    return inter / float(union) if union > 0 else 0.0


def _centroid_dist(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 99999.0
    cx_a = (a[0] + a[2]) / 2.0
    cy_a = (a[1] + a[3]) / 2.0
    cx_b = (b[0] + b[2]) / 2.0
    cy_b = (b[1] + b[3]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


class PlateTrack:
    """Temporary multi-frame recognition state for a single vehicle/plate track."""

    def __init__(self, track_id: str, vehicle_bbox: list[int], global_plate_bbox: list[int], vehicle_type: str, now: float | None = None) -> None:
        self.track_id = track_id
        self.created_at = now if now is not None else time.time()
        self.last_seen = self.created_at
        self.vehicle_bbox = vehicle_bbox
        self.global_plate_bbox = global_plate_bbox
        self.vehicle_type = vehicle_type
        
        self.observations: list[dict[str, Any]] = []
        self.is_finalized = False
        self.final_plate = ""
        self.final_confidence = 0.0
        self.db_logged = False

    def add_observation(self, plate_number: str, confidence: float, valid: bool, status: str = "", now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        self.last_seen = ts
        if plate_number:
            self.observations.append({
                "plate_number": plate_number,
                "confidence": float(confidence),
                "valid": bool(valid),
                "status": status,
                "timestamp": ts,
            })

    def evaluate_voting(self, min_observations: int = 3, min_confidence: float = 0.45) -> tuple[str, float, bool]:
        """Perform confidence-weighted voting across collected observations.
        
        Returns tuple: (best_plate, best_confidence, is_confirmed)
        """
        if self.is_finalized:
            return self.final_plate, self.final_confidence, True

        if not self.observations:
            return "", 0.0, False

        # Group observations by plate_number
        plate_groups: dict[str, list[dict[str, Any]]] = {}
        for obs in self.observations:
            p = obs["plate_number"]
            if not p:
                continue
            if p not in plate_groups:
                plate_groups[p] = []
            plate_groups[p].append(obs)

        if not plate_groups:
            return "", 0.0, False

        best_plate = ""
        best_score = -1.0
        best_avg_conf = 0.0

        for plate, obs_list in plate_groups.items():
            valid_bonus = 1.5 if is_valid_indian_plate(plate) else 0.8
            weighted_score = sum(obs["confidence"] * valid_bonus for obs in obs_list)
            avg_conf = sum(obs["confidence"] for obs in obs_list) / float(len(obs_list))
            
            # Prioritize total weighted score + observation frequency
            if weighted_score > best_score:
                best_score = weighted_score
                best_plate = plate
                best_avg_conf = avg_conf

        plate_obs_count = len(plate_groups.get(best_plate, []))
        is_valid = is_valid_indian_plate(best_plate)

        # Finalize only when consistent observations exist, confidence is sufficient, and format is valid
        if plate_obs_count >= min_observations and best_avg_conf >= min_confidence and is_valid:
            self.is_finalized = True
            self.final_plate = best_plate
            self.final_confidence = round(best_avg_conf, 2)
            return self.final_plate, self.final_confidence, True

        return best_plate, round(best_avg_conf, 2), False


class VehiclePlateTracker:
    """Coordinate multi-frame tracking and candidate aggregation across video streams."""

    def __init__(
        self,
        min_observations: int | None = None,
        min_confidence: float | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.min_observations = min_observations if min_observations is not None else Config.TRACKER_MIN_OBSERVATIONS
        self.min_confidence = min_confidence if min_confidence is not None else Config.TRACKER_CONFIDENCE_THRESHOLD
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else Config.TRACKER_TIMEOUT_SECONDS
        
        self.tracks: dict[str, PlateTrack] = {}
        self._next_id = 100

    def process_detection(
        self,
        vehicle_bbox: list[int] | None,
        global_plate_bbox: list[int] | None,
        vehicle_type: str,
        ocr_result: dict[str, Any],
        now: float | None = None,
    ) -> dict[str, Any]:
        """Update tracker with frame detection, associate track, run voting, and return tracking state."""
        current_time = now if now is not None else time.time()

        # 1. Clean up stale tracks
        self._purge_stale_tracks(current_time)

        if not vehicle_bbox and not global_plate_bbox:
            return {
                "track_id": None,
                "plate_number": "",
                "confidence": 0.0,
                "is_finalized": False,
                "is_newly_finalized": False,
                "observation_count": 0,
            }

        target_bbox = global_plate_bbox if global_plate_bbox else vehicle_bbox
        matched_track = self._find_matching_track(target_bbox)

        if matched_track is None:
            self._next_id += 1
            track_id = f"tr_{self._next_id}"
            matched_track = PlateTrack(
                track_id=track_id,
                vehicle_bbox=vehicle_bbox or [0, 0, 0, 0],
                global_plate_bbox=global_plate_bbox or [0, 0, 0, 0],
                vehicle_type=vehicle_type,
                now=current_time,
            )
            self.tracks[track_id] = matched_track

        # Update track state with new observation
        if vehicle_bbox:
            matched_track.vehicle_bbox = vehicle_bbox
        if global_plate_bbox:
            matched_track.global_plate_bbox = global_plate_bbox

        plate_text = ocr_result.get("plate_number") or ocr_result.get("text") or ""
        ocr_conf = float(ocr_result.get("ocr_confidence") or ocr_result.get("confidence") or 0.0)
        is_valid = bool(ocr_result.get("valid", is_valid_indian_plate(plate_text)))
        status = str(ocr_result.get("validation_status") or "")

        was_finalized_before = matched_track.is_finalized

        if plate_text:
            matched_track.add_observation(plate_text, ocr_conf, is_valid, status, now=current_time)

        best_plate, best_conf, is_finalized = matched_track.evaluate_voting(
            min_observations=self.min_observations,
            min_confidence=self.min_confidence,
        )

        is_newly_finalized = is_finalized and not was_finalized_before

        plate_obs_list = [o for o in matched_track.observations if o["plate_number"] == best_plate]
        obs_count = len(plate_obs_list) if best_plate else len(matched_track.observations)

        return {
            "track_id": matched_track.track_id,
            "plate_number": best_plate,
            "confidence": best_conf,
            "is_finalized": is_finalized,
            "is_newly_finalized": is_newly_finalized,
            "observation_count": obs_count,
            "track": matched_track,
        }

    def _find_matching_track(self, bbox: list[int]) -> PlateTrack | None:
        """Find an active track matching bbox via IoU or centroid distance."""
        best_track = None
        best_match_score = -1.0

        for track in self.tracks.values():
            ref_bbox = track.global_plate_bbox if track.global_plate_bbox != [0, 0, 0, 0] else track.vehicle_bbox
            iou_score = _iou(bbox, ref_bbox)
            dist = _centroid_dist(bbox, ref_bbox)

            if iou_score >= 0.25 or dist < 120.0:
                # Combined match score (higher IoU or closer distance)
                match_score = iou_score + (1.0 / (1.0 + dist / 100.0))
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_track = track

        return best_track

    def _purge_stale_tracks(self, now: float) -> None:
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if (now - track.last_seen) > self.timeout_seconds
        ]
        for tid in stale_ids:
            del self.tracks[tid]
