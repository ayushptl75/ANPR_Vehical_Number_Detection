import unittest

from modules.tracker import PlateTrack, VehiclePlateTracker


class TrackerTests(unittest.TestCase):
    def test_multi_frame_voting_resolves_conflicting_ocr_results(self) -> None:
        track = PlateTrack("tr_1", [100, 100, 400, 400], [150, 200, 350, 250], "car")

        # Frame 1: MH12AB1234 (conf 0.90)
        track.add_observation("MH12AB1234", 0.90, valid=True)
        plate, conf, is_final = track.evaluate_voting(min_observations=3, min_confidence=0.45)
        self.assertFalse(is_final)

        # Frame 2: MH12A8134 (conf 0.85 - corrupted OCR outlier)
        track.add_observation("MH12A8134", 0.85, valid=True)
        plate, conf, is_final = track.evaluate_voting(min_observations=3, min_confidence=0.45)
        self.assertFalse(is_final)

        # Frame 3: MH12AB1234 (conf 0.92)
        track.add_observation("MH12AB1234", 0.92, valid=True)
        plate, conf, is_final = track.evaluate_voting(min_observations=3, min_confidence=0.45)
        self.assertFalse(is_final)

        # Frame 4: MH12AB1234 (conf 0.94) -> 3rd observation for MH12AB1234 -> Finalized!
        track.add_observation("MH12AB1234", 0.94, valid=True)
        plate, conf, is_final = track.evaluate_voting(min_observations=3, min_confidence=0.45)

        self.assertTrue(is_final)
        self.assertEqual(plate, "MH12AB1234")
        self.assertGreaterEqual(conf, 0.90)

    def test_low_confidence_observations_do_not_finalize(self) -> None:
        track = PlateTrack("tr_2", [100, 100, 400, 400], [150, 200, 350, 250], "car")

        # 5 low confidence observations (0.20 < 0.45 min threshold)
        for _ in range(5):
            track.add_observation("DL1CG5692", 0.20, valid=True)

        plate, conf, is_final = track.evaluate_voting(min_observations=3, min_confidence=0.45)
        self.assertFalse(is_final)

    def test_vehicle_plate_tracker_associations_and_deduplication(self) -> None:
        tracker = VehiclePlateTracker(min_observations=3, min_confidence=0.45, timeout_seconds=5.0)

        bbox = [100, 100, 500, 500]
        plate_bbox = [150, 200, 350, 250]
        ocr_valid = {"plate_number": "KA01AB1234", "ocr_confidence": 0.95, "valid": True}

        # Frame 1
        res1 = tracker.process_detection(bbox, plate_bbox, "car", ocr_valid)
        self.assertFalse(res1["is_finalized"])
        self.assertFalse(res1["is_newly_finalized"])

        # Frame 2
        res2 = tracker.process_detection(bbox, plate_bbox, "car", ocr_valid)
        self.assertFalse(res2["is_finalized"])

        # Frame 3 -> Should finalize!
        res3 = tracker.process_detection(bbox, plate_bbox, "car", ocr_valid)
        self.assertTrue(res3["is_finalized"])
        self.assertTrue(res3["is_newly_finalized"])
        self.assertEqual(res3["plate_number"], "KA01AB1234")

        # Frame 4 -> Still finalized, but NOT newly finalized (prevents duplicate DB insertion)
        res4 = tracker.process_detection(bbox, plate_bbox, "car", ocr_valid)
        self.assertTrue(res4["is_finalized"])
        self.assertFalse(res4["is_newly_finalized"])

    def test_stale_track_cleanup(self) -> None:
        tracker = VehiclePlateTracker(min_observations=3, min_confidence=0.45, timeout_seconds=2.0)
        bbox = [100, 100, 300, 300]
        ocr = {"plate_number": "MH12AB1234", "ocr_confidence": 0.90, "valid": True}

        # Frame at t = 100.0
        res = tracker.process_detection(bbox, bbox, "car", ocr, now=100.0)
        track_id = res["track_id"]
        self.assertIn(track_id, tracker.tracks)

        # Process frame at t = 105.0 (> 2s timeout) -> Track should be purged
        tracker.process_detection(None, None, "car", {}, now=105.0)
        self.assertNotIn(track_id, tracker.tracks)


if __name__ == "__main__":
    unittest.main()
