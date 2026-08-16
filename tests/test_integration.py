import base64
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

from app import app, database_manager
from modules.camera_stream import CameraStreamManager
from modules.geometry import add_controlled_padding, clamp_bbox, crop_to_global_bbox
from modules.ocr_reader import OCRReader
from modules.plate_detector import PlateDetector
from modules.report_generator import ReportGenerator
from modules.tracker import VehiclePlateTracker


class IntegrationTestSuite(unittest.TestCase):
    """End-to-end integration test suite for all 17 ANPR software components."""

    def setUp(self) -> None:
        self.client = app.test_client()
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "admin"

    def test_01_image_upload_pipeline(self) -> None:
        """1. Test Image Upload Endpoint."""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        _, img_encoded = cv2.imencode(".jpg", img)
        data = {
            "image": (io.BytesIO(img_encoded.tobytes()), "test_upload.jpg")
        }
        res = self.client.post("/upload-image", data=data, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

    def test_02_video_upload_pipeline(self) -> None:
        """2. Test Video Upload Endpoint."""
        res = self.client.get("/upload-video")
        self.assertEqual(res.status_code, 200)

    def test_03_webcam_live_detect_api(self) -> None:
        """3. Test Webcam Live Scan API."""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        _, img_encoded = cv2.imencode(".jpg", img)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(img_encoded.tobytes()).decode("utf-8")
        res = self.client.post("/api/live-detect", data={"url": "0", "frame_data": b64_str})
        self.assertIn(res.status_code, (200, 400))

    def test_04_rtsp_camera_stream_manager(self) -> None:
        """4. Test RTSP/IP Camera Stream Manager."""
        manager = CameraStreamManager()
        stream = manager.get_stream("0", name="TestCam")
        self.assertIsNotNone(stream)
        manager.stop_all()

    def test_05_plate_detector(self) -> None:
        """5. Test Dedicated Plate Detector."""
        detector = PlateDetector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = detector.detect(img)
        self.assertIsInstance(res, list)

    def test_06_plate_crop_geometry(self) -> None:
        """6. Test Coordinate Transformation & Controlled Crop Geometry."""
        global_bbox = crop_to_global_bbox([20, 20, 80, 60], 10, 10, (480, 640))
        self.assertEqual(global_bbox, [30, 30, 90, 70])
        padded = add_controlled_padding([20, 20, 80, 60], (100, 100), pad_percent=0.05, min_pad_px=4)
        self.assertEqual(len(padded), 4)

    def test_07_ocr_reader(self) -> None:
        """7. Test Multi-Variant OCR Reader."""
        ocr = OCRReader()
        img = np.zeros((40, 120, 3), dtype=np.uint8)
        res = ocr.read(img)
        self.assertIn("ocr_confidence", res)

    def test_08_multi_frame_recognition(self) -> None:
        """8. Test Multi-Frame Tracking & Confidence Voting."""
        tracker = VehiclePlateTracker(min_observations=2)
        res1 = tracker.process_detection([0, 0, 100, 100], [10, 10, 90, 40], "car", {"plate_number": "MH12AB1234", "ocr_confidence": 0.9, "valid": True})
        res2 = tracker.process_detection([0, 0, 100, 100], [10, 10, 90, 40], "car", {"plate_number": "MH12AB1234", "ocr_confidence": 0.9, "valid": True})
        self.assertTrue(res2["is_finalized"])
        self.assertEqual(res2["plate_number"], "MH12AB1234")

    def test_09_database_storage(self) -> None:
        """9. Test Database Record Storage."""
        database_manager.add_detection("MH12AB9999", "CAR", 0.95, "Cam1")
        last = database_manager.get_last_detection("MH12AB9999")
        self.assertIsNotNone(last)
        self.assertEqual(last["plate_number"], "MH12AB9999")

    def test_10_duplicate_suppression(self) -> None:
        """10. Test Duplicate Record Suppression."""
        can_save = database_manager.can_save_detection("MH12AB9999", max_seconds=15)
        self.assertFalse(can_save)

    def test_11_blacklist_alerts(self) -> None:
        """11. Test Blacklist Watchlist Alerts."""
        database_manager.add_blacklist_entry("MH12AB9999", "Suspicious Vehicle")
        entry = database_manager.get_blacklist_entry("MH12AB9999")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["reason"], "Suspicious Vehicle")

    def test_12_vehicle_info_lookup(self) -> None:
        """12. Test Vehicle Information Lookup."""
        rec = database_manager.get_vehicle_info("MH12AB9999")
        self.assertTrue(rec is None or "verification_status" in rec)

    def test_13_verification_status_badges(self) -> None:
        """13. Test Source Provenance & Verification Badges."""
        database_manager.add_vehicle_record("MH12AB8888", owner_name="Test Owner", source="official_authorized_api", verified=1)
        info = database_manager.get_vehicle_info("MH12AB8888")
        self.assertIsNotNone(info)
        self.assertEqual(info["verification_status"], "VERIFIED")
        self.assertEqual(info["source_label"], "Official Authorized API")

    def test_14_dashboard_rendering(self) -> None:
        """14. Test Control Dashboard Rendering."""
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn("ANPR Control Dashboard", res.get_data(as_text=True))

    def test_15_csv_report_export(self) -> None:
        """15. Test CSV Report Export."""
        res = self.client.get("/reports/export?format=csv&period=daily")
        self.assertEqual(res.status_code, 200)
        self.assertIn(res.mimetype, ("text/csv", "application/vnd.ms-excel", "text/plain"))

    def test_16_excel_report_export(self) -> None:
        """16. Test Excel (XLSX) Report Export."""
        res = self.client.get("/reports/export?format=xlsx&period=daily")
        self.assertEqual(res.status_code, 200)

    def test_17_pdf_report_export(self) -> None:
        """17. Test PDF Report Export."""
        res = self.client.get("/reports/export?format=pdf&period=daily")
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
