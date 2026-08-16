import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from modules.video_processor import VideoProcessor


class VideoTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        mock_vehicle = SimpleNamespace(
            detect=lambda img: [{"bbox": [10, 10, 200, 200], "confidence": 0.95, "label": "car"}]
        )
        mock_plate = SimpleNamespace(
            detect=lambda crop: [{"bbox": [20, 20, 180, 60], "confidence": 0.95}]
        )
        mock_ocr = SimpleNamespace(
            read=lambda crop: {
                "plate_number": "MH12AB1234",
                "ocr_confidence": 0.92,
                "raw_text": "MH12AB1234",
                "validation_status": "VALID_REGISTRATION",
                "processing_variant": "clahe",
                "valid": True,
            }
        )
        self.processor = VideoProcessor(
            vehicle_detector=mock_vehicle,
            plate_detector=mock_plate,
            ocr_reader=mock_ocr,
        )

    def test_process_static_image_returns_pipeline_metadata(self) -> None:
        # Create temporary dummy image
        img_path = Path("static/output/test_sample.jpg")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(img_path), np.zeros((300, 400, 3), dtype=np.uint8))

        res = self.processor.process_static_image(str(img_path))
        self.assertEqual(res["plate_number"], "MH12AB1234")
        self.assertEqual(res["detected_vehicle_type"], "CAR")
        self.assertTrue(res["plate_selected"])
        self.assertTrue(Path(res["processed_image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
