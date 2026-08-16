import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from modules.anpr_pipeline import ANPRPipeline


class ImagePipelineTests(unittest.TestCase):
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
        self.pipeline = ANPRPipeline(
            vehicle_detector=mock_vehicle,
            plate_detector=mock_plate,
            ocr_reader=mock_ocr,
        )

    def test_process_image_success_flow(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        res = self.pipeline.process_image(image)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["plate_number"], "MH12AB1234")
        self.assertEqual(res["ocr_confidence"], 0.92)
        self.assertEqual(res["detected_vehicle_type"], "CAR")
        self.assertIn("vehicle_crop", res)
        self.assertIn("plate_crop", res)
        self.assertIn("annotated_frame", res)

    def test_process_image_rejects_empty_frame(self) -> None:
        res = self.pipeline.process_image(np.array([]))
        self.assertEqual(res["status"], "FAILED")
        self.assertEqual(res["reason"], "Empty image provided")

    def test_process_image_saves_debugging_artifacts(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        self.pipeline.process_image(image)

        output_dir = Path("static/output")
        self.assertTrue((output_dir / "original.jpg").exists())
        self.assertTrue((output_dir / "vehicle_detected.jpg").exists())
        self.assertTrue((output_dir / "cropped_plate.jpg").exists())
        self.assertTrue((output_dir / "enhanced_plate.jpg").exists())


if __name__ == "__main__":
    unittest.main()
