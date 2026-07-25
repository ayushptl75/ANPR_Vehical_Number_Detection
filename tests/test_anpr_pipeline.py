import unittest
from types import SimpleNamespace

import numpy as np

from modules.anpr_pipeline import ANPRPipeline


class ANPRPipelineTests(unittest.TestCase):
    def test_returns_failed_when_ocr_confidence_is_low(self) -> None:
        pipeline = ANPRPipeline(
            vehicle_detector=SimpleNamespace(detect=lambda frame: [{"bbox": [0, 0, 200, 200], "confidence": 0.95, "label": "car"}]),
            plate_detector=SimpleNamespace(detect=lambda frame: [{"bbox": [20, 20, 180, 60], "confidence": 0.95}]),
            ocr_reader=SimpleNamespace(read=lambda image: {"text": "gj01ab1234", "confidence": 0.5, "bboxes": []}),
            llm_reader=None,
        )

        result = pipeline.process_frame(np.zeros((240, 320, 3), dtype=np.uint8), camera_state="Gujarat")

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["reason"], "Low OCR confidence")


if __name__ == "__main__":
    unittest.main()
