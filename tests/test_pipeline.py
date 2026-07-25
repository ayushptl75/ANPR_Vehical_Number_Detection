import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from modules.anpr_pipeline import ANPRPipeline


class ANPRPipelineTests(unittest.TestCase):
    def test_pipeline_returns_success_with_tight_plate_crop_and_debug_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            pipeline = ANPRPipeline(
                vehicle_detector=SimpleNamespace(detect=lambda frame: [{"bbox": [0, 0, 200, 200], "confidence": 0.95, "label": "car"}]),
                plate_detector=SimpleNamespace(detect=lambda frame: [{"bbox": [40, 60, 150, 95], "confidence": 0.91}]),
                ocr_reader=SimpleNamespace(read=lambda image: {"text": "DL1CG5692", "confidence": 0.95, "bboxes": []}),
                llm_reader=SimpleNamespace(normalize=lambda text: (text, True)),
            )
            pipeline.output_dir = Path(tmpdir)
            image = np.zeros((240, 320, 3), dtype=np.uint8)

            result = pipeline.process_frame(image, camera_state="Delhi")

            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["plate"], "DL1CG5692")
            self.assertTrue((Path(tmpdir) / "cropped_plate.jpg").exists())
            self.assertTrue((Path(tmpdir) / "enhanced_plate.jpg").exists())


if __name__ == "__main__":
    unittest.main()
