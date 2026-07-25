import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from modules.plate_detector import PlateDetector


class PlateDetectorTests(unittest.TestCase):
    def test_prefers_tight_plate_bbox_and_filters_large_candidates(self) -> None:
        detector = PlateDetector(model_path="")
        image = np.zeros((240, 320, 3), dtype=np.uint8)

        candidates = [
            {"bbox": [0, 80, 320, 200], "confidence": 0.95},
            {"bbox": [80, 120, 190, 155], "confidence": 0.97},
            {"bbox": [70, 125, 180, 160], "confidence": 0.76},
        ]

        selected = detector._select_best_detection(candidates, image.shape)

        self.assertIsNotNone(selected)
        self.assertEqual(selected["bbox"], [80, 120, 190, 155])

    def test_resolves_model_path_from_workspace_candidates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            model_file = Path(tmpdir) / "models" / "best.pt"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_bytes(b"dummy")

            detector = PlateDetector(model_path="")
            resolved = detector._resolve_model_path("best.pt", search_roots=[Path(tmpdir)])

            self.assertEqual(resolved, str(model_file))


if __name__ == "__main__":
    unittest.main()
