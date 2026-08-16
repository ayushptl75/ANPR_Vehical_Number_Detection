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

    def test_missing_model_file_triggers_fallback_status(self) -> None:
        detector = PlateDetector(model_path="non_existent_model_weights.pt")
        self.assertFalse(detector.is_dedicated)
        self.assertIn(detector.model_status, {"file_not_found", "no_path"})

    def test_is_valid_bbox_filters_invalid_geometries(self) -> None:
        detector = PlateDetector(model_path="")
        shape = (480, 640, 3)

        # Valid plate candidate (e.g. 150x40 px -> ratio 3.75)
        self.assertTrue(detector._is_valid_bbox([100, 100, 250, 140], shape, conf=0.85))

        # Invalid: full frame background crop
        self.assertFalse(detector._is_valid_bbox([0, 0, 638, 478], shape, conf=0.95))

        # Invalid: undersized width
        self.assertFalse(detector._is_valid_bbox([100, 100, 140, 120], shape, conf=0.85))

        # Invalid: square aspect ratio (e.g. 100x100 -> ratio 1.0)
        self.assertFalse(detector._is_valid_bbox([100, 100, 200, 200], shape, conf=0.85))

        # Invalid: out of frame bounds
        self.assertFalse(detector._is_valid_bbox([-10, 100, 250, 140], shape, conf=0.85))


if __name__ == "__main__":
    unittest.main()
