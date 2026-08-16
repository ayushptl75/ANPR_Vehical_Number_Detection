import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts.evaluate_anpr import ANPREvaluator, calculate_bbox_iou, calculate_levenshtein_distance, calculate_ocr_accuracy


class EvaluationTests(unittest.TestCase):
    def test_levenshtein_distance_and_ocr_accuracy(self) -> None:
        self.assertEqual(calculate_levenshtein_distance("MH12AB1234", "MH12AB1234"), 0)
        self.assertEqual(calculate_levenshtein_distance("MH12A81234", "MH12AB1234"), 1)
        self.assertEqual(calculate_ocr_accuracy("MH12AB1234", "MH12AB1234"), 100.0)
        self.assertEqual(calculate_ocr_accuracy("MH12A81234", "MH12AB1234"), 90.0)

    def test_bbox_iou_calculation(self) -> None:
        box1 = [100, 100, 200, 200]
        box2 = [100, 100, 200, 200]
        self.assertAlmostEqual(calculate_bbox_iou(box1, box2), 1.0)

        box3 = [150, 100, 250, 200]
        iou = calculate_bbox_iou(box1, box3)
        self.assertTrue(0.0 < iou < 1.0)

    def test_anpr_evaluator_runs_and_generates_reports(self) -> None:
        mock_vehicle = SimpleNamespace(
            detect=lambda img: [{"bbox": [100, 100, 540, 380], "confidence": 0.95, "label": "car"}]
        )
        mock_plate = SimpleNamespace(
            detect=lambda crop: [{"bbox": [220, 280, 420, 330], "confidence": 0.95}]
        )
        mock_ocr = SimpleNamespace(
            read=lambda crop: {
                "plate_number": "MH12AB1234",
                "ocr_confidence": 0.95,
                "raw_text": "MH12AB1234",
                "validation_status": "VALID_REGISTRATION",
                "valid": True,
            }
        )
        from modules.anpr_pipeline import ANPRPipeline
        mock_pipeline = ANPRPipeline(mock_vehicle, mock_plate, mock_ocr)

        evaluator = ANPREvaluator(pipeline=mock_pipeline)
        results = evaluator.run_evaluation()

        self.assertIn("overall_metrics", results)
        self.assertEqual(results["overall_metrics"]["total_categories_tested"], 12)
        self.assertTrue(Path("reports/anpr_evaluation_results.csv").exists())
        self.assertTrue(Path("reports/anpr_evaluation_summary.md").exists())
        self.assertTrue(Path("eval_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
