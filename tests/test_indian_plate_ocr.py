import unittest
from types import SimpleNamespace

from modules.ocr_reader import OCRReader
from modules.utils import (
    is_valid_indian_plate,
    normalize_plate_text,
    positionally_correct_plate_text,
    validate_indian_plate_with_details,
)


class IndianPlateOCRTests(unittest.TestCase):
    def test_normalization_removes_noise_spaces_punctuation(self) -> None:
        raw_inputs = [
            "  MH 12 - AB - 1234 \n",
            "DL-1CG--5692.\t",
            "22 BH 1234 A\r\n",
        ]
        expected = ["MH12AB1234", "DL1CG5692", "22BH1234A"]
        for inp, exp in zip(raw_inputs, expected):
            self.assertEqual(normalize_plate_text(inp), exp)

    def test_valid_indian_plate_schemas(self) -> None:
        valid_plates = [
            "MH12AB1234",  # Standard 2-digit district
            "DL1CG5692",   # Single-digit district
            "22BH1234A",   # BH Series
            "KA01MB9876",  # Standard 2-digit district
        ]
        for plate in valid_plates:
            self.assertTrue(is_valid_indian_plate(plate), f"Plate {plate} should be valid")

    def test_position_aware_character_confusion_corrections(self) -> None:
        # 0/O, 1/I, 5/S, 8/B confusion scenarios
        scenarios = [
            ("MHI2AB1234", "MH12AB1234"),    # I -> 1 in district
            ("MH12ABS234", "MH12AB5234"),    # S -> 5 in 4-digit suffix
            ("DLICG5692", "DL1CG5692"),      # I -> 1 in single-digit district
        ]
        for corrupted, expected in scenarios:
            corrected = positionally_correct_plate_text(corrupted)
            self.assertEqual(corrected, expected, f"Failed for {corrupted}")
            self.assertTrue(is_valid_indian_plate(corrected))

    def test_validate_indian_plate_with_details(self) -> None:
        # Valid plate
        res1 = validate_indian_plate_with_details("MH12AB1234", confidence=0.92)
        self.assertTrue(res1["is_valid"])
        self.assertEqual(res1["validation_status"], "VALID_REGISTRATION")
        self.assertFalse(res1["was_corrected"])

        # Format corrected plate
        res2 = validate_indian_plate_with_details("MHI2AB1234", confidence=0.88)
        self.assertTrue(res2["is_valid"])
        self.assertEqual(res2["validation_status"], "FORMAT_CORRECTED")
        self.assertTrue(res2["was_corrected"])

        # Low confidence plate
        res3 = validate_indian_plate_with_details("INVALIDTEXT123", confidence=0.20, min_confidence=0.45)
        self.assertFalse(res3["is_valid"])
        self.assertEqual(res3["validation_status"], "LOW_CONFIDENCE")

    def test_ocr_reader_structured_response(self) -> None:
        reader = OCRReader()
        reader.reader = SimpleNamespace(readtext=lambda img: [(None, "dl1cg5692", 0.95)])

        result = reader.read(None)

        self.assertIn("plate_number", result)
        self.assertIn("ocr_confidence", result)
        self.assertIn("validation_status", result)
        self.assertIn("processing_variant", result)

        self.assertEqual(result["plate_number"], "DL1CG5692")
        self.assertEqual(result["ocr_confidence"], 0.95)
        self.assertEqual(result["validation_status"], "VALID_REGISTRATION")


if __name__ == "__main__":
    unittest.main()
