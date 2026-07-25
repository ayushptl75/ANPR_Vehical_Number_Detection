import unittest
from types import SimpleNamespace

from modules.ocr_reader import OCRReader


class OCRReaderTests(unittest.TestCase):
    def test_returns_plate_like_text_with_lower_confidence(self) -> None:
        reader = OCRReader()
        reader.reader = SimpleNamespace(readtext=lambda image: [(None, "ka01ab1234", 0.38)])

        result = reader.read(None)

        self.assertEqual(result["text"], "KA01AB1234")
        self.assertGreaterEqual(result["confidence"], 0.35)


if __name__ == "__main__":
    unittest.main()
