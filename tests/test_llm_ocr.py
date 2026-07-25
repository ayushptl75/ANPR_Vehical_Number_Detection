import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules.llm_ocr import LLMPlateOCR


class LLMPlateOCRTests(unittest.TestCase):
    def test_returns_valid_plate_from_llm_response(self) -> None:
        client = LLMPlateOCR(enabled=True, api_key="test-key", model="gpt-4o-mini", base_url="https://example.test")

        with patch.object(client, "_post_payload", return_value={"choices": [{"message": {"content": "GJ01AB1234"}}]}):
            result = client.read_plate("dummy")

        self.assertEqual(result["text"], "GJ01AB1234")
        self.assertGreaterEqual(result["confidence"], 0.85)

    def test_returns_none_when_llm_unavailable(self) -> None:
        client = LLMPlateOCR(enabled=False)
        result = client.read_plate("dummy")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
