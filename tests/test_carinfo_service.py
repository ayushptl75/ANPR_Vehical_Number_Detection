"""Unit tests for CarInfo Vehicle Information API Service."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from services.carinfo_service import CarInfoService, get_vehicle_information


class TestCarInfoService(unittest.TestCase):
    """Test suite for CarInfoService."""

    def setUp(self) -> None:
        self.service = CarInfoService()
        self.service.test_mode = True  # Ensure test mode is active for unit tests

    def test_plate_normalization(self) -> None:
        """Test normalization of plate text (spaces, lowercase, character correction)."""
        self.assertEqual(self.service.normalize_plate_number("gj 05 ab 1234"), "GJ05AB1234")
        self.assertEqual(self.service.normalize_plate_number("GJ05AB1234"), "GJ05AB1234")

    def test_plate_validation(self) -> None:
        """Test Indian plate format validation."""
        self.assertTrue(self.service.validate_plate_number("GJ05AB1234"))
        self.assertTrue(self.service.validate_plate_number("KA01AB1234"))
        self.assertFalse(self.service.validate_plate_number("INVALID123"))

    def test_mock_response_vehicle_info(self) -> None:
        """Test fetching vehicle information in test mode returns expected schema."""
        info = self.service.get_vehicle_information("GJ05AB1234")
        self.assertTrue(info["verified"])
        self.assertEqual(info["status"], "VERIFIED")
        self.assertEqual(info["registration_number"], "GJ05AB1234")
        self.assertEqual(info["manufacturer"], "Hyundai")
        self.assertEqual(info["model"], "Creta SX")
        self.assertEqual(info["fuel_type"], "PETROL")
        self.assertIn("Active", info["insurance_status"])
        self.assertIn("Active", info["fitness_status"])
        self.assertIn("Active", info["pucc_status"])
        self.assertEqual(info["owner_information"], "Ketan Patel")

    def test_caching_mechanism(self) -> None:
        """Test debouncing / caching prevents duplicate API queries."""
        info1 = self.service.get_vehicle_information("GJ05AB1234")
        self.assertFalse(info1.get("is_cached", False))

        info2 = self.service.get_vehicle_information("GJ05AB1234")
        self.assertTrue(info2.get("is_cached", False))

    def test_low_confidence_filtering(self) -> None:
        """Test low confidence plate recognition skips API call."""
        info = self.service.get_vehicle_information("GJ05AB1234", ocr_confidence=0.1, plate_confidence=0.1)
        self.assertEqual(info["status"], "LOW CONFIDENCE")

        invalid_info = self.service.get_vehicle_information("123")
        self.assertEqual(invalid_info["status"], "INVALID FORMAT")


    @patch("services.carinfo_service.requests")
    def test_live_api_success(self, mock_requests: MagicMock) -> None:
        """Test live CarInfo API HTTP request parsing."""
        self.service.test_mode = False
        self.service.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "result": {
                "vehicle_class": "Motor Car",
                "manufacturer": "Tata",
                "model": "Harrier",
                "fuel_type": "DIESEL",
                "registration_date": "2023-01-01",
                "insurance_status": "Active",
                "insurance_expiry": "2026-01-01",
                "fitness_status": "Active",
                "pucc_status": "Active",
                "owner_name": "Test Owner",
            },
        }
        mock_requests.get.return_value = mock_response

        info = self.service.get_vehicle_information("MH12CD5678", force_refresh=True)
        self.assertTrue(info["verified"])
        self.assertEqual(info["manufacturer"], "Tata")
        self.assertEqual(info["model"], "Harrier")

    @patch("services.carinfo_service.requests")
    def test_live_api_http_errors(self, mock_requests: MagicMock) -> None:
        """Test live API HTTP error code handling (401, 404, 429, 500)."""
        self.service.test_mode = False
        self.service.api_key = "test_key"

        # 404 Not Found
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_requests.get.return_value = mock_response_404

        info = self.service.get_vehicle_information("DL01XX9999", force_refresh=True)
        self.assertFalse(info["verified"])
        self.assertEqual(info["status"], "API ERROR")
        self.assertIn("not found", info["error"].lower())

    def test_module_level_function(self) -> None:
        """Test module-level helper get_vehicle_information."""
        info = get_vehicle_information("GJ05AB1234")
        self.assertTrue(info["verified"])


if __name__ == "__main__":
    unittest.main()
