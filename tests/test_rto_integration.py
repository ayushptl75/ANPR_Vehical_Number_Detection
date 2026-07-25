import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config
from modules.database_manager import DatabaseManager


class RTOIntegrationTests(unittest.TestCase):
    def test_fetch_and_save_rto_record_uses_provider_headers(self) -> None:
        db = DatabaseManager()
        response = SimpleNamespace(
            status_code=200,
            json=lambda: {
                "plate_number": "DL01AB1234",
                "vehicle_type": "Car",
                "manufacturer": "Honda",
                "model": "City",
                "owner_name": "Test User",
            },
        )

        with patch("requests.get", return_value=response) as mock_get, \
             patch.object(config.Config, "RTO_API_URL", "https://example.test/api"), \
             patch.object(config.Config, "RTO_API_PROVIDER", "demo"), \
             patch.object(config.Config, "RTO_API_KEY", "secret-key"), \
             patch.object(config.Config, "RTO_API_CLIENT_ID", "client-id"), \
             patch.object(config.Config, "RTO_API_CLIENT_SECRET", "client-secret"):
            result = db.fetch_and_save_rto_record("DL01AB1234")

        self.assertIsNotNone(result)
        kwargs = mock_get.call_args.kwargs
        self.assertEqual(kwargs["params"]["plate"], "DL01AB1234")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(kwargs["headers"]["X-Client-Id"], "client-id")
        self.assertEqual(kwargs["headers"]["X-Client-Secret"], "client-secret")
        self.assertIsNotNone(db.get_vehicle_record("DL01AB1234"))


if __name__ == "__main__":
    unittest.main()
