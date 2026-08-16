import unittest

from app import app, database_manager


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_dashboard_stats_keys(self) -> None:
        stats = database_manager.get_dashboard_stats()
        self.assertIn("total_vehicles_today", stats)
        self.assertIn("unique_vehicles_today", stats)
        self.assertIn("total_entries", stats)
        self.assertIn("blacklist_count", stats)
        self.assertIn("verification_rate", stats)
        self.assertIn("live_camera_status", stats)

    def test_dashboard_route_renders_without_error(self) -> None:
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "admin"

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("ANPR Control Dashboard", html)
        self.assertIn("Scans Today", html)
        self.assertIn("Verification Rate", html)


if __name__ == "__main__":
    unittest.main()
