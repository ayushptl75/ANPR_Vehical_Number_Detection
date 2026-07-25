import unittest

from modules.database_manager import DatabaseManager


class LiveScanHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = DatabaseManager()
        with self.db._connect() as conn:
            conn.execute("DELETE FROM live_scan_events")
            conn.commit()

    def test_add_and_retrieve_live_scan_events(self) -> None:
        self.db.add_live_scan_event(
            plate_number="KA01AB1234",
            vehicle_type="Car",
            confidence=0.92,
            camera_name="Webcam",
            source_url="0",
            gate_open=True,
            status="Approved",
        )

        rows = self.db.get_live_scan_events(limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plate_number"], "KA01AB1234")
        self.assertTrue(rows[0]["gate_open"])
        self.assertEqual(rows[0]["status"], "Approved")


if __name__ == "__main__":
    unittest.main()
