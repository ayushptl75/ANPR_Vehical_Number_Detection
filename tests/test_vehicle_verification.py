import unittest

from modules.database_manager import DatabaseManager


class VehicleVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = DatabaseManager()

    def test_non_existent_plate_returns_none_without_fabrication(self) -> None:
        # Non-existent plate should NOT return a fabricated record or auto-create dummy owner
        plate = "UNREGISTERED9999"
        info = self.db.get_vehicle_info(plate)
        self.assertIsNone(info)

    def test_seed_and_local_database_records_tagged_as_not_verified(self) -> None:
        # Local database record (e.g. KA01AB1234) should be tagged source='local_database', verified=0
        info = self.db.get_vehicle_info("KA01AB1234")
        self.assertIsNotNone(info)
        self.assertEqual(info["verification_status"], "NOT VERIFIED")
        self.assertEqual(info["source_label"], "Local Database")
        self.assertFalse(info["verified"])

    def test_add_vehicle_record_defaults_to_local_unverified(self) -> None:
        plate = "MH14AB9999"
        self.db.add_vehicle_record(plate, vehicle_type="Car", owner_name="Local Owner")

        info = self.db.get_vehicle_info(plate)
        self.assertIsNotNone(info)
        self.assertEqual(info["verification_status"], "NOT VERIFIED")
        self.assertEqual(info["source"], "local_database")
        self.assertFalse(info["verified"])

    def test_approve_rto_import_tags_as_imported_dataset_not_verified(self) -> None:
        import_id = self.db.add_rto_import({"plate_number": "GJ01AB5555", "owner_name": "CSV Owner", "vehicle_type": "Truck"})
        success = self.db.approve_rto_import(import_id)
        self.assertTrue(success)

        info = self.db.get_vehicle_info("GJ01AB5555")
        self.assertIsNotNone(info)
        self.assertEqual(info["source"], "imported_dataset")
        self.assertEqual(info["verification_status"], "NOT VERIFIED")
        self.assertEqual(info["source_label"], "Imported Dataset")
        self.assertFalse(info["verified"])


if __name__ == "__main__":
    unittest.main()
