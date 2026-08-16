import unittest
import numpy as np

from modules.geometry import add_controlled_padding, validate_plate_crop_dimensions
from modules.preprocessing import PlatePreprocessor
from services.image_processor import correct_plate_perspective


class PreprocessingTests(unittest.TestCase):
    def test_validate_plate_crop_dimensions(self) -> None:
        # Valid plate crop: 150x40 -> ratio 3.75
        valid_crop = np.zeros((40, 150, 3), dtype=np.uint8)
        self.assertTrue(validate_plate_crop_dimensions(valid_crop, min_width=60, min_height=18, min_aspect=2.0, max_aspect=6.0))

        # Invalid: too small width
        small_width = np.zeros((40, 50, 3), dtype=np.uint8)
        self.assertFalse(validate_plate_crop_dimensions(small_width, min_width=60, min_height=18))

        # Invalid: too small height
        small_height = np.zeros((10, 150, 3), dtype=np.uint8)
        self.assertFalse(validate_plate_crop_dimensions(small_height, min_width=60, min_height=18))

        # Invalid: square aspect ratio (100x100 -> ratio 1.0)
        square_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertFalse(validate_plate_crop_dimensions(square_crop, min_aspect=2.0, max_aspect=6.0))

    def test_independent_xy_padding(self) -> None:
        shape = (480, 640, 3)
        bbox = [100, 100, 200, 140]  # width 100, height 40

        # pad_x_percent = 0.10 -> 10px, pad_y_percent = 0.05 -> max(4, 2) = 4px
        padded = add_controlled_padding(bbox, shape, pad_x_percent=0.10, pad_y_percent=0.05, min_pad_px=4)
        self.assertEqual(padded, [90, 96, 210, 144])

    def test_generate_variants_preserves_original_image(self) -> None:
        preprocessor = PlatePreprocessor(enable_perspective=False)
        original = np.ones((50, 180, 3), dtype=np.uint8) * 128
        copy_original = original.copy()

        variants = preprocessor.generate_variants(original)

        self.assertGreaterEqual(len(variants), 5)
        # Verify original input array was not mutated
        np.testing.assert_array_equal(original, copy_original)
        # Verify variant names
        names = [v["name"] for v in variants]
        self.assertIn("original", names)
        self.assertIn("grayscale", names)
        self.assertIn("upscaled", names)
        self.assertIn("clahe", names)

    def test_perspective_correction_runs_safely(self) -> None:
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        warped = correct_plate_perspective(img)
        self.assertIsNotNone(warped)
        self.assertGreater(warped.size, 0)


if __name__ == "__main__":
    unittest.main()
