import unittest

from modules.geometry import (
    add_controlled_padding,
    clamp_bbox,
    crop_to_global_bbox,
    is_bbox_center_inside,
    scale_bbox,
)


class GeometryTests(unittest.TestCase):
    def test_clamp_bbox_bounds(self) -> None:
        shape = (480, 640, 3)
        # Out of bounds left/top
        self.assertEqual(clamp_bbox([-10, -5, 200, 150], shape), [0, 0, 200, 150])
        # Out of bounds right/bottom
        self.assertEqual(clamp_bbox([100, 100, 700, 500], shape), [100, 100, 640, 480])

    def test_crop_to_global_bbox_offset(self) -> None:
        shape = (1080, 1920, 3)
        # Vehicle at [500, 300, 1200, 800]
        # Plate detected inside vehicle crop at [50, 100, 250, 160]
        global_box = crop_to_global_bbox([50, 100, 250, 160], offset_x=500, offset_y=300, frame_shape=shape)
        self.assertEqual(global_box, [550, 400, 750, 460])

    def test_scale_bbox_resolution_mapping(self) -> None:
        # Scale 640x640 inference to 1280x720 camera frame
        src_shape = (640, 640, 3)
        dst_shape = (720, 1280, 3)
        bbox_in = [100, 100, 300, 200]

        # scale_x = 1280/640 = 2.0, scale_y = 720/640 = 1.125 (round(112.5) -> 112)
        scaled = scale_bbox(bbox_in, src_shape, dst_shape)
        self.assertEqual(scaled, [200, 112, 600, 225])

    def test_scale_bbox_1080p(self) -> None:
        # Scale 640x640 inference to 1920x1080 camera frame
        src_shape = (640, 640, 3)
        dst_shape = (1080, 1920, 3)
        bbox_in = [100, 100, 200, 200]

        # scale_x = 1920/640 = 3.0, scale_y = 1080/640 = 1.6875
        scaled = scale_bbox(bbox_in, src_shape, dst_shape)
        self.assertEqual(scaled, [300, 169, 600, 338])

    def test_add_controlled_padding(self) -> None:
        shape = (480, 640, 3)
        bbox = [100, 100, 200, 140]  # width 100, height 40

        # pad_percent=0.05 -> pad_x = max(4, 5) = 5, pad_y = max(4, 2) = 4
        padded = add_controlled_padding(bbox, shape, pad_percent=0.05, min_pad_px=4)
        self.assertEqual(padded, [95, 96, 205, 144])

    def test_is_bbox_center_inside(self) -> None:
        outer = [100, 100, 500, 500]

        # Center inside
        self.assertTrue(is_bbox_center_inside([150, 150, 250, 250], outer))

        # Center outside
        self.assertFalse(is_bbox_center_inside([50, 50, 95, 95], outer))


if __name__ == "__main__":
    unittest.main()
