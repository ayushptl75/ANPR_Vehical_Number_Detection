"""Coordinate transformation, boundary clamping, and resolution scaling utilities."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np


def clamp_bbox(bbox: Sequence[int], frame_shape: tuple[int, int]) -> list[int]:
    """Clamp bounding box coordinates [x1, y1, x2, y2] to valid frame boundaries [0, W] and [0, H]."""
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    return [x1, y1, x2, y2]


def crop_to_global_bbox(
    crop_bbox: Sequence[int],
    offset_x: int,
    offset_y: int,
    frame_shape: tuple[int, int] | None = None,
) -> list[int]:
    """Convert local crop bounding box coordinates [x1, y1, x2, y2] to absolute global frame coordinates."""
    if not crop_bbox or len(crop_bbox) != 4:
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    gx1 = x1 + int(offset_x)
    gy1 = y1 + int(offset_y)
    gx2 = x2 + int(offset_x)
    gy2 = y2 + int(offset_y)

    global_box = [gx1, gy1, gx2, gy2]
    if frame_shape is not None:
        return clamp_bbox(global_box, frame_shape)
    return global_box


def scale_bbox(
    bbox: Sequence[int],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> list[int]:
    """Scale a bounding box from src_shape (src_h, src_w) to dst_shape (dst_h, dst_w)."""
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]

    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return [int(v) for v in bbox]

    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)

    x1, y1, x2, y2 = [float(v) for v in bbox]

    sx1 = int(round(x1 * scale_x))
    sy1 = int(round(y1 * scale_y))
    sx2 = int(round(x2 * scale_x))
    sy2 = int(round(y2 * scale_y))

    return clamp_bbox([sx1, sy1, sx2, sy2], dst_shape)


def add_controlled_padding(
    bbox: Sequence[int],
    frame_shape: tuple[int, int],
    pad_percent: float = 0.05,
    min_pad_px: int = 4,
    pad_x_percent: float | None = None,
    pad_y_percent: float | None = None,
) -> list[int]:
    """Add controlled, symmetrical padding around [x1, y1, x2, y2] without shifting center or cutting characters."""
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]

    px_pct = pad_x_percent if pad_x_percent is not None else pad_percent
    py_pct = pad_y_percent if pad_y_percent is not None else pad_percent

    x1, y1, x2, y2 = [int(v) for v in bbox]
    width_px = max(1, x2 - x1)
    height_px = max(1, y2 - y1)

    pad_x = max(int(min_pad_px), int(round(width_px * px_pct)))
    pad_y = max(int(min_pad_px), int(round(height_px * py_pct)))

    px1 = x1 - pad_x
    py1 = y1 - pad_y
    px2 = x2 + pad_x
    py2 = y2 + pad_y

    return clamp_bbox([px1, py1, px2, py2], frame_shape)


def is_bbox_center_inside(inner_bbox: Sequence[int], outer_bbox: Sequence[int]) -> bool:
    """Return True if the center point of inner_bbox lies within outer_bbox."""
    if not inner_bbox or not outer_bbox or len(inner_bbox) != 4 or len(outer_bbox) != 4:
        return False

    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox

    icx = (ix1 + ix2) / 2.0
    icy = (iy1 + iy2) / 2.0

    return ox1 <= icx <= ox2 and oy1 <= icy <= oy2


def validate_plate_crop_dimensions(
    crop: Any,
    min_width: int = 60,
    min_height: int = 18,
    min_aspect: float = 2.0,
    max_aspect: float = 6.0,
) -> bool:
    """Validate that a cropped plate image meets minimum dimension and aspect ratio requirements."""
    if crop is None:
        return False

    img = np.array(crop)
    if img.size == 0 or len(img.shape) < 2:
        return False

    height_px, width_px = img.shape[:2]
    if width_px < min_width or height_px < min_height:
        return False

    aspect_ratio = width_px / float(max(1, height_px))
    return min_aspect <= aspect_ratio <= max_aspect
