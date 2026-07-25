"""Image preprocessing helpers for license-plate OCR."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class PlatePreprocessor:
    """Apply plate-focused enhancement steps before OCR."""

    def preprocess(self, image: Any) -> np.ndarray:
        if image is None:
            return np.zeros((320, 320, 3), dtype=np.uint8)
        img = np.array(image)
        if img.size == 0:
            return np.zeros((320, 320, 3), dtype=np.uint8)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
        if gray.shape[1] < 320:
            scale = 320 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return gray
