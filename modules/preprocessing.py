"""Step-by-step plate preprocessing pipeline and multi-variant image generation for OCR."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from config import Config
from services.image_processor import correct_plate_perspective


class PlatePreprocessor:
    """Apply plate-focused enhancement steps before OCR without destroying original images."""

    def __init__(self, enable_perspective: bool | None = None) -> None:
        self.enable_perspective = (
            enable_perspective
            if enable_perspective is not None
            else Config.PERSPECTIVE_CORRECTION_ENABLED
        )

    def preprocess(self, image: Any) -> np.ndarray:
        """Step-by-step preprocessing pipeline: Perspective -> Grayscale -> Resize -> Noise Reduction -> CLAHE -> Threshold."""
        if image is None:
            return np.zeros((320, 320, 3), dtype=np.uint8)

        img = np.array(image)
        if img.size == 0:
            return np.zeros((320, 320, 3), dtype=np.uint8)

        # 1. Perspective correction (optional)
        if self.enable_perspective:
            img = correct_plate_perspective(img)

        # 2. Grayscale conversion
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 3. Cubic interpolation resize
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        # 4. Bilateral filter noise reduction
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # 5. CLAHE contrast enhancement
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # 6. Sharpening filter
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        gray = cv2.filter2D(gray, -1, kernel)

        # 7. Adaptive thresholding
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)

        if gray.shape[1] < 320:
            scale = 320 / float(gray.shape[1])
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        return gray

    def generate_variants(self, image: Any) -> list[dict[str, Any]]:
        """Generate multiple non-destructive image variants for OCR evaluation."""
        if image is None:
            return []

        base = np.array(image)
        if base.size == 0:
            return []

        # Perspective corrected base copy
        warped_base = correct_plate_perspective(base) if self.enable_perspective else base

        variants: list[dict[str, Any]] = []

        # 1. Original BGR/RGB
        variants.append({"name": "original", "image": base})

        # 2. Grayscale
        if len(warped_base.shape) == 3:
            gray = cv2.cvtColor(warped_base, cv2.COLOR_BGR2GRAY)
        else:
            gray = warped_base.copy()
        variants.append({"name": "grayscale", "image": gray})

        # 3. Upscaled Grayscale (2.5x Cubic)
        upscaled = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        variants.append({"name": "upscaled", "image": upscaled})

        # 4. CLAHE Contrast Boost
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(upscaled)
        variants.append({"name": "clahe", "image": clahe})

        # 5. Bilateral Filter Denoised
        bilateral = cv2.bilateralFilter(upscaled, 9, 75, 75)
        variants.append({"name": "bilateral", "image": bilateral})

        # 6. Sharpened
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(clahe, -1, kernel)
        variants.append({"name": "sharpened", "image": sharpened})

        # 7. Adaptive Threshold
        adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
        variants.append({"name": "adaptive_threshold", "image": adaptive})

        # 8. Otsu Threshold
        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append({"name": "otsu_threshold", "image": otsu})

        return variants
