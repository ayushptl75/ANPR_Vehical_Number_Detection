from __future__ import annotations

from typing import Any

import numpy as np

try:
    import easyocr
except Exception:  # pragma: no cover - optional dependency
    easyocr = None

from config import Config
from modules.utils import (
    clean_plate_text,
    get_logger,
    is_valid_indian_plate,
    positionally_correct_plate_text,
)
from services.image_processor import correct_plate_perspective, generate_plate_variants


class OCRService:
    """OCR service for plate text extraction."""

    def __init__(self) -> None:
        self.logger = get_logger("ocr_service")
        self.reader = None
        if easyocr is not None:
            try:
                self.reader = easyocr.Reader(["en"], gpu=False)
            except Exception as exc:
                self.logger.warning("EasyOCR init failed: %s", exc)
                self.reader = None

    def read_plate(self, image: Any) -> dict[str, Any]:
        """Read and normalize a license plate from the cropped plate image."""
        if image is None:
            return {"text": "", "confidence": 0.0, "raw_text": "", "valid": False}
        img = np.array(image)
        img = correct_plate_perspective(img)
        variants = generate_plate_variants(img)
        if self.reader is None:
            self.logger.warning("OCR reader not available")
            return {"text": "", "confidence": 0.0, "raw_text": "", "valid": False}

        best_result = {"text": "", "confidence": 0.0, "raw_text": "", "valid": False}
        for variant in variants:
            try:
                results = self.reader.readtext(
                    variant,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=1,
                )
            except Exception as exc:
                self.logger.warning("OCR readtext failed: %s", exc)
                continue
            for item in results:
                raw_text = item[1] or ""
                confidence = float(item[2] or 0.0)
                normalized = clean_plate_text(raw_text)
                corrected = positionally_correct_plate_text(normalized)
                valid = is_valid_indian_plate(corrected)
                score = confidence + (1.0 if valid else 0.0)
                if score > best_result.get("confidence", 0.0) or (valid and not best_result.get("valid", False)):
                    best_result = {
                        "text": corrected,
                        "confidence": confidence,
                        "raw_text": raw_text,
                        "valid": valid,
                    }
        return best_result
