"""OCR reading and preprocessing for number plates."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    import easyocr
except Exception:  # pragma: no cover - optional dependency
    easyocr = None

from config import Config
from modules.llm_ocr import LLMPlateOCR
from modules.utils import (
    clean_plate_text,
    get_logger,
    is_valid_indian_plate,
    is_valid_plate_text,
    positionally_correct_plate_text,
)
from services.image_processor import generate_plate_variants, correct_plate_perspective


class OCRReader:
    """Read text from cropped plate images using EasyOCR with preprocessing."""

    def __init__(self) -> None:
        self.logger = get_logger("ocr")
        self.reader = None
        self.llm_reader = LLMPlateOCR(
            enabled=Config.LLM_OCR_ENABLED,
            api_key=Config.LLM_OCR_API_KEY,
            model=Config.LLM_OCR_MODEL,
            base_url=Config.LLM_OCR_BASE_URL,
        )
        if easyocr is not None:
            try:
                self.reader = easyocr.Reader(["en"], gpu=False)
            except Exception as exc:
                self.logger.warning("EasyOCR init failed: %s", exc)
                self.reader = None

    def preprocess(self, image: Any) -> Any:
        """Apply a series of preprocessing steps to improve OCR fidelity."""
        if image is None:
            return np.zeros((128, 256, 3), dtype=np.uint8)
        try:
            img = np.array(image)
        except Exception:
            return np.zeros((128, 256, 3), dtype=np.uint8)
        if img.size == 0:
            return np.zeros((128, 256, 3), dtype=np.uint8)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharp_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, sharp_kernel)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return rgb

    def read(self, image: Any) -> dict[str, Any]:
        """Return OCR text and confidence after preprocessing."""
        preprocessed = self.preprocess(image)
        if self.llm_reader.enabled:
            llm_result = self.llm_reader.read_plate(image)
            if llm_result is not None:
                return llm_result

        if self.reader is None:
            self.logger.warning("OCR reader not available")
            return {"text": "", "confidence": 0.0, "raw_text": ""}
        try:
            results = self.reader.readtext(preprocessed)
            if not results:
                return {"text": "", "confidence": 0.0, "raw_text": ""}
            best = max(results, key=lambda item: float(item[2] or 0.0))
            raw_text = best[1] or ""
            confidence = float(best[2] or 0.0)
            normalized = clean_plate_text(raw_text)
            if confidence < Config.OCR_CONFIDENCE_THRESHOLD and not is_valid_plate_text(normalized):
                return {"text": "", "confidence": round(confidence, 2), "raw_text": raw_text}
            return {"text": normalized, "confidence": round(confidence, 2), "raw_text": raw_text}
        except Exception as exc:
            self.logger.warning("OCR failed: %s", exc)
            return {"text": "", "confidence": 0.0, "raw_text": ""}
