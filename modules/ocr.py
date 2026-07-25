"""OCR wrapper using PaddleOCR-compatible output structure."""
from __future__ import annotations

from typing import Any

import numpy as np

from modules.preprocessing import PlatePreprocessor
from modules.utils import clean_plate_text, get_logger, is_valid_indian_plate

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None


class OCRReader:
    """Perform OCR on the preprocessed plate crop."""

    def __init__(self) -> None:
        self.logger = get_logger("ocr")
        self.preprocessor = PlatePreprocessor()
        self.reader = None
        if PaddleOCR is not None:
            try:
                self.reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception as exc:
                self.logger.warning("PaddleOCR init failed: %s", exc)
                self.reader = None

    def read(self, image: Any) -> dict[str, Any]:
        processed = self.preprocessor.preprocess(image)
        if self.reader is None:
            return {"text": "", "confidence": 0.0, "bboxes": []}

        try:
            results = self.reader.ocr(np.array(processed), cls=True)
            if not results or not results[0]:
                return {"text": "", "confidence": 0.0, "bboxes": []}

            best = None
            for item in results[0]:
                if not item:
                    continue
                text = str(item[1][0]).strip()
                conf = float(item[1][1]) if len(item[1]) > 1 else 0.0
                normalized = clean_plate_text(text)
                if not normalized:
                    continue
                if is_valid_indian_plate(normalized):
                    best = {"text": normalized, "confidence": round(conf, 2), "bboxes": [item[0]]}
                    break
                if best is None:
                    best = {"text": normalized, "confidence": round(conf, 2), "bboxes": [item[0]]}

            if best is None:
                return {"text": "", "confidence": 0.0, "bboxes": []}
            return best
        except Exception as exc:
            self.logger.warning("PaddleOCR failed: %s", exc)
            return {"text": "", "confidence": 0.0, "bboxes": []}
