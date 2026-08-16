"""OCR reading and multi-variant evaluation for Indian license plates."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import easyocr
except Exception:  # pragma: no cover - optional dependency
    easyocr = None

from config import Config
from modules.llm_ocr import LLMPlateOCR
from modules.preprocessing import PlatePreprocessor
from modules.utils import (
    clean_plate_text,
    get_logger,
    is_valid_indian_plate,
    is_valid_plate_text,
    positionally_correct_plate_text,
    validate_indian_plate_with_details,
)


class OCRReader:
    """Read text from cropped plate images using EasyOCR with multi-variant preprocessing."""

    def __init__(self) -> None:
        self.logger = get_logger("ocr")
        self.preprocessor = PlatePreprocessor()
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

    def preprocess(self, image: Any) -> np.ndarray:
        """Apply step-by-step preprocessing pipeline to improve OCR fidelity."""
        return self.preprocessor.preprocess(image)

    def read(self, image: Any) -> dict[str, Any]:
        """Return structured OCR details by evaluating single or multi-variant preprocessed images."""
        if image is None:
            image = np.zeros((128, 256, 3), dtype=np.uint8)

        # Vision LLM fallback check
        if self.llm_reader.enabled:
            llm_result = self.llm_reader.read_plate(image)
            if llm_result is not None:
                return llm_result

        if self.reader is None:
            self.logger.warning("OCR reader not available")
            return {
                "plate_number": "",
                "ocr_confidence": 0.0,
                "validation_status": "NO_READER",
                "processing_variant": "none",
                "raw_text": "",
                "text": "",
                "confidence": 0.0,
                "variant": "none",
                "valid": False,
            }

        # Multi-variant evaluation
        if Config.PREPROCESSING_MULTI_VARIANT_ENABLED:
            variants = self.preprocessor.generate_variants(image)
        else:
            single = self.preprocess(image)
            variants = [{"name": "standard", "image": single}]

        candidates: list[dict[str, Any]] = []

        for variant in variants:
            var_img = variant.get("image")
            if var_img is None or not isinstance(var_img, np.ndarray) or var_img.size == 0:
                continue

            try:
                results = self.reader.readtext(var_img)
                if not results:
                    continue
                best_item = max(results, key=lambda item: float(item[2] or 0.0))
                raw_text = best_item[1] or ""
                confidence = float(best_item[2] or 0.0)

                val_info = validate_indian_plate_with_details(raw_text, confidence, Config.OCR_CONFIDENCE_THRESHOLD)
                score = confidence + (1.0 if val_info["is_valid"] else 0.0)

                candidates.append({
                    "raw_text": raw_text,
                    "plate_number": val_info["plate_number"],
                    "confidence": confidence,
                    "val_info": val_info,
                    "variant_name": variant.get("name", "unknown"),
                    "score": score,
                    "image": var_img,
                })
            except Exception as exc:
                self.logger.warning("OCR variant '%s' failed: %s", variant.get("name"), exc)

        if not candidates:
            return {
                "plate_number": "",
                "ocr_confidence": 0.0,
                "validation_status": "INVALID_FORMAT",
                "processing_variant": "none",
                "raw_text": "",
                "text": "",
                "confidence": 0.0,
                "variant": "none",
                "valid": False,
            }

        # Sort candidates: prioritize format validity first, then highest confidence score
        candidates.sort(key=lambda item: (item["val_info"]["is_valid"], item["score"]), reverse=True)
        winner = candidates[0]
        best_enhanced_img = winner.get("image")

        # Save debug images if enabled
        if Config.DEBUG_MODE:
            try:
                debug_dir = Path(Config.DEBUG_FOLDER)
                debug_dir.mkdir(parents=True, exist_ok=True)
                if isinstance(image, np.ndarray) and image.size != 0:
                    cv2.imwrite(str(debug_dir / "cropped_plate.jpg"), image)
                if isinstance(best_enhanced_img, np.ndarray) and best_enhanced_img.size != 0:
                    cv2.imwrite(str(debug_dir / "enhanced_plate.jpg"), best_enhanced_img)
            except Exception:
                pass

        val_details = winner["val_info"]
        final_plate = val_details["plate_number"]
        final_conf = winner["confidence"]

        if final_conf < Config.OCR_CONFIDENCE_THRESHOLD and not val_details["is_valid"]:
            return {
                "plate_number": "",
                "ocr_confidence": round(final_conf, 2),
                "validation_status": "LOW_CONFIDENCE",
                "processing_variant": winner["variant_name"],
                "raw_text": winner["raw_text"],
                "text": "",
                "confidence": round(final_conf, 2),
                "variant": winner["variant_name"],
                "valid": False,
            }

        return {
            "plate_number": final_plate,
            "ocr_confidence": round(final_conf, 2),
            "validation_status": val_details["validation_status"],
            "processing_variant": winner["variant_name"],
            "raw_text": winner["raw_text"],
            "text": final_plate,
            "confidence": round(final_conf, 2),
            "variant": winner["variant_name"],
            "valid": val_details["is_valid"],
        }
