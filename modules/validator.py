"""Validation helpers for Indian number plates."""
from __future__ import annotations

import re

from modules.utils import clean_plate_text, normalize_plate_text

INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")


class PlateValidator:
    """Validate OCR output against Indian plate patterns."""

    def validate(self, text: str) -> tuple[bool, str]:
        cleaned = clean_plate_text(text)
        if not cleaned:
            return False, ""
        if INDIAN_PLATE_PATTERN.match(cleaned):
            return True, cleaned
        if len(cleaned) >= 6 and len(cleaned) <= 12:
            corrected = self._normalize_common_confusions(cleaned)
            if INDIAN_PLATE_PATTERN.match(corrected):
                return True, corrected
        return False, cleaned

    def _normalize_common_confusions(self, text: str) -> str:
        mapping = {"O": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "G": "6", "Q": "0"}
        out = []
        for ch in text:
            out.append(mapping.get(ch, ch))
        return "".join(out)
