"""Minimal LLM normalizer for OCR text."""
from __future__ import annotations

from modules.validator import PlateValidator


class LLMNormalizer:
    """Only normalize OCR mistakes; never invent missing characters."""

    def __init__(self) -> None:
        self.validator = PlateValidator()

    def normalize(self, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        cleaned = text.strip().upper()
        mapping = {"O": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "G": "6", "Q": "0"}
        normalized = "".join(mapping.get(ch, ch) for ch in cleaned if ch.isalnum())
        valid, final = self.validator.validate(normalized)
        if valid:
            return final, True
        return normalized, False
