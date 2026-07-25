"""Core detector abstractions for vehicle and plate detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Detection:
    bbox: list[int]
    confidence: float
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"bbox": self.bbox, "confidence": self.confidence, "label": self.label}
