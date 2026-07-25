"""Optional LLM-based OCR helper for extracting license plates from cropped images."""
from __future__ import annotations

import json
import os
from typing import Any

import requests


class LLMPlateOCR:
    """Use an LLM or compatible API to read a plate from an image or OCR crop."""

    def __init__(self, enabled: bool = False, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.enabled = enabled
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "20"))

    def _post_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or not self.api_key:
            raise RuntimeError("LLM OCR is disabled or missing API key")
        if not self.base_url:
            raise RuntimeError("LLM base URL is not configured")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def read_plate(self, image: Any) -> dict[str, Any] | None:
        """Return an OCR-like result dictionary when the LLM is configured."""
        if not self.enabled:
            return None
        if not self.api_key or not self.base_url:
            return None

        prompt = (
            "Read the vehicle number plate from this image. "
            "Respond with only the plate text in uppercase, no extra text. "
            "If unclear, respond with INVALID."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You extract Indian vehicle registration plate text from images."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image}},
                ]},
            ],
            "temperature": 0.0,
        }

        try:
            response = self._post_payload(payload)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = str(content or "").strip().upper()
            if not text or text == "INVALID" or text == "UNKNOWN":
                return {"text": "", "confidence": 0.0, "raw_text": text, "valid": False}
            clean = "".join(ch for ch in text if ch.isalnum())
            return {"text": clean, "confidence": 0.92, "raw_text": text, "valid": True}
        except Exception:
            return {"text": "", "confidence": 0.0, "raw_text": "", "valid": False}
