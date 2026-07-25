"""Utility helpers used throughout the ANPR application."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import current_app

BASE_DIR = Path(__file__).resolve().parent.parent


def ensure_directories() -> None:
    """Create all required folders if they do not yet exist."""
    directories = [
        BASE_DIR / "uploads",
        BASE_DIR / "processed",
        BASE_DIR / "vehicle_images",
        BASE_DIR / "plate_images",
        BASE_DIR / "reports",
        BASE_DIR / "logs",
        BASE_DIR / "debug",
        BASE_DIR / "database",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_safe_filename(filename: str) -> str:
    """Return a sanitized and unique filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{stamp}_{safe}"


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 and a random salt."""
    salt = secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password hash."""
    if not hashed or "$" not in hashed:
        return False
    salt, digest = hashed.split("$", 1)
    expected = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return expected == digest


INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"),
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}$"),
]

LETTER_TO_DIGIT_SUBSTITUTIONS = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}

DIGIT_TO_LETTER_SUBSTITUTIONS = {v: k for k, v in LETTER_TO_DIGIT_SUBSTITUTIONS.items()}


def normalize_plate_text(text: str) -> str:
    """Normalize OCR output into an uppercase plate text without separators."""
    normalized = re.sub(r"[^A-Za-z0-9]", "", text.upper())
    return normalized[:12]


def positionally_correct_plate_text(text: str) -> str:
    """Apply position-based corrections for common Indian plate OCR confusions."""
    plate = list(normalize_plate_text(text))
    if len(plate) < 6:
        return "".join(plate)

    # First two characters should be letters.
    for idx in range(min(2, len(plate))):
        if plate[idx].isdigit():
            plate[idx] = DIGIT_TO_LETTER_SUBSTITUTIONS.get(plate[idx], plate[idx])

    # District/RTO code positions should be digits.
    for idx in range(2, min(4, len(plate))):
        if plate[idx].isalpha():
            plate[idx] = LETTER_TO_DIGIT_SUBSTITUTIONS.get(plate[idx], plate[idx])

    # Series letters tend to follow the district portion.
    for idx in range(4, max(4, len(plate) - 4)):
        if plate[idx].isdigit():
            plate[idx] = DIGIT_TO_LETTER_SUBSTITUTIONS.get(plate[idx], plate[idx])

    # Final numeric portion should be digits.
    for idx in range(max(4, len(plate) - 4), len(plate)):
        if plate[idx].isalpha():
            plate[idx] = LETTER_TO_DIGIT_SUBSTITUTIONS.get(plate[idx], plate[idx])

    return "".join(plate)


def is_valid_indian_plate(text: str) -> bool:
    """Return True when a normalized plate matches common Indian registration formats."""
    cleaned = normalize_plate_text(text)
    if len(cleaned) < 6 or len(cleaned) > 12:
        return False
    if any(pattern.match(cleaned) for pattern in INDIAN_PLATE_PATTERNS):
        return True
    corrected = positionally_correct_plate_text(cleaned)
    return any(pattern.match(corrected) for pattern in INDIAN_PLATE_PATTERNS)


def clean_plate_text(text: str) -> str:
    """Normalize OCR output into a plate-like string."""
    cleaned = normalize_plate_text(text)
    return cleaned[:12]


def is_valid_plate_text(text: str) -> bool:
    """Legacy compatibility wrapper for Indian plate validation."""
    return is_valid_indian_plate(text)


def format_timestamp(value: datetime | None = None) -> str:
    """Return a human-readable timestamp."""
    ts = value or datetime.utcnow()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def compute_toll_amount(vehicle_type: str) -> float:
    """Return a toll amount based on vehicle type."""
    vt = (vehicle_type or "").strip().lower()
    if vt in {"car", "sedan", "suv", "jeep", "van"}:
        return 80.0
    if vt in {"bike", "motorbike", "motorcycle", "two-wheeler"}:
        return 40.0
    if vt in {"truck", "bus", "lcv", "hcv"}:
        return 150.0
    return 100.0


def get_logger(name: str) -> logging.Logger:
    """Create a configured logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(BASE_DIR / "logs" / "anpr.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger
