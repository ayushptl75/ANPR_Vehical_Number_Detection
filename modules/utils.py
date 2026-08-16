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


# Recognized Indian License Plate Structural Schemas
INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$"),   # Standard 2-digit district: MH12AB1234
    re.compile(r"^[A-Z]{2}[0-9]{1}[A-Z]{1,3}[0-9]{4}$"),   # Single-digit district: DL1CG5692
    re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"),         # BH Series: 22BH1234A
    re.compile(r"^BH[0-9]{2}[A-Z]{2}[0-9]{4}$"),          # BH Series alternative
    re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,3}$"), # Short number suffix: MH12AB123
    re.compile(r"^[A-Z]{2,3}[0-9]{1,4}[A-Z]{0,2}$"),      # Vintage / Govt format
    re.compile(r"^[0-9]{2}[A-Z]{1,3}[0-9]{4}$"),           # Alternate registration format
]

LETTER_TO_DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

DIGIT_TO_LETTER_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def normalize_plate_text(text: str) -> str:
    """Sanitize OCR output into uppercase alphanumeric string, removing spaces, newlines, and punctuation."""
    if not text:
        return ""
    # Strip spaces, newlines, tabs, and non-alphanumeric characters
    sanitized = re.sub(r"[^A-Za-z0-9]", "", str(text).upper().strip())
    return sanitized[:12]


def positionally_correct_plate_text(text: str) -> str:
    """Apply position-aware character confusion correction without blind replacement."""
    cleaned = normalize_plate_text(text)
    if not cleaned or len(cleaned) < 5:
        return cleaned

    # Exact match for 10-char standard plate (XX00XX0000) or 9-char single-digit district (XX0XX0000)
    # Check if 10-char plate needs last-4 digit correction first
    chars = list(cleaned)
    n = len(chars)

    # Strategy 1: Standard 10-char / 4-digit suffix plate (e.g. MH12ABS234 -> MH12AB5234)
    if n >= 8:
        cand = list(chars)
        # First 2 chars -> State code (Letters)
        for i in range(min(2, n)):
            if cand[i].isdigit() and cand[i] in DIGIT_TO_LETTER_MAP:
                cand[i] = DIGIT_TO_LETTER_MAP[cand[i]]
        # Next 2 chars -> District code (Digits)
        for i in range(2, min(4, n)):
            if cand[i].isalpha() and cand[i] in LETTER_TO_DIGIT_MAP:
                cand[i] = LETTER_TO_DIGIT_MAP[cand[i]]
        # Suffix (last 4 chars) -> Digits
        for i in range(max(4, n - 4), n):
            if cand[i].isalpha() and cand[i] in LETTER_TO_DIGIT_MAP:
                cand[i] = LETTER_TO_DIGIT_MAP[cand[i]]
        
        test_str = "".join(cand)
        if any(pattern.match(test_str) for pattern in INDIAN_PLATE_PATTERNS[:3]):
            return test_str

    # Direct match if no 10-char correction was required
    if any(pattern.match(cleaned) for pattern in INDIAN_PLATE_PATTERNS):
        return cleaned

    # Strategy 2: Single-digit district format (e.g. DL1CG5692)
    if n >= 7:
        cand = list(chars)
        # First 2 chars -> State code
        for i in range(min(2, n)):
            if cand[i].isdigit() and cand[i] in DIGIT_TO_LETTER_MAP:
                cand[i] = DIGIT_TO_LETTER_MAP[cand[i]]
        # Index 2 -> 1-digit district
        if cand[2].isalpha() and cand[2] in LETTER_TO_DIGIT_MAP:
            cand[2] = LETTER_TO_DIGIT_MAP[cand[2]]
        # Index 3-4 -> Series letters
        for i in range(3, min(5, n - 4)):
            if cand[i].isdigit() and cand[i] in DIGIT_TO_LETTER_MAP:
                cand[i] = DIGIT_TO_LETTER_MAP[cand[i]]
        # Last 4 chars -> Digits
        for i in range(max(3, n - 4), n):
            if cand[i].isalpha() and cand[i] in LETTER_TO_DIGIT_MAP:
                cand[i] = LETTER_TO_DIGIT_MAP[cand[i]]

        test_str = "".join(cand)
        if any(pattern.match(test_str) for pattern in INDIAN_PLATE_PATTERNS):
            return test_str

    # Strategy 3: BH-series format (e.g. 22BH1234A)
    if n >= 8 and ("BH" in cleaned or "8H" in cleaned or "BH" in "".join(chars[2:4])):
        cand = list(chars)
        for i in range(min(2, n)):
            if cand[i].isalpha() and cand[i] in LETTER_TO_DIGIT_MAP:
                cand[i] = LETTER_TO_DIGIT_MAP[cand[i]]
        test_str = "".join(cand)
        if any(pattern.match(test_str) for pattern in INDIAN_PLATE_PATTERNS):
            return test_str

    return cleaned


def is_valid_indian_plate(text: str) -> bool:
    """Return True when a normalized plate matches recognized Indian registration schemas."""
    cleaned = normalize_plate_text(text)
    if len(cleaned) < 5 or len(cleaned) > 12:
        return False
    if any(pattern.match(cleaned) for pattern in INDIAN_PLATE_PATTERNS):
        return True
    corrected = positionally_correct_plate_text(cleaned)
    return any(pattern.match(corrected) for pattern in INDIAN_PLATE_PATTERNS)


def validate_indian_plate_with_details(text: str, confidence: float, min_confidence: float = 0.45) -> dict[str, Any]:
    """Validate plate text and return structured validation status metadata."""
    cleaned = normalize_plate_text(text)
    if not cleaned or len(cleaned) < 5:
        return {
            "plate_number": cleaned,
            "is_valid": False,
            "validation_status": "INVALID_FORMAT",
            "was_corrected": False,
        }

    direct_match = any(pattern.match(cleaned) for pattern in INDIAN_PLATE_PATTERNS)
    corrected = positionally_correct_plate_text(cleaned)
    corrected_match = any(pattern.match(corrected) for pattern in INDIAN_PLATE_PATTERNS)

    if confidence < min_confidence and not (direct_match or corrected_match):
        return {
            "plate_number": corrected if corrected_match else cleaned,
            "is_valid": False,
            "validation_status": "LOW_CONFIDENCE",
            "was_corrected": corrected != cleaned,
        }

    if direct_match:
        return {
            "plate_number": cleaned,
            "is_valid": True,
            "validation_status": "VALID_REGISTRATION",
            "was_corrected": False,
        }

    if corrected_match:
        return {
            "plate_number": corrected,
            "is_valid": True,
            "validation_status": "FORMAT_CORRECTED",
            "was_corrected": True,
        }

    return {
        "plate_number": cleaned,
        "is_valid": False,
        "validation_status": "INVALID_FORMAT",
        "was_corrected": False,
    }


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
