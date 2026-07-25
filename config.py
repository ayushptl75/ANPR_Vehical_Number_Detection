"""Application configuration and paths."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Central configuration for the ANPR service."""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'database' / 'anpr.db'}")

    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
    PROCESSED_FOLDER = os.getenv("PROCESSED_FOLDER", str(BASE_DIR / "processed"))
    VEHICLE_IMAGES_FOLDER = os.getenv("VEHICLE_IMAGES_FOLDER", str(BASE_DIR / "vehicle_images"))
    PLATE_IMAGES_FOLDER = os.getenv("PLATE_IMAGES_FOLDER", str(BASE_DIR / "plate_images"))
    REPORTS_FOLDER = os.getenv("REPORTS_FOLDER", str(BASE_DIR / "reports"))
    LOGS_FOLDER = os.getenv("LOGS_FOLDER", str(BASE_DIR / "logs"))

    OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.45"))
    LLM_OCR_ENABLED = os.getenv("LLM_OCR_ENABLED", "False").lower() in ("1", "true", "yes")
    LLM_OCR_API_KEY = os.getenv("LLM_API_KEY")
    LLM_OCR_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_OCR_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_OCR_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
    MAX_DUPLICATE_SECONDS = int(os.getenv("MAX_DUPLICATE_SECONDS", "15"))
    VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "yolov8n.pt")
    PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "yolov8n.pt")
    PLATE_CROP_PADDING = int(os.getenv("PLATE_CROP_PADDING", "5"))
    PLATE_CONFIDENCE_THRESHOLD = float(os.getenv("PLATE_CONFIDENCE_THRESHOLD", "0.35"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("1", "true", "yes")
    DEBUG_FOLDER = os.getenv("DEBUG_FOLDER", str(BASE_DIR / "debug"))

    VEHICLE_API_URL = os.getenv("VEHICLE_API_URL")
    VEHICLE_API_KEY = os.getenv("VEHICLE_API_KEY")
    VEHICLE_API_PROVIDER = os.getenv("VEHICLE_API_PROVIDER")
    VEHICLE_API_TIMEOUT = float(os.getenv("VEHICLE_API_TIMEOUT", "10"))

    RTO_API_PROVIDER = os.getenv("RTO_API_PROVIDER")
    RTO_API_URL = os.getenv("RTO_API_URL")
    RTO_API_KEY = os.getenv("RTO_API_KEY")
    RTO_API_CLIENT_ID = os.getenv("RTO_API_CLIENT_ID")
    RTO_API_CLIENT_SECRET = os.getenv("RTO_API_CLIENT_SECRET")
    RTO_API_TIMEOUT = float(os.getenv("RTO_API_TIMEOUT", "10"))

    DEFAULT_ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
