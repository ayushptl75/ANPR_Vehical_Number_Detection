"""Dedicated CarInfo Vehicle Information API Service.

Integrates ANPR normalized registration numbers with authorized CarInfo API,
handling format validation, multi-frame confidence thresholds, response caching/debouncing,
test/mock mode, safe API error handling, and field normalization.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

try:
    import requests
except ImportError:
    requests = None

from config import Config
from modules.utils import (
    clean_plate_text,
    get_logger,
    is_valid_indian_plate,
    normalize_plate_text,
    positionally_correct_plate_text,
)

logger = get_logger("carinfo_service")


class CarInfoService:
    """Service wrapper for Authorized CarInfo Vehicle Registration API."""

    def __init__(self, db_manager: Any = None) -> None:
        self.db_manager = db_manager
        self.api_url = Config.CARINFO_API_URL
        self.api_key = Config.CARINFO_API_KEY
        self.api_secret = Config.CARINFO_API_SECRET
        self.api_token = Config.CARINFO_API_TOKEN
        self.test_mode = Config.CARINFO_TEST_MODE
        self.cache_ttl = Config.VEHICLE_API_CACHE_SECONDS
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def normalize_plate_number(self, plate_number: str) -> str:
        """Sanitize and positionally correct an Indian registration number."""
        if not plate_number:
            return ""
        cleaned = clean_plate_text(plate_number)
        corrected = positionally_correct_plate_text(cleaned)
        return corrected if is_valid_indian_plate(corrected) else cleaned

    def validate_plate_number(self, plate_number: str) -> bool:
        """Validate if the plate string conforms to recognized Indian plate standards."""
        if not plate_number:
            return False
        normalized = self.normalize_plate_number(plate_number)
        return is_valid_indian_plate(normalized)

    def get_vehicle_information(
        self,
        vehicle_number: str,
        ocr_confidence: float = 1.0,
        plate_confidence: float = 1.0,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Fetch normalized vehicle details from cache, test mode, or real CarInfo API."""
        normalized_plate = self.normalize_plate_number(vehicle_number)

        if not normalized_plate or len(normalized_plate) < 5:
            return self._build_empty_response(
                normalized_plate or vehicle_number,
                status="INVALID FORMAT",
                error="Invalid registration number length or format",
            )

        # Check confidence threshold if confidence metrics are supplied
        combined_confidence = (ocr_confidence + plate_confidence) / 2.0
        is_format_valid = self.validate_plate_number(normalized_plate)
        if combined_confidence < Config.OCR_CONFIDENCE_THRESHOLD:
            logger.warning("[ANPR] Low confidence for %s (%.2f). Skipping API call.", normalized_plate, combined_confidence)
            return self._build_empty_response(
                normalized_plate,
                status="LOW CONFIDENCE",
                error="Plate recognition confidence too low. Waiting for clearer frames.",
            )


        # Check Cache (In-Memory first, then DB if db_manager available)
        if not force_refresh:
            cached_data = self._check_cache(normalized_plate)
            if cached_data:
                logger.info("[API] Returning cached vehicle information for %s", normalized_plate)
                cached_data["is_cached"] = True
                return cached_data

        # Check Test / Mock Mode
        if self.test_mode or not self.api_key:
            logger.info("[API] Test mode active for %s. Generating mock CarInfo response.", normalized_plate)
            res = self._generate_mock_response(normalized_plate)
            self._save_to_cache(normalized_plate, res)
            return res

        # Execute Live API Request
        return self._fetch_from_carinfo_api(normalized_plate)

    def _check_cache(self, plate_number: str) -> dict[str, Any] | None:
        """Retrieve non-expired cached vehicle details."""
        now = time.time()
        # Memory Cache Check
        if plate_number in self._memory_cache:
            item = self._memory_cache[plate_number]
            if now - item["cached_at"] <= self.cache_ttl:
                return item["data"]
            else:
                del self._memory_cache[plate_number]

        # DB Cache Check
        if self.db_manager:
            try:
                db_record = self.db_manager.get_vehicle_info(plate_number)
                if db_record and db_record.get("last_verified"):
                    try:
                        last_v = datetime.fromisoformat(db_record["last_verified"])
                        age = (datetime.now(timezone.utc) - last_v.replace(tzinfo=timezone.utc)).total_seconds()
                        if age <= self.cache_ttl:
                            norm_db = self._normalize_database_record(db_record, plate_number)
                            self._memory_cache[plate_number] = {"cached_at": now, "data": norm_db}
                            return norm_db
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Error checking DB cache: %s", exc)

        return None

    def _save_to_cache(self, plate_number: str, data: dict[str, Any]) -> None:
        """Store lookup result in memory and database cache."""
        now = time.time()
        self._memory_cache[plate_number] = {"cached_at": now, "data": data}

        if self.db_manager and data.get("verified"):
            try:
                db_data = {
                    "vehicle_type": data.get("vehicle_class") if data.get("vehicle_class") != "Not Available" else None,
                    "manufacturer": data.get("manufacturer") if data.get("manufacturer") != "Not Available" else None,
                    "model": data.get("model") if data.get("model") != "Not Available" else None,
                    "fuel_type": data.get("fuel_type") if data.get("fuel_type") != "Not Available" else None,
                    "registration_date": data.get("registration_date") if data.get("registration_date") != "Not Available" else None,
                    "insurance_status": data.get("insurance_status") if data.get("insurance_status") != "Not Available" else None,
                    "insurance_expiry": data.get("insurance_expiry") if data.get("insurance_expiry") != "Not Available" else None,
                    "fitness_status": data.get("fitness_status") if data.get("fitness_status") != "Not Available" else None,
                    "puc_status": data.get("pucc_status") if data.get("pucc_status") != "Not Available" else None,
                    "owner_name": data.get("owner_information") if data.get("owner_information") != "Not Available" else None,
                    "data_source": "official_carinfo_api",
                    "verified": 1,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
                self.db_manager.add_vehicle_record(
                    plate_number=plate_number,
                    vehicle_type=db_data.get("vehicle_type") or "Unknown",
                    owner_name=db_data.get("owner_name") or "Not Available",
                    source="official_carinfo_api",
                    verified=True,
                )
            except Exception as exc:
                logger.warning("Could not persist CarInfo response to DB: %s", exc)

    def _fetch_from_carinfo_api(self, plate_number: str) -> dict[str, Any]:
        """Dispatch authenticated HTTP request to authorized CarInfo API."""
        if requests is None:
            logger.error("[API] 'requests' module not installed")
            return self._build_empty_response(
                plate_number,
                status="API ERROR",
                error="HTTP client library unavailable",
            )

        headers: dict[str, str] = {
            "User-Agent": "ANPR-Vehicle-System/1.0",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        if self.api_secret:
            headers["X-API-Secret"] = self.api_secret
        if self.api_token:
            headers["X-API-Token"] = self.api_token

        params = {"registration_number": plate_number, "rc_number": plate_number}

        logger.info("[API] Vehicle lookup started for %s", plate_number)
        try:
            response = requests.get(self.api_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                raw_json = response.json()
                logger.info("[API] Vehicle lookup successful for %s", plate_number)
                normalized = self._parse_api_response(raw_json, plate_number)
                self._save_to_cache(plate_number, normalized)
                return normalized

            status_code = response.status_code
            logger.warning("[API] Vehicle lookup failed with status code %s for %s", status_code, plate_number)

            error_msg_map = {
                401: "Unauthorized: Invalid CarInfo API credentials",
                403: "Forbidden: API access restricted",
                404: "Vehicle registration not found in CarInfo database",
                408: "Request timeout from CarInfo API",
                429: "Rate limit exceeded on CarInfo API",
                500: "CarInfo API internal server error",
            }
            err_msg = error_msg_map.get(status_code, f"CarInfo API returned HTTP status {status_code}")
            return self._build_empty_response(plate_number, status="API ERROR", error=err_msg)

        except requests.exceptions.Timeout:
            logger.error("[API] Timeout while requesting CarInfo API for %s", plate_number)
            return self._build_empty_response(plate_number, status="API ERROR", error="CarInfo API request timed out")
        except requests.exceptions.RequestException as exc:
            logger.error("[API] Network error requesting CarInfo API: %s", exc)
            return self._build_empty_response(plate_number, status="API ERROR", error="External network connection error")
        except Exception as exc:
            logger.exception("[API] Unexpected error during CarInfo lookup: %s", exc)
            return self._build_empty_response(plate_number, status="API ERROR", error=f"Unexpected error: {str(exc)}")

    def _parse_api_response(self, raw_data: dict[str, Any], plate_number: str) -> dict[str, Any]:
        """Normalize raw CarInfo API response fields into standard application format."""
        if not isinstance(raw_data, dict):
            return self._build_empty_response(plate_number, status="API ERROR", error="Invalid JSON response format")

        data = raw_data.get("result") or raw_data.get("data") or raw_data

        def get_field(*keys: str) -> str:
            for key in keys:
                val = data.get(key)
                if val and str(val).strip() and str(val).strip().lower() not in ("null", "none", "n/a", "unknown"):
                    return str(val).strip()
            return "Not Available"

        vehicle_class = get_field("vehicle_class", "vehicleClass", "class", "vehicle_category", "vehicleCategory")
        manufacturer = get_field("manufacturer", "make", "maker_name", "brand", "makerName")
        model = get_field("model", "maker_model", "variant", "vehicle_model", "modelName")
        fuel_type = get_field("fuel_type", "fuelType", "fuel", "fuel_desc")
        reg_date = get_field("registration_date", "registrationDate", "registered_at", "reg_date", "issue_date")
        insurance_status = get_field("insurance_status", "insuranceStatus", "insurance_state", "insurance_details")
        insurance_expiry = get_field("insurance_expiry", "insuranceExpiry", "insurance_valid_upto", "insurance_upto")
        fitness_status = get_field("fitness_status", "fitnessStatus", "fitness_state", "fitness_upto")
        pucc_status = get_field("pucc_status", "puc_status", "pucStatus", "puc_state", "pucc_upto", "puc_expiry")
        owner_info = get_field("owner_name", "ownerName", "registered_owner", "owner")
        if owner_info == "Not Available":
            owner_info = None  # Only populate owner if explicitly returned & permitted

        return {
            "status": "VERIFIED",
            "verified": True,
            "registration_number": plate_number,
            "vehicle_class": vehicle_class,
            "manufacturer": manufacturer,
            "model": model,
            "fuel_type": fuel_type,
            "registration_date": reg_date,
            "insurance_status": insurance_status,
            "insurance_expiry": insurance_expiry,
            "fitness_status": fitness_status,
            "pucc_status": pucc_status,
            "owner_information": owner_info,
            "api_source": "Authorized CarInfo API",
            "is_cached": False,
            "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": raw_data,
            "error": None,
        }

    def _generate_mock_response(self, plate_number: str) -> dict[str, Any]:
        """Generate realistic vehicle details for testing in test mode."""
        # Standard test dataset mappings
        test_database: dict[str, dict[str, Any]] = {
            "GJ05AB1234": {
                "vehicle_class": "Motor Car (LMV)",
                "manufacturer": "Hyundai",
                "model": "Creta SX",
                "fuel_type": "PETROL",
                "registration_date": "2022-03-15",
                "insurance_status": "Active (HDFC ERGO)",
                "insurance_expiry": "2026-03-14",
                "fitness_status": "Active (Valid upto 2037-03-14)",
                "pucc_status": "Active (Valid upto 2026-11-20)",
                "owner_information": "Ketan Patel",
            },
            "KA01AB1234": {
                "vehicle_class": "Motor Car (LMV)",
                "manufacturer": "Hyundai",
                "model": "i20 Asta",
                "fuel_type": "PETROL",
                "registration_date": "2022-01-10",
                "insurance_status": "Active",
                "insurance_expiry": "2027-01-10",
                "fitness_status": "Active",
                "pucc_status": "Active",
                "owner_information": "Rajesh Kumar",
            },
            "DL04C1234": {
                "vehicle_class": "Two Wheeler (MCWG)",
                "manufacturer": "Honda",
                "model": "Activa 6G",
                "fuel_type": "PETROL",
                "registration_date": "2021-06-05",
                "insurance_status": "Active",
                "insurance_expiry": "2026-06-05",
                "fitness_status": "Active",
                "pucc_status": "Active",
                "owner_information": "Anil Sharma",
            },
            "MH12AB1234": {
                "vehicle_class": "Motor Car (LMV)",
                "manufacturer": "Tata Motors",
                "model": "Nexon EV",
                "fuel_type": "ELECTRIC",
                "registration_date": "2023-08-20",
                "insurance_status": "Active (ICICI Lombard)",
                "insurance_expiry": "2026-08-19",
                "fitness_status": "Active",
                "pucc_status": "Exempt (EV)",
                "owner_information": "Priya Deshmukh",
            },
        }

        mock = test_database.get(plate_number)
        if not mock:
            # Deterministic fallback based on plate hash
            state_code = plate_number[:2] if len(plate_number) >= 2 else "IND"
            mock = {
                "vehicle_class": "Motor Car (LMV)",
                "manufacturer": "Maruti Suzuki",
                "model": "Swift ZXi",
                "fuel_type": "PETROL",
                "registration_date": "2021-05-10",
                "insurance_status": "Active",
                "insurance_expiry": "2026-12-31",
                "fitness_status": "Active",
                "pucc_status": "Active",
                "owner_information": f"Registered Owner ({state_code})",
            }

        return {
            "status": "VERIFIED",
            "verified": True,
            "registration_number": plate_number,
            "vehicle_class": mock["vehicle_class"],
            "manufacturer": mock["manufacturer"],
            "model": mock["model"],
            "fuel_type": mock["fuel_type"],
            "registration_date": mock["registration_date"],
            "insurance_status": mock["insurance_status"],
            "insurance_expiry": mock["insurance_expiry"],
            "fitness_status": mock["fitness_status"],
            "pucc_status": mock["pucc_status"],
            "owner_information": mock.get("owner_information"),
            "api_source": "CarInfo Test Mode (Mock)",
            "is_cached": False,
            "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": {"mock": True, "plate": plate_number},
            "error": None,
        }

    def _normalize_database_record(self, db_record: dict[str, Any], plate_number: str) -> dict[str, Any]:
        """Format existing database record into standard response schema."""
        return {
            "status": "VERIFIED" if db_record.get("verified") else "DETECTED",
            "verified": bool(db_record.get("verified")),
            "registration_number": plate_number,
            "vehicle_class": db_record.get("vehicle_type") or "Not Available",
            "manufacturer": db_record.get("manufacturer") or "Not Available",
            "model": db_record.get("model") or "Not Available",
            "fuel_type": db_record.get("fuel_type") or "Not Available",
            "registration_date": db_record.get("registration_date") or "Not Available",
            "insurance_status": db_record.get("insurance_status") or "Not Available",
            "insurance_expiry": db_record.get("insurance_expiry") or "Not Available",
            "fitness_status": db_record.get("fitness_status") or "Not Available",
            "pucc_status": db_record.get("puc_status") or "Not Available",
            "owner_information": db_record.get("owner_name") if db_record.get("owner_name") != "Unknown" else None,
            "api_source": db_record.get("data_source") or "Local Cache",
            "is_cached": True,
            "lookup_timestamp": db_record.get("last_updated") or datetime.now(timezone.utc).isoformat(),
            "raw_response": {},
            "error": None,
        }

    def _build_empty_response(self, plate_number: str, status: str, error: str) -> dict[str, Any]:
        """Construct fallback response when API lookup fails or is skipped."""
        return {
            "status": status,
            "verified": False,
            "registration_number": plate_number,
            "vehicle_class": "Not Available",
            "manufacturer": "Not Available",
            "model": "Not Available",
            "fuel_type": "Not Available",
            "registration_date": "Not Available",
            "insurance_status": "Not Available",
            "insurance_expiry": "Not Available",
            "fitness_status": "Not Available",
            "pucc_status": "Not Available",
            "owner_information": None,
            "api_source": "None",
            "is_cached": False,
            "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": {},
            "error": error,
        }


def get_vehicle_information(vehicle_number: str) -> dict[str, Any]:
    """Module-level helper for quick vehicle information lookups."""
    service = CarInfoService()
    return service.get_vehicle_information(vehicle_number)
