from __future__ import annotations

from datetime import datetime
from typing import Any

try:
    import requests
except Exception:
    requests = None

from config import Config
from modules.database_manager import DatabaseManager
from modules.utils import clean_plate_text, get_logger, is_valid_plate_text


class RTOVehicleService:
    """Service layer for authorized RTO/VAHAN vehicle registration lookups."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.logger = get_logger("rto_vehicle_service")
        self.api_url = Config.RTO_API_URL
        self.provider = Config.RTO_API_PROVIDER
        self.api_key = Config.RTO_API_KEY
        self.client_id = Config.RTO_API_CLIENT_ID
        self.client_secret = Config.RTO_API_CLIENT_SECRET
        self.timeout = Config.RTO_API_TIMEOUT

    def get_vehicle_details(self, registration_number: str) -> dict[str, Any]:
        """Lookup registration data from an authorized RTO/VAHAN API."""
        plate_norm = clean_plate_text(registration_number)
        if not plate_norm:
            return {"status": "invalid_plate", "error": "empty_plate", "vehicle": None}
        if not is_valid_plate_text(plate_norm):
            return {"status": "invalid_plate", "error": "invalid_format", "vehicle": None}

        if not self.api_url:
            return {"status": "not_configured", "error": "api_not_configured", "vehicle": None}

        if requests is None:
            self.logger.warning("requests library not installed; cannot perform RTO lookup")
            return {"status": "error", "error": "requests_not_installed", "vehicle": None}

        try:
            headers: dict[str, str] = {}
            params: dict[str, Any] = {"plate": plate_norm}
            if self.provider:
                params["provider"] = self.provider
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            if self.client_id:
                headers["X-Client-Id"] = self.client_id
            if self.client_secret:
                headers["X-Client-Secret"] = self.client_secret

            response = requests.get(self.api_url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if not data:
                return {"status": "not_found", "error": "not_found", "vehicle": None}
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                return {"status": "not_found", "error": "invalid_response", "vehicle": None}

            vehicle = self._normalize_response(data, plate_norm)
            if not vehicle:
                return {"status": "not_found", "error": "not_found", "vehicle": None}

            self._cache_vehicle_data(vehicle)
            vehicle["data_source"] = self.provider or "rto_api"
            return {"status": "success", "vehicle": vehicle}
        except requests.exceptions.Timeout:
            return {"status": "error", "error": "timeout", "vehicle": None}
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            self.logger.warning("RTO API HTTP error %s for %s", status_code, plate_norm)
            return {"status": "error", "error": f"http_{status_code}", "vehicle": None}
        except Exception as exc:
            self.logger.exception("RTO lookup failed for %s: %s", plate_norm, exc)
            return {"status": "error", "error": "lookup_failed", "vehicle": None}

    def _normalize_response(self, data: dict[str, Any], plate_norm: str) -> dict[str, Any] | None:
        """Normalize API response fields into a permitted vehicle record."""
        if not isinstance(data, dict):
            return None

        vehicle = {
            "plate_number": plate_norm,
            "registered_vehicle_class": data.get("registered_vehicle_class") or data.get("vehicle_class") or data.get("vehicle_category") or data.get("class") or data.get("vehicleCategory") or None,
            "manufacturer": data.get("manufacturer") or data.get("make") or data.get("brand") or None,
            "model": data.get("model") or data.get("variant") or data.get("vehicle_model") or None,
            "fuel_type": data.get("fuel_type") or data.get("fuel") or data.get("fuelType") or None,
            "vehicle_color": data.get("vehicle_color") or data.get("color") or None,
            "registration_date": data.get("registration_date") or data.get("registered_at") or data.get("registrationDate") or None,
            "registration_expiry": data.get("registration_valid_until") or data.get("expiry_date") or data.get("registrationExpiry") or None,
            "rc_status": data.get("rc_status") or data.get("rcStatus") or data.get("status") or None,
            "insurance_provider": data.get("insurance_provider") or data.get("insuranceProvider") or data.get("insurer") or None,
            "insurance_status": data.get("insurance_status") or data.get("insuranceStatus") or None,
            "insurance_expiry": data.get("insurance_expiry_date") or data.get("insuranceExpiry") or data.get("insurance_expiry") or None,
            "puc_status": data.get("puc_status") or data.get("pucStatus") or None,
            "puc_expiry": data.get("puc_expiry_date") or data.get("pucExpiry") or None,
            "owner_name": data.get("owner_name") or data.get("ownerName") or data.get("registered_owner") or None,
            "data_source": self.provider or "rto_api",
            "last_updated": data.get("updated_at") or data.get("last_updated") or datetime.utcnow().isoformat(),
        }

        # Only keep fields that were actually returned or mapped
        return {k: v for k, v in vehicle.items() if v is not None}

    def _cache_vehicle_data(self, vehicle: dict[str, Any]) -> None:
        """Store permitted API response in the local database cache."""
        if not vehicle or not vehicle.get("plate_number"):
            return

        cache_data = {
            "vehicle_type": vehicle.get("registered_vehicle_class") or vehicle.get("manufacturer") or None,
            "manufacturer": vehicle.get("manufacturer"),
            "model": vehicle.get("model"),
            "vehicle_color": vehicle.get("vehicle_color"),
            "fuel_type": vehicle.get("fuel_type"),
            "registration_date": vehicle.get("registration_date"),
            "registration_state": vehicle.get("registration_state"),
            "insurance_provider": vehicle.get("insurance_provider"),
            "insurance_status": vehicle.get("insurance_status"),
            "insurance_expiry": vehicle.get("insurance_expiry"),
            "puc_status": vehicle.get("puc_status"),
            "puc_expiry": vehicle.get("puc_expiry"),
            "rc_status": vehicle.get("rc_status"),
            "owner_name": vehicle.get("owner_name"),
            "vehicle_color": vehicle.get("vehicle_color"),
            "data_source": vehicle.get("data_source") or self.provider or "rto_api",
            "last_updated": vehicle.get("last_updated") or datetime.utcnow().isoformat(),
            "registered_vehicle_class": vehicle.get("registered_vehicle_class"),
        }
        self.db.update_vehicle(vehicle["plate_number"], cache_data)


def get_vehicle_details(registration_number: str) -> dict[str, Any]:
    db = DatabaseManager()
    service = RTOVehicleService(db)
    return service.get_vehicle_details(registration_number)
