from __future__ import annotations

import os
from typing import Any

try:
    import requests
except Exception:
    requests = None

from modules.database_manager import DatabaseManager
from modules.utils import clean_plate_text, get_logger, is_valid_plate_text


class VehicleLookupService:
    """Lookup vehicle details from local DB or authorized external API."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.logger = get_logger("vehicle_lookup_service")
        self.api_url = os.getenv("VEHICLE_API_URL")
        self.api_key = os.getenv("VEHICLE_API_KEY")
        self.provider = os.getenv("VEHICLE_API_PROVIDER")
        self.timeout = float(os.getenv("VEHICLE_API_TIMEOUT", "10"))

    def lookup(self, plate: str) -> dict[str, Any]:
        """Lookup a normalized plate from local DB first, then external API if configured."""
        plate_norm = clean_plate_text(plate)
        if not plate_norm:
            return {"status": "invalid_plate", "error": "empty_plate", "vehicle": None}
        if not is_valid_plate_text(plate_norm):
            return {"status": "invalid_plate", "error": "invalid_format", "vehicle": None}

        vehicle = self.db.get_vehicle_info(plate_norm)
        if vehicle:
            vehicle["data_source"] = "local"
            return {"status": "local", "vehicle": vehicle}

        if not self.api_url:
            return {"status": "not_found", "vehicle": None}

        if requests is None:
            self.logger.warning("requests library is required for external vehicle lookup")
            return {"status": "error", "error": "requests_not_installed", "vehicle": None}

        try:
            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            params = {"plate": plate_norm}
            if self.provider:
                params["provider"] = self.provider

            response = requests.get(self.api_url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if not data or not isinstance(data, dict):
                return {"status": "not_found", "vehicle": None}

            mapped = self._map_api_response(data, plate_norm)
            if not mapped:
                return {"status": "not_found", "vehicle": None}

            self._save_vehicle_record(mapped)
            mapped["data_source"] = "external"
            return {"status": "external", "vehicle": mapped}
        except requests.exceptions.Timeout:
            return {"status": "error", "error": "timeout", "vehicle": None}
        except requests.exceptions.HTTPError as exc:
            self.logger.warning("Vehicle API HTTP error: %s", exc)
            return {"status": "error", "error": f"http_{exc.response.status_code}", "vehicle": None}
        except Exception as exc:
            self.logger.exception("Vehicle API lookup failed: %s", exc)
            return {"status": "error", "error": "lookup_failed", "vehicle": None}

    def _map_api_response(self, data: dict[str, Any], plate_norm: str) -> dict[str, Any] | None:
        """Map external API response fields into local vehicle record fields."""
        if not isinstance(data, dict):
            return None

        vehicle = {
            "plate_number": plate_norm,
            "vehicle_type": data.get("vehicle_type") or data.get("type") or data.get("vehicleCategory") or "Unknown",
            "manufacturer": data.get("manufacturer") or data.get("make") or data.get("brand"),
            "model": data.get("model") or data.get("variant") or data.get("vehicle_model"),
            "fuel_type": data.get("fuel_type") or data.get("fuel") or data.get("fuelType"),
            "vehicle_color": data.get("vehicle_color") or data.get("color"),
            "registration_date": data.get("registration_date") or data.get("registered_at") or data.get("registrationDate"),
            "registration_state": data.get("registration_state") or data.get("state") or data.get("registrationState"),
            "registration_expiry": data.get("registration_valid_until") or data.get("expiry_date") or data.get("registrationExpiry"),
            "rc_status": data.get("rc_status") or data.get("rcStatus") or data.get("status") or "Unknown",
            "insurance_provider": data.get("insurance_provider") or data.get("insuranceProvider"),
            "insurance_status": data.get("insurance_status") or data.get("insuranceStatus"),
            "insurance_expiry": data.get("insurance_expiry_date") or data.get("insuranceExpiry") or data.get("insurance_expiry"),
            "puc_status": data.get("puc_status") or data.get("pucStatus"),
            "puc_expiry": data.get("puc_expiry_date") or data.get("pucExpiry"),
            "owner_name": data.get("owner_name") or data.get("ownerName"),
            "data_source": self.provider or os.getenv("VEHICLE_API_PROVIDER") or "external_api",
            "last_updated": data.get("updated_at") or data.get("last_updated") or None,
        }
        return vehicle

    def _save_vehicle_record(self, vehicle: dict[str, Any]) -> None:
        """Persist permitted vehicle data into the local DB."""
        if not vehicle or not vehicle.get("plate_number"):
            return
        self.db.update_vehicle(vehicle["plate_number"], vehicle)
