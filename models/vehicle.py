from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Vehicle:
    registration_number: str | None = None
    vehicle_type: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    fuel_type: str | None = None
    color: str | None = None
    registration_date: str | None = None
    registration_valid_until: str | None = None
    rc_status: str | None = None
    insurance_provider: str | None = None
    insurance_status: str | None = None
    insurance_expiry_date: str | None = None
    puc_status: str | None = None
    puc_expiry_date: str | None = None
    owner_name: str | None = None
    data_source: str | None = None
    last_updated: str | None = None
