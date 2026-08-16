"""SQLite-backed database manager for detections, users, blacklist entries, and toll data with source provenance and verification tracking."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config import Config
from modules.utils import clean_plate_text, get_logger, hash_password, verify_password

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(Config.SQLALCHEMY_DATABASE_URL.replace("sqlite:///", ""))


class DatabaseManager:
    """Manage persistent storage and simple queries for the ANPR demo."""

    def __init__(self) -> None:
        self.logger = get_logger("database")
        self.db_path = DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        """Create the database schema, seed default data, and ensure migrations."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'admin'
                );

                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    vehicle_type TEXT,
                    manufacturer TEXT,
                    model TEXT,
                    variant TEXT,
                    vehicle_color TEXT,
                    fuel_type TEXT,
                    engine_number TEXT,
                    chassis_number TEXT,
                    registration_date TEXT,
                    registration_state TEXT,
                    insurance_provider TEXT,
                    insurance_status TEXT,
                    insurance_expiry TEXT,
                    puc_status TEXT,
                    puc_expiry TEXT,
                    fastag_status TEXT,
                    fastag_balance REAL,
                    permit_status TEXT,
                    fitness_status TEXT,
                    rc_status TEXT,
                    owner_name TEXT,
                    city TEXT,
                    district TEXT,
                    state TEXT,
                    data_source TEXT,
                    source TEXT DEFAULT 'local_database',
                    verified INTEGER DEFAULT 0,
                    verified_at TEXT,
                    last_updated TEXT
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    vehicle_type TEXT,
                    confidence REAL,
                    detection_date TEXT,
                    detection_time TEXT,
                    camera_name TEXT,
                    vehicle_image TEXT,
                    plate_image TEXT,
                    source_type TEXT DEFAULT '',
                    original_image TEXT,
                    processed_image TEXT,
                    cropped_plate TEXT,
                    lookup_status TEXT,
                    lookup_error TEXT,
                    detected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    video_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS live_scan_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    vehicle_type TEXT,
                    confidence REAL,
                    plate_confidence REAL,
                    camera_name TEXT,
                    source_url TEXT,
                    note TEXT,
                    gate_open INTEGER DEFAULT 0,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    camera_type TEXT,
                    url TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS blacklist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS fastag_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT UNIQUE NOT NULL,
                    balance REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'Active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS toll_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    entry_time TEXT,
                    exit_time TEXT,
                    vehicle_number TEXT,
                    vehicle_type TEXT,
                    fastag_balance_before REAL,
                    toll_amount REAL,
                    balance_after REAL,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Auto-migrations for vehicles table
            cursor = conn.execute("PRAGMA table_info(vehicles)")
            existing_veh_cols = {row[1] for row in cursor.fetchall()}
            if "source" not in existing_veh_cols:
                conn.execute("ALTER TABLE vehicles ADD COLUMN source TEXT DEFAULT 'local_database'")
            if "verified" not in existing_veh_cols:
                conn.execute("ALTER TABLE vehicles ADD COLUMN verified INTEGER DEFAULT 0")
            if "verified_at" not in existing_veh_cols:
                conn.execute("ALTER TABLE vehicles ADD COLUMN verified_at TEXT")
            if "raw_api_response" not in existing_veh_cols:
                conn.execute("ALTER TABLE vehicles ADD COLUMN raw_api_response TEXT")


            # Seed default admin user if missing
            admin_user = conn.execute("SELECT * FROM users WHERE username = ?", (Config.DEFAULT_ADMIN_USERNAME,)).fetchone()
            if not admin_user:
                conn.execute(
                    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                    (Config.DEFAULT_ADMIN_USERNAME, hash_password(Config.DEFAULT_ADMIN_PASSWORD), "admin"),
                )

            # Seed default vehicle records for local testing
            vehicle_seed = [
                (
                    "KA01AB1234",
                    "Car",
                    "Hyundai",
                    "i20",
                    "Asta",
                    "White",
                    "Petrol",
                    "ENG12345",
                    "CHAS12345",
                    "2022-01-10",
                    "Karnataka",
                    "Active",
                    "2027-01-10",
                    "Active",
                    "2027-01-10",
                    "Active",
                    250.0,
                    "Active",
                    "Active",
                    "Active",
                    "Rajesh Kumar",
                    "Bengaluru",
                    "Bengaluru",
                    "Karnataka",
                    "local_database",
                    0,
                ),
                (
                    "DL04C1234",
                    "Bike",
                    "Honda",
                    "Activa",
                    "Standard",
                    "Black",
                    "Petrol",
                    "ENG67890",
                    "CHAS67890",
                    "2021-06-05",
                    "Delhi",
                    "Active",
                    "2026-06-05",
                    "Active",
                    "2026-06-05",
                    "Active",
                    120.0,
                    "Active",
                    "Active",
                    "Active",
                    "Anil Sharma",
                    "Delhi",
                    "Delhi",
                    "Delhi",
                    "local_database",
                    0,
                ),
            ]
            conn.executemany(
                """
                INSERT OR IGNORE INTO vehicles (
                    plate_number, vehicle_type, manufacturer, model, variant, vehicle_color,
                    fuel_type, engine_number, chassis_number, registration_date, registration_state,
                    insurance_status, insurance_expiry, puc_status, puc_expiry, fastag_status,
                    fastag_balance, permit_status, fitness_status, rc_status, owner_name, city, district, state,
                    source, verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                vehicle_seed,
            )
            conn.execute("INSERT OR IGNORE INTO fastag_accounts (plate_number, balance, status) VALUES (?, ?, ?)", ("KA01AB1234", 400.0, "Active"))
            conn.execute("INSERT OR IGNORE INTO fastag_accounts (plate_number, balance, status) VALUES (?, ?, ?)", ("DL04C1234", 150.0, "Active"))
            conn.commit()

            # Ensure detection extra columns exist
            cursor = conn.execute("PRAGMA table_info(detections)")
            existing_det_cols = {row[1] for row in cursor.fetchall()}
            if "source_type" not in existing_det_cols:
                conn.execute("ALTER TABLE detections ADD COLUMN source_type TEXT DEFAULT ''")
            if "original_image" not in existing_det_cols:
                conn.execute("ALTER TABLE detections ADD COLUMN original_image TEXT")
            if "processed_image" not in existing_det_cols:
                conn.execute("ALTER TABLE detections ADD COLUMN processed_image TEXT")
            if "cropped_plate" not in existing_det_cols:
                conn.execute("ALTER TABLE detections ADD COLUMN cropped_plate TEXT")
            if "detected_at" not in existing_det_cols:
                conn.execute("ALTER TABLE detections ADD COLUMN detected_at TEXT")
                conn.execute("UPDATE detections SET detected_at = ? WHERE detected_at IS NULL OR detected_at = ''", (datetime.utcnow().isoformat(),))
            conn.commit()

            # Create staging table for RTO imports
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS rto_imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    payload TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    approved INTEGER DEFAULT 0
                );
                """
            )
            conn.commit()

    def authenticate_user(self, username: str, password: str) -> dict[str, Any] | None:
        """Authenticate an admin user."""
        self._initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                return None
            stored_hash = row["password"]
            ok = verify_password(password, stored_hash)
            if ok:
                return {"id": row["id"], "username": row["username"], "role": row["role"]}
            if username == Config.DEFAULT_ADMIN_USERNAME and password == Config.DEFAULT_ADMIN_PASSWORD:
                conn.execute(
                    "UPDATE users SET password = ? WHERE username = ?",
                    (hash_password(Config.DEFAULT_ADMIN_PASSWORD), username),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                if row and verify_password(password, row["password"]):
                    return {"id": row["id"], "username": row["username"], "role": row["role"]}
        return None

    def add_detection(
        self,
        plate_number: str,
        vehicle_type: str,
        confidence: float,
        camera_name: str,
        vehicle_image: str = "",
        plate_image: str = "",
        video_name: str = "",
        source_type: str = "",
        original_image: str = "",
        processed_image: str = "",
        cropped_plate: str = "",
        lookup_status: str = "",
        lookup_error: str = "",
    ) -> None:
        """Store an ANPR detection record."""
        now = datetime.utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO detections (
                    plate_number, vehicle_type, confidence, detection_date, detection_time,
                    camera_name, vehicle_image, plate_image, source_type, original_image, processed_image,
                    cropped_plate, lookup_status, lookup_error, video_name, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plate_number,
                    vehicle_type,
                    confidence,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    camera_name,
                    vehicle_image,
                    plate_image,
                    source_type,
                    original_image,
                    processed_image,
                    cropped_plate,
                    lookup_status,
                    lookup_error,
                    video_name,
                    now.isoformat(),
                ),
            )
            conn.commit()

    def add_vehicle_record(
        self,
        plate_number: str,
        vehicle_type: str = "Unknown",
        owner_name: str = "Unknown",
        registration_state: str | None = None,
        vehicle_color: str | None = None,
        fastag_status: str | None = None,
        fastag_balance: float | None = None,
        source: str = "local_database",
        verified: bool = False,
    ) -> None:
        """Create a local vehicle record tagged with source and verified status."""
        plate = clean_plate_text(plate_number)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO vehicles (
                    plate_number, vehicle_type, owner_name, registration_state, vehicle_color,
                    fastag_status, fastag_balance, source, verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plate,
                    vehicle_type,
                    owner_name,
                    registration_state,
                    vehicle_color,
                    fastag_status,
                    fastag_balance,
                    source,
                    1 if verified else 0,
                ),
            )
            conn.commit()

    def get_vehicle_record(self, plate_number: str) -> dict[str, Any] | None:
        """Fetch a local vehicle record."""
        normalized_plate = clean_plate_text(plate_number or "")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM vehicles WHERE plate_number = ?", (normalized_plate,)).fetchone()
        return dict(row) if row else None

    def get_vehicle_info(self, plate_number: str) -> dict[str, Any] | None:
        """Fetch vehicle registration record together with FASTag details and data provenance tags."""
        normalized_plate = clean_plate_text(plate_number or "")
        if not normalized_plate:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    v.*,
                    fa.balance AS fastag_balance,
                    fa.status AS fastag_status
                FROM vehicles v
                LEFT JOIN fastag_accounts fa ON v.plate_number = fa.plate_number
                WHERE v.plate_number = ?
                """,
                (normalized_plate,),
            ).fetchone()

        if not row:
            return None

        res = dict(row)
        src = res.get("source") or res.get("data_source") or "local_database"
        is_verified = bool(res.get("verified"))

        if src == "official_authorized_api" and is_verified:
            source_label = "Official Authorized API"
            status_label = "VERIFIED"
        elif src == "imported_dataset":
            source_label = "Imported Dataset"
            status_label = "NOT VERIFIED"
        elif src == "local_database":
            source_label = "Local Database"
            status_label = "NOT VERIFIED"
        else:
            source_label = "Unknown / Unverified"
            status_label = "NOT VERIFIED"

        res["source"] = src
        res["source_label"] = source_label
        res["verified"] = is_verified
        res["verification_status"] = status_label
        res["verification_message"] = "Official Authorized Registration" if is_verified else "Vehicle details unverified (local/dataset record)"
        return res

    def fetch_and_save_rto_record(self, plate_number: str) -> dict[str, Any] | None:
        """Attempt to fetch vehicle details from an official external RTO API and tag as verified."""
        import json

        normalized_plate = clean_plate_text(plate_number or "")
        api_url = getattr(Config, "RTO_API_URL", None)
        if not api_url or not normalized_plate:
            return None

        try:
            try:
                import requests
                headers = {}
                params = {"plate": plate_number}
                if getattr(Config, "RTO_API_PROVIDER", None):
                    params["provider"] = Config.RTO_API_PROVIDER
                if getattr(Config, "RTO_API_KEY", None):
                    headers["Authorization"] = f"Bearer {Config.RTO_API_KEY}"
                if getattr(Config, "RTO_API_CLIENT_ID", None):
                    headers["X-Client-Id"] = Config.RTO_API_CLIENT_ID
                if getattr(Config, "RTO_API_CLIENT_SECRET", None):
                    headers["X-Client-Secret"] = Config.RTO_API_CLIENT_SECRET
                resp = requests.get(api_url, headers=headers, params=params, timeout=5)
                if resp.status_code != 200:
                    return None
                data = resp.json()
            except Exception:
                from urllib import request as urlreq, parse
                params = {"plate": plate_number}
                if getattr(Config, "RTO_API_PROVIDER", None):
                    params["provider"] = Config.RTO_API_PROVIDER
                q = parse.urlencode(params)
                with urlreq.urlopen(f"{api_url}?{q}", timeout=5) as r:
                    raw = r.read()
                    data = json.loads(raw.decode("utf-8"))

            if not data or not isinstance(data, dict):
                return None

            plate_value = data.get("plate_number") or data.get("plate") or plate_number
            now_iso = datetime.utcnow().isoformat()

            data["source"] = "official_authorized_api"
            data["verified"] = 1
            data["verified_at"] = now_iso

            vehicle_fields = (
                "plate_number", "vehicle_type", "manufacturer", "model", "variant", "vehicle_color",
                "fuel_type", "engine_number", "chassis_number", "registration_date", "registration_state",
                "insurance_status", "insurance_expiry", "puc_status", "puc_expiry", "fastag_status",
                "fastag_balance", "permit_status", "fitness_status", "rc_status", "owner_name", "city",
                "district", "state", "source", "verified", "verified_at"
            )
            values = [data.get(k) for k in vehicle_fields]
            values[0] = plate_value
            values[-3] = "official_authorized_api"
            values[-2] = 1
            values[-1] = now_iso

            payload = json.dumps(data)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vehicles (
                        plate_number, vehicle_type, manufacturer, model, variant, vehicle_color,
                        fuel_type, engine_number, chassis_number, registration_date, registration_state,
                        insurance_status, insurance_expiry, puc_status, puc_expiry, fastag_status,
                        fastag_balance, permit_status, fitness_status, rc_status, owner_name, city,
                        district, state, source, verified, verified_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    values,
                )
                conn.execute(
                    "INSERT OR REPLACE INTO fastag_accounts (plate_number, balance, status, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    (plate_value, data.get("fastag_balance") or 0.0, data.get("fastag_status") or "Active"),
                )
                conn.execute("INSERT INTO rto_imports (plate_number, payload) VALUES (?, ?)", (plate_value, payload))
                conn.commit()

            return data
        except Exception:
            return None

    def approve_rto_import(self, import_id: int) -> bool:
        """Approve an RTO import and upsert into vehicles with source='imported_dataset' and verified=0."""
        import json
        rec = self.get_rto_import(import_id)
        if not rec:
            return False
        data = json.loads(rec["payload"])
        
        vehicle_fields = (
            "plate_number", "vehicle_type", "manufacturer", "model", "variant", "vehicle_color",
            "fuel_type", "engine_number", "chassis_number", "registration_date", "registration_state",
            "insurance_status", "insurance_expiry", "puc_status", "puc_expiry", "fastag_status",
            "fastag_balance", "permit_status", "fitness_status", "rc_status", "owner_name", "city",
            "district", "state"
        )
        values = [data.get(k) for k in vehicle_fields]
        plate_val = data.get("plate_number") or data.get("plate")
        values[0] = plate_val

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vehicles (
                    plate_number, vehicle_type, manufacturer, model, variant, vehicle_color,
                    fuel_type, engine_number, chassis_number, registration_date, registration_state,
                    insurance_status, insurance_expiry, puc_status, puc_expiry, fastag_status,
                    fastag_balance, permit_status, fitness_status, rc_status, owner_name, city, district, state,
                    source, verified
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'imported_dataset', 0)
                """,
                values,
            )
            fastag_balance = data.get("fastag_balance")
            fastag_status = data.get("fastag_status")
            if fastag_balance is not None or fastag_status is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO fastag_accounts (plate_number, balance, status, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    (plate_val, fastag_balance or 0.0, fastag_status or "Active"),
                )
            conn.execute("UPDATE rto_imports SET approved = 1 WHERE id = ?", (import_id,))
            conn.commit()
        return True

    def get_blacklist_entry(self, plate_number: str) -> dict[str, Any] | None:
        """Return a blacklist record for the given plate."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM blacklist WHERE plate_number = ?", (plate_number,)).fetchone()
        return dict(row) if row else None

    def get_last_detection(self, plate_number: str) -> dict[str, Any] | None:
        """Return the most recent detection for the given plate."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM detections WHERE plate_number = ? ORDER BY created_at DESC LIMIT 1",
                (plate_number,),
            ).fetchone()
        return dict(row) if row else None

    def can_save_detection(self, plate_number: str, max_seconds: int) -> bool:
        """Return whether a new detection for this plate should be stored."""
        last = self.get_last_detection(plate_number)
        if not last or not last.get("created_at"):
            return True
        try:
            last_time = datetime.fromisoformat(last["created_at"])
        except ValueError:
            return True
        return (datetime.utcnow() - last_time).total_seconds() > max_seconds

    def update_fastag_balance(self, plate_number: str, balance: float) -> None:
        """Update FASTag balance for a plate."""
        with self._connect() as conn:
            conn.execute("UPDATE fastag_accounts SET balance = ? WHERE plate_number = ?", (balance, plate_number))
            conn.execute("UPDATE vehicles SET fastag_balance = ? WHERE plate_number = ?", (balance, plate_number))
            conn.commit()

    def charge_toll(self, plate_number: str, toll_amount: float) -> dict[str, Any] | None:
        """Charge toll for a vehicle and store the transaction."""
        vehicle = self.get_vehicle_record(plate_number)
        if not vehicle:
            return None
        balance_before = float(vehicle.get("fastag_balance") or 0.0)
        balance_after = balance_before - toll_amount
        status = "Approved" if balance_after >= 0 else "Insufficient Balance"
        if status == "Approved":
            self.update_fastag_balance(plate_number, balance_after)
        transaction_id = f"TOLL-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{plate_number}"
        self.add_toll_transaction(
            transaction_id,
            plate_number,
            vehicle.get("vehicle_type", "Unknown"),
            balance_before,
            toll_amount,
            balance_after,
            status,
        )
        return {
            "transaction_id": transaction_id,
            "vehicle_number": plate_number,
            "vehicle_type": vehicle.get("vehicle_type", "Unknown"),
            "owner_name": vehicle.get("owner_name", "Unknown"),
            "puc_status": vehicle.get("puc_status", "Unknown"),
            "fastag_balance_before": balance_before,
            "toll_amount": toll_amount,
            "balance_after": balance_after,
            "status": status,
            "vehicle": vehicle,
        }

    def add_toll_transaction(self, transaction_id: str, vehicle_number: str, vehicle_type: str, balance_before: float, toll_amount: float, balance_after: float, status: str) -> None:
        """Store a toll plaza transaction."""
        now = datetime.utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO toll_transactions (transaction_id, entry_time, exit_time, vehicle_number, vehicle_type, fastag_balance_before, toll_amount, balance_after, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (transaction_id, now.isoformat(), now.isoformat(), vehicle_number, vehicle_type, balance_before, toll_amount, balance_after, status),
            )
            conn.commit()

    def get_dashboard_stats(self) -> dict[str, Any]:
        """Compute dashboard summary statistics."""
        with self._connect() as conn:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            total_entries = conn.execute("SELECT COUNT(*) AS count FROM detections").fetchone()["count"]
            today_vehicles = conn.execute("SELECT COUNT(*) AS count FROM detections WHERE detection_date = ?", (today,)).fetchone()["count"]
            unique_today = conn.execute("SELECT COUNT(DISTINCT plate_number) AS count FROM detections WHERE detection_date = ?", (today,)).fetchone()["count"]
            blacklist_count = conn.execute("SELECT COUNT(*) AS count FROM blacklist").fetchone()["count"]
            
            verified_count = conn.execute("SELECT COUNT(*) AS count FROM vehicles WHERE verified = 1").fetchone()["count"]
            total_vehicles = conn.execute("SELECT COUNT(*) AS count FROM vehicles").fetchone()["count"]
            verification_rate = round((verified_count / float(max(1, total_vehicles))) * 100.0, 1)

        return {
            "total_vehicles_today": today_vehicles,
            "unique_vehicles_today": unique_today,
            "total_entries": total_entries,
            "blacklist_count": blacklist_count,
            "verified_count": verified_count,
            "verification_rate": verification_rate,
            "live_camera_status": "Connected",
            "todays_reports": today_vehicles,
        }

    def get_recent_detections(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent detections for the UI."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def search_detections(self, plate: str | None = None, date: str | None = None, vehicle_type: str | None = None) -> list[dict[str, Any]]:
        """Search detections using supplied filters."""
        query = "SELECT * FROM detections WHERE 1=1"
        params: list[Any] = []
        if plate:
            query += " AND plate_number LIKE ?"
            params.append(f"%{plate}%")
        if date:
            query += " AND detection_date = ?"
            params.append(date)
        if vehicle_type:
            query += " AND vehicle_type LIKE ?"
            params.append(f"%{vehicle_type}%")
        query += " ORDER BY id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def add_blacklist_entry(self, plate_number: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO blacklist (plate_number, reason) VALUES (?, ?)", (plate_number, reason))
            conn.commit()

    def get_blacklist_entries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM blacklist ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    def delete_blacklist_entry(self, blacklist_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM blacklist WHERE id = ?", (blacklist_id,))
            conn.commit()

    def add_live_scan_event(self, plate_number: str, vehicle_type: str, confidence: float, camera_name: str = "Camera", source_url: str = "", note: str = "", gate_open: bool = False, status: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_scan_events (plate_number, vehicle_type, confidence, camera_name, source_url, note, gate_open, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (plate_number, vehicle_type, confidence, camera_name, source_url, note, int(gate_open), status),
            )
            conn.commit()

    def get_live_scan_events(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM live_scan_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_reports(self, period: str = "daily") -> list[dict[str, Any]]:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with self._connect() as conn:
            if period == "weekly":
                start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
                rows = conn.execute("SELECT detection_date AS period, COUNT(*) AS count FROM detections WHERE detection_date >= ? GROUP BY detection_date ORDER BY detection_date", (start,)).fetchall()
            elif period == "monthly":
                rows = conn.execute("SELECT substr(detection_date, 1, 7) AS period, COUNT(*) AS count FROM detections GROUP BY substr(detection_date, 1, 7) ORDER BY period", ()).fetchall()
            else:
                rows = conn.execute("SELECT detection_date AS period, COUNT(*) AS count FROM detections WHERE detection_date = ? GROUP BY detection_date", (today,)).fetchall()
        return [dict(row) for row in rows]

    def add_rto_import(self, data: dict[str, Any]) -> int:
        import json
        plate = data.get("plate_number") or data.get("plate")
        payload = json.dumps(data)
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO rto_imports (plate_number, payload) VALUES (?, ?)", (plate, payload))
            conn.commit()
            return cur.lastrowid

    def list_rto_imports(self, only_pending: bool = True) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if only_pending:
                rows = conn.execute("SELECT * FROM rto_imports WHERE approved = 0 ORDER BY id DESC").fetchall()
            else:
                rows = conn.execute("SELECT * FROM rto_imports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    def get_rto_import(self, import_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM rto_imports WHERE id = ?", (import_id,)).fetchone()
        return dict(row) if row else None

    def get_toll_transactions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM toll_transactions ORDER BY id DESC LIMIT 20").fetchall()
        return [dict(row) for row in rows]
