"""Alerting utilities for blacklist events and suspicious detections."""
from __future__ import annotations

import os
import platform
from datetime import datetime
from typing import Any

from modules.utils import get_logger


class AlertManager:
    """Manage alerts, notifications, and an optional local alarm."""

    def __init__(self) -> None:
        self.logger = get_logger("alerts")
        self.alerts: list[dict[str, Any]] = []

    def trigger_alert(self, plate_number: str, reason: str) -> None:
        """Create an alert entry and trigger an alarm if available."""
        alert = {
            "plate_number": plate_number,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.alerts.append(alert)
        self.logger.warning("Alert triggered for %s: %s", plate_number, reason)
        self.play_alarm()

    def play_alarm(self) -> None:
        """Play a simple alarm sound if the host supports it."""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)
            elif platform.system() == "Darwin":
                os.system("afplay /System/Library/Sounds/Glass.aiff")
            else:
                os.system("paplay /usr/share/sounds/alsa/Front_Center.wav >/dev/null 2>&1")
        except Exception:
            return

    def get_alerts(self) -> list[dict[str, Any]]:
        """Return recent alerts."""
        return self.alerts[-10:]
