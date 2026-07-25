"""Camera helpers for local, IP, and RTSP sources."""
from __future__ import annotations

from typing import Any


class CameraManager:
    """Manages camera configuration and status."""

    def __init__(self) -> None:
        self.sources: list[dict[str, Any]] = [
            {"name": "Webcam", "type": "local", "url": "0"},
            {"name": "IP Camera", "type": "ip", "url": "http://192.168.1.100/mjpg"},
            {"name": "RTSP Camera", "type": "rtsp", "url": "rtsp://user:pass@camera.local/stream"},
        ]

    def get_status(self) -> dict[str, Any]:
        """Return the current camera status summary."""
        return {
            "available": len(self.sources),
            "active_source": self.sources[0]["name"],
            "status": "Connected",
        }

    def list_sources(self) -> list[dict[str, Any]]:
        """Return supported camera sources."""
        return self.sources

    def add_source(self, name: str, camera_type: str, url: str) -> None:
        """Store a new camera source."""
        self.sources.append({"name": name, "type": camera_type, "url": url})
