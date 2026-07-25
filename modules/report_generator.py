"""Report export helpers for CSV, Excel, and PDF output."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from modules.utils import get_logger


class ReportGenerator:
    """Generate downloadable reports using CSV, Excel, and simple PDF stubs."""

    def __init__(self) -> None:
        self.logger = get_logger("reports")
        self.output_dir = Path(Config.REPORTS_FOLDER)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, rows: list[dict[str, Any]], fmt: str = "csv", period: str = "daily") -> str:
        """Export a report to the requested file format."""
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        path = self.output_dir / f"anpr_report_{period}_{stamp}.{fmt.lower()}"
        if fmt == "csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["period", "count"])
                writer.writeheader()
                writer.writerows(rows)
        elif fmt == "xlsx":
            df = pd.DataFrame(rows)
            df.to_excel(path, index=False)
        elif fmt == "pdf":
            path.write_text("PDF export is not fully implemented in this demo build.", encoding="utf-8")
        else:
            raise ValueError("Unsupported export format")
        self.logger.info("Exported report %s", path)
        return str(path)
