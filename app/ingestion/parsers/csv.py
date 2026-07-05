"""CSV notes parser — converts rows into searchable narrative text."""

import csv
from pathlib import Path

from app.core.exceptions import IngestionError


def parse_csv(file_path: Path) -> str:
    try:
        with file_path.open(encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise IngestionError("CSV has no headers", {"file": file_path.name})

            rows = []
            for idx, row in enumerate(reader, start=1):
                parts = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
                if parts:
                    rows.append(f"[Row {idx}] " + " | ".join(parts))

            if not rows:
                raise IngestionError("CSV contains no data rows", {"file": file_path.name})
            return "\n".join(rows)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Failed to parse CSV: {exc}", {"file": file_path.name}) from exc
