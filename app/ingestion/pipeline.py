"""Document ingestion orchestration."""

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import IngestionError, ValidationError
from app.core.logging import get_logger
from app.ingestion.parsers import parse_csv, parse_markdown, parse_pdf, parse_text

logger = get_logger(__name__)

PARSERS = {
    "pdf": parse_pdf,
    "txt": parse_text,
    "md": parse_markdown,
    "markdown": parse_markdown,
    "csv": parse_csv,
}


@dataclass
class IngestedDocument:
    document_id: str
    filename: str
    file_type: str
    content: str
    content_hash: str
    char_count: int
    ingested_at: str
    source_path: str


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_file(filename: str, file_size: int) -> str:
    settings = get_settings()
    ext = Path(filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extension_set:
        raise ValidationError(
            f"Unsupported file type: .{ext}",
            {"allowed": sorted(settings.allowed_extension_set)},
        )
    if file_size > settings.max_upload_bytes:
        raise ValidationError(
            f"File exceeds {settings.max_upload_size_mb}MB limit",
            {"size_bytes": file_size},
        )
    if ext not in PARSERS:
        raise ValidationError(f"No parser registered for .{ext}")
    return ext


def save_upload(filename: str, data: bytes, upload_dir: Path | None = None) -> Path:
    settings = get_settings()
    target_dir = upload_dir or Path("data/uploads")
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    dest = target_dir / safe_name
    dest.write_bytes(data)
    logger.info("file_saved", filename=filename, path=str(dest), size=len(data))
    return dest


def ingest_file(file_path: Path, original_filename: str | None = None) -> IngestedDocument:
    filename = original_filename or file_path.name
    ext = file_path.suffix.lstrip(".").lower()
    file_size = file_path.stat().st_size

    validate_file(filename, file_size)
    parser = PARSERS[ext]

    logger.info("ingestion_started", filename=filename, file_type=ext)
    content = parser(file_path)

    if not content.strip():
        raise IngestionError("Document is empty after parsing", {"file": filename})

    doc = IngestedDocument(
        document_id=str(uuid.uuid4()),
        filename=filename,
        file_type=ext,
        content=content,
        content_hash=_compute_hash(content),
        char_count=len(content),
        ingested_at=datetime.now(timezone.utc).isoformat(),
        source_path=str(file_path),
    )
    logger.info(
        "ingestion_complete",
        document_id=doc.document_id,
        filename=filename,
        char_count=doc.char_count,
    )
    return doc


def ingest_bytes(filename: str, data: bytes) -> IngestedDocument:
    if not data:
        raise ValidationError("Uploaded file is empty")
    path = save_upload(filename, data)
    try:
        return ingest_file(path, original_filename=filename)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def cleanup_upload(file_path: str) -> None:
    path = Path(file_path)
    if path.exists():
        path.unlink(missing_ok=True)


def archive_document(source_path: str, archive_dir: Path | None = None) -> None:
    """Move processed upload to archive for audit trail."""
    src = Path(source_path)
    if not src.exists():
        return
    dest_dir = archive_dir or Path("data/archive")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest_dir / src.name))
