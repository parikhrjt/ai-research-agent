"""PDF document parser."""

from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import IngestionError


def parse_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        if not pages:
            raise IngestionError("PDF contains no extractable text", {"file": file_path.name})
        return "\n\n".join(pages)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF: {exc}", {"file": file_path.name}) from exc
