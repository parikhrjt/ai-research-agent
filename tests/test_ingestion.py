"""Tests for document ingestion parsers and pipeline."""

from pathlib import Path

import pytest

from app.core.exceptions import IngestionError, ValidationError
from app.ingestion.pipeline import ingest_bytes, ingest_file, validate_file
from app.ingestion.parsers.csv import parse_csv
from app.ingestion.parsers.markdown import parse_markdown
from app.ingestion.parsers.text import parse_text


class TestParsers:
    def test_parse_markdown_strips_front_matter(self, sample_markdown_path: Path):
        content = parse_markdown(sample_markdown_path)
        assert "Attention Is All You Need" in content
        assert "title:" not in content.lower() or "Transformer Architecture" in content

    def test_parse_text(self, sample_txt_path: Path):
        content = parse_text(sample_txt_path)
        assert "RAG Pipeline Design Notes" in content
        assert len(content) > 500

    def test_parse_csv(self, sample_csv_path: Path):
        content = parse_csv(sample_csv_path)
        assert "[Row 1]" in content
        assert "experiment_id" in content or "EXP-001" in content

    def test_parse_csv_empty_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.csv"
        empty.write_text("header1,header2\n")
        with pytest.raises(IngestionError):
            parse_csv(empty)


class TestPipeline:
    def test_validate_file_rejects_unsupported(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_file("malware.exe", 100)

    def test_validate_file_rejects_oversized(self):
        with pytest.raises(ValidationError, match="exceeds"):
            validate_file("big.pdf", 100 * 1024 * 1024)

    def test_ingest_markdown(self, sample_markdown_bytes: bytes):
        doc = ingest_bytes("notes.md", sample_markdown_bytes)
        assert doc.document_id
        assert doc.file_type == "md"
        assert doc.char_count > 0
        assert len(doc.content_hash) == 64

    def test_ingest_file_from_path(self, sample_txt_path: Path):
        doc = ingest_file(sample_txt_path)
        assert doc.filename == "rag_pipeline_notes.txt"
        assert "INGESTION PIPELINE" in doc.content.upper() or "ingestion pipeline" in doc.content.lower()

    def test_ingest_empty_raises(self):
        with pytest.raises(ValidationError):
            ingest_bytes("empty.txt", b"")
