"""Tests for document chunking strategies."""

from app.chunking.splitter import chunk_text


class TestChunking:
    def test_recursive_chunking(self):
        text = "Paragraph one.\n\n" + "Sentence. " * 200
        chunks = chunk_text(text, "doc-1", "txt", {"filename": "test.txt"})
        assert len(chunks) > 1
        assert all(c.document_id == "doc-1" for c in chunks)
        assert all(c.content.strip() for c in chunks)

    def test_markdown_header_chunking(self, sample_markdown_path):
        from app.ingestion.parsers.markdown import parse_markdown
        text = parse_markdown(sample_markdown_path)
        chunks = chunk_text(text, "doc-md", "md", {"filename": "notes.md"})
        assert len(chunks) >= 3
        strategies = {c.metadata.get("strategy") for c in chunks}
        assert "markdown_headers" in strategies

    def test_csv_row_batching(self, sample_csv_path):
        from app.ingestion.parsers.csv import parse_csv
        text = parse_csv(sample_csv_path)
        chunks = chunk_text(text, "doc-csv", "csv", {"filename": "data.csv"})
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("strategy") == "csv_row_batch"

    def test_chunk_indices_are_sequential(self):
        text = "Word " * 500
        chunks = chunk_text(text, "doc-seq", "txt")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_unique(self):
        text = "Content block. " * 100
        chunks = chunk_text(text, "doc-uniq", "txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
