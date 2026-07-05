"""Tests for RAG citations and pipeline (no LLM calls)."""

from app.rag.citations import (
    build_context,
    extract_cited_indices,
    filter_used_citations,
)
from app.vectorstore.base import RetrievedChunk


def _make_chunk(idx: int, content: str, filename: str = "paper.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk_{idx}",
        document_id="doc-1",
        content=content,
        score=0.85 - idx * 0.05,
        metadata={"filename": filename, "chunk_index": idx},
    )


class TestCitations:
    def test_build_context_numbering(self):
        chunks = [_make_chunk(0, "Transformers use attention."), _make_chunk(1, "BLEU score was 28.4.")]
        context, citations = build_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert len(citations) == 2
        assert citations[0].filename == "paper.md"

    def test_extract_cited_indices(self):
        answer = "The model achieved [1] BLEU and used [2] attention heads [1]."
        indices = extract_cited_indices(answer)
        assert indices == [1, 2]

    def test_filter_used_citations(self):
        chunks = [_make_chunk(0, "A"), _make_chunk(1, "B"), _make_chunk(2, "C")]
        _, citations = build_context(chunks)
        answer = "Based on [1] and [3], we conclude..."
        used = filter_used_citations(answer, citations)
        assert len(used) == 2
        assert {c.index for c in used} == {1, 3}

    def test_no_citations_returns_all(self):
        chunks = [_make_chunk(0, "Some text")]
        _, citations = build_context(chunks)
        used = filter_used_citations("No citation markers here.", citations)
        assert len(used) == 1


class TestRAGState:
    def test_retrieve_node_filters_low_scores(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE", "chroma")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_rag_chroma")
        monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")

        from app.core.config import get_settings
        get_settings.cache_clear()

        from app.rag.graph import RAGPipeline

        class MockStore:
            def similarity_search(self, embedding, top_k):
                return [
                    _make_chunk(0, "High relevance content about transformers."),
                    RetrievedChunk("c2", "d1", "Low relevance.", 0.1, {"filename": "x.txt"}),
                ]

        class MockEmbed:
            def embed_query(self, q):
                return [0.1] * 384

        pipeline = RAGPipeline()
        pipeline._store = MockStore()
        pipeline._embeddings = MockEmbed()

        state = pipeline._retrieve({"question": "What is a transformer?"})
        assert len(state["retrieved_chunks"]) == 1
        assert state["retrieved_chunks"][0].score >= 0.5

    def test_generate_no_context(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE", "chroma")
        from app.rag.graph import RAGPipeline

        pipeline = RAGPipeline()
        result = pipeline._generate({"retrieved_chunks": [], "context": "", "citations": [], "question": "test"})
        assert "don't have enough information" in result["answer"].lower()
