"""Tests for FastAPI endpoints."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["VECTOR_STORE"] = "chroma"
os.environ["CHROMA_PERSIST_DIR"] = "/tmp/test_api_chroma"


class MockLLM:
    def generate(self, prompt: str, system: str | None = None) -> str:
        return "Based on the context [1], the Transformer achieved 28.4 BLEU."

    def health_check(self) -> bool:
        return True


class MockEmbeddings:
    dimension = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1)] * 384 for i in range(len(texts))]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 384


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("app.rag.graph.get_embedding_provider", lambda: MockEmbeddings())

    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.vectorstore.chroma import get_chroma_store
    import app.rag.graph as graph_mod

    get_chroma_store.cache_clear()
    graph_mod._pipeline = None

    from app.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_llm(client, monkeypatch):
    """Client with mocked LLM to avoid Ollama dependency in /ask tests."""
    import app.rag.graph as graph_mod

    graph_mod._pipeline = None
    monkeypatch.setattr("app.rag.graph.get_llm_provider", lambda: MockLLM())
    yield client
    graph_mod._pipeline = None


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data


class TestUploadEndpoint:
    def test_upload_markdown(self, client: TestClient, sample_markdown_bytes: bytes):
        response = client.post(
            "/upload",
            files={"file": ("notes.md", sample_markdown_bytes, "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "notes.md"
        assert data["chunk_count"] > 0
        assert data["document_id"]

    def test_upload_rejects_empty(self, client: TestClient):
        response = client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_rejects_unsupported(self, client: TestClient):
        response = client.post(
            "/upload",
            files={"file": ("bad.exe", b"data", "application/octet-stream")},
        )
        assert response.status_code == 400


class TestAskEndpoint:
    def test_ask_requires_min_length(self, client: TestClient):
        response = client.post("/ask", json={"question": "ab"})
        assert response.status_code == 422

    def test_ask_with_no_documents(self, client_with_llm: TestClient):
        response = client_with_llm.post("/ask", json={"question": "What is a transformer?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["citations"], list)

    def test_ask_after_upload(self, client_with_llm: TestClient, sample_markdown_bytes: bytes):
        client_with_llm.post(
            "/upload",
            files={"file": ("notes.md", sample_markdown_bytes, "text/markdown")},
        )
        response = client_with_llm.post(
            "/ask",
            json={"question": "What BLEU score did the Transformer achieve?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["retrieved_count"] >= 0
