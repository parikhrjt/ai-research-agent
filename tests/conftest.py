"""Shared test fixtures."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("VECTOR_STORE", "chroma")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture
def sample_markdown_path() -> Path:
    return SAMPLES_DIR / "transformer_architecture.md"


@pytest.fixture
def sample_txt_path() -> Path:
    return SAMPLES_DIR / "rag_pipeline_notes.txt"


@pytest.fixture
def sample_csv_path() -> Path:
    return SAMPLES_DIR / "experiment_results.csv"


@pytest.fixture
def sample_markdown_bytes(sample_markdown_path) -> bytes:
    return sample_markdown_path.read_bytes()


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
