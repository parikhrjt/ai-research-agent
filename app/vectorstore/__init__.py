"""Vector store factory."""

from functools import lru_cache

from app.core.config import get_settings
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma import ChromaVectorStore, get_chroma_store
from app.vectorstore.pgvector import PgVectorStore, get_pgvector_store


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store == "chroma":
        return get_chroma_store()
    return get_pgvector_store()


def initialize_vector_store() -> VectorStore:
    store = get_vector_store()
    store.initialize()
    return store
