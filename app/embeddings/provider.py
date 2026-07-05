"""Embedding generation using local sentence-transformers."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider:
    """Singleton-style embedding provider with lazy model loading."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._dimension = settings.embedding_dimension
        self._model: SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("embedding_model_loaded", model=self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def similarity(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb))


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()
