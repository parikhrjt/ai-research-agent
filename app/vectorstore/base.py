"""Vector store abstractions and implementations."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def initialize(self) -> None: ...

    def upsert_chunks(
        self,
        chunks: list,
        embeddings: list[list[float]],
        document_meta: dict,
    ) -> int: ...

    def similarity_search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...

    def delete_document(self, document_id: str) -> int: ...

    def list_documents(self) -> list[dict]: ...

    def health_check(self) -> bool: ...
