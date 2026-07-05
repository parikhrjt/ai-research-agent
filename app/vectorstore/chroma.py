"""ChromaDB vector store implementation (local, no PostgreSQL required)."""

import json
from functools import lru_cache
from pathlib import Path

import chromadb

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.vectorstore.base import RetrievedChunk

logger = get_logger(__name__)

COLLECTION_NAME = "research_chunks"
DOC_COLLECTION = "research_documents"


class ChromaVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._chunks = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._documents = self._client.get_or_create_collection(name=DOC_COLLECTION)

    def initialize(self) -> None:
        settings = get_settings()
        logger.info("chroma_initialized", persist_dir=settings.chroma_persist_dir)

    def upsert_chunks(self, chunks, embeddings, document_meta: dict) -> int:
        try:
            self._documents.upsert(
                ids=[document_meta["document_id"]],
                documents=[json.dumps(document_meta)],
                metadatas=[{
                    "filename": document_meta["filename"],
                    "file_type": document_meta["file_type"],
                    "char_count": document_meta["char_count"],
                }],
            )

            ids = [c.chunk_id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [
                {
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "filename": document_meta["filename"],
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ]
            self._chunks.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("chunks_upserted", document_id=document_meta["document_id"], count=len(chunks))
            return len(chunks)
        except Exception as exc:
            raise VectorStoreError(f"Chroma upsert failed: {exc}") from exc

    def similarity_search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        try:
            results = self._chunks.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            if not results["ids"] or not results["ids"][0]:
                return chunks

            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        document_id=meta.get("document_id", ""),
                        content=results["documents"][0][i],
                        score=score,
                        metadata=meta,
                    )
                )
            return chunks
        except Exception as exc:
            raise VectorStoreError(f"Chroma search failed: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        try:
            existing = self._chunks.get(where={"document_id": document_id})
            if existing["ids"]:
                self._chunks.delete(ids=existing["ids"])
            self._documents.delete(ids=[document_id])
            return 1
        except Exception as exc:
            raise VectorStoreError(f"Chroma delete failed: {exc}") from exc

    def list_documents(self) -> list[dict]:
        try:
            docs = self._documents.get(include=["metadatas", "documents"])
            result = []
            for i, doc_id in enumerate(docs["ids"]):
                meta = docs["metadatas"][i] if docs["metadatas"] else {}
                full_meta = json.loads(docs["documents"][i]) if docs["documents"] else {}
                chunk_count = len(
                    self._chunks.get(where={"document_id": doc_id}, include=[])["ids"]
                )
                result.append({
                    "document_id": doc_id,
                    "filename": meta.get("filename", full_meta.get("filename", "")),
                    "file_type": meta.get("file_type", ""),
                    "char_count": meta.get("char_count", 0),
                    "chunk_count": chunk_count,
                    "ingested_at": full_meta.get("ingested_at", ""),
                })
            return result
        except Exception as exc:
            raise VectorStoreError(f"Chroma list failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            self._chunks.count()
            return True
        except Exception:
            return False


@lru_cache
def get_chroma_store() -> ChromaVectorStore:
    return ChromaVectorStore()
