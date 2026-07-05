"""PostgreSQL + pgvector vector store implementation."""

import json
import uuid
from functools import lru_cache

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.vectorstore.base import RetrievedChunk

logger = get_logger(__name__)


class PgVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._dsn = settings.database_url
        self._dimension = settings.embedding_dimension

    def _connect(self):
        conn = psycopg2.connect(self._dsn)
        register_vector(conn)
        return conn

    def initialize(self) -> None:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id UUID PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        char_count INTEGER NOT NULL,
                        ingested_at TIMESTAMPTZ NOT NULL,
                        metadata JSONB DEFAULT '{{}}'
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id UUID NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding vector({self._dimension}),
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
                    "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                )
                conn.commit()
            logger.info("pgvector_initialized")
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize pgvector: {exc}") from exc

    def upsert_chunks(self, chunks, embeddings, document_meta: dict) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding count mismatch")

        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (document_id, filename, file_type, content_hash,
                                           char_count, ingested_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        content_hash = EXCLUDED.content_hash,
                        char_count = EXCLUDED.char_count
                    """,
                    (
                        document_meta["document_id"],
                        document_meta["filename"],
                        document_meta["file_type"],
                        document_meta["content_hash"],
                        document_meta["char_count"],
                        document_meta["ingested_at"],
                        json.dumps(document_meta.get("extra", {})),
                    ),
                )

                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO chunks (chunk_id, document_id, content, chunk_index,
                                            embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.content,
                            chunk.chunk_index,
                            embedding,
                            json.dumps(chunk.metadata),
                        ),
                    )
                conn.commit()
            logger.info("chunks_upserted", document_id=document_meta["document_id"], count=len(chunks))
            return len(chunks)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert chunks: {exc}") from exc

    def similarity_search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        try:
            with self._connect() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.document_id, c.content, c.metadata,
                           1 - (c.embedding <=> %s::vector) AS score,
                           d.filename
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )
                rows = cur.fetchall()

            results = []
            for row in rows:
                meta = row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}")
                meta["filename"] = row["filename"]
                results.append(
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        document_id=str(row["document_id"]),
                        content=row["content"],
                        score=float(row["score"]),
                        metadata=meta,
                    )
                )
            return results
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
                deleted = cur.rowcount
                conn.commit()
            return deleted
        except Exception as exc:
            raise VectorStoreError(f"Delete failed: {exc}") from exc

    def list_documents(self) -> list[dict]:
        try:
            with self._connect() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(
                    """
                    SELECT d.document_id, d.filename, d.file_type, d.char_count,
                           d.ingested_at, COUNT(c.chunk_id) AS chunk_count
                    FROM documents d
                    LEFT JOIN chunks c ON c.document_id = d.document_id
                    GROUP BY d.document_id
                    ORDER BY d.ingested_at DESC
                    """
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            raise VectorStoreError(f"List documents failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False


@lru_cache
def get_pgvector_store() -> PgVectorStore:
    return PgVectorStore()
