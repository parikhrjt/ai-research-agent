"""LangGraph RAG pipeline — retrieve, generate, cite."""

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.chunking.splitter import chunk_text
from app.core.config import get_settings
from app.core.logging import get_logger
from app.embeddings.provider import get_embedding_provider
from app.ingestion.pipeline import IngestedDocument, ingest_bytes, ingest_file
from app.llm import get_llm_provider
from app.rag.citations import Citation, build_context, filter_used_citations
from app.rag.prompts import RAG_PROMPT, SYSTEM_PROMPT
from app.vectorstore import get_vector_store
from app.vectorstore.base import RetrievedChunk

logger = get_logger(__name__)


class RAGState(TypedDict):
    question: str
    query_embedding: list[float]
    retrieved_chunks: list[RetrievedChunk]
    context: str
    citations: list[Citation]
    answer: str


@dataclass
class AskResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_count: int = 0


@dataclass
class IngestResult:
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    char_count: int


class RAGPipeline:
    def __init__(self) -> None:
        self._embeddings = get_embedding_provider()
        self._store = get_vector_store()
        self._llm = get_llm_provider()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RAGState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        return graph.compile()

    def _retrieve(self, state: RAGState) -> dict:
        settings = get_settings()
        embedding = self._embeddings.embed_query(state["question"])
        chunks = self._store.similarity_search(embedding, settings.retrieval_top_k)
        filtered = [c for c in chunks if c.score >= settings.retrieval_score_threshold]
        context, citations = build_context(filtered)
        logger.info("retrieval_complete", question=state["question"][:80], chunks=len(filtered))
        return {
            "query_embedding": embedding,
            "retrieved_chunks": filtered,
            "context": context,
            "citations": citations,
        }

    def _generate(self, state: RAGState) -> dict:
        if not state["retrieved_chunks"]:
            return {
                "answer": "I don't have enough information in the uploaded documents to answer this.",
                "citations": [],
            }

        prompt = RAG_PROMPT.format(context=state["context"], question=state["question"])
        answer = self._llm.generate(prompt, system=SYSTEM_PROMPT)
        used_citations = filter_used_citations(answer, state["citations"])
        logger.info("generation_complete", citation_count=len(used_citations))
        return {"answer": answer, "citations": used_citations}

    def ask(self, question: str) -> AskResult:
        result = self._graph.invoke({"question": question})
        return AskResult(
            answer=result["answer"],
            citations=result.get("citations", []),
            retrieved_count=len(result.get("retrieved_chunks", [])),
        )

    def ingest_document(self, doc: IngestedDocument) -> IngestResult:
        chunks = chunk_text(
            text=doc.content,
            document_id=doc.document_id,
            file_type=doc.file_type,
            base_metadata={"filename": doc.filename, "file_type": doc.file_type},
        )
        embeddings = self._embeddings.embed_texts([c.content for c in chunks])
        doc_meta = {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "content_hash": doc.content_hash,
            "char_count": doc.char_count,
            "ingested_at": doc.ingested_at,
        }
        count = self._store.upsert_chunks(chunks, embeddings, doc_meta)
        return IngestResult(
            document_id=doc.document_id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunk_count=count,
            char_count=doc.char_count,
        )

    def ingest_bytes(self, filename: str, data: bytes) -> IngestResult:
        doc = ingest_bytes(filename, data)
        return self.ingest_document(doc)

    def ingest_path(self, path: str, filename: str | None = None) -> IngestResult:
        from pathlib import Path
        doc = ingest_file(Path(path), original_filename=filename)
        return self.ingest_document(doc)

    def list_documents(self) -> list[dict]:
        return self._store.list_documents()


_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
