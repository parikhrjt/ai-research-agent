"""Intelligent document chunking strategies."""

from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict


def _base_splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(
    text: str,
    document_id: str,
    file_type: str,
    base_metadata: dict | None = None,
) -> list[DocumentChunk]:
    """Chunk document text using format-aware strategies."""
    meta = base_metadata or {}
    chunks: list[DocumentChunk] = []

    if file_type in ("md", "markdown"):
        chunks = _chunk_markdown(text, document_id, meta)
    elif file_type == "csv":
        chunks = _chunk_csv(text, document_id, meta)
    else:
        chunks = _chunk_recursive(text, document_id, meta)

    logger.info(
        "chunking_complete",
        document_id=document_id,
        file_type=file_type,
        chunk_count=len(chunks),
    )
    return chunks


def _chunk_recursive(text: str, document_id: str, meta: dict) -> list[DocumentChunk]:
    splitter = _base_splitter()
    texts = splitter.split_text(text)
    return _to_chunks(texts, document_id, meta, strategy="recursive")


def _chunk_markdown(text: str, document_id: str, meta: dict) -> list[DocumentChunk]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS)
    header_docs = header_splitter.split_text(text)

    splitter = _base_splitter()
    all_texts: list[tuple[str, dict]] = []

    for doc in header_docs:
        sub_chunks = splitter.split_text(doc.page_content)
        header_meta = {k: v for k, v in doc.metadata.items()}
        for sub in sub_chunks:
            all_texts.append((sub, header_meta))

    if not all_texts:
        return _chunk_recursive(text, document_id, meta)

    chunks = []
    for i, (content, header_meta) in enumerate(all_texts):
        chunk_meta = {**meta, **header_meta, "strategy": "markdown_headers"}
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}_{i}",
                document_id=document_id,
                content=content,
                chunk_index=i,
                metadata=chunk_meta,
            )
        )
    return chunks


def _chunk_csv(text: str, document_id: str, meta: dict) -> list[DocumentChunk]:
    """Keep CSV rows grouped in batches for context preservation."""
    settings = get_settings()
    rows = text.split("\n")
    batch_size = max(5, settings.chunk_size // 200)
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    chunks = []
    for i, batch in enumerate(batches):
        content = "\n".join(batch)
        if content.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_{i}",
                    document_id=document_id,
                    content=content,
                    chunk_index=i,
                    metadata={**meta, "strategy": "csv_row_batch", "row_batch": i},
                )
            )
    return chunks


def _to_chunks(texts: list[str], document_id: str, meta: dict, strategy: str) -> list[DocumentChunk]:
    chunks = []
    for i, content in enumerate(texts):
        if content.strip():
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_{i}",
                    document_id=document_id,
                    content=content,
                    chunk_index=i,
                    metadata={**meta, "strategy": strategy},
                )
            )
    return chunks
