"""Citation extraction and formatting."""

import re
from dataclasses import dataclass

from app.vectorstore.base import RetrievedChunk


@dataclass
class Citation:
    index: int
    chunk_id: str
    document_id: str
    filename: str
    excerpt: str
    score: float
    chunk_index: int | None = None


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Format retrieved chunks as numbered context blocks with citation metadata."""
    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "unknown")
        chunk_idx = chunk.metadata.get("chunk_index")
        header = f"[{i}] Source: {filename}"
        if chunk_idx is not None:
            header += f" (chunk {chunk_idx})"
        context_parts.append(f"{header}\n{chunk.content}")
        citations.append(
            Citation(
                index=i,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=filename,
                excerpt=_truncate(chunk.content, 300),
                score=round(chunk.score, 4),
                chunk_index=int(chunk_idx) if chunk_idx is not None else None,
            )
        )

    return "\n\n---\n\n".join(context_parts), citations


def extract_cited_indices(answer: str) -> list[int]:
    """Parse [1], [2] style citations from the LLM answer."""
    matches = re.findall(r"\[(\d+)\]", answer)
    return sorted({int(m) for m in matches})


def filter_used_citations(answer: str, citations: list[Citation]) -> list[Citation]:
    """Return only citations referenced in the answer."""
    used = set(extract_cited_indices(answer))
    if not used:
        return citations
    return [c for c in citations if c.index in used]


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rsplit(" ", 1)[0] + "..."
