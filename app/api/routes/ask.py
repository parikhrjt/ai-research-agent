"""RAG question-answering endpoint."""

from fastapi import APIRouter

from app.api.schemas import AskRequest, AskResponse, CitationResponse
from app.core.logging import get_logger
from app.rag import get_rag_pipeline

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    pipeline = get_rag_pipeline()
    result = pipeline.ask(request.question)

    citations = [
        CitationResponse(
            index=c.index,
            filename=c.filename,
            excerpt=c.excerpt,
            score=c.score,
            chunk_index=c.chunk_index,
        )
        for c in result.citations
    ]

    logger.info("ask_complete", question=request.question[:80], citations=len(citations))
    return AskResponse(
        answer=result.answer,
        citations=citations,
        retrieved_count=result.retrieved_count,
    )
