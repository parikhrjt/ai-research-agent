"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: str
    llm_provider: str
    components: dict[str, bool]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    char_count: int
    message: str = "Document ingested successfully"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class CitationResponse(BaseModel):
    index: int
    filename: str
    excerpt: str
    score: float
    chunk_index: int | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_count: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str | None = None
    char_count: int | None = None
    chunk_count: int | None = None
    ingested_at: str | None = None


class ErrorResponse(BaseModel):
    error: str
    details: dict | None = None
