"""Document upload endpoint."""

from fastapi import APIRouter, File, UploadFile

from app.api.schemas import DocumentInfo, UploadResponse
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.rag import get_rag_pipeline

router = APIRouter()
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise ValidationError("Filename is required")

    data = await file.read()
    if not data:
        raise ValidationError("Uploaded file is empty")

    pipeline = get_rag_pipeline()
    result = pipeline.ingest_bytes(file.filename, data)

    logger.info("upload_complete", document_id=result.document_id, chunks=result.chunk_count)
    return UploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        file_type=result.file_type,
        chunk_count=result.chunk_count,
        char_count=result.char_count,
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    pipeline = get_rag_pipeline()
    docs = pipeline.list_documents()
    return [
        DocumentInfo(
            document_id=str(d.get("document_id", "")),
            filename=d.get("filename", ""),
            file_type=d.get("file_type"),
            char_count=d.get("char_count"),
            chunk_count=d.get("chunk_count"),
            ingested_at=str(d.get("ingested_at", "")) if d.get("ingested_at") else None,
        )
        for d in docs
    ]
