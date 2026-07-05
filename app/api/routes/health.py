"""Health check endpoint."""

from fastapi import APIRouter

from app import __version__
from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.llm import get_llm_provider
from app.vectorstore import get_vector_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    store = get_vector_store()
    llm = get_llm_provider()

    components = {
        "vector_store": store.health_check(),
        "llm": llm.health_check(),
    }
    all_healthy = all(components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=__version__,
        vector_store=settings.vector_store,
        llm_provider=settings.llm_provider,
        components=components,
    )
