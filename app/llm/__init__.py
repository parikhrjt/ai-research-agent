"""LLM provider factory."""

from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider, get_ollama_provider
from app.llm.openai import OpenAIProvider, get_openai_provider


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "openai":
        return get_openai_provider()
    return get_ollama_provider()
