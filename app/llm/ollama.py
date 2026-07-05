"""Ollama local LLM provider."""

from functools import lru_cache

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self._base_url}/api/chat",
                    json={"model": self._model, "messages": messages, "stream": False},
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                if not content:
                    raise LLMError("Ollama returned empty response")
                return content.strip()
        except httpx.HTTPError as exc:
            logger.error("ollama_request_failed", error=str(exc))
            raise LLMError(f"Ollama request failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


@lru_cache
def get_ollama_provider() -> OllamaProvider:
    return OllamaProvider()
