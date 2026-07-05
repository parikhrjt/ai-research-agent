"""OpenAI LLM provider (optional, configured via .env)."""

from functools import lru_cache

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider:
    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model
        self._base_url = "https://api.openai.com/v1"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self._api_key:
            raise LLMError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "messages": messages, "temperature": 0.2},
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
        except httpx.HTTPError as exc:
            logger.error("openai_request_failed", error=str(exc))
            raise LLMError(f"OpenAI request failed: {exc}") from exc

    def health_check(self) -> bool:
        return bool(self._api_key)


@lru_cache
def get_openai_provider() -> OpenAIProvider:
    return OpenAIProvider()
