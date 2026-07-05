"""LLM provider abstractions."""

from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...

    def health_check(self) -> bool: ...
