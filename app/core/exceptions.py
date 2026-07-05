"""Application-specific exception hierarchy."""

from typing import Any


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(AppError):
    """Raised when document parsing or ingestion fails."""


class VectorStoreError(AppError):
    """Raised when vector store operations fail."""


class LLMError(AppError):
    """Raised when LLM inference fails."""


class ValidationError(AppError):
    """Raised when input validation fails."""
