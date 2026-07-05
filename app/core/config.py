"""Centralized application configuration via environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "AI Research Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # Streamlit
    streamlit_port: int = 8501

    # Vector store
    vector_store: Literal["pgvector", "chroma"] = "pgvector"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "research"
    postgres_password: str = "research"
    postgres_db: str = "research_agent"
    database_url: str = "postgresql://research:research@localhost:5432/research_agent"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # RAG
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.3

    # LLM
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Upload
    max_upload_size_mb: int = 25
    allowed_extensions: str = "pdf,txt,md,markdown,csv"

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        return v.upper()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_extension_set(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
