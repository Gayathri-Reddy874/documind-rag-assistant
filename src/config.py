"""
Centralized application configuration.

All tunables are sourced from environment variables (via a `.env` file or the
runtime environment) so the same codebase can move between local development,
staging, and production without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider (Groq) ---
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq-hosted model used for generation",
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # --- Embeddings ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model for the vector store",
    )
    embedding_device: Literal["cpu", "cuda"] = "cpu"

    # --- Chunking ---
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Retrieval ---
    retriever_top_k: int = Field(default=4, gt=0)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Storage ---
    vector_store_dir: str = Field(default="data/vector_store")
    max_upload_mb: int = Field(default=25, gt=0)
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".docx", ".md", ".csv")

    # --- App / observability ---
    app_name: str = "Enterprise Document Intelligence Assistant"
    log_level: str = "INFO"
    environment: Literal["development", "staging", "production"] = "development"


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()

