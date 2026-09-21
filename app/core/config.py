"""
Centralized application configuration.

Every environment-dependent value lives here and ONLY here.
No other module should read os.environ directly.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "DeepResearch-RAG"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # --- Database ---
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/deepresearch"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- LLM ---
    # Originally spec'd as Mistral; switched to Groq during Phase 6 due to
    # Mistral's free-tier rate limits requiring billing to be enabled even
    # within the free monthly allowance. The LLM layer (app/llm/models.py)
    # is provider-abstracted, so this was a config-only change.
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # --- Web search (Phase 7) ---
    TAVILY_API_KEY: str | None = None

    # --- Embeddings / Reranking ---
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # --- Retrieval tuning ---
    TOP_K: int = 20
    RERANK_TOP_K: int = 5

    # --- Ingestion ---
    MAX_CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_MB: int = 25

    # --- Corrective RAG (Phase 8) ---
    MAX_RETRIES: int = 2


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance - avoids re-parsing .env on every call."""
    return Settings()