"""
RAG Document Assistant — Application Configuration
===================================================
Uses pydantic-settings to load configuration from environment variables
and ``.env`` files.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    PROJECT_NAME: str = "RAG Document Assistant"

    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "gemma4:e4b"

    # Vector DB Settings
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # Ingestion Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()
