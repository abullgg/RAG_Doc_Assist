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
    ANTHROPIC_API_KEY: str = ""  # Optional at startup; required for /ask
    MODEL_NAME: str = "claude-3-5-sonnet-20241022"

    # Vector DB Settings
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # Ingestion Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
