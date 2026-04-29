"""
Application settings loaded from environment variables or .env file.
All tuneable values live here — override via .env before launching the server.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    PROJECT_NAME: str = "RAG Document Assistant"

    # ── LLM ─────────────────────────────────────────────────────────────── #
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "gemma3:4b"

    # ── Embeddings ───────────────────────────────────────────────────────── #
    # Available BGE variants (EMBEDDING_DIMENSION must match the chosen model):
    #   BAAI/bge-large-en-v1.5 → 1024-dim  (best quality, slower)
    #   BAAI/bge-base-en-v1.5  → 768-dim   (recommended)
    #   BAAI/bge-small-en-v1.5 → 384-dim   (fastest)
    # After switching models, delete ./data/ to clear the stale index.
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768  # Must match the chosen model above

    # ── Vector Store ─────────────────────────────────────────────────────── #
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # ── Ingestion ────────────────────────────────────────────────────────── #
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200

    # ── Cross-Encoder Reranker ───────────────────────────────────────────── #
    # Reranking runs AFTER hybrid retrieval. Stage 1 retrieves RERANKER_TOP_N
    # candidates quickly (bi-encoder), then Stage 2 scores every (query, chunk)
    # pair with the cross-encoder and keeps only the best top_k for the LLM.
    #
    # Set RERANKER_ENABLED=false to skip reranking (faster, lower quality).
    #
    # Available models:
    #   BAAI/bge-reranker-base   — ~280 MB  (recommended)
    #   BAAI/bge-reranker-large  — ~1.3 GB  (highest accuracy, slower)
    #   cross-encoder/ms-marco-MiniLM-L-6-v2 — ~80 MB (fast alternative)
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    # Number of candidates fetched from hybrid search before reranking.
    # Should be ≥ top_k sent with each /ask request (typically 3-5).
    RERANKER_TOP_N: int = 10

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()
