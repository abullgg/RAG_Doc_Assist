"""
Application settings loaded from environment variables or .env file.
All tuneable values live here — override via .env before launching the server.
"""

from typing import Dict, Literal
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Chunk-size presets (characters)
# ---------------------------------------------------------------------------

CHUNK_SIZE_PRESETS: Dict[str, int] = {
    "technical_spec": 1200,  # Dense specs / manuals → smaller = more precise
    "narrative": 1800,        # Flowing prose → larger keeps sentences together
    "data_heavy": 800,        # Table/list-heavy → smallest, tables are atomic
    "default": 1500,          # Balanced default (was 2000)
}

DocType = Literal["technical_spec", "narrative", "data_heavy", "default"]


# ---------------------------------------------------------------------------
# Retrieval configuration model
# ---------------------------------------------------------------------------

class RetrievalConfig(BaseModel):
    """
    Controls the retrieval funnel for each /ask request.

    Attributes:
        max_tokens_for_context: Soft character ceiling for combined context fed
            to the LLM. Gemma 3 4B has an 8K token window; reserving ~3000
            chars for context leaves room for the system prompt and generation.
            (1 token ≈ 4 chars for English, so 3000 chars ≈ 750 tokens.)
        top_k_candidates: Final number of chunks passed to the LLM after
            reranking.
        reranker_top_n: Wider candidate pool fetched from hybrid search before
            the cross-encoder sees them.
    """

    max_tokens_for_context: int = 3000
    top_k_candidates: int = 5
    reranker_top_n: int = 10


# ---------------------------------------------------------------------------
# Main settings class
# ---------------------------------------------------------------------------

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
    # CHUNK_SIZE / CHUNK_OVERLAP are the fallback values when no doc_type
    # preset is specified. Prefer CHUNK_SIZE_PRESETS for new uploads.
    CHUNK_SIZE: int = 1500      # Lowered from 2000 for better Gemma 3 4B fit
    CHUNK_OVERLAP: int = 200

    # Header detection
    HEADER_DETECTION_ENABLED: bool = True
    HEADER_CONFIDENCE_THRESHOLD: float = 0.75

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

    # ── Context budget ───────────────────────────────────────────────────── #
    MAX_CONTEXT_CHARS: int = 3000  # Soft ceiling; trimmed from lowest-confidence end

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()
