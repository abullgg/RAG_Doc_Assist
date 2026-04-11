"""
Embedding Service
=================
Wraps the ``sentence-transformers`` library to produce dense vector
embeddings using the **all-MiniLM-L6-v2** model (384-dimensional).
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.errors import EmbeddingError

logger = logging.getLogger(__name__)

# Model identifier — lightweight, fast, and broadly effective
_MODEL_NAME: str = "all-MiniLM-L6-v2"
_EMBEDDING_DIMENSION: int = 384


class EmbeddingService:
    """
    Generates sentence-level embeddings using a pre-trained transformer.

    The model is loaded **once** at instantiation and reused across calls.
    """

    def __init__(self) -> None:
        logger.info("Loading sentence-transformer model '%s' …", _MODEL_NAME)
        try:
            self.model: SentenceTransformer = SentenceTransformer(_MODEL_NAME)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{_MODEL_NAME}': {exc}"
            ) from exc
        logger.info(
            "Model loaded successfully (dimension=%d)", _EMBEDDING_DIMENSION
        )

    # ------------------------------------------------------------------ #
    #  Single-text embedding
    # ------------------------------------------------------------------ #

    def embed_text(self, text: str) -> np.ndarray:
        """
        Compute the embedding for a single piece of text.

        Args:
            text: The input string to embed.

        Returns:
            A 1-D ``numpy.ndarray`` of shape ``(384,)`` with dtype ``float32``.

        Raises:
            EmbeddingError: If encoding fails.
        """
        logger.debug("Embedding single text (%d chars)", len(text))
        try:
            embedding: np.ndarray = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embedding.astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed text: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Batch embedding
    # ------------------------------------------------------------------ #

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of text chunks.

        A ``tqdm`` progress bar is displayed during encoding.

        Args:
            chunks: List of text strings to embed.

        Returns:
            A 2-D ``numpy.ndarray`` of shape ``(len(chunks), 384)``
            with dtype ``float32``.

        Raises:
            EmbeddingError: If encoding fails.
        """
        logger.info("Embedding %d chunks …", len(chunks))
        try:
            embeddings: np.ndarray = self.model.encode(
                chunks,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32,
            )
            embeddings = embeddings.astype(np.float32)

            logger.info(
                "Embedding complete — shape %s, dtype %s",
                embeddings.shape,
                embeddings.dtype,
            )
            return embeddings
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(chunks)} chunks: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Metadata
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_embedding_dimension() -> int:
        """Return the dimensionality of the embedding vectors (384)."""
        return _EMBEDDING_DIMENSION
