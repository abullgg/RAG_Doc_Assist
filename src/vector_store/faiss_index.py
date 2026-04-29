"""
FAISS Retrieval Service
-----------------------
Manages a FAISS IndexFlatIP (inner product) index for dense-vector similarity search.

Embeddings are L2-normalised at encode time, so inner product equals cosine
similarity. Scores fall in the range (-1, 1] where 1.0 is a perfect match.

A chunk store (dict of FAISS position → (doc_id, chunk_text)) maps search
results back to the original passages and their source documents.
"""

import logging
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.core.config import settings
from src.utils.errors import RetrievalError

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Wraps FAISS and pairs every indexed vector with its source (doc_id, chunk_text).

    Args:
        dimension: Embedding vector size. Must match the active embedding model.
    """

    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION) -> None:
        self.dimension: int = dimension
        # Maps FAISS vector position → (doc_id, chunk_text)
        self._chunk_store: Dict[int, Tuple[str, str]] = {}

    def add_to_index(
        self,
        embeddings: np.ndarray,
        doc_id: str,
        chunks: List[str],
        existing_index: Optional[faiss.IndexFlatIP] = None,
    ) -> faiss.IndexFlatIP:
        """
        Add embeddings to the FAISS index and record their source chunks.

        Creates a new IndexFlatIP if no existing index is provided.

        Args:
            embeddings:     2-D float32 array of shape (n, dimension).
            doc_id:         ID of the document these chunks belong to.
            chunks:         Original text chunks — must have the same length as embeddings.
            existing_index: Existing FAISS index to append to. Creates a new one if None.

        Returns:
            The updated (or newly created) FAISS index.

        Raises:
            RetrievalError: If the add operation fails.
        """
        try:
            if existing_index is None:
                logger.info("Creating new FAISS IndexFlatIP (dim=%d)", self.dimension)
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = existing_index

            start_pos: int = index.ntotal
            index.add(embeddings)

            for i, chunk in enumerate(chunks):
                self._chunk_store[start_pos + i] = (doc_id, chunk)

            logger.info(
                "Added %d vectors for doc '%s' — index now contains %d vectors",
                len(chunks),
                doc_id,
                index.ntotal,
            )
            return index

        except Exception as exc:
            raise RetrievalError(
                f"Failed to add {len(chunks)} vectors to index: {exc}"
            ) from exc

    def search(
        self,
        query_embedding: np.ndarray,
        faiss_index: faiss.IndexFlatIP,
        top_k: int = 3,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Find the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: 1-D or 2-D float32 query vector from embed_query().
            faiss_index:     The FAISS index to search.
            top_k:           Number of results to return.
            filter_doc_id:   If set, only return chunks from this document.

        Returns:
            List of dicts with keys: chunk, doc_id, score.

        Raises:
            RetrievalError: If the FAISS search fails.
        """
        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            effective_k = faiss_index.ntotal if filter_doc_id else min(top_k, faiss_index.ntotal)
            if effective_k == 0:
                return []

            logger.info(
                "Searching FAISS index (%d vectors) for top-%d (filter_doc_id=%s)",
                faiss_index.ntotal,
                effective_k,
                filter_doc_id,
            )

            scores_raw, indices = faiss_index.search(query_embedding, effective_k)

            results: List[Dict[str, object]] = []
            for rank, (score, idx) in enumerate(zip(scores_raw[0], indices[0])):
                if idx == -1:
                    continue  # FAISS sentinel for "no result"

                doc_id, chunk_text = self._chunk_store.get(int(idx), ("unknown", ""))

                if filter_doc_id and doc_id != filter_doc_id:
                    continue

                results.append({
                    "chunk": chunk_text,
                    "doc_id": doc_id,
                    "score": round(float(score), 4),
                })
                logger.debug(
                    "  Rank %d: doc=%s, score=%.4f, preview='%s…'",
                    rank + 1, doc_id, score, chunk_text[:80],
                )

                if len(results) >= top_k:
                    break

            logger.info("Returning %d search results", len(results))
            return results

        except Exception as exc:
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

    def get_all_chunks(self) -> List[Tuple[str, str]]:
        """
        Return all (doc_id, text) pairs ordered by FAISS index position.
        Used by HybridRetriever to build the BM25 corpus.
        """
        return [self._chunk_store[i] for i in range(len(self._chunk_store))]

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        faiss_index: faiss.IndexFlatIP,
        hybrid_retriever,
        top_k: int = 3,
        filter_doc_id: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Run hybrid search (FAISS + BM25) via HybridRetriever.
        Falls back to pure semantic search if BM25 is unavailable.

        Returns:
            Tuple of (chunks, scores).
        """
        try:
            chunks, scores = hybrid_retriever.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                faiss_index=faiss_index,
                top_k=top_k,
                filter_doc_id=filter_doc_id,
            )
            return chunks, scores
        except Exception as exc:
            logger.error("Hybrid search failed: %s — falling back to semantic", exc)
            results = self.search(query_embedding, faiss_index, top_k, filter_doc_id=filter_doc_id)
            return [r["chunk"] for r in results], [r["score"] for r in results]

    @staticmethod
    def get_index_stats(faiss_index: Optional[faiss.IndexFlatIP]) -> Dict[str, object]:
        """Return total vector count and dimension of the FAISS index."""
        if faiss_index is None:
            return {"total_vectors": 0, "dimension": 0}
        return {"total_vectors": faiss_index.ntotal, "dimension": faiss_index.d}
