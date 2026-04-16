"""
FAISS Retrieval Service
=======================
Manages a FAISS ``IndexFlatL2`` index for dense-vector similarity search.
Stores the original chunk texts alongside their index positions so that
search results include the actual source passages.
"""

import logging
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.utils.errors import RetrievalError

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Thin wrapper around FAISS that pairs every indexed vector with
    its source ``(doc_id, chunk_text)`` metadata.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        # Internal store: maps a global vector index → (doc_id, chunk_text)
        self._chunk_store: Dict[int, Tuple[str, str]] = {}

    # ------------------------------------------------------------------ #
    #  Index Management
    # ------------------------------------------------------------------ #

    def add_to_index(
        self,
        embeddings: np.ndarray,
        doc_id: str,
        chunks: List[str],
        existing_index: Optional[faiss.IndexFlatL2] = None,
    ) -> faiss.IndexFlatL2:
        """
        Insert *embeddings* into a FAISS index (creating one if needed)
        and record the matching chunk texts for later retrieval.

        Args:
            embeddings:     2-D float32 array of shape ``(n, dim)``.
            doc_id:         The document ID these chunks belong to.
            chunks:         The original text chunks, same length as *embeddings*.
            existing_index: An existing FAISS index to append to; if ``None``
                            a new ``IndexFlatL2`` is created.

        Returns:
            The (possibly newly created) FAISS index.

        Raises:
            RetrievalError: If the operation fails.
        """
        try:
            dimension: int = embeddings.shape[1]

            if existing_index is None:
                logger.info("Creating new FAISS IndexFlatL2 (dim=%d)", dimension)
                index = faiss.IndexFlatL2(dimension)
            else:
                index = existing_index

            # Record the starting offset so we can map vector positions → chunks
            start_pos: int = index.ntotal

            # Add vectors to the index
            index.add(embeddings)

            # Map each new vector position to its source chunk
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

    # ------------------------------------------------------------------ #
    #  Search
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: np.ndarray,
        faiss_index: faiss.IndexFlatL2,
        top_k: int = 3,
    ) -> List[Dict[str, object]]:
        """
        Find the *top_k* most similar chunks to *query_embedding*.

        Similarity is computed as ``1 / (1 + L2_distance)`` so that
        values are in (0, 1] with 1 = perfect match.

        Args:
            query_embedding: 1-D or 2-D float32 query vector.
            faiss_index:     The FAISS index to search.
            top_k:           Number of results to return.

        Returns:
            A list of dicts, each containing:
                - ``chunk``      : the original text passage
                - ``doc_id``     : source document ID
                - ``score``      : similarity score in (0, 1]
                - ``distance``   : raw L2 distance

        Raises:
            RetrievalError: If the search operation fails.
        """
        try:
            # FAISS expects a 2-D query matrix
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Clamp top_k to what the index actually holds
            effective_k: int = min(top_k, faiss_index.ntotal)

            logger.info(
                "Searching FAISS index (%d vectors) for top-%d matches",
                faiss_index.ntotal,
                effective_k,
            )

            distances, indices = faiss_index.search(query_embedding, effective_k)

            results: List[Dict[str, object]] = []
            for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    # FAISS returns -1 when there are fewer vectors than top_k
                    continue

                doc_id, chunk_text = self._chunk_store.get(
                    int(idx), ("unknown", "")
                )

                # Convert L2 distance → similarity score
                similarity: float = 1.0 / (1.0 + float(dist))

                results.append(
                    {
                        "chunk": chunk_text,
                        "doc_id": doc_id,
                        "score": round(similarity, 4),
                        "distance": round(float(dist), 4),
                    }
                )
                logger.debug(
                    "  Rank %d: doc=%s, score=%.4f, dist=%.4f, preview='%s…'",
                    rank + 1,
                    doc_id,
                    similarity,
                    dist,
                    chunk_text[:80],
                )

            logger.info("Returning %d search results", len(results))
            return results

        except Exception as exc:
            raise RetrievalError(
                f"FAISS search failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Hybrid Search
    # ------------------------------------------------------------------ #

    def get_all_chunks(self) -> List[str]:
        """Return all text chunks correctly ordered by FAISS integer index."""
        return [self._chunk_store[i][1] for i in range(len(self._chunk_store))]

    def hybrid_search(self, query_embedding, query_text, faiss_index, hybrid_retriever, top_k=3):
        """
        Hybrid search: semantic + keyword combined
        
        Args:
            query_embedding: 384-dim vector
            query_text: Raw question text
            faiss_index: FAISS index
            hybrid_retriever: HybridRetriever instance
            top_k: Results to return
        
        Returns:
            Tuple of (chunks, scores)
        """
        try:
            chunks, scores = hybrid_retriever.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                faiss_index=faiss_index,
                top_k=top_k
            )
            return chunks, scores
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            # Fallback to semantic-only
            results = self.search(query_embedding, faiss_index, top_k)
            return [r["chunk"] for r in results], [r["score"] for r in results]

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_index_stats(
        faiss_index: Optional[faiss.IndexFlatL2],
    ) -> Dict[str, object]:
        """
        Return basic statistics about the FAISS index.

        Args:
            faiss_index: The FAISS index (may be ``None``).

        Returns:
            A dict with ``total_vectors`` and ``dimension``.
        """
        if faiss_index is None:
            return {"total_vectors": 0, "dimension": 0}

        return {
            "total_vectors": faiss_index.ntotal,
            "dimension": faiss_index.d,
        }
