"""
Persistent Storage
------------------
Saves and loads the RAG application state to disk so indexed documents
survive server restarts.

Files written to ./data/:
  faiss_index.bin — the serialised FAISS index
  metadata.json   — {doc_id: metadata} for all indexed documents
  chunks.pkl      — FAISS position → (doc_id, chunk_text) mapping

Dimension guard:
  On load, the stored index dimension is checked against the configured
  EMBEDDING_DIMENSION. If they differ (e.g. after switching embedding models),
  the stale files are deleted and the server starts with an empty index.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import faiss

from src.core.config import settings

logger = logging.getLogger(__name__)


class PersistentStorage:
    """
    Saves and loads RAG state (FAISS index, document metadata, chunk store).

    Never raises on load failures — always returns empty state so the server
    starts cleanly even when files are missing or corrupted.

    Args:
        data_dir: Directory to store persistence files. Defaults to "data".
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.faiss_path = self.data_dir / "faiss_index.bin"
        self.meta_path = self.data_dir / "metadata.json"
        self.chunks_path = self.data_dir / "chunks.pkl"

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.error("Failed to create data directory: %s", exc)

    def save_state(
        self,
        faiss_index: Optional[faiss.IndexFlatIP],
        indexed_documents: Dict[str, dict],
        chunk_store: Dict[int, Tuple[str, str]],
    ) -> bool:
        """
        Write FAISS index, document metadata, and chunk store to disk.

        Args:
            faiss_index:        The active FAISS index (skipped if None).
            indexed_documents:  {doc_id: metadata} registry.
            chunk_store:        FAISS position → (doc_id, chunk_text) mapping.

        Returns:
            True if all files were saved successfully.
        """
        try:
            if faiss_index is not None:
                faiss.write_index(faiss_index, str(self.faiss_path))

            with open(self.meta_path, "w", encoding="utf-8") as fh:
                json.dump(indexed_documents, fh, indent=2)

            with open(self.chunks_path, "wb") as fh:
                pickle.dump(chunk_store, fh)

            logger.info("RAG state saved to '%s'", self.data_dir)
            return True

        except Exception as exc:
            logger.error("Failed to save state: %s", exc)
            return False

    def load_state(
        self,
    ) -> Tuple[Optional[faiss.IndexFlatIP], Dict[str, dict], Dict[int, Tuple[str, str]]]:
        """
        Load persisted RAG state from disk.

        If the stored FAISS index dimension doesn't match EMBEDDING_DIMENSION
        (which happens when you change the embedding model), all stale files are
        deleted and empty state is returned.

        Returns:
            Tuple of (faiss_index, indexed_documents, chunk_store).
            Any missing or unreadable file returns its empty default.
        """
        faiss_index = None
        indexed_documents: Dict[str, dict] = {}
        chunk_store: Dict[int, Tuple[str, str]] = {}

        try:
            if self.faiss_path.exists():
                faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(
                    "Loaded FAISS index from '%s' (dim=%d, vectors=%d)",
                    self.faiss_path,
                    faiss_index.d,
                    faiss_index.ntotal,
                )

                if faiss_index.d != settings.EMBEDDING_DIMENSION:
                    logger.warning(
                        "Stored index dimension (%d) doesn't match configured dimension (%d). "
                        "Clearing stale index — all documents must be re-uploaded.",
                        faiss_index.d,
                        settings.EMBEDDING_DIMENSION,
                    )
                    self.clear_state()
                    return None, {}, {}

            if self.meta_path.exists():
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    indexed_documents = json.load(fh)
                logger.info(
                    "Loaded metadata from '%s' (%d documents)",
                    self.meta_path,
                    len(indexed_documents),
                )

            if self.chunks_path.exists():
                with open(self.chunks_path, "rb") as fh:
                    chunk_store = pickle.load(fh)
                logger.info(
                    "Loaded chunk store from '%s' (%d entries)",
                    self.chunks_path,
                    len(chunk_store),
                )

        except Exception as exc:
            logger.error(
                "Failed to load state from disk — starting with empty state: %s", exc
            )
            return None, {}, {}

        return faiss_index, indexed_documents, chunk_store

    def clear_state(self) -> bool:
        """
        Delete all persisted state files.

        Use this when switching embedding models to force a clean re-index.

        Returns:
            True if the operation succeeded.
        """
        try:
            for path in (self.faiss_path, self.meta_path, self.chunks_path):
                if path.exists():
                    path.unlink()
            logger.info("Cleared all persisted state from '%s'", self.data_dir)
            return True
        except Exception as exc:
            logger.error("Failed to clear persisted state: %s", exc)
            return False
