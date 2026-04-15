"""
Persistent Storage module for RAG state.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import faiss

logger = logging.getLogger(__name__)

class PersistentStorage:
    """
    Handles saving and loading the RAG application state safely to disk.
    Does not crash on failures, only logs issues and defaults to empty state.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.faiss_path = self.data_dir / "faiss_index.bin"
        self.meta_path = self.data_dir / "metadata.json"
        self.chunks_path = self.data_dir / "chunks.pkl"
        
        # Ensure data directory exists
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
            
    def save_state(self, faiss_index: Optional[faiss.IndexFlatL2], indexed_documents: Dict[str, dict], chunk_store: Dict[int, Tuple[str, str]]) -> bool:
        """
        Save FAISS index, metadata, and chunks to disk.
        Returns True if successful, False otherwise.
        """
        try:
            # 1. Save FAISS index
            if faiss_index is not None:
                faiss.write_index(faiss_index, str(self.faiss_path))
            
            # 2. Save document metadata
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(indexed_documents, f, indent=2)
                
            # 3. Save chunks
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(chunk_store, f)
                
            logger.info("Successfully saved RAG state to disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to save state to disk: {e}")
            return False

    def load_state(self) -> Tuple[Optional[faiss.IndexFlatL2], Dict[str, dict], Dict[int, Tuple[str, str]]]:
        """
        Load FAISS index, metadata, and chunks from disk.
        Returns empty safe defaults if files do not exist or fail to load.
        """
        faiss_index = None
        indexed_documents = {}
        chunk_store = {}
        
        try:
            if self.faiss_path.exists():
                faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(f"Loaded FAISS index from {self.faiss_path}")
            
            if self.meta_path.exists():
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    indexed_documents = json.load(f)
                logger.info(f"Loaded metadata from {self.meta_path}")
                
            if self.chunks_path.exists():
                with open(self.chunks_path, 'rb') as f:
                    chunk_store = pickle.load(f)
                logger.info(f"Loaded chunk store from {self.chunks_path}")
                
        except Exception as e:
            logger.error(f"Failed to load state from disk (returning empty state): {e}")
            return None, {}, {}
            
        return faiss_index, indexed_documents, chunk_store

    def clear_state(self) -> bool:
        """Clear the persisted state from disk manually. Used for testing/reset."""
        try:
            if self.faiss_path.exists(): self.faiss_path.unlink()
            if self.meta_path.exists(): self.meta_path.unlink()
            if self.chunks_path.exists(): self.chunks_path.unlink()
            logger.info("Cleared persisted state.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False
