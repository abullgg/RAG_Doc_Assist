import faiss
import numpy as np
import os
from src.core.config import settings

class FAISSIndex:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        # In MVP, metadata might be stored as a simple pickle or json
        pass

    def load(self, path: str):
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        pass

    def add(self, embeddings: np.ndarray, metadata: list):
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 3):
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return distances, indices
