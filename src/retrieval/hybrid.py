"""
Hybrid retriever combinning FAISS Semantic Search with BM25 Keyword Search.
"""

import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Blends FAISS semantic scores with BM25 keyword match scores.
    """
    
    def __init__(self):
        self._bm25: BM25Okapi = None
        self._chunks: List[str] = []
        
    def index_chunks(self, chunk_list: List[str]) -> None:
        """
        Build the BM25 index from a raw list of text chunks.
        """
        if not chunk_list:
            logger.warning("Empty chunk list provided to HybridRetriever; BM25 index not built.")
            self._bm25 = None
            self._chunks = []
            return
            
        self._chunks = chunk_list
        tokenized_corpus = [doc.lower().split() for doc in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 keyword index over {len(self._chunks)} chunks.")
        
    def search_semantic(self, query_embedding, faiss_index, top_k=5) -> Tuple[List[str], List[float]]:
        """ Call existing FAISS lookup but returning raw texts and scores. """
        import numpy as np
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        effective_k = min(top_k, faiss_index.ntotal)
        if effective_k == 0:
            return [], []
            
        distances, indices = faiss_index.search(query_embedding, effective_k)
        
        chunks = []
        scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self._chunks):
                continue
            chunks.append(self._chunks[idx])
            # Normalize distance to similarity score
            scores.append(1.0 / (1.0 + float(dist)))
            
        return chunks, scores
        
    def search_keyword(self, query_text: str, top_k=5) -> Tuple[List[str], List[float]]:
        """ Score via BM25 matching. """
        if not self._bm25 or not query_text:
            return [], []
            
        tokenized_query = query_text.lower().split()
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        chunks = []
        scores = []
        for idx in top_k_indices:
            if doc_scores[idx] > 0:
                chunks.append(self._chunks[idx])
                scores.append(float(doc_scores[idx]))
                
        return chunks, scores
        
    def hybrid_search(self, query_embedding, query_text: str, faiss_index, top_k=3) -> Tuple[List[str], List[float]]:
        """ Merge BM25 and Semantic searches with a weighted scoring mechanism. """
        if not self._bm25 or not self._chunks:
            logger.warning("BM25 index missing. Falling back entirely to semantic search.")
            return self.search_semantic(query_embedding, faiss_index, top_k=top_k)
            
        sem_chunks, sem_scores = self.search_semantic(query_embedding, faiss_index, top_k=5)
        kw_chunks, kw_scores = self.search_keyword(query_text, top_k=5)
        
        # Normalize semantic scores 0-1
        max_sem = max(sem_scores) if sem_scores else 1.0
        norm_sem = {c: s/max_sem for c, s in zip(sem_chunks, sem_scores)}
        
        # Normalize keyword scores 0-1
        max_kw = max(kw_scores) if kw_scores else 1.0
        norm_kw = {c: s/max_kw for c, s in zip(kw_chunks, kw_scores)}
        
        all_chunks = set(sem_chunks) | set(kw_chunks)
        
        final_scores = {}
        for chunk in all_chunks:
            sem = norm_sem.get(chunk, 0.0)
            kw = norm_kw.get(chunk, 0.0)
            # Final scoring metric
            final_scores[chunk] = (sem * 0.7) + (kw * 0.3)
            
        # Sort descending and snap top K
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        res_chunks = [item[0] for item in sorted_items]
        res_scores = [item[1] for item in sorted_items]
        
        return res_chunks, res_scores
