"""
Uploader Task
=============
Asynchronous task logic for background processing large document uploads.
Uses a batched embedding approach to provide granular progress tracking.
"""

import logging
import asyncio
from typing import List
import numpy as np

import src.main as state
from src.ingestion.processor import DocumentProcessor
from src.embeddings.embedding import EmbeddingService

logger = logging.getLogger(__name__)

async def process_upload_task(
    job_id: str, 
    raw_bytes: bytes, 
    filename: str, 
    content_type: str,
    processor: DocumentProcessor,
    embedding_service: EmbeddingService
) -> None:
    """
    Executes the ingestion → embedding → FAISS indexing steps.
    Regularly updates the global job_tracker with current percentage.
    """
    tracker = state.job_tracker
    tracker.update_status(job_id, status="processing", progress=10)
    
    try:
        # 1. Extract Text
        text = await processor.extract_text(raw_bytes, filename, content_type)
        if not text.strip():
            raise ValueError(f"No readable text found in '{filename}'.")
            
        tracker.update_status(job_id, status="processing", progress=20)
        
        # 2. Chunk text
        chunks = processor.chunk_text(text)
        if not chunks:
            raise ValueError(f"Text from '{filename}' produced no usable chunks.")
            
        tracker.update_status(job_id, status="processing", progress=25)
        
        # 3. Embed chunks in batches for progress tracking
        all_embeddings = []
        # Update progress between 25% and 85%
        # Process in batches of 10% to emit regular updates cleanly
        batch_size = max(1, len(chunks) // 10)
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Since model.encode is heavily blocking CPU, yielding event loop allows other API requests to process
            emb = await asyncio.to_thread(embedding_service.embed_chunks, batch)
            all_embeddings.append(emb)
            
            ratio_done = (i + len(batch)) / len(chunks)
            curr_prog = min(int(25 + (ratio_done * 60)), 85)  # Scales 25 -> 85
            tracker.update_status(job_id, status="processing", progress=curr_prog)
            
        embeddings = np.vstack(all_embeddings)
        tracker.update_status(job_id, status="processing", progress=90)
        
        # 4. Generate Document ID (using Job ID so tracking is straightforward)
        doc_id = job_id
        
        # 5. Add to FAISS Index
        state.faiss_index = state.retrieval_service.add_to_index(
            embeddings=embeddings,
            doc_id=doc_id,
            chunks=chunks,
            existing_index=state.faiss_index,
        )
        
        # 6. Global Registry
        state.indexed_documents[doc_id] = {
            "filename": filename,
            "chunks_created": len(chunks),
            "text_length": len(text),
        }
        
        tracker.update_status(job_id, status="processing", progress=95)
        
        # 7. Persistent Storage sync
        state.storage.save_state(
            state.faiss_index, 
            state.indexed_documents, 
            state.retrieval_service._chunk_store
        )
        
        tracker.mark_complete(job_id, len(chunks))
        
    except Exception as exc:
        logger.error(f"Background Task {job_id} failed: {exc}")
        tracker.mark_failed(job_id, str(exc))
