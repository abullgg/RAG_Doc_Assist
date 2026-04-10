"""
Services package for the RAG Document Assistant.

Exports:
    - DocumentProcessor: Handles file ingestion and text chunking.
    - EmbeddingService: Generates vector embeddings via sentence-transformers.
    - RetrievalService: Manages FAISS index operations and similarity search.
    - LLMService: Interfaces with the Claude API for answer generation.
"""

from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService

__all__ = [
    "DocumentProcessor",
    "EmbeddingService",
    "RetrievalService",
    "LLMService",
]
