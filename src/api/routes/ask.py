"""
POST /ask — Question-Answering Endpoint
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from src.embeddings.embedding import EmbeddingService
from src.embeddings.embedding import EmbeddingService
from src.generation.llm import LLMService
from src.models.schemas import AskRequest, AskResponse
from src.utils.errors import EmbeddingError, RetrievalError, LLMServiceError
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()

# Re-use the same service singletons that upload.py creates.
# EmbeddingService is heavy (loads the model), so we import lazily to
# avoid loading the model twice if upload.py was imported first.
# avoid loading the model twice if upload.py was imported first.
_embedding_service: Optional[EmbeddingService] = None
_llm_service: Optional[LLMService] = None


def _get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def _get_llm_service() -> LLMService:
    """Lazily initialise — server can start without ANTHROPIC_API_KEY."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a Question.

    Embeds the question, retrieves the top-K most relevant chunks from
    FAISS, and sends them to Claude to generate a grounded answer.
    """
    logger.info(
        "===== Ask request: '%s' (top_k=%d) =====",
        request.question,
        request.top_k,
    )

    # --- Pre-flight checks ------------------------------------------------
    if not state.indexed_documents:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Upload a document first.",
        )

    if state.faiss_index is None or state.faiss_index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="The FAISS index is empty. Upload a document first.",
        )

    try:
        embedding_svc = _get_embedding_service()

        # 1. Embed the question
        query_embedding: np.ndarray = embedding_svc.embed_text(request.question)

        # 2. Hybrid / Semantic Search
        use_hybrid = True
        if use_hybrid:
            source_texts, similarities = state.retrieval_service.hybrid_search(
                query_embedding=query_embedding,
                query_text=request.question,
                faiss_index=state.faiss_index,
                hybrid_retriever=state.hybrid_retriever,
                top_k=request.top_k
            )
        else:
            results: List[Dict] = state.retrieval_service.search(
                query_embedding=query_embedding,
                faiss_index=state.faiss_index,
                top_k=request.top_k,
            )
            source_texts = [r["chunk"] for r in results]
            similarities = [r["score"] for r in results]

        if not source_texts:
            return AskResponse(
                answer="No relevant information was found in the indexed documents.",
                sources=[],
                confidence=0.0,
            )

        # 3. Build context from retrieved chunks
        context: str = "\n\n---\n\n".join(
            f"[Source {i + 1}]:\n{chunk}"
            for i, chunk in enumerate(source_texts)
        )

        # 4. Call Claude
        llm = _get_llm_service()
        answer: str = llm.generate_answer(
            question=request.question,
            context=context,
        )

        # 5. Compute average confidence from similarity scores
        avg_confidence: float = round(
            sum(similarities) / len(similarities), 4
        )

        logger.info(
            "Answer generated — %d sources, avg_confidence=%.4f",
            len(source_texts),
            avg_confidence,
        )

        return AskResponse(
            answer=answer,
            sources=source_texts,
            confidence=avg_confidence,
        )

    except HTTPException:
        raise
    except LLMServiceError as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (EmbeddingError, RetrievalError) as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error answering question")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing your question: {exc}",
        ) from exc
