"""
POST /ask — Question-answering endpoint.

Embeds the question, runs hybrid retrieval (FAISS + BM25),
and sends retrieved passages to the local Ollama LLM for a grounded answer.

Pipeline
--------
1. embed_query()        -- BGE query prefix applied (via state.embedding_service)
2. hybrid_search()      -- FAISS cosine + BM25 fused; fetches RERANKER_TOP_N candidates
3. CrossEncoderReranker -- scores every (query, chunk) pair; keeps top_k best
4. build context        -- join top_k chunks with source labels
5. LLMService           -- grounded answer generation via Ollama

Step 3 is skipped when RERANKER_ENABLED=false in config / .env,
or when state.reranker is None (startup didn't initialize it).
"""

import asyncio
import logging
from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from src.core.config import settings
from src.generation.llm import LLMService
from src.models.schemas import AskRequest, AskResponse
from src.utils.errors import EmbeddingError, RetrievalError, LLMServiceError
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()

# ML model singletons (embedding_service, reranker) live in src.main and are
# initialized ONCE in the startup event. Access them via state.embedding_service
# and state.reranker. Only the LLM client (no GPU weights in-process) stays local.
_llm_service: Optional[LLMService] = None


def _get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question against indexed documents.

    Retrieves the most relevant passages from the FAISS + BM25 index,
    optionally reranks them with a cross-encoder, and returns an answer
    grounded strictly in those passages.
    """
    logger.info(
        "===== Ask request: '%s' (top_k=%d, reranker=%s) =====",
        request.question,
        request.top_k,
        settings.RERANKER_ENABLED,
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

    if state.embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not ready. Server may still be starting up.",
        )

    try:
        # Step 1: Embed the query
        # embed_query() applies the BGE retrieval prefix so the vector is
        # in the correct query space for cosine similarity against passages.
        query_embedding: np.ndarray = state.embedding_service.embed_query(request.question)

        # Step 2: Hybrid retrieval (Stage 1)
        # When reranking is enabled we fetch RERANKER_TOP_N candidates --
        # a broader net -- so the cross-encoder has enough material to work
        # with. Without reranking we fetch exactly request.top_k.
        retrieve_k: int = settings.RERANKER_TOP_N if state.reranker else request.top_k

        source_texts, similarities = state.retrieval_service.hybrid_search(
            query_embedding=query_embedding,
            query_text=request.question,
            faiss_index=state.faiss_index,
            hybrid_retriever=state.hybrid_retriever,
            top_k=retrieve_k,
            filter_doc_id=request.document_id,
        )

        if not source_texts:
            return AskResponse(
                answer="No relevant information was found in the indexed documents.",
                sources=[],
                confidence=0.0,
            )

        # Step 3: Cross-encoder reranking (Stage 2)
        # The cross-encoder scores every (query, chunk) pair jointly using
        # full self-attention, producing much more accurate relevance scores
        # than the bi-encoder cosine similarity from Stage 1.
        #
        # CrossEncoder.predict() is CPU-bound, so we offload it to a thread
        # pool to keep the FastAPI event loop responsive.
        if state.reranker:
            ranked_pairs: List[Tuple[str, float]] = await asyncio.to_thread(
                state.reranker.rerank,
                request.question,
                source_texts,
                request.top_k,
            )
            source_texts = [text for text, _ in ranked_pairs]
            similarities = [score for _, score in ranked_pairs]
            logger.info(
                "Reranking complete -- %d candidates -> %d chunks (top score=%.4f)",
                retrieve_k,
                len(source_texts),
                similarities[0] if similarities else 0.0,
            )

        # Step 4: Build context
        context: str = "\n\n---\n\n".join(
            f"[Source {i + 1}]:\n{chunk}"
            for i, chunk in enumerate(source_texts)
        )

        # Step 5: Generate answer
        llm = _get_llm_service()
        answer: str = llm.generate_answer(
            question=request.question,
            context=context,
        )

        avg_confidence: float = round(
            sum(similarities) / len(similarities), 4
        )

        logger.info(
            "Answer generated -- %d sources, avg_confidence=%.4f",
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
