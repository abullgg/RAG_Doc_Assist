"""
RAG Document Assistant — FastAPI Server
=======================================
A Retrieval-Augmented Generation API that lets users upload documents
(PDF / TXT), indexes them with FAISS, and answers questions by
retrieving relevant chunks and sending them to Claude.

Endpoints:
    GET  /health   → system health check
    POST /upload   → ingest a document
    POST /ask      → ask a question against indexed documents
"""

import logging
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.llm_service import LLMService

# ------------------------------------------------------------------ #
#  Bootstrap
# ------------------------------------------------------------------ #

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  FastAPI application
# ------------------------------------------------------------------ #

app = FastAPI(
    title="RAG Document Assistant",
    description=(
        "Upload PDF / TXT documents, index them with FAISS, "
        "and ask questions answered by Claude."
    ),
    version="1.0.0",
)

# ------------------------------------------------------------------ #
#  Global State
# ------------------------------------------------------------------ #

# Registry of all indexed documents: doc_id → metadata dict
indexed_documents: Dict[str, dict] = {}

# The shared FAISS index (created on first upload)
faiss_index: Optional[faiss.IndexFlatL2] = None

# ------------------------------------------------------------------ #
#  Service Singletons
# ------------------------------------------------------------------ #

document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService()

# LLMService is initialised lazily (only when /ask is called)
# so the server can still start without an ANTHROPIC_API_KEY for
# upload-only workflows.
_llm_service: Optional[LLMService] = None


def _get_llm_service() -> LLMService:
    """Lazily initialise and return the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ------------------------------------------------------------------ #
#  Pydantic Models
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    """Body schema for POST /ask."""
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of chunks to retrieve")


class AskResponse(BaseModel):
    """Response schema for POST /ask."""
    answer: str
    sources: List[str]
    confidence: float


class UploadResponse(BaseModel):
    """Response schema for POST /upload."""
    document_id: str
    filename: str
    chunks_created: int
    status: str
    message: str


class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str
    documents_indexed: int


# ================================================================== #
#  ENDPOINTS
# ================================================================== #


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    **Health Check**

    Returns the server status and the number of documents currently indexed.
    """
    logger.info("Health check — %d documents indexed", len(indexed_documents))
    return HealthResponse(
        status="healthy",
        documents_indexed=len(indexed_documents),
    )


# ------------------------------------------------------------------ #
#  POST /upload
# ------------------------------------------------------------------ #

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    **Upload & Index a Document**

    Accepts a PDF or TXT file, extracts text, splits it into overlapping
    chunks, embeds each chunk, and adds the vectors to the FAISS index.
    """
    global faiss_index

    filename: str = file.filename or "unknown"
    logger.info("===== Upload request: '%s' =====", filename)

    # --- Validate file type ------------------------------------------------
    allowed_extensions = (".pdf", ".txt")
    if not filename.lower().endswith(allowed_extensions):
        logger.warning("Rejected file '%s': unsupported type", filename)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{filename}'. "
                f"Allowed types: {', '.join(allowed_extensions)}"
            ),
        )

    try:
        # 1. Extract text
        text: str = await document_processor.extract_text(file)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No readable text found in '{filename}'.",
            )

        # 2. Chunk the text
        chunks: List[str] = document_processor.chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Text from '{filename}' produced no usable chunks.",
            )

        # 3. Embed the chunks
        embeddings: np.ndarray = embedding_service.embed_chunks(chunks)

        # 4. Generate a document ID
        doc_id: str = document_processor.generate_document_id()

        # 5. Add to FAISS index
        faiss_index = retrieval_service.add_to_index(
            embeddings=embeddings,
            doc_id=doc_id,
            chunks=chunks,
            existing_index=faiss_index,
        )

        # 6. Record in the global registry
        indexed_documents[doc_id] = {
            "filename": filename,
            "chunks_created": len(chunks),
            "text_length": len(text),
        }

        logger.info(
            "Document '%s' indexed as %s — %d chunks",
            filename,
            doc_id,
            len(chunks),
        )

        return UploadResponse(
            document_id=doc_id,
            filename=filename,
            chunks_created=len(chunks),
            status="success",
            message=f"Document '{filename}' uploaded and indexed successfully.",
        )

    except HTTPException:
        raise  # re-raise validation errors as-is
    except ValueError as exc:
        logger.error("Validation error during upload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing '%s'", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing '{filename}': {exc}",
        ) from exc


# ------------------------------------------------------------------ #
#  POST /ask
# ------------------------------------------------------------------ #

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    **Ask a Question**

    Embeds the question, retrieves the top-K most relevant chunks from
    FAISS, and sends them to Claude to generate a grounded answer.
    """
    logger.info("===== Ask request: '%s' (top_k=%d) =====", request.question, request.top_k)

    # --- Pre-flight checks -------------------------------------------------
    if not indexed_documents:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Upload a document first.",
        )

    if faiss_index is None or faiss_index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="The FAISS index is empty. Upload a document first.",
        )

    try:
        # 1. Embed the question
        query_embedding: np.ndarray = embedding_service.embed_text(request.question)

        # 2. Search FAISS
        results: List[dict] = retrieval_service.search(
            query_embedding=query_embedding,
            faiss_index=faiss_index,
            top_k=request.top_k,
        )

        if not results:
            return AskResponse(
                answer="No relevant information was found in the indexed documents.",
                sources=[],
                confidence=0.0,
            )

        # 3. Build context from retrieved chunks
        source_texts: List[str] = [r["chunk"] for r in results]
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
            sum(r["score"] for r in results) / len(results), 4
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
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error answering question")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing your question: {exc}",
        ) from exc


# ================================================================== #
#  Entrypoint
# ================================================================== #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
