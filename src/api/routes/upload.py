"""
POST /upload — Document Ingestion Endpoint
"""

import logging
from typing import List

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.core.config import settings
from src.ingestion.processor import DocumentProcessor
from src.embeddings.embedding import EmbeddingService
from src.embeddings.embedding import EmbeddingService
from src.models.schemas import UploadResponse
from src.utils.errors import DocumentProcessingError, EmbeddingError, RetrievalError
import src.main as state

logger = logging.getLogger(__name__)

router = APIRouter()

# Service singletons (created once when this module is first imported)
_processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)
_embedding_service = EmbeddingService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload & Index a Document.

    Accepts a PDF or TXT file, extracts text, splits it into overlapping
    chunks, embeds each chunk, and adds the vectors to the FAISS index.
    """
    filename: str = file.filename or "unknown"
    logger.info("===== Upload request: '%s' =====", filename)

    # --- Validate file type -----------------------------------------------
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
        text: str = await _processor.extract_text(file)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No readable text found in '{filename}'.",
            )

        # 2. Chunk the text
        chunks: List[str] = _processor.chunk_text(text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Text from '{filename}' produced no usable chunks.",
            )

        # 3. Embed the chunks
        embeddings: np.ndarray = _embedding_service.embed_chunks(chunks)

        # 4. Generate a document ID
        doc_id: str = _processor.generate_document_id()

        # 5. Add to FAISS index
        state.faiss_index = state.retrieval_service.add_to_index(
            embeddings=embeddings,
            doc_id=doc_id,
            chunks=chunks,
            existing_index=state.faiss_index,
        )

        # 6. Record in the global registry
        state.indexed_documents[doc_id] = {
            "filename": filename,
            "chunks_created": len(chunks),
            "text_length": len(text),
        }

        logger.info(
            "Document '%s' indexed as %s — %d chunks",
            filename, doc_id, len(chunks),
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
    except DocumentProcessingError as exc:
        logger.error("Document processing error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingError, RetrievalError) as exc:
        logger.error("Pipeline error during upload: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing '%s'", filename)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while processing '{filename}': {exc}",
        ) from exc
