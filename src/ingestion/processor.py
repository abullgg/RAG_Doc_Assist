"""
Document Processor
==================
Handles file ingestion (PDF / TXT), text extraction, chunking with
character-level overlap, and unique document ID generation.
"""

import io
import logging
import uuid
from typing import List

from fastapi import UploadFile
from PyPDF2 import PdfReader

from src.utils.errors import DocumentProcessingError

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts text from uploaded files and splits it into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ #
    #  Text Extraction
    # ------------------------------------------------------------------ #

    async def extract_text(self, file_upload: UploadFile) -> str:
        """
        Read the uploaded file and return its full text content.

        Supports:
            - application/pdf  → extracted via PyPDF2
            - text/plain (.txt) → decoded as UTF-8 (latin-1 fallback)

        Args:
            file_upload: A FastAPI ``UploadFile`` object.

        Returns:
            The extracted text as a single string.

        Raises:
            DocumentProcessingError: If the file type is unsupported or empty.
        """
        filename: str = file_upload.filename or ""
        content_type: str = file_upload.content_type or ""
        logger.info(
            "Extracting text from '%s' (content_type=%s)", filename, content_type
        )

        raw_bytes: bytes = await file_upload.read()

        if not raw_bytes:
            raise DocumentProcessingError(
                f"Uploaded file '{filename}' is empty."
            )

        # --- PDF ---
        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            return self._extract_pdf(raw_bytes, filename)

        # --- TXT ---
        if filename.lower().endswith(".txt") or "text" in content_type:
            return self._extract_txt(raw_bytes, filename)

        raise DocumentProcessingError(
            f"Unsupported file type: '{filename}' ({content_type}). "
            "Only PDF and TXT files are accepted."
        )

    # ------------------------------------------------------------------ #
    #  Private Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_pdf(raw_bytes: bytes, filename: str) -> str:
        """Parse PDF bytes and concatenate all page texts."""
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages_text: List[str] = []
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
                    logger.debug(
                        "Page %d of '%s': extracted %d chars",
                        page_num + 1,
                        filename,
                        len(page_text),
                    )

            full_text = "\n".join(pages_text)
            logger.info(
                "PDF '%s': %d pages, %d chars total",
                filename,
                len(reader.pages),
                len(full_text),
            )
            return full_text

        except Exception as exc:
            logger.error("Failed to parse PDF '%s': %s", filename, exc)
            raise DocumentProcessingError(
                f"Could not parse PDF '{filename}': {exc}"
            ) from exc

    @staticmethod
    def _extract_txt(raw_bytes: bytes, filename: str) -> str:
        """Decode plain-text bytes with a UTF-8 fallback to latin-1."""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed for '%s'; falling back to latin-1",
                filename,
            )
            text = raw_bytes.decode("latin-1")

        logger.info("TXT '%s': %d chars extracted", filename, len(text))
        return text

    # ------------------------------------------------------------------ #
    #  Chunking
    # ------------------------------------------------------------------ #

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> List[str]:
        """
        Split *text* into chunks of approximately *chunk_size* characters
        with *overlap* characters shared between consecutive chunks.

        Empty or whitespace-only chunks are silently dropped.

        Args:
            text:       The full document text.
            chunk_size: Max characters per chunk (defaults to instance setting).
            overlap:    Characters repeated across boundaries (defaults to
                        instance setting).

        Returns:
            A list of non-empty text chunks.
        """
        size = chunk_size or self.chunk_size
        lap = overlap or self.chunk_overlap

        if not text or not text.strip():
            logger.warning("chunk_text received empty text — returning no chunks")
            return []

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            # Advance by (chunk_size - overlap) to create the overlap window
            start += size - lap

        logger.info(
            "Chunked %d chars → %d chunks (size=%d, overlap=%d)",
            text_length,
            len(chunks),
            size,
            lap,
        )
        return chunks

    # ------------------------------------------------------------------ #
    #  ID Generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_document_id() -> str:
        """
        Generate a unique identifier for a newly ingested document.

        Returns:
            A UUID-4 string (e.g. ``'a3f8b1c2-...'``).
        """
        doc_id = str(uuid.uuid4())
        logger.debug("Generated document ID: %s", doc_id)
        return doc_id
