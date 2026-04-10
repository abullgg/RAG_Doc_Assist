"""
Document Processor Service
==========================
Handles file ingestion (PDF / TXT), text extraction, chunking with
character-level overlap, and unique document ID generation.
"""

import uuid
import logging
from typing import List

from fastapi import UploadFile
from PyPDF2 import PdfReader
import io

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts text from uploaded files and splits it into overlapping chunks."""

    # ------------------------------------------------------------------ #
    #  Text Extraction
    # ------------------------------------------------------------------ #

    async def extract_text(self, file_upload: UploadFile) -> str:
        """
        Read the uploaded file and return its full text content.

        Supports:
            - application/pdf  → extracted via PyPDF2
            - text/plain (.txt) → decoded as UTF-8

        Args:
            file_upload: A FastAPI ``UploadFile`` object.

        Returns:
            The extracted text as a single string.

        Raises:
            ValueError: If the file type is unsupported or the file is empty.
        """
        filename = file_upload.filename or ""
        content_type = file_upload.content_type or ""
        logger.info("Extracting text from '%s' (content_type=%s)", filename, content_type)

        raw_bytes: bytes = await file_upload.read()

        if not raw_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        # --- PDF -----------------------------------------------------------
        if filename.lower().endswith(".pdf") or "pdf" in content_type:
            return self._extract_pdf(raw_bytes, filename)

        # --- TXT -----------------------------------------------------------
        if filename.lower().endswith(".txt") or "text" in content_type:
            return self._extract_txt(raw_bytes, filename)

        raise ValueError(
            f"Unsupported file type: '{filename}' ({content_type}). "
            "Only PDF and TXT files are accepted."
        )

    # ------------------------------------------------------------------ #
    #  Private helpers
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
            raise ValueError(f"Could not parse PDF '{filename}': {exc}") from exc

    @staticmethod
    def _extract_txt(raw_bytes: bytes, filename: str) -> str:
        """Decode plain-text bytes with a UTF-8 fallback to latin-1."""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed for '%s'; falling back to latin-1", filename
            )
            text = raw_bytes.decode("latin-1")
        logger.info("TXT '%s': %d chars extracted", filename, len(text))
        return text

    # ------------------------------------------------------------------ #
    #  Chunking
    # ------------------------------------------------------------------ #

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        """
        Split *text* into chunks of approximately *chunk_size* characters with
        *overlap* characters shared between consecutive chunks.

        Empty or whitespace-only chunks are silently dropped.

        Args:
            text:       The full document text.
            chunk_size: Maximum number of characters per chunk.
            overlap:    Number of characters to repeat at the start of each
                        subsequent chunk (provides context continuity).

        Returns:
            A list of non-empty text chunks.
        """
        if not text or not text.strip():
            logger.warning("chunk_text received empty text — returning no chunks")
            return []

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            # Only keep chunks that contain actual content
            if chunk.strip():
                chunks.append(chunk)

            # Advance by (chunk_size - overlap) to create the overlap window
            start += chunk_size - overlap

        logger.info(
            "Chunked %d chars → %d chunks (size=%d, overlap=%d)",
            text_length,
            len(chunks),
            chunk_size,
            overlap,
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
