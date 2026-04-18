"""
Document Processor
==================
Handles file ingestion (PDF / TXT), text extraction, semantic chunking with
LangChain's RecursiveCharacterTextSplitter, and unique document ID generation.
"""

import io
import logging
import uuid
from typing import List

from fastapi import UploadFile
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.errors import DocumentProcessingError

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts text from uploaded files and splits it into semantic chunks."""

    def __init__(self) -> None:
        # Initialize RecursiveCharacterTextSplitter with hierarchical separators.
        # The splitter tries each separator in order, only falling back to the
        # next one when a chunk still exceeds chunk_size.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,           # Target chunk size in characters
            chunk_overlap=200,         # Overlap for cross-boundary context
            separators=[
                "\n\n",                # 1. Split on paragraphs (preferred)
                "\n",                  # 2. Split on lines
                ". ",                  # 3. Split on sentences
                " ",                   # 4. Split on words (rare)
                ""                     # 5. Split on characters (last resort)
            ],
            length_function=len,
            is_separator_regex=False,
        )
        logger.info("DocumentProcessor initialized with LangChain semantic splitter")

    # ------------------------------------------------------------------ #
    #  Text Extraction
    # ------------------------------------------------------------------ #

    async def extract_text(self, raw_bytes: bytes, filename: str, content_type: str) -> str:
        """
        Read the bytes and return its full text content.

        Supports:
            - application/pdf  → extracted via pdfplumber (layout-aware)
            - text/plain (.txt) → decoded as UTF-8 (latin-1 fallback)

        Args:
            raw_bytes: Raw file bytes loaded into memory.
            filename: File name string.
            content_type: MIME string.

        Returns:
            The extracted text as a single string.

        Raises:
            DocumentProcessingError: If the file type is unsupported or empty.
        """
        logger.info(
            "Extracting text from '%s' (content_type=%s)", filename, content_type
        )

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
        """Parse PDF bytes using pdfplumber (layout-aware) and concatenate all page texts."""
        try:
            pages_text: List[str] = []
            num_pages = 0
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                num_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
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
                "PDF '%s': %d pages, %d chars total (pdfplumber)",
                filename,
                num_pages,
                len(full_text),
            )
            return full_text

        except Exception as exc:
            logger.error("Failed to parse PDF '%s' with pdfplumber: %s", filename, exc)
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

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks using LangChain's RecursiveCharacterTextSplitter.

        The splitter respects paragraph boundaries, sentence boundaries, and
        document structure — headers stay with their content instead of being
        orphaned into separate chunks.

        Args:
            text: The full document text.

        Returns:
            A list of non-empty text chunks.

        Raises:
            DocumentProcessingError: If text is empty or chunking fails.
        """
        if not text or not text.strip():
            raise DocumentProcessingError("Cannot chunk empty text")

        try:
            chunks = self.splitter.split_text(text)

            logger.info(
                "Created %d semantic chunks from text of length %d",
                len(chunks),
                len(text),
            )
            for i, chunk in enumerate(chunks):
                logger.debug(
                    "Chunk %d: %d chars, starts with: %s...",
                    i,
                    len(chunk),
                    chunk[:50],
                )

            return chunks
        except Exception as exc:
            logger.error("Error chunking text: %s", exc)
            raise DocumentProcessingError(
                f"Failed to chunk text: {exc}"
            ) from exc

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
