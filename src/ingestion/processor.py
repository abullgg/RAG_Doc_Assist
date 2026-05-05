"""
Document Processor
------------------
Handles file ingestion (PDF / TXT): text and table extraction,
semantic chunking, and unique document ID generation.
"""

import io
import logging
import uuid
import re
from typing import List

from fastapi import UploadFile
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from src.utils.errors import DocumentProcessingError

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Extracts text from uploaded files and splits it into semantic chunks."""

    def __init__(self) -> None:
        # Pass 1: Semantic header splitting
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ],
            return_each_line=False,
        )
        
        # Pass 2: Size enforcement safety net
        self.recursive_splitter = RecursiveCharacterTextSplitter(
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
        logger.info("DocumentProcessor initialized with 2-pass semantic splitter")

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
    def _table_to_markdown(table: list) -> str:
        """
        Convert a pdfplumber table (list of rows, each a list of cell strings)
        into a GitHub-style markdown table string.

        Returns an empty string if the table is empty or contains only blank cells.
        """
        if not table:
            return ""

        # Normalise cells: None → "", strip whitespace
        cleaned: List[List[str]] = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]

        # Skip tables that are entirely blank
        if all(all(cell == "" for cell in row) for row in cleaned):
            return ""

        # Pad every row to the same column count
        col_count = max(len(row) for row in cleaned)
        padded = [row + [""] * (col_count - len(row)) for row in cleaned]

        lines: List[str] = []
        # Header (first row)
        lines.append("| " + " | ".join(padded[0]) + " |")
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
        # Data rows
        for row in padded[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _extract_pdf(raw_bytes: bytes, filename: str) -> str:
        """
        Parse PDF bytes and return the full text content.

        Strategy (per page):
        1. ``page.extract_text()``  — always runs first; preserves the existing
           prose/paragraph flow that already delivers >92 % accuracy on text docs.
        2. ``page.extract_tables()`` — runs additionally and silently; any detected
           tables are converted to markdown and **appended** after the page text.
           Failures are caught and ignored so text-only docs are never affected.
        """
        try:
            pages_text: List[str] = []
            num_pages = 0

            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                num_pages = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages):
                    parts: List[str] = []

                    # ── 1. Existing flat text extraction (unchanged behaviour) ──
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        parts.append(page_text)
                        logger.debug(
                            "Page %d/%d '%s': %d chars (text)",
                            page_num + 1, num_pages, filename, len(page_text),
                        )

                    # ── 2. Additive table extraction (new) ──
                    try:
                        tables = page.extract_tables() or []
                        table_mds: List[str] = []
                        for table in tables:
                            md = DocumentProcessor._table_to_markdown(table)
                            if md:
                                table_mds.append(md)

                        if table_mds:
                            # Label the block clearly so Gemma recognises structure
                            block = "[TABLE]\n" + "\n\n[TABLE]\n".join(table_mds)
                            parts.append(block)
                            logger.debug(
                                "Page %d/%d '%s': %d table(s) extracted as markdown",
                                page_num + 1, num_pages, filename, len(table_mds),
                            )

                    except Exception as tbl_exc:
                        # Table extraction failure never blocks text extraction
                        logger.warning(
                            "Page %d/%d '%s': table extraction skipped (%s)",
                            page_num + 1, num_pages, filename, tbl_exc,
                        )

                    if parts:
                        pages_text.append("\n\n".join(parts))

            full_text = "\n\n".join(pages_text)
            logger.info(
                "PDF '%s': %d pages, %d chars total",
                filename, num_pages, len(full_text),
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

    def normalize_headers(self, text: str) -> str:
        """
        Normalize all header formats to Markdown.
        Handles: numbered sections, ALL-CAPS, underlined, title-case isolated lines.
        """
        lines = text.split('\n')
        normalized_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            stripped = line.strip()
            if not stripped:
                normalized_lines.append(line)
                continue
                
            # 1. Already Markdown
            if stripped.startswith('#'):
                normalized_lines.append(line)
                continue
                
            # 2. Numbered Sections (1. Introduction, 2.1 Methods, 1) Skills)
            if re.match(r'^\d+(\.\d+)*[\.\)]\s+[A-Z]', stripped):
                normalized_lines.append(f"# {stripped}")
                continue
                
            # 3. ALL-CAPS Headers
            if len(stripped) > 3 and stripped.isupper() and not stripped.endswith('.'):
                normalized_lines.append(f"# {stripped}")
                continue
                
            # 4. Underlined Headers (followed by --- or ===)
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                if re.match(r'^[-=]{3,}$', next_line):
                    normalized_lines.append(f"# {stripped}")
                    skip_next = True  # Skip the --- line
                    continue
            
            # 5. Title-Case Isolated Lines
            is_title_case = stripped[0].isupper()
            is_short = len(stripped) < 50
            is_isolated = (
                i == 0 or 
                lines[i - 1].strip() == '' or
                i == len(lines) - 1 or
                lines[i + 1].strip() == ''
            )
            
            if is_title_case and is_short and is_isolated:
                # Avoid matching short sentences
                if not re.search(r'[,\.]$', stripped) and not stripped.startswith(('The ', 'A ', 'An ', 'the ', 'a ', 'an ')):
                    normalized_lines.append(f"# {stripped}")
                    continue
            
            normalized_lines.append(line)
            
        return '\n'.join(normalized_lines)

    def chunk_text(self, text: str) -> List[str]:
        """
        2-pass semantic chunking:
        1. Normalize headers to Markdown
        2. Split by semantic boundaries (MarkdownHeaderTextSplitter)
        3. Enforce size limits (RecursiveCharacterTextSplitter)
        """
        if not text or not text.strip():
            raise DocumentProcessingError("Cannot chunk empty text")

        try:
            # Step 1: Normalize headers
            normalized_text = self.normalize_headers(text)
            logger.info("Headers normalized to Markdown format")
            
            # Step 2: First pass - Semantic splitting
            try:
                semantic_chunks = self.markdown_splitter.split_text(normalized_text)
                logger.info("MarkdownHeaderTextSplitter created %d semantic chunks", len(semantic_chunks))
            except Exception as e:
                logger.warning("Markdown splitting failed: %s, using full text as single chunk", e)
                semantic_chunks = [{"page_content": normalized_text, "metadata": {}}]
            
            # Step 3: Second pass - Size enforcement
            final_chunks = []
            for i, chunk in enumerate(semantic_chunks):
                # Handle Document objects (Langchain) vs raw strings/dicts
                chunk_text = getattr(chunk, 'page_content', chunk.get("page_content", chunk) if isinstance(chunk, dict) else chunk)
                chunk_metadata = getattr(chunk, 'metadata', chunk.get("metadata", {}) if isinstance(chunk, dict) else {})
                
                # Re-attach metadata safely so it is passed to LLM context
                header_context = " > ".join(str(v) for v in chunk_metadata.values())
                if header_context:
                    chunk_text = f"[{header_context}]\n{chunk_text}"
                
                chunk_size = len(chunk_text)
                
                if chunk_size > 2000:
                    logger.info("Chunk %d exceeds 2000 chars (%d), recursively splitting", i, chunk_size)
                    sub_chunks = self.recursive_splitter.split_text(chunk_text)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk_text)
            
            # Log final statistics
            logger.info("Final result: %d chunks", len(final_chunks))
            if final_chunks:
                sizes = [len(c) for c in final_chunks]
                logger.info("Chunk sizes - Min: %d, Max: %d, Avg: %d", min(sizes), max(sizes), sum(sizes)//len(sizes))
            
            return final_chunks
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
