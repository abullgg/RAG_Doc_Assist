"""
Custom Exceptions for the RAG Document Assistant.
==================================================
Each exception maps to a specific failure domain so that API routes
can catch them and return the correct HTTP status code.
"""


class DocumentProcessingError(Exception):
    """
    Raised when document ingestion fails.

    Examples: corrupt PDF, unsupported file type, empty file,
    encoding errors during text extraction.
    """

    def __init__(self, message: str = "Document processing failed.") -> None:
        self.message = message
        super().__init__(self.message)


class EmbeddingError(Exception):
    """
    Raised when the embedding model fails to encode text.

    Examples: model not loaded, input too long, numpy dtype mismatch.
    """

    def __init__(self, message: str = "Embedding generation failed.") -> None:
        self.message = message
        super().__init__(self.message)


class RetrievalError(Exception):
    """
    Raised when FAISS index operations fail.

    Examples: index not initialised, dimension mismatch, search on
    an empty index.
    """

    def __init__(self, message: str = "Retrieval operation failed.") -> None:
        self.message = message
        super().__init__(self.message)


class LLMServiceError(Exception):
    """
    Raised when the LLM API call fails.

    Examples: missing API key, rate-limited, network timeout,
    malformed response from Claude.
    """

    def __init__(self, message: str = "LLM service error.") -> None:
        self.message = message
        super().__init__(self.message)
