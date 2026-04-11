"""
Pydantic Schemas
================
Request and response models for every API endpoint, plus shared
base models for documents and chunks.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Shared / Base Models
# ------------------------------------------------------------------ #

class DocumentBase(BaseModel):
    """Base schema for a document reference."""
    filename: str
    metadata: Optional[Dict] = None


class DocumentCreate(DocumentBase):
    """Schema used internally when creating a new document record."""
    content: str


class ChunkResponse(BaseModel):
    """A single retrieved chunk with its similarity score."""
    text: str
    score: float
    metadata: Dict


# ------------------------------------------------------------------ #
#  /health
# ------------------------------------------------------------------ #

class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str
    documents_indexed: int


# ------------------------------------------------------------------ #
#  /upload
# ------------------------------------------------------------------ #

class UploadResponse(BaseModel):
    """Response schema for POST /upload."""
    document_id: str
    filename: str
    chunks_created: int
    status: str
    message: str


# ------------------------------------------------------------------ #
#  /ask
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    """Body schema for POST /ask."""
    question: str = Field(
        ..., min_length=1, description="The question to answer"
    )
    top_k: int = Field(
        default=3, ge=1, le=20, description="Number of chunks to retrieve"
    )


class AskResponse(BaseModel):
    """Response schema for POST /ask."""
    answer: str
    sources: List[str]
    confidence: float
