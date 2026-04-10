from pydantic import BaseModel
from typing import List, Optional

class DocumentBase(BaseModel):
    filename: str
    metadata: Optional[dict] = None

class DocumentCreate(DocumentBase):
    content: str

class ChunkResponse(BaseModel):
    text: str
    score: float
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResponse]

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[ChunkResponse]
