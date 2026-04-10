from fastapi import APIRouter
from src.models.schemas import ChatRequest, ChatResponse, ChunkResponse

router = APIRouter()

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    # Placeholder for MVP RAG logic
    # 1. Retrieve top-K chunks from FAISS
    # 2. Call Anthropic Claude API
    # 3. Return response with sources
    
    dummy_source = ChunkResponse(text="Source context...", score=0.85, metadata={"filename": "doc.pdf"})
    return ChatResponse(
        answer="According to the document, the answer is...",
        sources=[dummy_source]
    )
