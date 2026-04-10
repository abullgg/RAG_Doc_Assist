# RAG Document Assistant: Project Requirements & Expectations
Version: 1.0  
Last Updated: April 2026  

## Phase 1: MVP (Minimum Viable Product)
Goal: "I built a working RAG system"

### Core Features:
- Upload a single PDF/document
- Chunk the document into manageable pieces
- Generate embeddings for each chunk
- Store embeddings in local FAISS vector database
- Accept a user question via API
- Retrieve top-K relevant chunks using similarity search
- Send context + question to Claude/Gemini API
- Return LLM-generated answer

### Technologies:
- **Language**: Python 3.10+
- **Backend Framework**: FastAPI
- **Vector Database**: FAISS (local)
- **Embeddings Model**: sentence-transformers
- **LLM**: Claude API (Anthropic)

### API Endpoints:
```
POST /upload
  Input: PDF file
  Output: Document indexed

POST /ask
  Input: { "question": "..." }
  Output: { "answer": "...", "sources": [...] }

GET /health
  Output: { "status": "healthy" }
```
