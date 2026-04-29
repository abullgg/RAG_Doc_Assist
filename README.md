# RAG Document Assistant

Upload a PDF or TXT → ask questions → get answers grounded strictly in your document.

Runs entirely on your machine. No cloud. No API keys.

---

## How It Works

```
Upload PDF / TXT
      │
      ▼
Extract text + tables  →  Chunk  →  Embed passages (BGE 768d)  →  FAISS + BM25 index
                                                                            │
                                          Question  →  embed_query() (BGE + prefix)
                                                                            │
                                                  Stage 1 — Hybrid search (FAISS 70% + BM25 30%)
                                                  Retrieve RERANKER_TOP_N candidates
                                                                            │
                                                  Stage 2 — Cross-encoder rerank
                                                  Score every (query, chunk) pair jointly
                                                  Keep top_k highest-scoring chunks
                                                                            │
                                                            Context  →  Ollama (Gemma 3 4B)  →  Answer
```

**Two-stage retrieval** separates speed from accuracy. Stage 1 (bi-encoder + BM25) casts a wide net fast. Stage 2 (cross-encoder) rescores every candidate pair with full attention and keeps only the most relevant chunks for the LLM.

**Hybrid retrieval** fuses FAISS vector search (70%) with BM25 keyword scoring (30%) so exact terms, acronyms, and numeric codes are never missed by pure semantic search.

**Table-aware ingestion** — PDF tables are extracted and converted to markdown so the LLM receives structured data, not collapsed rows.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js (App Router) |
| LLM | Ollama — `gemma3:4b` (local) |
| Bi-encoder Embeddings | `BAAI/bge-base-en-v1.5` · 768-dim |
| Vector Store | FAISS `IndexFlatIP` |
| Keyword Search | BM25 (`rank-bm25`) |
| Cross-Encoder Reranker | `BAAI/bge-reranker-base` |
| PDF Parsing | pdfplumber |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |

---

## Project Structure

```
src/
├── main.py                  # FastAPI app, global state, CORS
├── core/config.py           # Settings via Pydantic + .env
├── api/routes/
│   ├── upload.py            # POST /upload
│   ├── ask.py               # POST /ask  (embed → hybrid → rerank → LLM)
│   ├── health.py            # GET  /health
│   └── status.py            # GET  /status/{job_id}
├── ingestion/processor.py   # PDF/TXT extraction, table → markdown, chunking
├── embeddings/embedding.py  # BGE bi-encoder (embed_query / embed_chunks)
├── vector_store/            # FAISS index management
├── retrieval/
│   ├── hybrid.py            # Stage 1 — BM25 + FAISS hybrid search
│   └── reranker.py          # Stage 2 — cross-encoder reranking
├── generation/llm.py        # Ollama client + system prompt
├── persistence/storage.py   # Save/load index to ./data/
└── tasks/job_tracker.py     # Async upload job tracking

frontend/
├── app/page.tsx             # Chat UI (upload, thread, sources)
├── app/globals.css          # Dark design system
└── lib/api.ts               # Typed API client
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com/) installed and running

### Backend

```bash
git clone https://github.com/abullgg/RAG_Doc_Assist.git
cd RAG_Doc_Assist
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Pull the LLM

```bash
ollama pull gemma3:4b
```

### Environment

Create `.env` in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=gemma3:4b
FAISS_INDEX_PATH=./data/faiss_index
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Run

```bash
# 1. Ollama
ollama serve

# 2. Backend
uvicorn src.main:app --reload

# 3. Frontend
cd frontend && npm run dev
```

- API: `http://localhost:8000` · Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:3000`

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + indexed document count |
| `POST` | `/upload` | Upload a PDF or TXT file |
| `GET` | `/status/{job_id}` | Poll upload job progress |
| `POST` | `/ask` | Ask a question against the document |

`POST /ask` body:

```json
{
  "question": "What are the key findings?",
  "document_id": "optional-uuid",
  "top_k": 3
}
```

Returns: `answer`, `sources[]`, `confidence`.

---

## Changing the Embedding Model

> ⚠️ Switching models requires re-uploading all documents.

1. Update `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` in `.env`
2. Delete `./data/`
3. Restart the backend

The persistence layer automatically detects a dimension mismatch and clears the stale index on startup.

---

## License

Do whatever you want with it.
