# RAG Document Assistant

Upload a PDF or TXT → it gets extracted, chunked, embedded, and indexed → ask questions in a chat interface → get answers grounded strictly in your document.

No cloud dependencies. Runs entirely on your machine using Ollama for LLM inference and FAISS for vector search.

## How It Works

```
PDF/TXT  →  Extract Text + Tables  →  Chunk (RecursiveCharacterTextSplitter)
                                               ↓
                                       Embed (all-MiniLM-L6-v2, 384d)
                                               ↓
                                       Index (FAISS + BM25)
                                               ↓
               Question  →  Hybrid Search (70% semantic + 30% keyword)
                                               ↓
                               Top-K chunks → Ollama (Gemma 3 4B) → Answer
```

**Hybrid retrieval** combines FAISS cosine similarity with BM25 keyword matching (weighted 70/30). This catches both semantically similar passages and exact keyword matches that pure vector search would miss.

**Table-aware ingestion**: `pdfplumber` runs both `extract_text()` and `extract_tables()` per page. Detected tables are converted to GitHub-style markdown and appended alongside the prose text so the LLM receives structured data instead of collapsed rows.

## Tech Stack

| Layer | Tech |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Next.js 16 (App Router) |
| LLM | Ollama (default: `gemma3:4b`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector Store | FAISS (IndexFlatL2) |
| Keyword Search | BM25 (rank-bm25) |
| PDF Parsing | pdfplumber (text + table extraction) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Markdown Rendering | react-markdown + remark-gfm |
| Persistence | Pickle-based state persistence to disk |

## Project Structure

```
src/
├── main.py                  # FastAPI app entry point, global state, CORS
├── api/routes/
│   ├── upload.py            # POST /upload — file ingestion + job tracking
│   ├── ask.py               # POST /ask — question answering with doc isolation
│   ├── health.py            # GET /health
│   └── status.py            # GET /status/{job_id}
├── ingestion/
│   └── processor.py         # PDF/TXT extraction, table → markdown, chunking
├── embeddings/
│   └── embedding.py         # Sentence-transformer wrapper
├── vector_store/
│   └── faiss_index.py       # FAISS index management + chunk storage
├── retrieval/
│   └── hybrid.py            # BM25 + FAISS hybrid search
├── generation/
│   └── llm.py               # Ollama client + table-aware system prompt
├── persistence/
│   └── storage.py           # Save/load FAISS index + chunks to disk
├── core/
│   └── config.py            # Pydantic settings from .env
├── models/
│   └── schemas.py           # Request/response Pydantic models
├── tasks/
│   └── job_tracker.py       # Async upload job tracking
└── utils/
    ├── logger.py
    └── errors.py

frontend/                    # Next.js 16 App Router frontend
├── app/
│   ├── page.tsx             # Main chat interface
│   ├── globals.css          # Obsidian Flux dark design system
│   └── layout.tsx
└── lib/
    └── api.ts               # Typed API client for all 4 endpoints
```

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
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Pull the LLM model

```bash
ollama pull gemma3:4b
```

You can swap this for any Ollama model by setting `MODEL_NAME` in `.env`.

### Environment Variables

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=gemma3:4b
FAISS_INDEX_PATH=./data/faiss_index
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

## Run

**1. Start Ollama:**

```bash
ollama serve
```

**2. Start the backend:**

```bash
uvicorn src.main:app --reload
```

API available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

**3. Start the frontend:**

```bash
cd frontend
npm run dev
```

Opens at `http://localhost:3000`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + indexed document count |
| `POST` | `/upload` | Upload PDF/TXT for processing |
| `GET` | `/status/{job_id}` | Poll upload job progress |
| `POST` | `/ask` | Ask a question against uploaded documents |

### `POST /ask`

```json
{
  "question": "What are the key findings?",
  "document_id": "optional-uuid-to-scope-query",
  "top_k": 3
}
```

Returns the LLM answer, confidence score, and source chunks.

## Key Design Decisions

- **Document isolation**: Queries are scoped to a specific `document_id` to prevent cross-document contamination in the FAISS index
- **Table-aware extraction**: Each PDF page runs both `extract_text()` and `extract_tables()` — tables are converted to markdown and appended so the LLM receives structured data
- **Hybrid search > pure vector**: BM25 catches exact term matches that embeddings miss (acronyms, proper nouns, numeric codes)
- **Table-aware system prompt**: Gemma is explicitly instructed to recognise and reconstruct four table formats (markdown, flat/collapsed text, key-value pairs, CSV) and output answers as markdown tables
- **Local-first**: No API keys, no data leaves your machine. Ollama runs the LLM on your own hardware
- **Separation of concerns**: Next.js frontend communicates with the backend only over HTTP — no shared imports
- **Persistence**: FAISS index and chunk data survive server restarts via pickle serialization to `./data/`

## License

Do whatever you want with it.
