# RAG Document Assistant

Upload a PDF or TXT → it gets chunked, embedded, and indexed → ask questions → get answers grounded in your document. That's it.

No cloud dependencies. Runs entirely on your machine using Ollama for LLM inference and FAISS for vector search.

## How It Works

```
PDF/TXT  →  Extract Text  →  Chunk (RecursiveCharacterTextSplitter)
                                      ↓
                              Embed (all-MiniLM-L6-v2, 384d)
                                      ↓
                              Index (FAISS + BM25)
                                      ↓
              Question  →  Hybrid Search (70% semantic + 30% keyword)
                                      ↓
                              Top-K chunks → Ollama LLM → Answer
```

**Hybrid retrieval** combines FAISS cosine similarity with BM25 keyword matching (weighted 70/30). This catches both semantically similar passages and exact keyword matches that pure vector search would miss.

## Tech Stack

| Layer | Tech |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| LLM | Ollama (default: `gemma3:4b`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector Store | FAISS (IndexFlatL2) |
| Keyword Search | BM25 (rank-bm25) |
| PDF Parsing | pdfplumber |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Persistence | Pickle-based state persistence to disk |

## Project Structure

```
src/
├── main.py                  # FastAPI app entry point, global state
├── api/routes/
│   ├── upload.py            # POST /upload — file ingestion
│   ├── ask.py               # POST /ask — question answering
│   ├── health.py            # GET /health
│   └── status.py            # GET /status/{job_id}
├── ingestion/
│   └── processor.py         # PDF/TXT extraction + semantic chunking
├── embeddings/
│   └── embedding.py         # Sentence-transformer wrapper
├── vector_store/
│   └── faiss_index.py       # FAISS index management + chunk storage
├── retrieval/
│   └── hybrid.py            # BM25 + FAISS hybrid search
├── generation/
│   └── llm.py               # Ollama LLM client + grounded prompting
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

app.py                       # Streamlit frontend (standalone, HTTP only)
```

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Install

```bash
git clone https://github.com/<your-username>/RAG-doc_assist.git
cd RAG-doc_assist
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
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Run

**1. Start the backend:**

```bash
uvicorn src.main:app --reload
```

API available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

**2. Start the frontend:**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload PDF/TXT for processing |
| `GET` | `/status/{job_id}` | Poll upload job progress |
| `POST` | `/ask` | Ask a question against uploaded documents |

### `POST /upload`

Multipart file upload. Returns `job_id` and processing status.

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

- **Document isolation**: Queries can be scoped to a specific `document_id` to prevent cross-document contamination
- **Hybrid search > pure vector search**: BM25 catches exact term matches that embeddings miss (acronyms, proper nouns, codes)
- **Local-first**: No API keys needed for the core pipeline. Ollama runs the LLM on your own hardware
- **Separation of concerns**: Streamlit frontend only talks to the backend over HTTP — zero shared imports
- **Persistence**: FAISS index and chunk data survive server restarts via pickle serialization to `./data/`

## License

Do whatever you want with it.
