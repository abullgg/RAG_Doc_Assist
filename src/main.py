"""
RAG Document Assistant — FastAPI Server
=======================================
Single entry point for the application. Initialises logging,
creates the FastAPI app, mounts routers, and manages global state
(indexed documents registry + FAISS index).

Run with:
    uvicorn src.main:app --reload
"""

import logging
from typing import Dict, Optional

import faiss
from fastapi import FastAPI

from src.core.config import settings
from src.utils.logger import setup_logging
from src.api.routes import upload, ask, health

# ------------------------------------------------------------------ #
#  Bootstrap
# ------------------------------------------------------------------ #

setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Global State  (shared across route modules via import)
# ------------------------------------------------------------------ #

# Registry of all indexed documents:  doc_id → metadata dict
indexed_documents: Dict[str, dict] = {}

# The shared FAISS index (created on first upload)
faiss_index: Optional[faiss.IndexFlatL2] = None

# ------------------------------------------------------------------ #
#  FastAPI Application
# ------------------------------------------------------------------ #

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Upload PDF / TXT documents, index them with FAISS, "
        "and ask questions answered by Claude."
    ),
    version="1.0.0",
)

# Mount routers
app.include_router(health.router, tags=["monitoring"])
app.include_router(upload.router, tags=["upload"])
app.include_router(ask.router, tags=["query"])


@app.get("/")
async def root():
    """Root endpoint — API welcome message."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API.",
        "documentation": "/docs",
    }


# ------------------------------------------------------------------ #
#  Entrypoint
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
