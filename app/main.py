"""FastAPI application: routers, static files, nothing else.

    uv run uvicorn app.main:app --reload --port 8000

Requires the embedding service on :8001 and Postgres on :5433. GET /api/health
says which of the two is missing.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import admin, chat, documents
from app.config import settings

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="DIU Admission RAG")
app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/api/ui-config")
def ui_config() -> dict:
    """What the page needs to know that only the server knows."""
    return {
        "corpus_is_synthetic": settings.CORPUS_IS_SYNTHETIC,
        "max_history_turns": settings.MAX_HISTORY_TURNS,
    }
