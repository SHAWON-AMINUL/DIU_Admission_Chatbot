"""Local BGE-M3 embedding microservice.

Both the ingest pipeline and the query-side chat endpoint call this service.
That is the entire point of it existing: the model weights, the normalization
setting, and the tokenizer live in exactly one process, so the vectors written
at ingest time and the vectors computed at query time cannot silently disagree.

Runs entirely offline after the first model download. No API keys, no rate
limits, no per-request cost.

    uv run uvicorn app.embeddings.service:app --port 8001
"""

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("embeddings")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the service accepts traffic.

    Loading BGE-M3 onto the GPU takes longer than a typical HTTP client
    timeout. Loading it lazily on the first /embed call meant whichever post
    was ingested first always timed out, while later ones succeeded -- a
    confusing, intermittent-looking failure. Paying the cost at startup makes
    "the service is up" mean "the service can answer".
    """
    get_model()
    yield


app = FastAPI(title="DIU Admission - Embedding Service", lifespan=lifespan)

_model: SentenceTransformer | None = None


def _pick_device() -> str:
    if settings.EMBEDDING_DEVICE != "auto":
        return settings.EMBEDDING_DEVICE
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


def get_model() -> SentenceTransformer:
    """Load the model once, on first use."""
    global _model
    if _model is None:
        device = _pick_device()
        log.info("loading %s on %s", settings.EMBEDDING_MODEL, device)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        log.info("model ready (dim=%s)", _model.get_embedding_dimension())
    return _model


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model_name: str
    dim: int
    embeddings: list[list[float]]


class TokenCountRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict:
    """Cheap liveness check that does not force a model load."""
    return {
        "status": "ok",
        "model": settings.EMBEDDING_MODEL,
        "model_name_tag": settings.MODEL_NAME_TAG,
        "dim": settings.EMBEDDING_DIM,
        "device": _pick_device(),
        "loaded": _model is not None,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed texts. Always L2-normalized.

    Normalization is applied here rather than by callers so that every vector
    in the database is unit length. Cosine distance does not strictly require
    it, but it keeps stored distances comparable across batches and leaves the
    door open to switching to the faster inner-product operator later.
    """
    model = get_model()
    vectors = model.encode(
        req.texts,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    )
    return EmbedResponse(
        model_name=settings.MODEL_NAME_TAG,
        dim=int(vectors.shape[1]),
        embeddings=vectors.tolist(),
    )


@app.post("/token_count")
def token_count(req: TokenCountRequest) -> dict:
    """Count BGE-M3 subword tokens.

    The chunker calls this instead of counting words or using tiktoken.
    Bangla produces far more subword tokens per character than English, so any
    other counter would size Bangla chunks wrongly.
    """
    tokenizer = get_model().tokenizer
    counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in req.texts]
    return {"counts": counts}
