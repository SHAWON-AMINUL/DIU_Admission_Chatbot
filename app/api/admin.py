"""Operational endpoints.

The health check reports each dependency separately and never raises. A single
boolean would say "unhealthy" for both a stopped Postgres and an unloaded
embedding model, which are different problems with different fixes -- and the
whole point of this endpoint is that a broken demo is diagnosable at a glance
rather than by reading two sets of logs.
"""

from fastapi import APIRouter

from app.config import settings
from app.database.connection import get_conn
from app.embeddings.client import EmbeddingClient

router = APIRouter(prefix="/api", tags=["admin"])


def _database_health() -> tuple[bool, dict | str]:
    try:
        with get_conn() as conn:
            posts, chunks, vectors = conn.execute(
                """
                SELECT (SELECT COUNT(*) FROM posts WHERE deleted_at IS NULL),
                       (SELECT COUNT(*) FROM ai_content),
                       (SELECT COUNT(*) FROM ai_content_vectors)
                """
            ).fetchone()
        return True, {"posts": posts, "chunks": chunks, "vectors": vectors}
    except Exception as exc:
        return False, str(exc)


def _embedding_health() -> tuple[bool, dict | str]:
    # Short timeout: this is a liveness probe, not a request worth waiting on.
    client = EmbeddingClient(timeout=5.0)
    try:
        return True, client.health()
    except Exception as exc:
        return False, str(exc)
    finally:
        client.close()


@router.get("/health")
def health() -> dict:
    db_ok, db_detail = _database_health()
    embed_ok, embed_detail = _embedding_health()

    return {
        "database": {"ok": db_ok, "detail": db_detail},
        "embedding_service": {"ok": embed_ok, "detail": embed_detail},
        "config": {
            "model_name": settings.MODEL_NAME_TAG,
            "dim": settings.EMBEDDING_DIM,
            "chunk_target_tokens": settings.CHUNK_TARGET_TOKENS,
            "chunk_overlap_tokens": settings.CHUNK_OVERLAP_TOKENS,
        },
    }
