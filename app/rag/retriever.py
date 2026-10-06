"""Hybrid retrieval: run every branch, fuse the rankings, return chunks.

This module is now only wiring -- embed the query, fan out to the branches,
fuse, rehydrate. The dense SQL lives in cosine_search, the lexical SQL in
bm25_search, and the ranking rule in hybrid_search, where it is tested without
a database. Adding a branch means adding a name to BRANCHES, nothing else.

There is deliberately no SQL/structured branch. The schema holds no structured
admission tables (no programs, fees, deadlines), so a third branch would have
nothing to query. It can be added when those tables exist.
"""

from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.database.connection import get_conn
from app.embeddings.client import EmbeddingClient
from app.rag import bm25_search, cosine_search
from app.rag.hybrid_search import fuse
from app.rag.rows import ChunkRow


@dataclass(frozen=True)
class Hit:
    ai_content_id: int
    post_id: int
    post_title: str
    chunk_index: int
    chunk_text: str
    score: float
    vector_rank: int | None
    text_rank: int | None


def search(
    query: str, top_k: int | None = None, client: EmbeddingClient | None = None
) -> list[Hit]:
    """Retrieve chunks for a query. Returns fused, ranked Hits."""
    k = top_k or settings.RETRIEVAL_TOP_K
    owns_client = client is None
    client = client or EmbeddingClient()

    try:
        # No pre-translation: BGE-M3 matches a Bangla query against English
        # chunks natively, so the raw query is embedded as typed.
        _, vectors = client.embed([query])
        # np.array, not the raw JSON list -- see the note in pipeline.py.
        query_vector = np.asarray(vectors[0], dtype=np.float32)

        with get_conn() as conn:
            branches = {
                "vector": cosine_search.search(conn, query_vector, k),
                "text": bm25_search.search(conn, query, k),
            }
    finally:
        if owns_client:
            client.close()

    # One row per chunk, whichever branch found it. Both branches select the
    # same columns, so either copy rehydrates a hit identically.
    by_id: dict[int, ChunkRow] = {
        row.ai_content_id: row for rows in branches.values() for row in rows
    }
    ranked = {name: [row.ai_content_id for row in rows] for name, rows in branches.items()}

    return [
        Hit(
            ai_content_id=f.key,
            post_id=by_id[f.key].post_id,
            post_title=by_id[f.key].post_title,
            chunk_index=by_id[f.key].chunk_index,
            chunk_text=by_id[f.key].chunk_text,
            score=f.score,
            vector_rank=f.ranks["vector"],
            text_rank=f.ranks["text"],
        )
        for f in fuse(ranked, k=settings.RRF_K)[:k]
    ]
