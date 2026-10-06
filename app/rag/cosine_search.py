"""Dense branch: nearest chunks by cosine distance over pgvector.

Filtered to one model_name. Vectors from two different embedding models share
the column but do not share a space -- comparing them returns confident
nonsense -- and the partial HNSW index in migrations/001_schema.sql is built
per model, so this filter is also what keeps the query on the index.
"""

from collections.abc import Sequence

import numpy as np

from app.config import settings
from app.rag.rows import ChunkRow


def search(conn, query_vector: np.ndarray, k: int) -> Sequence[ChunkRow]:
    rows = conn.execute(
        """
        SELECT c.id, c.post_id, p.title, c.chunk_index, c.chunk_text
          FROM ai_content_vectors v
          JOIN ai_content c ON c.id = v.ai_content_id
          JOIN posts p      ON p.id = c.post_id
         WHERE v.model_name = %s
           AND p.deleted_at IS NULL
         ORDER BY v.vector <=> %s
         LIMIT %s
        """,
        (settings.MODEL_NAME_TAG, query_vector, k),
    ).fetchall()
    return [ChunkRow(*r) for r in rows]
