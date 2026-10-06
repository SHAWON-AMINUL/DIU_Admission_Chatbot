"""Lexical branch: exact-term matching, so a query can hit on words the dense
model paraphrases away ("BDT 6,500", "Fall 2026", a program code).

NAMING, honestly: this is PostgreSQL full-text search ranked by ts_rank_cd,
not BM25. ts_rank_cd weights term density and proximity but has no document
length normalization and no IDF saturation, so it ranks differently from real
BM25. The file is named for the role it plays in the hybrid, not the formula.
Swapping in true BM25 means either the pg_search/ParadeDB extension or an
external index; the seam is this function's signature, which would not change.

KNOWN LIMITATION, accepted for the MVP: the 'english' text search
configuration does not stem or usefully tokenize Bangla, so for a Bangla query
this branch contributes ~nothing and retrieval degrades to vector-only.
BGE-M3's cross-lingual embeddings make that workable -- but it must not be
described as working hybrid search in Bangla.
"""

from collections.abc import Sequence

from app.rag.rows import ChunkRow


def search(conn, query: str, k: int) -> Sequence[ChunkRow]:
    rows = conn.execute(
        """
        SELECT c.id, c.post_id, p.title, c.chunk_index, c.chunk_text
          FROM ai_content c
          JOIN posts p ON p.id = c.post_id
         WHERE p.deleted_at IS NULL
           AND c.search_vector @@ websearch_to_tsquery('english', %s)
         ORDER BY ts_rank_cd(c.search_vector,
                             websearch_to_tsquery('english', %s)) DESC
         LIMIT %s
        """,
        (query, query, k),
    ).fetchall()
    return [ChunkRow(*r) for r in rows]
