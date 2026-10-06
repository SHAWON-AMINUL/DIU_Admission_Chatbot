"""The ingest pipeline.

Implements steps 5-8 of the agreed workflow, triggered once a post has been
approved (director level) and verified:

    5. status -> 'AI training in progress'
    6. body is chunked                        -> ai_content
    7. each chunk is embedded                 -> ai_content_vectors
    8. success -> 'AI training complete'
       failure -> fail_history row, fail_count += 1, 'AI training failed'

Scope notes, agreed and deliberate:
  * Attachments (post_files) are NOT ingested. Body text only.
  * No retry/resume/re-ingest-on-edit logic. A failed run is parked for a
    human. This is an MVP demonstration pipeline.
  * The one exception is the DELETE below: re-running on the same post must
    not duplicate chunks.
"""

import hashlib
import logging
from datetime import UTC, datetime

import numpy as np

from app.database.connection import get_conn
from app.embeddings.client import EmbeddingClient
from app.ingestion.chunker import chunk_post

log = logging.getLogger("ingestion")

# A post is eligible for ingest only when it has cleared BOTH gates.
ELIGIBLE_SQL = """
    approval_status = 'approved'
    AND verify_status = 'verified'
    AND deleted_at IS NULL
"""


class IngestError(Exception):
    """Raised with the chunk index that failed, where one is known."""

    def __init__(self, message: str, chunk_index: int | None = None):
        super().__init__(message)
        self.chunk_index = chunk_index


def _set_status(conn, post_id: int, status: str, **extra) -> None:
    sets = ["status = %s"]
    params: list = [status]
    for column, value in extra.items():
        sets.append(f"{column} = %s")
        params.append(value)
    params.append(post_id)
    conn.execute(f"UPDATE posts SET {', '.join(sets)} WHERE id = %s", params)


def _record_failure(post_id: int, reason: str, chunk_index: int | None) -> None:
    """Failure bookkeeping runs in its own transaction.

    The work transaction has already rolled back by the time we get here, so
    this must not ride on it or the failure record would roll back too.

    Never raises: a failure while recording a failure must not mask the
    original error.
    """
    try:
        with get_conn() as conn:
            # A nonexistent post_id would violate the FK below, so check first.
            if conn.execute(
                "SELECT 1 FROM posts WHERE id = %s", (post_id,)
            ).fetchone() is None:
                log.error("cannot record failure: post %s does not exist", post_id)
                return

            conn.execute(
                """
                INSERT INTO fail_history (post_id, fail_reason, fail_chunk_index)
                VALUES (%s, %s, %s)
                """,
                (post_id, reason[:2000], chunk_index),
            )
            conn.execute(
                """
                UPDATE posts
                   SET status = 'AI training failed',
                       fail_count = fail_count + 1,
                       claimed_at = NULL
                 WHERE id = %s
                """,
                (post_id,),
            )
    except Exception:
        log.exception("could not record failure for post %s", post_id)


def ingest_post(post_id: int, client: EmbeddingClient | None = None) -> dict:
    """Run the full pipeline for one post. Returns a small result summary."""
    owns_client = client is None
    client = client or EmbeddingClient()

    try:
        # --- Load and gate ------------------------------------------------
        with get_conn() as conn:
            row = conn.execute(
                f"""
                SELECT id, title, body, status, approval_status, verify_status
                  FROM posts
                 WHERE id = %s AND {ELIGIBLE_SQL}
                 FOR UPDATE
                """,
                (post_id,),
            ).fetchone()

            if row is None:
                raise IngestError(
                    f"post {post_id} is not eligible: it must be approved, "
                    "verified, and not deleted"
                )

            _, title, body, *_ = row
            if not (body or "").strip():
                raise IngestError(f"post {post_id} has an empty body")

            _set_status(
                conn,
                post_id,
                "AI training in progress",
                claimed_at=datetime.now(UTC),
            )

        # --- Chunk --------------------------------------------------------
        chunks = chunk_post(title=title, body=body, count_tokens=client.token_counts)
        if not chunks:
            raise IngestError(f"post {post_id} produced no chunks")

        # --- Embed --------------------------------------------------------
        # One call for the whole post: batching happens inside the service.
        try:
            model_tag, vectors = client.embed([c.text for c in chunks])
        except Exception as exc:
            raise IngestError(f"embedding failed: {exc}") from exc

        if len(vectors) != len(chunks):
            raise IngestError(
                f"embedding service returned {len(vectors)} vectors "
                f"for {len(chunks)} chunks"
            )

        # --- Store --------------------------------------------------------
        # One transaction: chunks and vectors land together or not at all.
        with get_conn() as conn:
            # Idempotency. Without this, re-running the pipeline on a post
            # silently doubles its chunks -- which shows up in a demo as the
            # bot repeating itself. ai_content_vectors cascades from here.
            conn.execute("DELETE FROM ai_content WHERE post_id = %s", (post_id,))

            for chunk, vector in zip(chunks, vectors, strict=True):
                content_id = conn.execute(
                    """
                    INSERT INTO ai_content (post_id, chunk_text, chunk_index, token_count)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (post_id, chunk.text, chunk.chunk_index, chunk.token_count),
                ).fetchone()[0]

                conn.execute(
                    """
                    INSERT INTO ai_content_vectors (ai_content_id, model_name, vector)
                    VALUES (%s, %s, %s)
                    """,
                    # np.array, not the raw JSON list: psycopg would otherwise
                    # try to adapt a Python list as a float8[] array, which the
                    # vector column rejects.
                    (content_id, model_tag, np.asarray(vector, dtype=np.float32)),
                )

            digest = hashlib.sha256((body or "").encode("utf-8")).hexdigest()
            _set_status(
                conn,
                post_id,
                "AI training complete",
                claimed_at=None,
                content_hash=digest,
            )

        log.info("post %s ingested: %s chunks", post_id, len(chunks))
        return {
            "post_id": post_id,
            "status": "AI training complete",
            "chunks": len(chunks),
            "model_name": model_tag,
            "tokens": sum(c.token_count for c in chunks),
        }

    except IngestError as exc:
        log.warning("post %s failed: %s", post_id, exc)
        _record_failure(post_id, str(exc), exc.chunk_index)
        return {"post_id": post_id, "status": "AI training failed", "error": str(exc)}

    except Exception as exc:  # unexpected: still must not leave the row claimed
        log.exception("post %s failed unexpectedly", post_id)
        _record_failure(post_id, f"unexpected: {exc}", None)
        return {"post_id": post_id, "status": "AI training failed", "error": str(exc)}

    finally:
        if owns_client:
            client.close()


def reclaim_stale(timeout_minutes: int = 30) -> list[int]:
    """Reset posts stuck in 'AI training in progress' past their lease.

    A worker that is killed mid-run (crash, Ctrl-C, machine restart) leaves
    its post claimed forever. Without this, that post is invisible to every
    later run: it is neither complete nor eligible.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            UPDATE posts
               SET status = 'AI training failed',
                   claimed_at = NULL
             WHERE status = 'AI training in progress'
               AND claimed_at < NOW() - make_interval(mins => %s)
             RETURNING id
            """,
            (timeout_minutes,),
        ).fetchall()
    ids = [r[0] for r in rows]
    if ids:
        log.warning("reclaimed %s stale post(s): %s", len(ids), ids)
    return ids


def eligible_post_ids() -> list[int]:
    """Approved + verified posts that have not been ingested yet."""
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT id FROM posts
             WHERE {ELIGIBLE_SQL}
               AND status NOT IN ('AI training complete', 'AI training in progress')
             ORDER BY id
            """
        ).fetchall()
    return [r[0] for r in rows]


def ingest_all_eligible() -> list[dict]:
    client = EmbeddingClient()
    try:
        return [ingest_post(pid, client=client) for pid in eligible_post_ids()]
    finally:
        client.close()
