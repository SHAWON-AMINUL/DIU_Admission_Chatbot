"""Every SQL statement about documents, in one place.

Endpoints used to open connections, write SQL, unpack tuples by position and
raise HTTP errors -- four jobs in one function, none of them testable apart.
The repository takes an already-open connection and returns schema objects; it
raises no HTTPException, because whether a missing document is a 404 or a
retry is the caller's judgement, not the storage layer's.

Still psycopg rather than an ORM: the schema is owned by the DB/Data teammate
and is scheduled to change. Duplicating their table definitions in an ORM here
would drift the moment they alter a column. `dict_row` gets the safety that
actually mattered -- fields addressed by name, not by index.
"""

from psycopg.rows import dict_row

from app.schemas.document import ChunkOut, FailureOut, IngestStats, PostSummary

# Scalar subqueries rather than joins + GROUP BY: joining chunks and vectors
# together multiplies rows, and SUM(DISTINCT ...) to compensate would silently
# collapse two chunks that happen to have the same token count.
_SUMMARY_SQL = """
    SELECT p.id, p.title, p.body, p.status, p.approval_status,
           p.verify_status, p.fail_count, p.created_at, p.published_at,
           p.approved_at, p.verified_at,
           (SELECT COUNT(*) FROM ai_content c
             WHERE c.post_id = p.id) AS chunk_count,
           (SELECT COUNT(*) FROM ai_content_vectors v
              JOIN ai_content c2 ON c2.id = v.ai_content_id
             WHERE c2.post_id = p.id) AS vector_count,
           (SELECT COALESCE(SUM(c.token_count), 0) FROM ai_content c
             WHERE c.post_id = p.id) AS token_total
      FROM posts p
     WHERE p.deleted_at IS NULL
     ORDER BY p.id DESC
"""

_INGEST_FIELDS = ("chunk_count", "vector_count", "token_total")


def _to_summary(row: dict) -> PostSummary:
    stats = IngestStats(**{f: int(row[f]) for f in _INGEST_FIELDS})
    return PostSummary(
        **{k: v for k, v in row.items() if k not in _INGEST_FIELDS}, ingest=stats
    )


class PostRepository:
    """Reads and writes documents. Holds a connection; owns no transaction."""

    def __init__(self, conn):
        self._conn = conn

    def commit(self) -> None:
        """Commit now, rather than waiting for the request to end.

        Needed before queuing a `BackgroundTasks` job: FastAPI runs a
        `yield`-dependency's exit code (which is where this connection would
        otherwise commit) only after background tasks finish, so a task that
        opens its own connection would otherwise never see this request's
        writes.
        """
        self._conn.commit()

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        return self._conn.cursor(row_factory=dict_row).execute(sql, params).fetchall()

    def _row(self, sql: str, params: tuple = ()) -> dict | None:
        return self._conn.cursor(row_factory=dict_row).execute(sql, params).fetchone()

    # --- reads ------------------------------------------------------------

    def list_summaries(self) -> list[PostSummary]:
        return [_to_summary(r) for r in self._rows(_SUMMARY_SQL)]

    def approval_status(self, post_id: int) -> str | None:
        """None means no such live document -- distinct from 'pending'."""
        row = self._row(
            "SELECT approval_status FROM posts WHERE id = %s AND deleted_at IS NULL",
            (post_id,),
        )
        return row["approval_status"] if row else None

    def chunks(self, post_id: int) -> list[ChunkOut]:
        return [
            ChunkOut(**r)
            for r in self._rows(
                """
                SELECT c.id, c.chunk_index, c.chunk_text, c.token_count,
                       v.model_name, (v.id IS NOT NULL) AS has_vector
                  FROM ai_content c
                  LEFT JOIN ai_content_vectors v ON v.ai_content_id = c.id
                 WHERE c.post_id = %s
                 ORDER BY c.chunk_index
                """,
                (post_id,),
            )
        ]

    def failures(self, post_id: int) -> list[FailureOut]:
        return [
            FailureOut(reason=r["fail_reason"], chunk_index=r["fail_chunk_index"], at=r["fail_time"])
            for r in self._rows(
                """
                SELECT fail_reason, fail_chunk_index, fail_time
                  FROM fail_history
                 WHERE post_id = %s
                 ORDER BY fail_time DESC
                """,
                (post_id,),
            )
        ]

    # --- writes -----------------------------------------------------------

    def _any_user_id(self) -> int | None:
        """Stand-in for the authenticated user.

        There is no auth yet and `users` has no role column, so "who approved
        this" cannot actually be answered. Every write below records the first
        user in the table. This is a placeholder for the Backend teammate's
        auth, not an access-control decision -- see docs/schema-change-requests.md.
        """
        row = self._row("SELECT id FROM users ORDER BY id LIMIT 1")
        return row["id"] if row else None

    def create(self, title: str, body: str, category_ids: list[int]) -> int | None:
        author = self._any_user_id()
        if author is None:
            return None

        post_id = self._row(
            """
            INSERT INTO posts (title, body, created_by, status)
            VALUES (%s, %s, %s, 'Draft')
            RETURNING id
            """,
            (title, body, author),
        )["id"]

        for category_id in category_ids:
            self._conn.execute(
                """
                INSERT INTO category_post (category_id, post_id, created_by)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (category_id, post_id, author),
            )
        return post_id

    def publish(self, post_id: int) -> bool:
        return self._row(
            """
            UPDATE posts SET status = 'published', published_at = NOW()
             WHERE id = %s AND deleted_at IS NULL
             RETURNING id
            """,
            (post_id,),
        ) is not None

    def approve(self, post_id: int) -> bool:
        return self._row(
            """
            UPDATE posts
               SET approval_status = 'approved', approved_at = NOW(), approved_by = %s
             WHERE id = %s AND deleted_at IS NULL
             RETURNING id
            """,
            (self._any_user_id(), post_id),
        ) is not None

    def mark_verified(self, post_id: int) -> bool:
        return self._row(
            """
            UPDATE posts
               SET status = 'verified', verify_status = 'verified',
                   verified_at = NOW(), verified_by = %s
             WHERE id = %s AND deleted_at IS NULL
             RETURNING id
            """,
            (self._any_user_id(), post_id),
        ) is not None
