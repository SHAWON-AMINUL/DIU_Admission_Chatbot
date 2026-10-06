"""Raw psycopg3 access.

Deliberately not SQLAlchemy: this slice touches four tables with a handful of
queries, and the schema is owned by the DB/Data teammate. Duplicating someone
else's table definitions in an ORM here would drift the moment they alter a
column -- which is already scheduled to happen.
"""

import atexit
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Teach this connection how to send/receive pgvector's vector type."""
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=8,
            configure=_configure,
            open=True,
        )
        # Without this, short-lived scripts exit while pool worker threads are
        # still alive and psycopg prints "couldn't stop thread" warnings.
        atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn():
    """Yield a pooled connection inside a transaction.

    Commits on clean exit, rolls back on exception.
    """
    with get_pool().connection() as conn:
        yield conn
