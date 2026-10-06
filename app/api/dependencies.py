"""What the endpoints ask FastAPI for.

Every dependency here is overridable via `app.dependency_overrides`, which is
the whole point: it is what lets the endpoint tests run against an in-memory
repository with Postgres switched off, while production wiring stays in one
readable place.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

from app.database.connection import get_conn
from app.database.repositories import PostRepository


class SearchFn(Protocol):
    """Retrieve chunks for a query, best first."""

    def __call__(self, query: str, top_k: int) -> Sequence: ...


def get_search() -> Callable:
    """The hybrid retriever.

    Imported lazily so that importing the app does not pull in numpy, pgvector
    and an HTTP client -- and so a test can replace retrieval without any of
    them being reachable.
    """
    from app.rag.retriever import search

    return search


def get_post_repository() -> Iterator[PostRepository]:
    """One connection and one transaction per request.

    The connection is returned to the pool when the request ends, so an
    endpoint can never leak one by forgetting to close it.
    """
    with get_conn() as conn:
        yield PostRepository(conn)


def get_chat_fn() -> Callable:
    """The non-streaming RAG flow. Overridable, like get_search."""
    from app.rag.chain import achat

    return achat


def get_stream_fn() -> Callable:
    """The streaming RAG flow."""
    from app.rag.chain import astream_chat

    return astream_chat
