"""Presents the existing hybrid retriever to LangChain as a BaseRetriever.

An adapter, deliberately: the dense branch, the lexical branch and the RRF
fusion are already tested and already work, and langchain-postgres' PGVector
could not read this schema anyway -- it owns its own tables, and the schema
here belongs to another teammate.

Nothing in app/rag/retriever.py changes. This module only reshapes the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config import settings

if TYPE_CHECKING:
    from app.rag.retriever import Hit


def _default_search(query: str, top_k: int) -> Sequence:
    """The actual search import is deferred until invoke() is called.

    This allows importing HybridRetriever without pulling in numpy, psycopg,
    pgvector, or httpx. Combined with the TYPE_CHECKING guard on Hit, it
    makes unit tests runnable with no Postgres, embedding service, or Ollama.
    """
    from app.rag.retriever import search

    return search(query, top_k=top_k)


class HybridRetriever(BaseRetriever):
    """The project's hybrid search, wearing a LangChain interface."""

    # A field rather than a module-level import, so a test can substitute
    # retrieval wholesale without Postgres or the embedding service.
    search_fn: Callable = _default_search
    top_k: int = Field(default_factory=lambda: settings.GENERATION_TOP_K)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                page_content=hit.chunk_text,
                # vector_rank/text_rank travel with the chunk on purpose: they
                # are the existing answer to "why did this chunk come back?",
                # and an answer is exactly where that question gets asked.
                metadata={
                    "ai_content_id": hit.ai_content_id,
                    "post_id": hit.post_id,
                    "post_title": hit.post_title,
                    "chunk_index": hit.chunk_index,
                    "score": hit.score,
                    "vector_rank": hit.vector_rank,
                    "text_rank": hit.text_rank,
                },
            )
            for hit in self.search_fn(query, top_k=self.top_k)
        ]
