"""Wire shapes for the query side: search, chat, and chat's SSE stream.

SearchRequest/SearchResponse are the retrieval-only contract for /api/search.
ChatMessage/ChatRequest/ChatResponse are the contract for /api/chat and
/api/chat/stream, which run the full RAG chain and return a grounded answer
alongside the SearchHit list it was grounded on. hit_from_document is the one
Document -> SearchHit conversion both the search and chat endpoints use, so
the citations an applicant sees are always exactly the chunks retrieval
returned.

Before generation was wired in, this module held only the search shapes, on
the reasoning that putting an LLM in front of retrieval before retrieval was
measurably good would only hide retrieval problems behind fluent prose. That
retrieval-only slice shipped and was validated first; generation was added
once retrieval was trusted on its own.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    post_id: int
    post_title: str
    chunk_index: int
    chunk_text: str
    score: float
    # Which branch found this, and at what position. None means that branch
    # missed it entirely -- the first thing to look at when a result is wrong.
    vector_rank: int | None = None
    text_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Bounded because history is interpolated into the prompt verbatim.
    history: list[ChatMessage] = Field(
        default_factory=list, max_length=settings.MAX_HISTORY_TURNS * 2
    )


class ChatResponse(BaseModel):
    answer: str
    # SearchHit reused rather than a parallel model: the fields are identical,
    # and two shapes for one concept drift the moment either endpoint changes.
    sources: list[SearchHit]


def hit_from_document(doc) -> SearchHit:
    """The wire shape for one retrieved chunk.

    Lives here, not in app/rag/, because app/rag/ must not import the wire
    contract -- keeping that arrow one-way is what lets retrieval be tested
    without FastAPI and the endpoints be tested without retrieval.
    """
    return SearchHit(
        post_id=doc.metadata["post_id"],
        post_title=doc.metadata["post_title"],
        chunk_index=doc.metadata["chunk_index"],
        chunk_text=doc.page_content,
        score=doc.metadata["score"],
        vector_rank=doc.metadata.get("vector_rank"),
        text_rank=doc.metadata.get("text_rank"),
    )
