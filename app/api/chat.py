"""Query-side endpoints: retrieval-only search, plus chat and its stream.

/search is retrieval only. /chat and /chat/stream run the full RAG chain and
return a grounded answer. None of the three endpoints holds retrieval or
generation logic of its own; each validates, calls its injected dependency,
and lets the response model (or SSE framing) shape the output.
"""

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import SearchFn, get_chat_fn, get_search, get_stream_fn
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    hit_from_document,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# Shown to the (unauthenticated, applicant-facing) client instead of the real
# exception, which can otherwise carry hosts, ports, usernames and internal
# paths. The real exception is always logged server-side via logger.exception.
_UNAVAILABLE_DETAIL = "The assistant is temporarily unavailable. Please try again."


@router.post("/search", response_model=SearchResponse)
def run_search(
    payload: SearchRequest,
    search: Annotated[SearchFn, Depends(get_search)],
) -> SearchResponse:
    hits = search(payload.query, top_k=payload.top_k)
    return SearchResponse(
        query=payload.query,
        hits=[SearchHit.model_validate(h, from_attributes=True) for h in hits],
    )


@router.post("/chat", response_model=ChatResponse)
async def run_chat(
    payload: ChatRequest,
    chat: Annotated[Callable, Depends(get_chat_fn)],
) -> ChatResponse:
    try:
        result = await chat(
            payload.question,
            [(message.role, message.content) for message in payload.history],
        )
    except Exception:
        # Same non-revealing detail as the stream path, so the two chat
        # endpoints fail the same way instead of one leaking internals and
        # the other staying silent about them.
        logger.exception("run_chat failed")
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from None
    return ChatResponse(
        answer=result["answer"],
        sources=[hit_from_document(doc) for doc in result["sources"]],
    )


def _sse(payload: dict) -> str:
    """One SSE frame. ensure_ascii=False keeps Bangla readable on the wire."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def run_chat_stream(
    payload: ChatRequest,
    stream: Annotated[Callable, Depends(get_stream_fn)],
) -> StreamingResponse:
    history = [(message.role, message.content) for message in payload.history]

    async def frames() -> AsyncIterator[str]:
        try:
            async for kind, data in stream(payload.question, history):
                if kind == "sources":
                    yield _sse(
                        {
                            "type": "sources",
                            "sources": [
                                hit_from_document(doc).model_dump() for doc in data
                            ],
                        }
                    )
                else:
                    yield _sse({"type": "token", "text": data})
        except Exception:
            # The 200 and its headers are already on the wire, so this cannot
            # become a 500. The client is told in-band or not at all -- but
            # the in-band detail must be a fixed, non-revealing string: the
            # real exception (host, port, DB user, internal paths) is only
            # ever logged server-side, never rendered to an unauthenticated,
            # applicant-facing page.
            logger.exception("run_chat_stream failed")
            yield _sse({"type": "error", "detail": _UNAVAILABLE_DETAIL})
            return
        yield _sse({"type": "done"})

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        # Without this an intervening proxy buffers the whole answer and
        # streaming silently degrades to a long pause then a wall of text.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
