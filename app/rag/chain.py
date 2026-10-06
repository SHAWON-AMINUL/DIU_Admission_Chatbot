"""The RAG flow: classify, contextualize, retrieve, answer.

Written as one async generator rather than a single composed Runnable. Two
reasons, both practical: the "never call the model with no documents" rule is
plainly visible as an early return but awkward inside a chain of .assign()
calls, and streaming reads straight off .astream() instead of being recovered
by parsing astream_events output.

The pieces are still LCEL -- prompt | llm | parser -- and achat() is
implemented by consuming astream_chat(), so the streaming and non-streaming
endpoints cannot drift into answering differently.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from app.llm import get_chat_model
from app.rag import trace
from app.rag.diagnosis import diagnose
from app.rag.langchain_retriever import HybridRetriever
from app.rag.language import (
    ANSWER_LANGUAGE,
    REFUSALS,
    QuestionLanguage,
    classify_heuristic,
)
from app.rag.prompts import (
    ANSWER_PROMPT,
    CLASSIFY_PROMPT,
    CONTEXTUALIZE_PROMPT,
    LANGUAGE_RULES,
    format_docs,
    format_history,
)

_LABELS: tuple[QuestionLanguage, ...] = ("bangla_script", "banglish", "english")


def _log_diagnosis(chat_id: str | None, docs, answer: str, refusal: str, **kw) -> None:
    """Record why the cycle did not answer, or that it did.

    Always written, never only on failure: "no cause found" and "the
    diagnosis never ran" look identical in a log that stays silent on
    success.
    """
    causes = diagnose(docs, answer, refusal, **kw)
    trace.record(
        chat_id,
        "diagnosis",
        causes=", ".join(cause for cause, _ in causes) or "none",
        **{cause: evidence for cause, evidence in causes},
    )


def _elapsed(started: float) -> float:
    """Milliseconds since a perf_counter mark, rounded for the log."""
    return round((time.perf_counter() - started) * 1000, 1)


def _retrieval_score_summary(docs: Sequence[Any], *, branch: str | None = None) -> str:
    """Readable score and rank summary for a retrieval pass."""
    if branch == "vector":
        hits = [doc for doc in docs if doc.metadata.get("vector_rank") is not None]
    elif branch == "text":
        hits = [doc for doc in docs if doc.metadata.get("text_rank") is not None]
    else:
        hits = list(docs)

    if not hits:
        return "none"
    lines = []
    for i, doc in enumerate(hits, start=1):
        score = doc.metadata.get("score")
        vector_rank = doc.metadata.get("vector_rank")
        text_rank = doc.metadata.get("text_rank")
        lines.append(
            f"[{i}] post {doc.metadata.get('post_id')} "
            f"chunk {doc.metadata.get('chunk_index')} "
            f"score={score} vector_rank={vector_rank} text_rank={text_rank}"
        )
    return "\n".join(lines)


def _retrieval_chunk_summary(docs: Sequence[Any], *, branch: str) -> str:
    """Readable chunk text for one retrieval branch in the cycle log."""
    if branch == "vector":
        hits = [doc for doc in docs if doc.metadata.get("vector_rank") is not None]
    else:
        hits = [doc for doc in docs if doc.metadata.get("text_rank") is not None]

    if not hits:
        return "none"
    return "\n\n".join(
        f"[{i}] post {doc.metadata.get('post_id')} "
        f"chunk {doc.metadata.get('chunk_index')}\n{doc.page_content}"
        for i, doc in enumerate(hits, start=1)
    )


async def _timed(coro) -> tuple[Any, float]:
    """Await a coroutine and report how long it took, in milliseconds.

    Retrieval and classification run concurrently, so their stage lines are
    written after the gather in a fixed order rather than whenever each
    happens to finish -- a cycle log whose stage order shuffles run to run is
    not a cycle log.
    """
    started = time.perf_counter()
    result = await coro
    return result, round((time.perf_counter() - started) * 1000, 1)


async def _classify(question: str, llm: BaseChatModel) -> QuestionLanguage:
    """Lexicon first, model only when the lexicon has no evidence.

    The heuristic is free and covers the common cases; the model covers the
    Banglish vocabulary the lexicon has never seen. Guessing English on no
    evidence would silently reintroduce the bug this whole path exists for.
    """
    detected = classify_heuristic(question)
    if detected is not None:
        return detected

    raw = await (CLASSIFY_PROMPT | llm | StrOutputParser()).ainvoke(
        {"question": question}
    )
    lowered = raw.strip().lower()
    for label in _LABELS:
        if label in lowered:
            return label
    # A small model asked for one word sometimes returns a sentence. English is
    # the safe default: a wrong language is bad, an unhandled KeyError is a 500.
    return "english"


async def astream_chat(
    question: str,
    history: Sequence[tuple[str, str]] = (),
    *,
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | None = None,
    chat_id: str | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ("sources", docs) once, then ("token", text) repeatedly.

    With a chat_id, every stage of the cycle is appended to that chat's log
    (see app.rag.trace); without one nothing is written.
    """
    retriever = retriever if retriever is not None else HybridRetriever()
    llm = llm if llm is not None else get_chat_model()
    cycle_started = time.perf_counter()
    trace.record(chat_id, "question", question=question, history_turns=len(history))

    # Skipped entirely without history: it is a second model round-trip, and a
    # first question has nothing to be rewritten against.
    standalone = question
    if history:
        contextualize = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
        rewritten, ms = await _timed(
            contextualize.ainvoke(
                {"history": format_history(history), "question": question}
            )
        )
        standalone = rewritten.strip() or question
        # Logged because the chunks below were retrieved against this, not
        # against what the user typed: a bad retrieval on an elliptical
        # follow-up is inexplicable without the rewrite beside it.
        trace.record(chat_id, "contextualize", standalone=standalone, ms=ms)

    # Retrieval does not need the language and the language does not need
    # retrieval, so the classifier's round-trip hides inside time retrieval was
    # already spending rather than adding to it.
    #
    # The language is read from the user's own question, never from standalone
    # -- that is model output and may have drifted into another language.
    (docs, retrieve_ms), (kind, classify_ms) = await asyncio.gather(
        _timed(retriever.ainvoke(standalone)),
        _timed(_classify(question, llm)),
    )
    trace.record(chat_id, "classify", language=kind, ms=classify_ms)
    vector_hits = [doc for doc in docs if doc.metadata.get("vector_rank") is not None]
    text_hits = [doc for doc in docs if doc.metadata.get("text_rank") is not None]
    trace.record(
        chat_id,
        "vector_retrieval",
        docs=len(vector_hits),
        retrieval_score=_retrieval_score_summary(vector_hits, branch="vector"),
        chunks=_retrieval_chunk_summary(vector_hits, branch="vector"),
        ms=retrieve_ms,
    )
    trace.record(
        chat_id,
        "ts_retrieval",
        docs=len(text_hits),
        retrieval_score=_retrieval_score_summary(text_hits, branch="text"),
        chunks=_retrieval_chunk_summary(text_hits, branch="text"),
        ms=retrieve_ms,
    )
    trace.record(
        chat_id,
        "retrieve",
        docs=len(docs),
        # Titles, not just ids: a bad answer is usually a retrieval problem,
        # and "which posts were in front of the model" is the first thing
        # you need to see -- an id list makes you go look each one up.
        sources="\n".join(
            f"[{i}] post {doc.metadata.get('post_id')} "
            f"chunk {doc.metadata.get('chunk_index')} "
            f"-- {doc.metadata.get('post_title')}"
            for i, doc in enumerate(docs, start=1)
        ),
        retrieved_text="\n\n".join(
            f"[{i}] post {doc.metadata.get('post_id')} "
            f"chunk {doc.metadata.get('chunk_index')}\n{doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        ) or "none",
        retrieval_score=_retrieval_score_summary(docs),
        ms=retrieve_ms,
    )
    trace.record(
        chat_id,
        "reranking",
        docs=len(docs),
        best_retrieval_score=max(
            (doc.metadata.get("score") or 0.0 for doc in docs), default=0.0
        ),
        retrieval_score=_retrieval_score_summary(docs),
        ms=retrieve_ms,
    )
    yield "sources", docs

    language = ANSWER_LANGUAGE[kind]

    if not docs:
        # Logged as its own stage: a refusal with no model call is the system
        # working, and without this line it reads like a cycle that died.
        trace.record(chat_id, "refusal", language=language)
        _log_diagnosis(chat_id, docs, REFUSALS[language], REFUSALS[language])
        trace.record(chat_id, "done", ms=_elapsed(cycle_started))
        yield "token", REFUSALS[language]
        return

    # Not piped through StrOutputParser here: that would keep only .content
    # and silently drop the model's reasoning trace. Streaming the raw chunk
    # instead exposes additional_kwargs["reasoning_content"] -- populated only
    # when reasoning=True (see get_chat_model) -- as its own "thinking" event,
    # kept separate from "token" so the reasoning trace can never end up
    # concatenated into the answer text a caller assembles from this stream.
    answer = ANSWER_PROMPT | llm
    generate_started = time.perf_counter()
    answer_parts: list[str] = []
    thinking_chars = 0
    stop_reason = None
    try:
        async for chunk in answer.astream(
            {
                "context": format_docs(docs),
                "question": standalone,
                "language_rule": LANGUAGE_RULES[language],
                "refusal": REFUSALS[language],
            }
        ):
            stop_reason = chunk.response_metadata.get("done_reason") or stop_reason
            thinking = chunk.additional_kwargs.get("reasoning_content")
            if thinking:
                thinking_chars += len(thinking)
                yield "thinking", thinking
            if chunk.content:
                answer_parts.append(chunk.content)
                yield "token", chunk.content
    except Exception as exc:
        # A cycle that died mid-generation is the one worth reading, and the
        # endpoint only records that something failed, not where.
        trace.record(
            chat_id,
            "llm_output",
            thinking_chars=thinking_chars,
            ms=_elapsed(generate_started),
            error=f"{type(exc)._name_}: {exc}",
            answer="".join(answer_parts),
        )
        _log_diagnosis(
            chat_id,
            docs,
            "".join(answer_parts),
            REFUSALS[language],
            error=f"{type(exc)._name_}: {exc}",
        )
        trace.record(chat_id, "done", ms=_elapsed(cycle_started), failed=True)
        raise

    trace.record(
        chat_id,
        "llm_output",
        thinking_chars=thinking_chars,
        ms=_elapsed(generate_started),
        # Last field on purpose: it is the only multi-line one, so the block
        # stays scannable down the left edge above it.
        answer="".join(answer_parts),
    )
    _log_diagnosis(
        chat_id,
        docs,
        "".join(answer_parts),
        REFUSALS[language],
        truncated=stop_reason == "length",
    )
    trace.record(chat_id, "done", ms=_elapsed(cycle_started))


async def achat(
    question: str,
    history: Sequence[tuple[str, str]] = (),
    *,
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """The whole answer at once. Consumes astream_chat so there is one flow."""
    sources: list = []
    tokens: list[str] = []

    async for kind, payload in astream_chat(
        question, history, retriever=retriever, llm=llm, chat_id=chat_id
    ):
        if kind == "sources":
            sources = payload
        elif kind == "token":
            tokens.append(payload)
        # "thinking" is deliberately dropped here: achat()'s contract is the
        # final answer text, and mixing the reasoning trace into it would
        # reintroduce exactly the chain-of-thought-as-answer bug reasoning=True
        # exists to avoid.

    return {"answer": "".join(tokens), "sources": sources}