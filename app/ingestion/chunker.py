"""Chunking for admission content.

Strategy, in order:

1. Split the body on structural boundaries (blank lines / headings). Admission
   content is naturally sectioned -- "Eligibility", "Fees", "Deadlines" -- and
   a boundary-respecting split keeps a fee table from bleeding into a deadline.
2. Only if a section is still over the token budget, slide a window across it
   with overlap, so a fact spanning the cut survives in at least one chunk.
3. Prefix every chunk with the post title.

Step 3 matters more than the size tuning. A chunk reading "The fee is BDT
6,500 per credit" is nearly unretrievable for the query "CSE tuition", because
the program name lives in the title, not the chunk body. Prefixing puts it in
the embedded text.

Token counting is injected rather than done here. In production that is
BGE-M3's own tokenizer, reached through the embedding service: Bangla expands
to far more subword tokens per character than English, so a word count or
tiktoken would size Bangla chunks wrongly. Passing it in as a function is also
what lets every rule above be tested without a model or a network call.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.config import settings

# Counts tokens for a batch of texts, returning one count per text, in order.
# Batched rather than one-at-a-time because the real implementation is an HTTP
# round trip and a post can hold hundreds of sentences.
TokenCounter = Callable[[Sequence[str]], Sequence[int]]


@dataclass
class Chunk:
    chunk_index: int
    text: str
    token_count: int


# A blank line, or a line that looks like a heading ("## Fees", "Fees:").
_SECTION_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")  # '।' is the Bangla full stop


def _split_sections(body: str) -> list[str]:
    return [s.strip() for s in _SECTION_SPLIT.split(body or "") if s.strip()]


def _split_sentences(section: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(section) if s.strip()]


def _window_oversized(
    section: str,
    count_tokens: TokenCounter,
    target: int,
    overlap: int,
) -> list[str]:
    """Slide a sentence-aligned window across a section that is too big.

    Windows are built from whole sentences so chunks never start or end
    mid-sentence, which reads badly in a cited answer.
    """
    sentences = _split_sentences(section)
    if len(sentences) <= 1:
        # A single enormous sentence: nothing sentence-aligned to do.
        return [section]

    counts = count_tokens(sentences)

    windows: list[str] = []
    start = 0
    while start < len(sentences):
        total = 0
        end = start
        while end < len(sentences) and total + counts[end] <= target:
            total += counts[end]
            end += 1
        if end == start:  # one sentence alone exceeds target; take it whole
            end = start + 1

        windows.append(" ".join(sentences[start:end]))

        if end >= len(sentences):
            break

        # Step back far enough to carry ~`overlap` tokens into the next window.
        back = 0
        carried = 0
        while end - back - 1 > start and carried < overlap:
            carried += counts[end - back - 1]
            back += 1
        start = end - back

    return windows


def chunk_post(
    title: str,
    body: str,
    count_tokens: TokenCounter,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Split one post into embeddable chunks, each prefixed with the title."""
    target = target_tokens or settings.CHUNK_TARGET_TOKENS
    overlap = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    sections = _split_sections(body)
    if not sections:
        return []

    # Merge adjacent small sections up to the budget, and window the big ones.
    section_counts = count_tokens(sections)

    pieces: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for section, count in zip(sections, section_counts, strict=True):
        if count > target:
            if buffer:
                pieces.append("\n\n".join(buffer))
                buffer, buffer_tokens = [], 0
            pieces.extend(_window_oversized(section, count_tokens, target, overlap))
        elif buffer_tokens + count <= target:
            buffer.append(section)
            buffer_tokens += count
        else:
            pieces.append("\n\n".join(buffer))
            buffer, buffer_tokens = [section], count

    if buffer:
        pieces.append("\n\n".join(buffer))

    # Prefix the title, then measure what will actually be embedded.
    texts = [f"{title}\n\n{piece}" for piece in pieces]
    final_counts = count_tokens(texts)

    return [
        Chunk(chunk_index=i, text=text, token_count=n)
        for i, (text, n) in enumerate(zip(texts, final_counts, strict=True))
    ]
