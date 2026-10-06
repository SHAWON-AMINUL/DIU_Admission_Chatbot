"""The row shape every retrieval branch returns.

Both branches select the same five columns, so this is what lets the fusion
step treat them interchangeably -- and it replaces positional r[0]..r[4]
indexing, where inserting a column silently shifts every field downstream.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRow:
    ai_content_id: int
    post_id: int
    post_title: str
    chunk_index: int
    chunk_text: str
