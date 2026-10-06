"""Reciprocal Rank Fusion over any number of ranked branches.

Pure: it takes ranked keys and returns ranked keys. It knows nothing about
Postgres, pgvector, embeddings or HTTP, which is what makes the fusion rule
itself testable and what lets a new branch (BM25, a reranker) join without
touching the retrieval code.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fused[K]:
    key: K
    score: float
    # Every branch name is present; None means that branch did not return it.
    # An absent key and a key ranked last are different facts about retrieval.
    ranks: Mapping[str, int | None]


def fuse[K](branches: Mapping[str, Sequence[K]], k: int) -> list[Fused[K]]:
    """Fuse ranked branches by RRF: score(d) = sum of 1 / (k + rank(d)).

    Each branch value is ordered best-first. Rank is 1-based.

    Ranks rather than raw scores, because the branches' scales (cosine
    distance vs ts_rank_cd) are not comparable and normalizing them against
    each other would be inventing a relationship that isn't there.
    """
    scores: dict[K, float] = {}
    ranks: dict[K, dict[str, int | None]] = {}

    for branch, ranked in branches.items():
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, dict.fromkeys(branches))[branch] = rank

    return [
        Fused(key=key, score=score, ranks=ranks[key])
        for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
