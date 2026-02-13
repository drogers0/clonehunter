from __future__ import annotations

import math

from clonehunter.core.types import Embedding
from clonehunter.model.interfaces import VectorIndex


class BruteIndex(VectorIndex):
    def __init__(self) -> None:
        self._vectors: dict[str, Embedding] = {}

    def build(self, vectors: list[Embedding], ids: list[str]) -> None:
        self._vectors = dict(zip(ids, vectors, strict=True))

    def query(self, vector: Embedding, k: int) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for id_, candidate in self._vectors.items():
            score = cosine_similarity(vector, candidate)
            scored.append((id_, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def cosine_similarity(a: Embedding, b: Embedding) -> float:
    if a.dim != b.dim:
        raise ValueError("Embedding dimensions do not match")
    dot = sum(x * y for x, y in zip(a.vector, b.vector, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a.vector)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b.vector)) or 1.0
    return dot / (norm_a * norm_b)
