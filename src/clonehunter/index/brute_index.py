from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from clonehunter.core.types import Embedding
from clonehunter.model.interfaces import VectorIndex


class BruteIndex(VectorIndex):
    def __init__(self) -> None:
        self._ids: list[str] = []
        # float32 matches typical embedding precision and halves memory vs float64
        self._matrix: NDArray[np.float32] | None = None  # (N, D), raw vectors
        self._norms: NDArray[np.float32] | None = None  # (N,), precomputed L2 norms

    def build(self, vectors: list[Embedding], ids: list[str]) -> None:
        self._ids = list(ids)
        if not vectors:
            self._matrix = None
            self._norms = None
            return
        self._matrix = np.asarray([v.vector for v in vectors], dtype=np.float32)
        norms = cast(NDArray[np.float32], np.linalg.norm(self._matrix, axis=1))
        self._norms = cast(NDArray[np.float32], np.where(norms == 0, 1.0, norms))

    def query(self, vector: Embedding, k: int) -> list[tuple[str, float]]:
        if self._matrix is None or self._norms is None or len(self._ids) == 0:
            return []
        q = np.asarray(vector.vector, dtype=np.float32)
        norm_q = float(np.linalg.norm(q))
        if norm_q == 0.0:
            norm_q = 1.0
        # dot(a, b) / (norm_a * norm_b) — matches original computation order
        dots = self._matrix @ q
        scores = dots / (self._norms * norm_q)
        # Stable descending sort to match Python's stable sort behavior
        top_idx = np.argsort(-scores, kind="stable")[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in top_idx]
