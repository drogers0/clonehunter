from __future__ import annotations

from typing import Any, cast

import numpy as np

from clonehunter.core.types import Embedding
from clonehunter.model.interfaces import VectorIndex


class FaissIndex(VectorIndex):
    def __init__(self, nlist: int = 128, nprobe: int = 8) -> None:
        try:
            import faiss  # type: ignore  # DO NOT SKIP TEST
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss is not installed; install faiss-cpu to use FaissIndex"
            ) from exc
        self._faiss: Any = faiss
        self._index: Any | None = None
        self._ids: list[str] = []
        self._nlist = nlist
        self._nprobe = nprobe

    def build(self, vectors: list[Embedding], ids: list[str]) -> None:
        if not vectors:
            self._index = None
            self._ids = []
            return
        dim = vectors[0].dim
        mat = np.array([v.vector for v in vectors], dtype=np.float32)
        self._faiss.normalize_L2(mat)

        if len(vectors) < self._nlist:
            index = self._faiss.IndexFlatIP(dim)
            index.add(mat)
        else:
            quantizer = self._faiss.IndexFlatIP(dim)
            index = self._faiss.IndexIVFFlat(
                quantizer, dim, self._nlist, self._faiss.METRIC_INNER_PRODUCT
            )
            index.train(mat)
            index.add(mat)
            index.nprobe = self._nprobe

        self._index = index
        self._ids = list(ids)

    def query(self, vector: Embedding, k: int) -> list[tuple[str, float]]:
        if self._index is None:
            return []
        vec = np.array([vector.vector], dtype=np.float32)
        self._faiss.normalize_L2(vec)
        distances, indices = self._index.search(vec, k)
        indices_list = cast(list[int], indices[0].tolist())
        distances_list = cast(list[float], distances[0].tolist())
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices_list, distances_list, strict=True):
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append((self._ids[idx], float(score)))
        return results
