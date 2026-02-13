import sys
import types

import numpy as np
import pytest
from numpy.typing import NDArray

from clonehunter.core.types import Embedding
from clonehunter.index.faiss_index import FaissIndex


def test_faiss_index_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    FloatArray = NDArray[np.float32]
    IntArray = NDArray[np.int64]

    class DummyIndex:
        def __init__(self, dim: int) -> None:
            self._dim = dim
            self._vectors: FloatArray | None = None

        def add(self, mat: FloatArray) -> None:
            self._vectors = mat

        def search(self, vec: FloatArray, k: int) -> tuple[FloatArray, IntArray]:
            scores: FloatArray = np.array([[1.0 for _ in range(k)]], dtype=np.float32)
            indices: IntArray = np.array([[0 for _ in range(k)]], dtype=np.int64)
            return scores, indices

    class DummyIVF(DummyIndex):
        def __init__(self, quantizer: DummyIndex, dim: int, nlist: int, metric: int) -> None:
            super().__init__(dim)
            self.nlist = nlist
            self.metric = metric
            self.nprobe = 0

        def train(self, mat: FloatArray) -> None:
            _ = mat

    def _index_flat_ip(dim: int) -> DummyIndex:
        return DummyIndex(dim)

    def _index_ivf_flat(quantizer: DummyIndex, dim: int, nlist: int, metric: int) -> DummyIVF:
        return DummyIVF(quantizer, dim, nlist, metric)

    def _normalize(mat: FloatArray) -> FloatArray:
        return mat

    dummy_faiss = types.SimpleNamespace(
        IndexFlatIP=_index_flat_ip,
        IndexIVFFlat=_index_ivf_flat,
        METRIC_INNER_PRODUCT=0,
        normalize_L2=_normalize,
    )
    monkeypatch.setitem(sys.modules, "faiss", dummy_faiss)

    index = FaissIndex(nlist=1, nprobe=1)
    vectors = [Embedding(vector=[1.0, 0.0], dim=2)]
    index.build(vectors, ["a"])
    results = index.query(Embedding(vector=[1.0, 0.0], dim=2), 1)
    assert results[0][0] == "a"
