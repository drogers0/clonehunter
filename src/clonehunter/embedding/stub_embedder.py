from __future__ import annotations

import hashlib
import math

from clonehunter.core.types import Embedding, SnippetRef


class StubEmbedder:
    """Deterministic embedder for tests and local smoke runs."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, snippets: list[SnippetRef]) -> list[Embedding]:
        embeddings: list[Embedding] = []
        for snippet in snippets:
            digest = hashlib.sha256(snippet.text.encode("utf-8")).digest()
            values = [b / 255.0 for b in digest[: self._dim]]
            norm = math.sqrt(sum(v * v for v in values)) or 1.0
            normalized = [v / norm for v in values]
            embeddings.append(Embedding(vector=normalized, dim=self._dim))
        return embeddings
