from pathlib import Path

from clonehunter.core.types import Embedding
from clonehunter.embedding.cache import EmbeddingCache


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(str(tmp_path))
    key = "abc"
    emb = Embedding(vector=[0.1, 0.2], dim=2)
    cache.set_many({key: emb})
    loaded = cache.get_many([key])
    assert key in loaded
    assert loaded[key].vector == emb.vector
    assert loaded[key].dim == emb.dim
