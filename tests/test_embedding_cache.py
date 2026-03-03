import json
import math
from pathlib import Path

from clonehunter.core.types import Embedding
from clonehunter.embedding.cache import EmbeddingCache


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(str(tmp_path))
    key = "abc"
    emb = Embedding(vector=[0.1, 0.2, 0.3], dim=3)
    cache.set_many({key: emb})
    loaded = cache.get_many([key])
    assert key in loaded
    assert loaded[key].dim == emb.dim
    # float32 BLOB storage means slight precision change vs float64
    for a, b in zip(loaded[key].vector, emb.vector, strict=True):
        assert math.isclose(a, b, rel_tol=1e-6)
    cache.close()


def test_embedding_cache_overwrite(tmp_path: Path) -> None:
    cache = EmbeddingCache(str(tmp_path))
    key = "k1"
    vec1 = Embedding(vector=[1.0, 2.0], dim=2)
    vec2 = Embedding(vector=[3.0, 4.0], dim=2)
    cache.set_many({key: vec1})
    cache.set_many({key: vec2})
    loaded = cache.get_many([key])
    assert loaded[key].dim == 2
    for a, b in zip(loaded[key].vector, vec2.vector, strict=True):
        assert math.isclose(a, b, rel_tol=1e-6)
    cache.close()


def test_embedding_cache_missing_keys(tmp_path: Path) -> None:
    cache = EmbeddingCache(str(tmp_path))
    cache.set_many({"a": Embedding(vector=[1.0], dim=1)})
    loaded = cache.get_many(["a", "b", "c"])
    assert "a" in loaded
    assert "b" not in loaded
    assert "c" not in loaded
    cache.close()


def test_embedding_cache_empty_keys(tmp_path: Path) -> None:
    cache = EmbeddingCache(str(tmp_path))
    loaded = cache.get_many([])
    assert loaded == {}
    cache.close()


def test_embedding_cache_batch_write_read(tmp_path: Path) -> None:
    cache = EmbeddingCache(str(tmp_path))
    items = {f"key_{i}": Embedding(vector=[float(i), float(i + 1)], dim=2) for i in range(50)}
    cache.set_many(items)
    loaded = cache.get_many(items.keys())
    assert len(loaded) == 50
    for k, emb in items.items():
        assert k in loaded
        for a, b in zip(loaded[k].vector, emb.vector, strict=True):
            assert math.isclose(a, b, rel_tol=1e-6)
    cache.close()


def test_embedding_cache_json_migration(tmp_path: Path) -> None:
    """Old JSON cache files should be lazily migrated into SQLite."""
    key = "abc123"
    safe_key = key.replace("/", "_")
    json_path = tmp_path / f"{safe_key}.json"
    json_path.write_text(json.dumps({"vector": [0.5, 0.25], "dim": 2}), encoding="utf-8")

    cache = EmbeddingCache(str(tmp_path))
    loaded = cache.get_many([key])
    assert key in loaded
    assert loaded[key].dim == 2
    for a, b in zip(loaded[key].vector, [0.5, 0.25], strict=True):
        assert math.isclose(a, b, rel_tol=1e-6)

    # Second read should come from SQLite, not JSON
    cache2 = EmbeddingCache(str(tmp_path))
    loaded2 = cache2.get_many([key])
    assert key in loaded2
    cache.close()
    cache2.close()


def test_embedding_cache_schema_version_mismatch(tmp_path: Path) -> None:
    """A DB with a different schema version should be renamed and recreated."""
    import sqlite3

    db_path = tmp_path / "embeddings.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 999")
    conn.execute("CREATE TABLE dummy (id INTEGER)")
    conn.commit()
    conn.close()

    cache = EmbeddingCache(str(tmp_path))
    # Should have recreated the DB; old one should be backed up
    cache.set_many({"test": Embedding(vector=[1.0], dim=1)})
    loaded = cache.get_many(["test"])
    assert "test" in loaded
    cache.close()

    # Verify the old DB was renamed
    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1
