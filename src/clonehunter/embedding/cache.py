from __future__ import annotations

import array as _array
import json
import os
import sqlite3
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

from clonehunter.core.types import Embedding

_SCHEMA_VERSION = 1
_CHUNK_SIZE = 900  # Safe below SQLite's default SQLITE_MAX_VARIABLE_NUMBER (999)


class EmbeddingCache:
    """SQLite-backed embedding cache with float32 BLOB storage."""

    def __init__(self, root: str) -> None:
        self.root = root
        root_path = Path(os.path.expanduser(root))
        root_path.mkdir(parents=True, exist_ok=True)
        self._db_path = root_path / "embeddings.sqlite3"
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")

            row = conn.execute("PRAGMA user_version").fetchone()
            version: int = row[0] if row is not None else 0

            if version == 0:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS embeddings "
                    "(key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
                )
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.commit()
            elif version != _SCHEMA_VERSION:
                conn.close()
                backup = self._db_path.with_suffix(f".v{version}-{int(time.time())}.bak")
                self._db_path.rename(backup)
                return self._open()

            return conn
        except sqlite3.DatabaseError:
            conn.close()
            backup = self._db_path.with_suffix(f".corrupt-{int(time.time())}.bak")
            if self._db_path.exists():
                self._db_path.rename(backup)
            return self._open()

    def _json_path(self, key: str) -> Path:
        """Path to a legacy JSON cache file for migration."""
        safe = key.replace("/", "_")
        return Path(os.path.expanduser(self.root)) / f"{safe}.json"

    def get_many(self, keys: Iterable[str]) -> dict[str, Embedding]:
        key_list = list(keys)
        if not key_list:
            return {}

        results: dict[str, Embedding] = {}
        conn = self._conn

        for i in range(0, len(key_list), _CHUNK_SIZE):
            chunk = key_list[i : i + _CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            cursor = conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({placeholders})",
                chunk,
            )
            for row in cursor:
                rkey: str = row[0]
                rdim: int = row[1]
                rblob: bytes = row[2]
                buf = _array.array("f")
                buf.frombytes(rblob)
                results[rkey] = Embedding(vector=buf.tolist(), dim=rdim)

        # Lazy migration: check old JSON files for cache misses
        missed = [k for k in key_list if k not in results]
        if missed:
            migrated = self._migrate_json(missed)
            results.update(migrated)

        return results

    def _migrate_json(self, keys: list[str]) -> dict[str, Embedding]:
        """Load from legacy JSON cache files and migrate into SQLite."""
        found: dict[str, Embedding] = {}
        for key in keys:
            json_path = self._json_path(key)
            if not json_path.exists():
                continue
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                vec = payload["vector"]
                dim = payload["dim"]
                if isinstance(vec, list) and isinstance(dim, int):
                    found[key] = Embedding(vector=cast(Sequence[float], vec), dim=dim)
            except Exception:
                continue
        if found:
            self.set_many(found)
        return found

    def set_many(self, items: dict[str, Embedding]) -> None:
        if not items:
            return
        rows: list[tuple[str, int, bytes]] = []
        for key, emb in items.items():
            buf = _array.array("f", emb.vector)
            rows.append((key, emb.dim, buf.tobytes()))

        self._conn.executemany(
            "INSERT INTO embeddings(key, dim, vec) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            rows,
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
