from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from clonehunter.core.types import Embedding


@dataclass(frozen=True, slots=True)
class EmbeddingCache:
    root: str

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_")
        return Path(os.path.expanduser(self.root)) / f"{safe}.json"

    def get_many(self, keys: Iterable[str]) -> dict[str, Embedding]:
        results: dict[str, Embedding] = {}
        for key in keys:
            path = self._path(key)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            results[key] = Embedding(vector=payload["vector"], dim=payload["dim"])
        return results

    def set_many(self, items: dict[str, Embedding]) -> None:
        root = Path(os.path.expanduser(self.root))
        root.mkdir(parents=True, exist_ok=True)
        for key, emb in items.items():
            path = self._path(key)
            path.write_text(
                json.dumps({"vector": list(emb.vector), "dim": emb.dim}), encoding="utf-8"
            )
