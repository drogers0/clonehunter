from __future__ import annotations

import hashlib


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_cache_key(
    model_name: str, model_revision: str, max_tokens: int, snippet_hash: str
) -> str:
    payload = f"{model_name}:{model_revision}:{max_tokens}:{snippet_hash}"
    return hash_text(payload)
