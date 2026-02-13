from __future__ import annotations

import re


def lexical_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
