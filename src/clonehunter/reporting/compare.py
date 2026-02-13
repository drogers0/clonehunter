from __future__ import annotations

from dataclasses import dataclass

from clonehunter.core.types import CandidateMatch
from clonehunter.similarity.ranking import best_match


@dataclass(frozen=True, slots=True)
class CompareData:
    kind_a: str
    kind_b: str
    span_a: dict[str, int]
    span_b: dict[str, int]
    similarity: float
    text_a: str
    text_b: str


def select_compare(matches: list[CandidateMatch]) -> CompareData | None:
    best = best_match(matches)
    if best is None:
        return None
    return CompareData(
        kind_a=best.snippet_a.kind,
        kind_b=best.snippet_b.kind,
        span_a={"start_line": best.snippet_a.start_line, "end_line": best.snippet_a.end_line},
        span_b={"start_line": best.snippet_b.start_line, "end_line": best.snippet_b.end_line},
        similarity=best.similarity,
        text_a=best.snippet_a.text,
        text_b=best.snippet_b.text,
    )
