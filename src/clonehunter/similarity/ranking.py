from __future__ import annotations

from clonehunter.core.types import CandidateMatch, SnippetRef


def span_len(snippet: SnippetRef) -> int:
    return max(0, snippet.end_line - snippet.start_line + 1)


def kind_rank(match: CandidateMatch) -> int:
    a = match.snippet_a.kind
    b = match.snippet_b.kind
    if a == "FUNC" and b == "FUNC":
        return 3
    if "FUNC" in (a, b):
        return 2
    if a == "WIN" and b == "WIN":
        return 1
    return 0


def best_match(matches: list[CandidateMatch]) -> CandidateMatch | None:
    if not matches:
        return None

    def _rank(match: CandidateMatch) -> tuple[int, int, float]:
        len_a = span_len(match.snippet_a)
        len_b = span_len(match.snippet_b)
        return (kind_rank(match), min(len_a, len_b), match.similarity)

    return max(matches, key=_rank)
