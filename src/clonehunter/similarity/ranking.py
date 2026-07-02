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

    def _rank(match: CandidateMatch) -> tuple[int, int, float, int, int, int, int]:
        len_a = span_len(match.snippet_a)
        len_b = span_len(match.snippet_b)
        # Total, order-independent tie-break: on equal (kind_rank, min_len, similarity),
        # prefer the lexicographically smallest snippet span (negated so max() picks it).
        return (
            kind_rank(match),
            min(len_a, len_b),
            match.similarity,
            -match.snippet_a.start_line,
            -match.snippet_a.end_line,
            -match.snippet_b.start_line,
            -match.snippet_b.end_line,
        )

    return max(matches, key=_rank)
