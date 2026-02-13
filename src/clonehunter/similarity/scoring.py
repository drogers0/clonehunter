from __future__ import annotations

from clonehunter.core.types import CandidateMatch


def best_score(matches: list[CandidateMatch]) -> float:
    if not matches:
        return 0.0
    return max(match.similarity for match in matches)
