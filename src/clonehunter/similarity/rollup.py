from __future__ import annotations

from collections import defaultdict

from clonehunter.core.config import Thresholds
from clonehunter.core.types import CandidateMatch, Finding, SnippetRef
from clonehunter.similarity.lexical import lexical_similarity
from clonehunter.similarity.occurrences import SelfCloneOccurrences, is_self_clone
from clonehunter.similarity.ranking import kind_rank
from clonehunter.similarity.scoring import best_score


def rollup_findings(matches: list[CandidateMatch], thresholds: Thresholds) -> list[Finding]:
    grouped: dict[tuple[str, str], list[CandidateMatch]] = defaultdict(list)
    filtered = _filter_overlapping_matches(matches)
    filtered = _filter_lexical_matches(filtered, thresholds.lexical_min_ratio)
    filtered = _dedupe_matches(filtered)
    for match in filtered:
        match = _normalize_orientation(match)
        key = _fn_pair_key(match)
        grouped[key].append(match)

    findings: list[Finding] = []
    for _key, group in grouped.items():
        func_a = group[0].snippet_a.function
        func_b = group[0].snippet_b.function
        score = best_score(group)
        reasons = _reasons(group, thresholds)
        if reasons:
            findings.append(
                Finding(
                    function_a=func_a,
                    function_b=func_b,
                    score=score,
                    duplicated_lines=_duplicated_lines(group),
                    evidence=group,
                    reasons=reasons,
                    metadata={},
                )
            )
    return findings


def _dedupe_matches(matches: list[CandidateMatch]) -> list[CandidateMatch]:
    # Remove symmetric duplicates and identical span pairs, collapsing across snippet kinds.
    # Keep the strongest similarity; on ties, prefer FUNC/FUNC over other kinds.
    best: dict[tuple[tuple[str, int, int], tuple[str, int, int]], CandidateMatch] = {}
    order: list[tuple[tuple[str, int, int], tuple[str, int, int]]] = []

    for match in matches:
        a = match.snippet_a
        b = match.snippet_b
        a_key = (a.function.identity, a.start_line, a.end_line)
        b_key = (b.function.identity, b.start_line, b.end_line)
        key = (a_key, b_key) if a_key <= b_key else (b_key, a_key)
        if key not in best:
            best[key] = match
            order.append(key)
        else:
            if match.similarity > best[key].similarity or (
                match.similarity == best[key].similarity and kind_rank(match) > kind_rank(best[key])
            ):
                best[key] = match
    return [best[key] for key in order]


def _fn_pair_key(match: CandidateMatch) -> tuple[str, str]:
    a = match.snippet_a.function.identity
    b = match.snippet_b.function.identity
    return (a, b) if a <= b else (b, a)


def _normalize_orientation(match: CandidateMatch) -> CandidateMatch:
    # Candidate generation queries each snippet independently, so a/b ordering
    # within a pair is arbitrary. Canonicalize it here -- the single place every
    # per-side aggregator (duplicated lines, evidence bounds, diff rendering)
    # branches from -- so all downstream consumers see a consistent split.
    # NOTE: this is per-pair, not a global 2-partition of the group; a self-clone
    # group with 3+ occurrences chained by pairwise matches would still over-report
    # per-side (a shared region on both sides) if aggregated with plain min/max. That
    # N-way case is now handled by connected-component clustering via
    # SelfCloneOccurrences, used by _duplicated_lines below and by the HTML reporter's
    # evidence bounds / hidden-line markers.
    a, b = match.snippet_a, match.snippet_b
    if _is_canonical_order(a, b):
        return match
    return CandidateMatch(
        snippet_a=b, snippet_b=a, similarity=match.similarity, evidence=match.evidence
    )


def _is_canonical_order(a: SnippetRef, b: SnippetRef) -> bool:
    if a.function.identity != b.function.identity:
        return a.function.identity < b.function.identity
    return a.start_line <= b.start_line


def _reasons(matches: list[CandidateMatch], thresholds: Thresholds) -> list[str]:
    func_hits = [m for m in matches if m.snippet_a.kind == "FUNC" or m.snippet_b.kind == "FUNC"]
    win_hits = [m for m in matches if m.snippet_a.kind == "WIN" or m.snippet_b.kind == "WIN"]
    exp_hits = [m for m in matches if m.snippet_a.kind == "EXP" or m.snippet_b.kind == "EXP"]

    reasons: list[str] = []
    if func_hits and best_score(func_hits) >= thresholds.func:
        reasons.append("func_threshold")
    if exp_hits and best_score(exp_hits) >= thresholds.exp:
        reasons.append("exp_threshold")
    if len(win_hits) >= thresholds.min_window_hits:
        reasons.append("min_window_hits")
    return reasons


def _filter_overlapping_matches(matches: list[CandidateMatch]) -> list[CandidateMatch]:
    filtered: list[CandidateMatch] = []
    for match in matches:
        snippet_a = match.snippet_a
        snippet_b = match.snippet_b
        func_a = match.snippet_a.function
        func_b = match.snippet_b.function

        # Allow self-clones only when the matched ranges in the function are disjoint.
        if func_a.identity == func_b.identity:
            snippet_overlap = _overlap_len(
                snippet_a.start_line, snippet_a.end_line, snippet_b.start_line, snippet_b.end_line
            )
            if snippet_overlap:
                continue
            filtered.append(match)
            continue

        # Functions that overlap in the same file represent structural containment, not duplication.
        if func_a.file.path == func_b.file.path:
            function_overlap = _overlap_len(
                func_a.start_line, func_a.end_line, func_b.start_line, func_b.end_line
            )
            if function_overlap:
                continue
        filtered.append(match)
    return filtered


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start > end:
        return 0
    return end - start + 1


def _filter_lexical_matches(
    matches: list[CandidateMatch], min_ratio: float
) -> list[CandidateMatch]:
    if min_ratio <= 0:
        return matches
    filtered: list[CandidateMatch] = []
    for match in matches:
        ratio = lexical_similarity(match.snippet_a.text, match.snippet_b.text)
        if ratio >= min_ratio:
            filtered.append(match)
    return filtered


def _duplicated_lines(matches: list[CandidateMatch]) -> int:
    if not matches:
        return 0
    if is_self_clone(matches):
        return SelfCloneOccurrences(matches).duplicated_lines()
    spans_a = [(m.snippet_a.start_line, m.snippet_a.end_line) for m in matches]
    spans_b = [(m.snippet_b.start_line, m.snippet_b.end_line) for m in matches]
    return min(_covered_lines(spans_a), _covered_lines(spans_b))


def _covered_lines(spans: list[tuple[int, int]]) -> int:
    if not spans:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
            continue
        if end > merged[-1][1]:
            merged[-1][1] = end
    return sum(end - start + 1 for start, end in merged)
