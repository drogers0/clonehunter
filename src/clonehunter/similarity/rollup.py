from __future__ import annotations

from collections import defaultdict

from clonehunter.core.config import Thresholds
from clonehunter.core.types import CandidateMatch, Finding
from clonehunter.similarity.lexical import lexical_similarity
from clonehunter.similarity.ranking import kind_rank
from clonehunter.similarity.scoring import best_score


def rollup_findings(matches: list[CandidateMatch], thresholds: Thresholds) -> list[Finding]:
    grouped: dict[tuple[str, str], list[CandidateMatch]] = defaultdict(list)
    filtered = _filter_overlapping_functions(matches)
    filtered = _filter_overlapping_windows(filtered)
    filtered = _filter_lexical_matches(filtered, thresholds.lexical_min_ratio)
    filtered = _dedupe_matches(filtered)
    for match in filtered:
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


def _filter_overlapping_windows(matches: list[CandidateMatch]) -> list[CandidateMatch]:
    filtered: list[CandidateMatch] = []
    for match in matches:
        a = match.snippet_a
        b = match.snippet_b
        if a.function.identity == b.function.identity:
            same_span = a.start_line == b.start_line and a.end_line == b.end_line
            if same_span:
                continue
            overlap = _overlap_len(a.start_line, a.end_line, b.start_line, b.end_line)
            if overlap:
                # Drop overlapping self-matches across different snippet kinds (e.g., FUNC vs WIN).
                if a.kind != b.kind:
                    continue
                # Drop any overlapping windows within the same function.
                if a.kind == "WIN":
                    continue
        filtered.append(match)
    return filtered


def _filter_overlapping_functions(matches: list[CandidateMatch]) -> list[CandidateMatch]:
    filtered: list[CandidateMatch] = []
    for match in matches:
        func_a = match.snippet_a.function
        func_b = match.snippet_b.function
        if func_a.file.path == func_b.file.path and func_a.identity != func_b.identity:
            overlap = _overlap_len(
                func_a.start_line, func_a.end_line, func_b.start_line, func_b.end_line
            )
            if overlap:
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
