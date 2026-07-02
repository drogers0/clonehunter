from __future__ import annotations

from clonehunter.core.types import CandidateMatch

Interval = tuple[int, int]


def is_self_clone(matches: list[CandidateMatch]) -> bool:
    if not matches:
        return False
    m = matches[0]
    return m.snippet_a.function.identity == m.snippet_b.function.identity


def _merge_overlapping(spans: set[Interval]) -> list[Interval]:
    # Merge only on strict overlap (next.start <= current.end), NOT adjacency, so two
    # touching-but-distinct occurrences on opposite sides are never collapsed.
    merged: list[Interval] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


class SelfCloneOccurrences:
    """Connected occurrence-components for a self-clone group. Order-independent.

    Pools every snippet span from both sides of a self-clone group's matches, merges
    strictly-overlapping spans into physical "occurrences", then unions the occurrences
    connected by a match into components (union-find). This resolves N-way chained
    self-clones (e.g. R1<->R2<->R3 via two pairwise matches sharing R2) into the correct
    global partition, instead of the two-sided min/max used for cross-function findings.
    """

    def __init__(self, matches: list[CandidateMatch]) -> None:
        spans = {(m.snippet_a.start_line, m.snippet_a.end_line) for m in matches}
        spans |= {(m.snippet_b.start_line, m.snippet_b.end_line) for m in matches}
        self._occ: list[Interval] = _merge_overlapping(spans)
        self._parent = list(range(len(self._occ)))
        for m in matches:
            self._union(
                self._index(m.snippet_a.start_line, m.snippet_a.end_line),
                self._index(m.snippet_b.start_line, m.snippet_b.end_line),
            )

    def _index(self, start: int, end: int) -> int:
        # Occurrences are sorted and disjoint; each raw span lies fully inside exactly one.
        # Note: if a stray window strictly overlaps two otherwise-distinct sub-regions, the
        # pool merges them into one occurrence -- a benign, cosmetic edge case (see plan).
        for i, (o_start, o_end) in enumerate(self._occ):
            if o_start <= start and end <= o_end:
                return i
        return -1  # unreachable for spans that produced the occurrences

    def _find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def _union(self, a: int, b: int) -> None:
        if a < 0 or b < 0:
            return
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def occurrence_for(self, start: int, end: int) -> Interval:
        i = self._index(start, end)
        return self._occ[i] if i >= 0 else (start, end)

    def duplicated_lines(self) -> int:
        components: dict[int, list[int]] = {}
        for i, (start, end) in enumerate(self._occ):
            components.setdefault(self._find(i), []).append(end - start + 1)
        return sum(sum(lens) - max(lens) for lens in components.values())
