from clonehunter.core.config import Thresholds
from clonehunter.core.types import CandidateMatch, FileRef, FunctionRef, SnippetRef
from clonehunter.similarity.rollup import rollup_findings


def test_rollup_min_window_hits():
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file, qualified_name="a", start_line=1, end_line=5, code="pass", code_hash="a"
    )
    fn_b = FunctionRef(
        file=file, qualified_name="b", start_line=10, end_line=14, code="pass", code_hash="b"
    )
    a1 = SnippetRef(
        kind="WIN", function=fn_a, start_line=1, end_line=3, text="a1", snippet_hash="a1"
    )
    b1 = SnippetRef(
        kind="WIN", function=fn_b, start_line=10, end_line=12, text="b1", snippet_hash="b1"
    )
    a2 = SnippetRef(
        kind="WIN", function=fn_a, start_line=2, end_line=4, text="a2", snippet_hash="a2"
    )
    b2 = SnippetRef(
        kind="WIN", function=fn_b, start_line=11, end_line=13, text="b2", snippet_hash="b2"
    )
    matches = [
        CandidateMatch(snippet_a=a1, snippet_b=b1, similarity=0.5, evidence=""),
        CandidateMatch(snippet_a=a2, snippet_b=b2, similarity=0.5, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0),
    )
    assert findings


def test_rollup_filters_overlapping_windows_same_function():
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=10, code="pass", code_hash="c"
    )
    a = SnippetRef(kind="WIN", function=fn, start_line=1, end_line=5, text="a", snippet_hash="a")
    b = SnippetRef(kind="WIN", function=fn, start_line=4, end_line=8, text="b", snippet_hash="b")
    match = CandidateMatch(snippet_a=a, snippet_b=b, similarity=1.0, evidence="")
    findings = rollup_findings([match], Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1))
    assert findings == []


def test_rollup_keeps_non_overlapping_windows_same_function() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=40, code="pass", code_hash="c"
    )
    a = SnippetRef(
        kind="WIN", function=fn, start_line=1, end_line=10, text="same", snippet_hash="a"
    )
    b = SnippetRef(
        kind="WIN", function=fn, start_line=21, end_line=30, text="same", snippet_hash="b"
    )
    match = CandidateMatch(snippet_a=a, snippet_b=b, similarity=1.0, evidence="")
    findings = rollup_findings(
        [match],
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1, lexical_min_ratio=0.0),
    )
    assert findings


def test_rollup_drops_identical_windows_same_function() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=30, code="pass", code_hash="c"
    )
    a = SnippetRef(kind="WIN", function=fn, start_line=5, end_line=24, text="a", snippet_hash="a")
    b = SnippetRef(kind="WIN", function=fn, start_line=5, end_line=24, text="b", snippet_hash="b")
    match = CandidateMatch(snippet_a=a, snippet_b=b, similarity=1.0, evidence="")
    findings = rollup_findings([match], Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1))
    assert findings == []


def test_rollup_drops_identical_func_self_match() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=10, code="pass", code_hash="c"
    )
    a = SnippetRef(kind="FUNC", function=fn, start_line=1, end_line=10, text="a", snippet_hash="a")
    b = SnippetRef(kind="FUNC", function=fn, start_line=1, end_line=10, text="b", snippet_hash="b")
    match = CandidateMatch(snippet_a=a, snippet_b=b, similarity=1.0, evidence="")
    findings = rollup_findings([match], Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1))
    assert findings == []


def test_rollup_drops_overlapping_func_win_same_function() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=30, code="pass", code_hash="c"
    )
    func = SnippetRef(
        kind="FUNC", function=fn, start_line=1, end_line=30, text="f", snippet_hash="f"
    )
    win = SnippetRef(kind="WIN", function=fn, start_line=5, end_line=24, text="w", snippet_hash="w")
    match = CandidateMatch(snippet_a=func, snippet_b=win, similarity=1.0, evidence="")
    findings = rollup_findings([match], Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1))
    assert findings == []


def test_rollup_drops_overlapping_expansions_same_function() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=60, code="pass", code_hash="c"
    )
    a = SnippetRef(kind="EXP", function=fn, start_line=10, end_line=35, text="a", snippet_hash="a")
    b = SnippetRef(kind="EXP", function=fn, start_line=20, end_line=45, text="b", snippet_hash="b")
    match = CandidateMatch(snippet_a=a, snippet_b=b, similarity=1.0, evidence="")
    findings = rollup_findings(
        [match],
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1, lexical_min_ratio=0.0),
    )
    assert findings == []


def test_rollup_drops_overlapping_functions_same_file() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_outer = FunctionRef(
        file=file, qualified_name="outer", start_line=1, end_line=40, code="pass", code_hash="o"
    )
    fn_inner = FunctionRef(
        file=file, qualified_name="inner", start_line=10, end_line=20, code="pass", code_hash="i"
    )
    outer = SnippetRef(
        kind="FUNC", function=fn_outer, start_line=1, end_line=40, text="o", snippet_hash="o"
    )
    inner = SnippetRef(
        kind="FUNC", function=fn_inner, start_line=10, end_line=20, text="i", snippet_hash="i"
    )
    match = CandidateMatch(snippet_a=outer, snippet_b=inner, similarity=1.0, evidence="")
    findings = rollup_findings(
        [match],
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1, lexical_min_ratio=0.0),
    )
    assert findings == []


def test_rollup_applies_lexical_filter() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file, qualified_name="a", start_line=1, end_line=5, code="pass", code_hash="a"
    )
    fn_b = FunctionRef(
        file=file, qualified_name="b", start_line=10, end_line=14, code="pass", code_hash="b"
    )
    a = SnippetRef(
        kind="WIN",
        function=fn_a,
        start_line=1,
        end_line=3,
        text="def alpha():\n    return 1",
        snippet_hash="a1",
    )
    b = SnippetRef(
        kind="WIN",
        function=fn_b,
        start_line=10,
        end_line=12,
        text="def beta():\n    return 2",
        snippet_hash="b1",
    )
    match = CandidateMatch(snippet_a=a, snippet_b=b, similarity=0.99, evidence="")
    findings = rollup_findings(
        [match],
        Thresholds(
            func=0.9,
            win=0.9,
            exp=0.9,
            min_window_hits=1,
            lexical_min_ratio=0.6,
        ),
    )
    assert findings == []


def test_rollup_distinct_functions_with_same_code_hash() -> None:
    file_a = FileRef(path="a.py", content_hash="h", language="python")
    file_b = FileRef(path="b.py", content_hash="h", language="python")
    file_c = FileRef(path="c.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file_a, qualified_name="a", start_line=1, end_line=2, code="pass", code_hash="same"
    )
    fn_b = FunctionRef(
        file=file_b, qualified_name="b", start_line=1, end_line=2, code="pass", code_hash="same"
    )
    fn_c = FunctionRef(
        file=file_c, qualified_name="c", start_line=1, end_line=2, code="pass", code_hash="same"
    )
    a = SnippetRef(
        kind="FUNC", function=fn_a, start_line=1, end_line=2, text="pass", snippet_hash="a"
    )
    b = SnippetRef(
        kind="FUNC", function=fn_b, start_line=1, end_line=2, text="pass", snippet_hash="b"
    )
    c = SnippetRef(
        kind="FUNC", function=fn_c, start_line=1, end_line=2, text="pass", snippet_hash="c"
    )
    matches = [
        CandidateMatch(snippet_a=a, snippet_b=b, similarity=0.99, evidence=""),
        CandidateMatch(snippet_a=a, snippet_b=c, similarity=0.99, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1, lexical_min_ratio=0.0),
    )
    assert len(findings) == 2


def test_rollup_normalizes_self_clone_orientation() -> None:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=3000, code="pass", code_hash="c"
    )
    # Two occurrences of the same duplicated block within one function, found by two
    # different sliding windows that queried each other in opposite directions.
    early_1 = SnippetRef(
        kind="WIN", function=fn, start_line=1800, end_line=1820, text="s", snippet_hash="w1"
    )
    late_1 = SnippetRef(
        kind="WIN", function=fn, start_line=2100, end_line=2120, text="s", snippet_hash="w2"
    )
    late_2 = SnippetRef(
        kind="WIN", function=fn, start_line=2105, end_line=2125, text="s", snippet_hash="w3"
    )
    early_2 = SnippetRef(
        kind="WIN", function=fn, start_line=1805, end_line=1825, text="s", snippet_hash="w4"
    )
    matches = [
        CandidateMatch(snippet_a=early_1, snippet_b=late_1, similarity=0.9, evidence=""),
        # Orientation flipped relative to the pair above -- candidate generation
        # queries each snippet independently, so this is realistic, not contrived.
        CandidateMatch(snippet_a=late_2, snippet_b=early_2, similarity=0.9, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0),
    )
    assert len(findings) == 1
    evidence = findings[0].evidence
    max_end_a = max(m.snippet_a.end_line for m in evidence)
    min_start_b = min(m.snippet_b.start_line for m in evidence)
    assert max_end_a < min_start_b, "side A and side B evidence bounds must be disjoint"


def test_rollup_nway_chained_self_clone_duplicated_lines() -> None:
    # Three occurrences of one duplicated block, chained by two pairwise matches that
    # share the long middle region R2 (R1<->R2 and R2<->R3, no direct R1<->R3 edge). Per
    # #22's per-pair canonicalization alone, R2 lands on side A of one match and side B
    # of the other, so the old per-side min/max double-counts it: min(covered_a=107,
    # covered_b=107) = 107. The occurrence-component fix must instead treat {R1, R2, R3}
    # as one connected component and count sum-max.
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=3000, code="pass", code_hash="c"
    )
    r1 = SnippetRef(
        kind="EXP", function=fn, start_line=1000, end_line=1005, text="s", snippet_hash="r1"
    )
    r2 = SnippetRef(
        kind="EXP", function=fn, start_line=2000, end_line=2100, text="s", snippet_hash="r2"
    )
    r3 = SnippetRef(
        kind="EXP", function=fn, start_line=2900, end_line=2905, text="s", snippet_hash="r3"
    )
    # Orientation-flipped variant of the second edge, and a slightly shifted middle span
    # (R2' overlaps R2) to exercise both canonicalization and occurrence merging.
    r2b = SnippetRef(
        kind="EXP", function=fn, start_line=2005, end_line=2105, text="s", snippet_hash="r2b"
    )
    matches = [
        CandidateMatch(snippet_a=r1, snippet_b=r2, similarity=0.95, evidence=""),
        CandidateMatch(snippet_a=r3, snippet_b=r2b, similarity=0.95, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=1, lexical_min_ratio=0.0),
    )
    assert len(findings) == 1
    # merged middle occurrence is 2000-2105 (106 lines): sum - max = (6 + 106 + 6) - 106 = 12
    assert findings[0].duplicated_lines == 12


def test_rollup_two_occurrence_self_clone_duplicated_lines_unchanged() -> None:
    # Guard: the clean 2-occurrence self-clone case (same fixture as
    # test_rollup_normalizes_self_clone_orientation) must produce the exact same
    # duplicated_lines value under the new occurrence-component path as the old
    # min(covered_a, covered_b) path: sum - max = (26 + 26) - 26 = 26.
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=3000, code="pass", code_hash="c"
    )
    early_1 = SnippetRef(
        kind="WIN", function=fn, start_line=1800, end_line=1820, text="s", snippet_hash="w1"
    )
    late_1 = SnippetRef(
        kind="WIN", function=fn, start_line=2100, end_line=2120, text="s", snippet_hash="w2"
    )
    late_2 = SnippetRef(
        kind="WIN", function=fn, start_line=2105, end_line=2125, text="s", snippet_hash="w3"
    )
    early_2 = SnippetRef(
        kind="WIN", function=fn, start_line=1805, end_line=1825, text="s", snippet_hash="w4"
    )
    matches = [
        CandidateMatch(snippet_a=early_1, snippet_b=late_1, similarity=0.9, evidence=""),
        CandidateMatch(snippet_a=late_2, snippet_b=early_2, similarity=0.9, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0),
    )
    assert len(findings) == 1
    assert findings[0].duplicated_lines == 26


def test_rollup_cross_function_multiwindow_duplicated_lines_unchanged() -> None:
    # Guard: cross-function findings never enter the self-clone occurrence-component
    # path (function identities differ), so duplicated_lines stays the existing
    # min(covered_a, covered_b) over independently-pooled per-side spans.
    # covered_a = len(1-10) + len(20-29) = 10 + 10 = 20 (disjoint, no merge).
    # covered_b = len(101-110) + len(120-139) = 10 + 20 = 30 (disjoint, no merge).
    # min(20, 30) = 20.
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file, qualified_name="a", start_line=1, end_line=50, code="pass", code_hash="ca"
    )
    fn_b = FunctionRef(
        file=file, qualified_name="b", start_line=100, end_line=150, code="pass", code_hash="cb"
    )
    a1 = SnippetRef(
        kind="WIN", function=fn_a, start_line=1, end_line=10, text="a1", snippet_hash="a1"
    )
    b1 = SnippetRef(
        kind="WIN", function=fn_b, start_line=101, end_line=110, text="b1", snippet_hash="b1"
    )
    a2 = SnippetRef(
        kind="WIN", function=fn_a, start_line=20, end_line=29, text="a2", snippet_hash="a2"
    )
    b2 = SnippetRef(
        kind="WIN", function=fn_b, start_line=120, end_line=139, text="b2", snippet_hash="b2"
    )
    matches = [
        CandidateMatch(snippet_a=a1, snippet_b=b1, similarity=0.95, evidence=""),
        CandidateMatch(snippet_a=a2, snippet_b=b2, similarity=0.95, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0),
    )
    assert len(findings) == 1
    assert findings[0].duplicated_lines == 20


def test_rollup_canonicalizes_cross_function_orientation() -> None:
    file = FileRef(path="z.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file, qualified_name="a_fn", start_line=1, end_line=20, code="pass", code_hash="ca"
    )
    fn_b = FunctionRef(
        file=file, qualified_name="b_fn", start_line=100, end_line=120, code="pass", code_hash="cb"
    )
    snip_a = SnippetRef(
        kind="FUNC", function=fn_a, start_line=1, end_line=20, text="s", snippet_hash="sa"
    )
    snip_b = SnippetRef(
        kind="FUNC", function=fn_b, start_line=100, end_line=120, text="s", snippet_hash="sb"
    )
    # Supplied B-before-A (the smaller identity "a_fn" is in snippet_b).
    matches = [CandidateMatch(snippet_a=snip_b, snippet_b=snip_a, similarity=0.95, evidence="")]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0),
    )
    assert len(findings) == 1
    assert findings[0].function_a.identity < findings[0].function_b.identity
