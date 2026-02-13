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
