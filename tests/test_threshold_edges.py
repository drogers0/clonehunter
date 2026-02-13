from clonehunter.core.config import Thresholds
from clonehunter.core.types import CandidateMatch, FileRef, FunctionRef, SnippetKind, SnippetRef
from clonehunter.similarity.rollup import rollup_findings


def _match(kind: SnippetKind, sim: float, a_start: int = 1, b_start: int = 10) -> CandidateMatch:
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file,
        qualified_name="a",
        start_line=a_start,
        end_line=a_start + 1,
        code="pass",
        code_hash="a",
    )
    fn_b = FunctionRef(
        file=file,
        qualified_name="b",
        start_line=b_start,
        end_line=b_start + 2,
        code="pass",
        code_hash="b",
    )
    a = SnippetRef(
        kind=kind,
        function=fn_a,
        start_line=a_start,
        end_line=a_start + 1,
        text="a",
        snippet_hash=f"a{a_start}",
    )
    b = SnippetRef(
        kind=kind,
        function=fn_b,
        start_line=b_start,
        end_line=b_start + 2,
        text="b",
        snippet_hash=f"b{b_start}",
    )
    return CandidateMatch(snippet_a=a, snippet_b=b, similarity=sim, evidence="")


def test_func_threshold_edge():
    thresholds = Thresholds(func=0.95, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0)
    at = rollup_findings([_match("FUNC", 0.95)], thresholds)
    below = rollup_findings([_match("FUNC", 0.9499)], thresholds)
    assert at
    assert not below


def test_win_threshold_edge():
    thresholds = Thresholds(func=0.95, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0)
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file, qualified_name="a", start_line=1, end_line=20, code="pass", code_hash="a"
    )
    fn_b = FunctionRef(
        file=file, qualified_name="b", start_line=30, end_line=50, code="pass", code_hash="b"
    )
    a1 = SnippetRef(
        kind="WIN", function=fn_a, start_line=1, end_line=3, text="a1", snippet_hash="a1"
    )
    b1 = SnippetRef(
        kind="WIN", function=fn_b, start_line=30, end_line=32, text="b1", snippet_hash="b1"
    )
    a2 = SnippetRef(
        kind="WIN", function=fn_a, start_line=4, end_line=6, text="a2", snippet_hash="a2"
    )
    b2 = SnippetRef(
        kind="WIN", function=fn_b, start_line=33, end_line=35, text="b2", snippet_hash="b2"
    )
    at = rollup_findings(
        [
            CandidateMatch(snippet_a=a1, snippet_b=b1, similarity=0.9, evidence=""),
            CandidateMatch(snippet_a=a2, snippet_b=b2, similarity=0.9, evidence=""),
        ],
        thresholds,
    )
    below = rollup_findings(
        [
            CandidateMatch(snippet_a=a1, snippet_b=b1, similarity=0.8999, evidence=""),
            CandidateMatch(snippet_a=a2, snippet_b=b2, similarity=0.8999, evidence=""),
        ],
        thresholds,
    )
    assert at
    assert below


def test_exp_threshold_edge():
    thresholds = Thresholds(func=0.95, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0)
    at = rollup_findings([_match("EXP", 0.9)], thresholds)
    below = rollup_findings([_match("EXP", 0.8999)], thresholds)
    assert at
    assert not below


def test_min_window_hits_edge():
    thresholds = Thresholds(func=0.95, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0)
    file = FileRef(path="x.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file, qualified_name="a", start_line=1, end_line=20, code="pass", code_hash="a"
    )
    fn_b = FunctionRef(
        file=file, qualified_name="b", start_line=30, end_line=50, code="pass", code_hash="b"
    )
    a1 = SnippetRef(
        kind="WIN", function=fn_a, start_line=1, end_line=3, text="a1", snippet_hash="a1"
    )
    b1 = SnippetRef(
        kind="WIN", function=fn_b, start_line=30, end_line=32, text="b1", snippet_hash="b1"
    )
    a2 = SnippetRef(
        kind="WIN", function=fn_a, start_line=4, end_line=6, text="a2", snippet_hash="a2"
    )
    b2 = SnippetRef(
        kind="WIN", function=fn_b, start_line=33, end_line=35, text="b2", snippet_hash="b2"
    )
    m1 = CandidateMatch(snippet_a=a1, snippet_b=b1, similarity=0.95, evidence="")
    m2 = CandidateMatch(snippet_a=a2, snippet_b=b2, similarity=0.95, evidence="")
    findings = rollup_findings([m1, m2], thresholds)
    assert findings
