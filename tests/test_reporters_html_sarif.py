import json
from pathlib import Path

from clonehunter.core.config import Thresholds
from clonehunter.core.types import (
    CandidateMatch,
    FileRef,
    Finding,
    FunctionRef,
    ScanResult,
    ScanStats,
    SnippetRef,
)
from clonehunter.reporting.html_reporter import HtmlReporter
from clonehunter.reporting.sarif_reporter import SarifReporter
from clonehunter.similarity.rollup import rollup_findings


def _sample_result() -> ScanResult:
    file_a = FileRef(path="fixtures/tiny_repo/a.py", content_hash="h", language="python")
    file_b = FileRef(path="fixtures/tiny_repo/b.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file_a, qualified_name="a", start_line=1, end_line=2, code="pass", code_hash="a"
    )
    fn_b = FunctionRef(
        file=file_b, qualified_name="b", start_line=10, end_line=12, code="pass", code_hash="b"
    )
    snip = SnippetRef(
        kind="FUNC", function=fn_a, start_line=1, end_line=2, text="pass", snippet_hash="s"
    )
    match = CandidateMatch(snippet_a=snip, snippet_b=snip, similarity=1.0, evidence="")
    finding = Finding(
        function_a=fn_a,
        function_b=fn_b,
        score=1.0,
        duplicated_lines=2,
        evidence=[match],
        reasons=["func"],
        metadata={},
    )
    return ScanResult(
        findings=[finding],
        stats=ScanStats(0, 0, 0, 0, 1, 0, 0),
        config_snapshot={},
        timing={},
    )


def test_html_reporter(tmp_path: Path) -> None:
    result = _sample_result()
    out = tmp_path / "report.html"
    HtmlReporter().write(result, str(out))
    text = out.read_text(encoding="utf-8")
    assert "CloneHunter Report" in text
    assert "Schema:" in text
    assert "2 duplicated lines" in text
    assert "fixtures/tiny_repo/a.py:1-2" in text or "fixtures\\tiny_repo\\a.py:1-2" in text


def test_html_reporter_marks_hidden_duplicated_lines(tmp_path: Path) -> None:
    file_a = FileRef(path="fixtures/tiny_repo/a.py", content_hash="h", language="python")
    file_b = FileRef(path="fixtures/tiny_repo/b.py", content_hash="h", language="python")
    code_a = "\n".join(f"a{i}" for i in range(1, 61))
    code_b = "\n".join(f"b{i}" for i in range(1, 61))
    fn_a = FunctionRef(
        file=file_a, qualified_name="a", start_line=1, end_line=60, code=code_a, code_hash="a"
    )
    fn_b = FunctionRef(
        file=file_b, qualified_name="b", start_line=1, end_line=60, code=code_b, code_hash="b"
    )
    before_a = SnippetRef(
        kind="WIN", function=fn_a, start_line=1, end_line=10, text="pre", snippet_hash="ba"
    )
    before_b = SnippetRef(
        kind="WIN", function=fn_b, start_line=1, end_line=10, text="pre", snippet_hash="bb"
    )
    mid_a = SnippetRef(
        kind="WIN", function=fn_a, start_line=20, end_line=30, text="mid", snippet_hash="ma"
    )
    mid_b = SnippetRef(
        kind="WIN", function=fn_b, start_line=20, end_line=30, text="mid", snippet_hash="mb"
    )
    after_a = SnippetRef(
        kind="WIN", function=fn_a, start_line=40, end_line=50, text="post", snippet_hash="aa"
    )
    after_b = SnippetRef(
        kind="WIN", function=fn_b, start_line=40, end_line=50, text="post", snippet_hash="ab"
    )
    finding = Finding(
        function_a=fn_a,
        function_b=fn_b,
        score=0.95,
        duplicated_lines=32,
        evidence=[
            CandidateMatch(before_a, before_b, 0.91, ""),
            CandidateMatch(mid_a, mid_b, 0.99, ""),
            CandidateMatch(after_a, after_b, 0.92, ""),
        ],
        reasons=["min_window_hits"],
        metadata={},
    )
    result = ScanResult(
        findings=[finding],
        stats=ScanStats(0, 0, 0, 0, 1, 0, 0),
        config_snapshot={},
        timing={},
    )
    out = tmp_path / "report.html"
    HtmlReporter().write(result, str(out))
    text = out.read_text(encoding="utf-8")
    assert "fixtures/tiny_repo/a.py:1-50" in text
    assert "fixtures/tiny_repo/b.py:1-50" in text
    assert text.count("&lt;10 lines not shown&gt;") == 2
    assert text.count("&lt;11 lines not shown&gt;") == 2


def test_html_reporter_diff_numbers_survive_blank_line_stripping(tmp_path: Path) -> None:
    file_a = FileRef(path="a.py", content_hash="h", language="python")
    file_b = FileRef(path="b.py", content_hash="h", language="python")
    code = "line1\n\nline3\nline4"
    fn_a = FunctionRef(
        file=file_a, qualified_name="a", start_line=10, end_line=13, code=code, code_hash="a"
    )
    fn_b = FunctionRef(
        file=file_b, qualified_name="b", start_line=50, end_line=53, code=code, code_hash="b"
    )
    snip_a = SnippetRef(
        kind="FUNC", function=fn_a, start_line=10, end_line=13, text=code, snippet_hash="sa"
    )
    snip_b = SnippetRef(
        kind="FUNC", function=fn_b, start_line=50, end_line=53, text=code, snippet_hash="sb"
    )
    match = CandidateMatch(snippet_a=snip_a, snippet_b=snip_b, similarity=1.0, evidence="")
    finding = Finding(
        function_a=fn_a,
        function_b=fn_b,
        score=1.0,
        duplicated_lines=4,
        evidence=[match],
        reasons=["func"],
        metadata={},
    )
    result = ScanResult(
        findings=[finding], stats=ScanStats(0, 0, 0, 0, 1, 0, 0), config_snapshot={}, timing={}
    )
    out = tmp_path / "report.html"
    HtmlReporter().write(result, str(out))
    text = out.read_text(encoding="utf-8")
    # code is "line1\n\nline3\nline4" starting at line 10 (A) / 50 (B): the blank
    # line at offset 1 is stripped, so "line3" is truly the 3rd physical line ->
    # 12 / 52, not the post-strip position (11 / 51) a naive running-offset would
    # produce.
    # Anchor to the line-number cell so the assertions can't accidentally match
    # other rendered content. "line3" must be labeled 12 / 52 (its true source
    # position), never 11 / 51 (what a post-strip running offset would produce).
    assert ">12</td>" in text
    assert ">52</td>" in text
    assert ">11</td>" not in text
    assert ">51</td>" not in text


def test_html_reporter_self_clone_evidence_bounds_disjoint(tmp_path: Path) -> None:
    file = FileRef(path="app/service.py", content_hash="h", language="python")
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
    result = ScanResult(
        findings=findings,
        stats=ScanStats(0, 0, 0, 0, len(findings), 0, 0),
        config_snapshot={},
        timing={},
    )
    out = tmp_path / "report.html"
    HtmlReporter().write(result, str(out))
    text = out.read_text(encoding="utf-8")
    assert "app/service.py:1800-1825" in text
    assert "app/service.py:2100-2125" in text


def test_html_reporter_nway_chained_self_clone_bounds_disjoint(tmp_path: Path) -> None:
    # Three equal-length occurrences chained via a shared middle region, same self-clone
    # shape as the rollup regression test but sized so the header assertion is
    # unambiguous. best_match must deterministically prefer the higher-similarity
    # R1<->R2 pair (kind_rank and min_len tie between the two matches), independent of
    # evidence ordering, so the header shows the two occurrences of that pair -- not a
    # min/max envelope spanning all three occurrences.
    file = FileRef(path="app/nway.py", content_hash="h", language="python")
    fn = FunctionRef(
        file=file, qualified_name="f", start_line=1, end_line=3000, code="pass", code_hash="c"
    )
    r1 = SnippetRef(
        kind="WIN", function=fn, start_line=1000, end_line=1020, text="s", snippet_hash="r1"
    )
    r2 = SnippetRef(
        kind="WIN", function=fn, start_line=2000, end_line=2020, text="s", snippet_hash="r2"
    )
    r3 = SnippetRef(
        kind="WIN", function=fn, start_line=2900, end_line=2920, text="s", snippet_hash="r3"
    )
    r2b = SnippetRef(
        kind="WIN", function=fn, start_line=2005, end_line=2025, text="s", snippet_hash="r2b"
    )
    matches = [
        CandidateMatch(snippet_a=r1, snippet_b=r2, similarity=0.95, evidence=""),
        CandidateMatch(snippet_a=r3, snippet_b=r2b, similarity=0.90, evidence=""),
    ]
    findings = rollup_findings(
        matches,
        Thresholds(func=0.9, win=0.9, exp=0.9, min_window_hits=2, lexical_min_ratio=0.0),
    )
    assert len(findings) == 1
    result = ScanResult(
        findings=findings,
        stats=ScanStats(0, 0, 0, 0, len(findings), 0, 0),
        config_snapshot={},
        timing={},
    )
    out = tmp_path / "report.html"
    HtmlReporter().write(result, str(out))
    text = out.read_text(encoding="utf-8")
    assert "app/nway.py:1000-1020" in text
    assert "app/nway.py:2000-2025" in text
    assert "app/nway.py:1000-2025" not in text


def test_sarif_reporter(tmp_path: Path) -> None:
    result = _sample_result()
    out = tmp_path / "report.sarif"
    SarifReporter().write(result, str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("version") == "2.1.0"
    assert "runs" in payload
    assert "properties" in payload
    assert payload["runs"][0]["results"][0]["properties"]["duplicated_lines"] == 2
