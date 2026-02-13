import json
from pathlib import Path

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


def test_sarif_reporter(tmp_path: Path) -> None:
    result = _sample_result()
    out = tmp_path / "report.sarif"
    SarifReporter().write(result, str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("version") == "2.1.0"
    assert "runs" in payload
    assert "properties" in payload
    assert payload["runs"][0]["results"][0]["properties"]["duplicated_lines"] == 2
