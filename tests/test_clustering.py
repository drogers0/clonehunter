from clonehunter.core.types import CandidateMatch, FileRef, Finding, FunctionRef, SnippetRef
from clonehunter.similarity.clustering import cluster_findings, filter_clusters


def _finding(a: str, b: str) -> Finding:
    file_a = FileRef(path=f"{a}.py", content_hash="h", language="python")
    file_b = FileRef(path=f"{b}.py", content_hash="h", language="python")
    fn_a = FunctionRef(
        file=file_a, qualified_name=a, start_line=1, end_line=2, code="pass", code_hash=a
    )
    fn_b = FunctionRef(
        file=file_b, qualified_name=b, start_line=1, end_line=2, code="pass", code_hash=b
    )
    snip = SnippetRef(
        kind="FUNC", function=fn_a, start_line=1, end_line=2, text="pass", snippet_hash=a
    )
    match = CandidateMatch(snippet_a=snip, snippet_b=snip, similarity=1.0, evidence="")
    return Finding(
        function_a=fn_a,
        function_b=fn_b,
        score=1.0,
        duplicated_lines=2,
        evidence=[match],
        reasons=["func"],
        metadata={},
    )


def test_cluster_min_size_filter():
    findings = [_finding("a", "b"), _finding("a", "b")]
    clustered = cluster_findings(findings)
    filtered = filter_clusters(clustered, min_size=2)
    assert len(filtered) == len(clustered)
