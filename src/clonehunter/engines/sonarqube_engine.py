from __future__ import annotations

import json
import os
from pathlib import Path

from clonehunter.core.types import (
    CandidateMatch,
    FileRef,
    Finding,
    FunctionRef,
    ScanRequest,
    ScanResult,
    ScanStats,
    SnippetRef,
)
from clonehunter.io.fingerprints import hash_text
from clonehunter.model.interfaces import Engine


class SonarQubeEngine(Engine):
    def scan(self, request: ScanRequest) -> ScanResult:
        report_path = os.environ.get("CLONEHUNTER_SONAR_REPORT", "").strip()
        if not report_path:
            raise RuntimeError(
                "SonarQube engine is not configured. Set CLONEHUNTER_SONAR_REPORT to a JSON file."
            )
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        findings: list[Finding] = []
        for issue in payload.get("duplications", []):
            a = _to_function(issue.get("a"))
            b = _to_function(issue.get("b"))
            if a is None or b is None:
                continue
            snip = SnippetRef(
                kind="FUNC",
                function=a,
                start_line=a.start_line,
                end_line=a.end_line,
                text=a.code,
                snippet_hash=a.code_hash,
            )
            match = CandidateMatch(
                snippet_a=snip, snippet_b=snip, similarity=1.0, evidence="sonarqube"
            )
            findings.append(
                Finding(
                    function_a=a,
                    function_b=b,
                    score=1.0,
                    duplicated_lines=min(_span_len(a), _span_len(b)),
                    evidence=[match],
                    reasons=["sonarqube"],
                    metadata={},
                )
            )

        stats = ScanStats(
            file_count=0,
            function_count=0,
            snippet_count=0,
            candidate_count=0,
            finding_count=len(findings),
            cache_hits=0,
            cache_misses=0,
        )
        return ScanResult(findings=findings, stats=stats, config_snapshot={}, timing={})


def _to_function(data: dict[str, object] | None) -> FunctionRef | None:
    if not data:
        return None
    file_path = str(data.get("path", ""))
    start = _to_int(data.get("start", 1), 1)
    end = _to_int(data.get("end", start), start)
    code = str(data.get("code", ""))
    file_ref = FileRef(path=file_path, content_hash="", language="python")
    return FunctionRef(
        file=file_ref,
        qualified_name=str(data.get("name", file_path)),
        start_line=start,
        end_line=end,
        code=code,
        code_hash=hash_text(code),
    )


def _to_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _span_len(func: FunctionRef) -> int:
    return max(0, func.end_line - func.start_line + 1)
