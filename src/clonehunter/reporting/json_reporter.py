from __future__ import annotations

import difflib
import json
from dataclasses import asdict

from clonehunter.core.types import CandidateMatch, Finding, FunctionRef, ScanResult
from clonehunter.reporting.compare import CompareData, select_compare
from clonehunter.reporting.schema import SCHEMA_VERSION


class JsonReporter:
    def write(self, result: ScanResult, out_path: str) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "findings": [_serialize_finding(finding) for finding in result.findings],
            "stats": asdict(result.stats),
            "config": result.config_snapshot,
            "timing": result.timing,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _serialize_finding(finding: Finding) -> dict[str, object]:
    function_a = _serialize_function(finding.function_a)
    function_b = _serialize_function(finding.function_b)
    return {
        "function_a": function_a,
        "function_b": function_b,
        "score": finding.score,
        "duplicated_lines": finding.duplicated_lines,
        "compare": _select_compare(finding.evidence),
        "reasons": finding.reasons,
        "metadata": finding.metadata,
    }


def _serialize_function(func: FunctionRef) -> dict[str, object]:
    file_ref = func.file
    return {
        "file": {
            "path": file_ref.path,
            "content_hash": file_ref.content_hash,
            "language": file_ref.language,
        },
        "qualified_name": func.qualified_name,
        "start_line": func.start_line,
        "end_line": func.end_line,
        "code_hash": func.code_hash,
    }


def _serialize_evidence(compare: CompareData) -> dict[str, object]:
    return {
        "kind_a": compare.kind_a,
        "kind_b": compare.kind_b,
        "span_a": compare.span_a,
        "span_b": compare.span_b,
        "similarity": compare.similarity,
        "diff": _diff_text(compare.text_a, compare.text_b),
    }


def _select_compare(matches: list[CandidateMatch]) -> dict[str, object] | None:
    compare = select_compare(matches)
    if compare is None:
        return None
    return _serialize_evidence(compare)


def _diff_text(text_a: str, text_b: str) -> str:
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()
    diff = list(difflib.unified_diff(lines_a, lines_b, lineterm="", n=3))
    return _truncate_diff(diff, max_lines=80, max_chars=4000)


def _truncate_diff(lines: list[str], max_lines: int, max_chars: int) -> str:
    if not lines:
        return ""
    if len(lines) <= max_lines and sum(len(line) for line in lines) <= max_chars:
        return "\n".join(lines)
    trimmed = lines[:max_lines]
    text = "\n".join(trimmed)
    if len(text) > max_chars:
        text = text[:max_chars]
    return f"{text}\n... diff truncated ..."
