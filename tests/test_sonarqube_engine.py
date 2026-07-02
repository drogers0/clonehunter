import json
from pathlib import Path

import pytest

from clonehunter.core.config import CloneHunterConfig
from clonehunter.engines.sonarqube_engine import ScanRequest, SonarQubeEngine


def test_sonarqube_engine_reads_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "duplications": [
            {
                "a": {"path": "a.py", "start": 1, "end": 2, "code": "pass", "name": "a"},
                "b": {"path": "b.py", "start": 1, "end": 2, "code": "pass", "name": "b"},
            }
        ]
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("CLONEHUNTER_SONAR_REPORT", str(report_path))

    engine = SonarQubeEngine()
    result = engine.scan(ScanRequest(paths=["."], config=CloneHunterConfig()))
    assert len(result.findings) == 1


def test_sonarqube_engine_pairs_evidence_with_b_not_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "duplications": [
            {
                "a": {
                    "path": "a.py",
                    "start": 1,
                    "end": 2,
                    "code": "code_a_body",
                    "name": "func_a",
                },
                "b": {
                    "path": "b.py",
                    "start": 10,
                    "end": 12,
                    "code": "code_b_body",
                    "name": "func_b",
                },
            }
        ]
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("CLONEHUNTER_SONAR_REPORT", str(report_path))

    engine = SonarQubeEngine()
    result = engine.scan(ScanRequest(paths=["."], config=CloneHunterConfig()))

    assert len(result.findings) == 1
    match = result.findings[0].evidence[0]
    assert match.snippet_a.function.file.path == "a.py"
    assert match.snippet_a.text == "code_a_body"
    assert match.snippet_b.function.file.path == "b.py"
    assert match.snippet_b.text == "code_b_body"
    assert match.snippet_b.start_line == 10
    assert match.snippet_b.end_line == 12
