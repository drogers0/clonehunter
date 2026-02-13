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
