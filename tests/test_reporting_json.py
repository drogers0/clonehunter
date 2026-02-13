import json
from pathlib import Path

from clonehunter.core.config import CloneHunterConfig, EmbedderConfig
from clonehunter.core.pipeline import run_pipeline
from clonehunter.reporting.json_reporter import JsonReporter


def test_json_reporter(tmp_path: Path) -> None:
    result = run_pipeline(
        ["fixtures/tiny_repo"], CloneHunterConfig(embedder=EmbedderConfig(name="stub"))
    )
    out = tmp_path / "report.json"
    JsonReporter().write(result, str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "schema_version" in payload
    assert "findings" in payload
    assert "stats" in payload
    if payload["findings"]:
        assert "duplicated_lines" in payload["findings"][0]
        compare = payload["findings"][0]["compare"]
        if compare:
            assert "diff" in compare
        assert "code" not in payload["findings"][0]["function_a"]
