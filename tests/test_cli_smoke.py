import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_scan_json(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m",
        "clonehunter",
        "scan",
        "fixtures/tiny_repo",
        "--format",
        "json",
        "--out",
        str(out),
    ]
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["CLONEHUNTER_EMBEDDER"] = "stub"
    subprocess.check_call(cmd, env=env)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "findings" in payload
