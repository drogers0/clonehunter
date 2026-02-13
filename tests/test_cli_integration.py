import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_sonarqube_error(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    cmd = [
        sys.executable,
        "-m",
        "clonehunter",
        "scan",
        "fixtures/tiny_repo",
        "--engine",
        "sonarqube",
        "--format",
        "json",
        "--out",
        str(tmp_path / "out.json"),
    ]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(cmd, env=env)
