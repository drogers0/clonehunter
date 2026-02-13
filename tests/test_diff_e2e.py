import json
import os
import subprocess
import sys
from pathlib import Path


def test_diff_end_to_end(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "b.py").write_text(
        """

def add_copy(values):
    total = 0
    for n in values:
        total += n
    return total
""",
        encoding="utf-8",
    )
    (repo / "a.py").write_text(
        """

def add(nums):
    total = 0
    for n in nums:
        total += n
    return total
""",
        encoding="utf-8",
    )

    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)
    subprocess.check_call(["git", "add", "a.py", "b.py"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)

    (repo / "a.py").write_text(
        """

def add(nums):
    total = 0
    for n in nums:
        total += n
    return total


def add_copy(values):
    total = 0
    for n in values:
        total += n
    return total
""",
        encoding="utf-8",
    )

    out = tmp_path / "diff.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["CLONEHUNTER_EMBEDDER"] = "stub"
    cmd = [
        sys.executable,
        "-m",
        "clonehunter",
        "diff",
        "--base",
        "HEAD",
        "--format",
        "json",
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd, cwd=repo, env=env)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "findings" in payload
    assert any(
        "b.py" in finding["function_a"]["file"]["path"]
        or "b.py" in finding["function_b"]["file"]["path"]
        for finding in payload["findings"]
    )
