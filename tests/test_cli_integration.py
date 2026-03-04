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


def test_cli_diff_sonarqube_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)
    subprocess.check_call(["git", "add", "a.py"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)

    (repo / "a.py").write_text("def a():\n    return 2\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["CLONEHUNTER_EMBEDDER"] = "stub"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clonehunter",
            "diff",
            ".",
            "--base",
            "HEAD",
            "--engine",
            "sonarqube",
            "--format",
            "json",
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SonarQube engine is not configured" in result.stderr


def test_cli_diff_outside_git_fails(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["CLONEHUNTER_EMBEDDER"] = "stub"

    workdir = tmp_path / "nogit"
    workdir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clonehunter",
            "diff",
            ".",
            "--base",
            "HEAD",
            "--format",
            "json",
            "--out",
            str(tmp_path / "out.json"),
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Failed to determine changed files" in result.stderr
