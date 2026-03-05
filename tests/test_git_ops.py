import os
import subprocess
from pathlib import Path

import pytest

from clonehunter.io.git import GitError, changed_files


def test_changed_files_outside_git_raises(tmp_path: Path):
    path = tmp_path / "repo"
    path.mkdir()
    cwd = os.getcwd()
    try:
        os.chdir(path)
        with pytest.raises(GitError) as exc_info:
            changed_files("HEAD")
        assert "Not a git repository" in str(exc_info.value)
    finally:
        os.chdir(cwd)


def test_changed_files_includes_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)

    tracked = repo / "tracked.py"
    tracked.write_text("def keep():\n    return 1\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "tracked.py"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)

    (repo / "new_file.py").write_text("def new_file():\n    return 2\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        files = changed_files("HEAD")
    finally:
        os.chdir(cwd)

    assert "new_file.py" in files


def test_changed_files_respects_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)

    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)

    (repo / "src" / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (repo / "tests" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "src/a.py", "tests/b.py"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo)

    (repo / "src" / "a.py").write_text("def a():\n    return 10\n", encoding="utf-8")
    (repo / "tests" / "b.py").write_text("def b():\n    return 20\n", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        files = changed_files("HEAD", ["src"])
    finally:
        os.chdir(cwd)

    assert files == ["src/a.py"]
