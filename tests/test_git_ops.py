import os
from pathlib import Path

from clonehunter.io.git import changed_files


def test_changed_files_outside_git(tmp_path: Path):
    path = tmp_path / "repo"
    path.mkdir()
    cwd = os.getcwd()
    try:
        os.chdir(path)
        files = changed_files("HEAD")
        assert isinstance(files, list)
        assert files == []
    finally:
        os.chdir(cwd)
