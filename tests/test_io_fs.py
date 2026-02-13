from pathlib import Path

import pytest

from clonehunter.io.fs import collect_files


def test_collect_files_excludes_venv(tmp_path: Path):
    root = tmp_path / "repo"
    venv = root / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "skip.py").write_text("print('skip')", encoding="utf-8")
    (root / "keep.py").write_text("print('keep')", encoding="utf-8")

    files = collect_files(
        [str(root)],
        include_globs=["**/*.py"],
        exclude_globs=["**/.venv/**", "**/site-packages/**"],
    )
    paths = {Path(f.path).name for f in files}
    assert "keep.py" in paths
    assert "skip.py" not in paths


def test_collect_files_explicit_path_respects_excludes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "repo"
    target = root / "src" / "keep.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('keep')", encoding="utf-8")

    monkeypatch.chdir(root)
    files = collect_files(
        ["src/keep.py"],
        include_globs=["**/*.py"],
        exclude_globs=["src/**"],
    )
    assert files == []


def test_collect_files_handles_non_utf8(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_bytes(b"\xff\xfe\xfa")
    files = collect_files([str(tmp_path)], include_globs=["**/*.py"], exclude_globs=[])
    assert len(files) == 1


def test_collect_files_non_python_are_text_language(tmp_path: Path):
    js = tmp_path / "sample.js"
    js.write_text("const x = 1;\n", encoding="utf-8")
    files = collect_files([str(tmp_path)], include_globs=["**/*.js"], exclude_globs=[])
    assert len(files) == 1
    assert files[0].language == "text"
