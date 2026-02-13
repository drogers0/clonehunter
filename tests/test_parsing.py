from pathlib import Path

from clonehunter.io.fs import collect_files
from clonehunter.parsing.python_ast import extract_functions


def test_extract_functions_from_fixture():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    names = {fn.qualified_name for fn in functions}
    assert "add_numbers" in names
    assert "sum_list" in names
    assert "wrapper" in names


def test_extract_functions_skips_invalid_syntax(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("def oops(:\n  pass\n", encoding="utf-8")
    files = collect_files([str(tmp_path)], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    assert functions == []
