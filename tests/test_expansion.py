from pathlib import Path

from clonehunter.io.fs import collect_files
from clonehunter.parsing.python_ast import extract_functions
from clonehunter.snippets.expansion import ExpansionParams, expand_calls


def test_expansion_generates_snippet():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    snippets = expand_calls(functions, ExpansionParams(enabled=True, depth=1, max_chars=10000))
    assert any(snippet.kind == "EXP" for snippet in snippets)


def test_expansion_resolves_imports_and_methods():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    snippets = expand_calls(functions, ExpansionParams(enabled=True, depth=1, max_chars=10000))
    combined = "\n".join(snippet.text for snippet in snippets)
    assert "helper_sum" in combined
    assert "total(self, items)" in combined


def test_expansion_respects_max_chars():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    snippets = expand_calls(functions, ExpansionParams(enabled=True, depth=1, max_chars=1))
    assert snippets == []


def test_expansion_resolves_duplicate_module_basenames(tmp_path: Path) -> None:
    (tmp_path / "pkg1").mkdir()
    (tmp_path / "pkg2").mkdir()
    (tmp_path / "pkg1" / "util.py").write_text(
        "def helper():\n    return 'MARKER_PKG1'\n", encoding="utf-8"
    )
    (tmp_path / "pkg2" / "util.py").write_text(
        "def helper():\n    return 'MARKER_PKG2'\n", encoding="utf-8"
    )
    (tmp_path / "main1.py").write_text(
        "from pkg1.util import helper\n\n\ndef caller_one():\n    return helper()\n",
        encoding="utf-8",
    )
    (tmp_path / "main2.py").write_text(
        "from pkg2.util import helper\n\n\ndef caller_two():\n    return helper()\n",
        encoding="utf-8",
    )
    files = collect_files([str(tmp_path)], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    snippets = expand_calls(functions, ExpansionParams(enabled=True, depth=1, max_chars=10000))

    by_caller = {snippet.function.qualified_name: snippet.text for snippet in snippets}
    assert "MARKER_PKG1" in by_caller["caller_one"]
    assert "MARKER_PKG2" not in by_caller["caller_one"]
    assert "MARKER_PKG2" in by_caller["caller_two"]
    assert "MARKER_PKG1" not in by_caller["caller_two"]
