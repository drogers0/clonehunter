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
