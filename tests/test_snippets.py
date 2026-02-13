import pytest

from clonehunter.io.fs import collect_files
from clonehunter.parsing.python_ast import extract_functions
from clonehunter.snippets.generators import (
    WindowParams,
    generate_function_snippets,
    generate_window_snippets,
)


def test_generate_function_snippets():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    snippets = generate_function_snippets(functions)
    assert len(snippets) == len(functions)
    assert all(snippet.kind == "FUNC" for snippet in snippets)


def test_window_snippets_min_nonempty():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    params = WindowParams(window_lines=3, stride_lines=2, min_nonempty=2)
    snippets = generate_window_snippets(functions, params)
    assert all(snippet.kind == "WIN" for snippet in snippets)
    assert snippets


def test_window_snippets_stride_validation():
    files = collect_files(["fixtures/tiny_repo"], ["**/*.py"], [])
    functions = [fn for file in files for fn in extract_functions(file)]
    params = WindowParams(window_lines=3, stride_lines=0, min_nonempty=1)
    with pytest.raises(ValueError):
        generate_window_snippets(functions, params)
