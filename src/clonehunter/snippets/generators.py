from __future__ import annotations

from dataclasses import dataclass

from clonehunter.core.types import FunctionRef, SnippetRef
from clonehunter.io.fingerprints import hash_text
from clonehunter.snippets.normalization import normalize_source


@dataclass(frozen=True, slots=True)
class WindowParams:
    window_lines: int
    stride_lines: int
    min_nonempty: int


def _make_snippet(function: FunctionRef, start: int, end: int, kind: str) -> SnippetRef:
    lines = function.code.splitlines()
    snippet_text = "\n".join(lines[start - 1 : end])
    normalized = normalize_source(snippet_text)
    snippet_hash = hash_text(
        f"{kind}:{function.file.path}:{function.start_line}:{function.end_line}:{function.code_hash}:{start}:{end}:{normalized}"
    )
    return SnippetRef(
        kind=kind,  # type: ignore[arg-type]
        function=function,
        start_line=function.start_line + start - 1,
        end_line=function.start_line + end - 1,
        text=normalized,
        snippet_hash=snippet_hash,
    )


def generate_function_snippets(functions: list[FunctionRef]) -> list[SnippetRef]:
    snippets: list[SnippetRef] = []
    for fn in functions:
        snippet_hash = hash_text(
            f"FUNC:{fn.file.path}:{fn.start_line}:{fn.end_line}:{fn.code_hash}"
        )
        snippets.append(
            SnippetRef(
                kind="FUNC",
                function=fn,
                start_line=fn.start_line,
                end_line=fn.end_line,
                text=normalize_source(fn.code),
                snippet_hash=snippet_hash,
            )
        )
    return snippets


def generate_window_snippets(
    functions: list[FunctionRef], params: WindowParams
) -> list[SnippetRef]:
    if params.window_lines <= 0:
        raise ValueError("window_lines must be > 0")
    if params.stride_lines <= 0:
        raise ValueError("stride_lines must be > 0")
    snippets: list[SnippetRef] = []
    for fn in functions:
        lines = fn.code.splitlines()
        if not lines:
            continue
        idx = 0
        while idx < len(lines):
            start = idx + 1
            end = min(idx + params.window_lines, len(lines))
            window_lines = lines[start - 1 : end]
            nonempty = sum(1 for line in window_lines if line.strip())
            if nonempty >= params.min_nonempty:
                snippets.append(_make_snippet(fn, start, end, "WIN"))
            idx += params.stride_lines
    return snippets
