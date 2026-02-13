from __future__ import annotations

import ast
from dataclasses import dataclass

from clonehunter.core.types import FileRef, FunctionRef
from clonehunter.io.fingerprints import hash_text


@dataclass(frozen=True, slots=True)
class ParsedFile:
    file: FileRef
    source: str
    tree: ast.AST


def parse_file(file: FileRef) -> ParsedFile:
    with open(file.path, encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=file.path)
    return ParsedFile(file=file, source=source, tree=tree)


def extract_functions(file: FileRef) -> list[FunctionRef]:
    try:
        parsed = parse_file(file)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    lines = parsed.source.splitlines()
    functions: list[FunctionRef] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _visit_callable(self, node: ast.AST) -> None:
            self._add_function(node)
            self.stack.append(getattr(node, "name", "<lambda>"))
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _visit_callable
        visit_AsyncFunctionDef = _visit_callable

        def _add_function(self, node: ast.AST) -> None:
            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)
            code = "\n".join(lines[start_line - 1 : end_line])
            qualified = ".".join([*self.stack, getattr(node, "name", "<lambda>")])
            functions.append(
                FunctionRef(
                    file=file,
                    qualified_name=qualified,
                    start_line=start_line,
                    end_line=end_line,
                    code=code,
                    code_hash=hash_text(code),
                )
            )

    Visitor().visit(parsed.tree)
    return functions
