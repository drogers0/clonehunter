from __future__ import annotations

import ast


def strip_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    class DocstringRemover(ast.NodeTransformer):
        def _strip(self, node: ast.stmt) -> ast.stmt:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                value = node.value
                if isinstance(value.value, str):
                    return ast.Pass()
            return node

        def _strip_body(self, node: ast.AST) -> ast.AST:
            self.generic_visit(node)
            body = getattr(node, "body", None)
            if body:
                body[0] = self._strip(body[0])
            return node

        visit_FunctionDef = _strip_body
        visit_AsyncFunctionDef = _strip_body
        visit_Module = _strip_body

    stripped = DocstringRemover().visit(tree)
    ast.fix_missing_locations(stripped)
    return ast.unparse(stripped)


def normalize_source(source: str) -> str:
    return strip_docstrings(source)
