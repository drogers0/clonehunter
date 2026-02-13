from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from clonehunter.core.types import FunctionRef, SnippetRef
from clonehunter.io.fingerprints import hash_text
from clonehunter.snippets.normalization import normalize_source


@dataclass(frozen=True, slots=True)
class ExpansionParams:
    enabled: bool
    depth: int
    max_chars: int


def expand_calls(functions: list[FunctionRef], params: ExpansionParams) -> list[SnippetRef]:
    if not params.enabled or params.depth <= 0:
        return []

    by_file: dict[str, list[FunctionRef]] = {}
    for fn in functions:
        by_file.setdefault(fn.file.path, []).append(fn)

    module_name_map = _module_name_map(by_file)
    module_functions = {file_path: _name_map(fns) for file_path, fns in by_file.items()}
    module_qualified = {
        file_path: {fn.qualified_name: fn for fn in fns} for file_path, fns in by_file.items()
    }
    module_classes = {file_path: _module_class_names(fns) for file_path, fns in by_file.items()}
    module_factories = {
        file_path: _factory_map_for_functions(fns) for file_path, fns in by_file.items()
    }
    local_files = [Path(p) for p in by_file]
    snippets: list[SnippetRef] = []
    for file_path, fns in by_file.items():
        name_map = _name_map(fns)
        qualified_map = {fn.qualified_name: fn for fn in fns}
        class_names = _class_names(qualified_map)
        imports = _collect_imports(Path(file_path), local_files)
        for fn in fns:
            expanded_text, helpers = _expand_for_function(
                fn,
                name_map,
                qualified_map,
                class_names,
                imports,
                module_name_map,
                module_functions,
                module_qualified,
                module_classes,
                module_factories,
                params,
            )
            if not helpers:
                continue
            normalized = normalize_source(expanded_text)
            snippet_hash = hash_text(
                f"EXP:{fn.file.path}:{fn.start_line}:{fn.end_line}:{fn.code_hash}:{','.join(helpers)}:{params.depth}:{params.max_chars}:{normalized}"
            )
            snippets.append(
                SnippetRef(
                    kind="EXP",
                    function=fn,
                    start_line=fn.start_line,
                    end_line=fn.end_line,
                    text=normalized,
                    snippet_hash=snippet_hash,
                )
            )
    return snippets


def _name_map(functions: list[FunctionRef]) -> dict[str, FunctionRef]:
    mapping: dict[str, FunctionRef] = {}
    for fn in functions:
        short = fn.qualified_name.split(".")[-1]
        mapping[short] = fn
    return mapping


def _expand_for_function(
    function: FunctionRef,
    name_map: dict[str, FunctionRef],
    qualified_map: dict[str, FunctionRef],
    class_names: set[str],
    imports: ImportMap,
    module_name_map: dict[str, str],
    module_functions: dict[str, dict[str, FunctionRef]],
    module_qualified: dict[str, dict[str, FunctionRef]],
    module_classes: dict[str, set[str]],
    module_factories: dict[str, dict[str, str]],
    params: ExpansionParams,
) -> tuple[str, list[str]]:
    helpers: list[str] = []
    expanded = function.code
    frontier = [function]
    visited: set[str] = {function.identity}
    class_name = _class_name(function)
    factory_map = _factory_map_for_functions(list(qualified_map.values()))
    local_class_map = _local_class_map(
        function,
        class_names,
        factory_map,
        imports,
        module_name_map,
        module_factories,
        module_classes,
    )

    for _ in range(params.depth):
        next_frontier: list[FunctionRef] = []
        for fn in frontier:
            for call in _collect_calls(fn.code):
                helper = _resolve_call(
                    call,
                    name_map,
                    qualified_map,
                    class_names,
                    imports,
                    module_name_map,
                    module_functions,
                    module_qualified,
                    class_name,
                    local_class_map,
                )
                if helper is None:
                    continue
                if helper.identity in visited:
                    continue
                addition = f"\n\n# expanded:{helper.qualified_name}\n{helper.code}"
                if len(expanded) + len(addition) > params.max_chars:
                    continue
                visited.add(helper.identity)
                helpers.append(helper.qualified_name)
                expanded += addition
                next_frontier.append(helper)
        frontier = next_frontier
    return expanded, helpers


@dataclass(frozen=True, slots=True)
class CallRef:
    kind: str
    base: str | None
    name: str


@dataclass(frozen=True, slots=True)
class ImportMap:
    module_aliases: dict[str, str]
    function_aliases: dict[str, tuple[str, str]]
    class_aliases: dict[str, tuple[str, str]]


def _collect_calls(source: str) -> set[CallRef]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    calls: set[CallRef] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            call = _call_from_node(node.func)
            if call:
                calls.add(call)
            self.generic_visit(node)

    Visitor().visit(tree)
    return calls


def _call_from_node(node: ast.AST) -> CallRef | None:
    if isinstance(node, ast.Name):
        return CallRef(kind="name", base=None, name=node.id)
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Name):
            return CallRef(kind="attr", base=base.id, name=node.attr)
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Name):
            return CallRef(kind="ctor", base=base.func.id, name=node.attr)
    return None


def _class_name(function: FunctionRef) -> str | None:
    parts = function.qualified_name.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return None


def _resolve_call(
    call: CallRef,
    name_map: dict[str, FunctionRef],
    qualified_map: dict[str, FunctionRef],
    class_names: set[str],
    imports: ImportMap,
    module_name_map: dict[str, str],
    module_functions: dict[str, dict[str, FunctionRef]],
    module_qualified: dict[str, dict[str, FunctionRef]],
    class_name: str | None,
    local_class_map: dict[str, tuple[str | None, str]],
) -> FunctionRef | None:
    if call.kind == "name":
        if call.name in name_map:
            return name_map[call.name]
        if call.name in imports.function_aliases:
            module_path, func_name = imports.function_aliases[call.name]
            return _resolve_module_function(
                module_path, func_name, module_name_map, module_functions
            )
        return None
    if call.kind == "attr":
        if call.base in {"self", "cls"} and class_name:
            qualified = f"{class_name}.{call.name}"
            return qualified_map.get(qualified)
        if call.base in local_class_map:
            module_path, cls_name = local_class_map[call.base]
            if module_path is None:
                qualified = f"{cls_name}.{call.name}"
                return qualified_map.get(qualified)
            return _resolve_class_method(
                module_path, cls_name, call.name, module_name_map, module_qualified
            )
        if call.base in imports.module_aliases:
            module_path = imports.module_aliases[call.base]
            return _resolve_module_function(
                module_path, call.name, module_name_map, module_functions
            )
    if call.kind == "ctor":
        cls_name = call.base or ""
        if cls_name in class_names:
            qualified = f"{cls_name}.{call.name}"
            local = qualified_map.get(qualified)
            if local:
                return local
        if cls_name in imports.class_aliases:
            module_path, imported_class = imports.class_aliases[cls_name]
            return _resolve_class_method(
                module_path, imported_class, call.name, module_name_map, module_qualified
            )
    return None


def _module_name_map(by_file: dict[str, list[FunctionRef]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for file_path in by_file:
        path = Path(file_path)
        mapping[path.name] = file_path
        mapping[path.stem] = file_path
    return mapping


def _resolve_module_function(
    module_path: str,
    func_name: str,
    module_name_map: dict[str, str],
    module_functions: dict[str, dict[str, FunctionRef]],
) -> FunctionRef | None:
    return cast(
        FunctionRef | None,
        _resolve_from_module(module_path, func_name, module_name_map, module_functions),
    )


def _resolve_class_method(
    module_path: str,
    class_name: str,
    method_name: str,
    module_name_map: dict[str, str],
    module_qualified: dict[str, dict[str, FunctionRef]],
) -> FunctionRef | None:
    qualified = _resolve_map_for_module(module_path, module_name_map, module_qualified)
    if qualified is None:
        return None
    return cast(dict[str, FunctionRef], qualified).get(f"{class_name}.{method_name}")


def _collect_imports(path: Path, local_files: list[Path]) -> ImportMap:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ImportMap(module_aliases={}, function_aliases={}, class_aliases={})
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ImportMap(module_aliases={}, function_aliases={}, class_aliases={})

    module_aliases: dict[str, str] = {}
    function_aliases: dict[str, tuple[str, str]] = {}
    class_aliases: dict[str, tuple[str, str]] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_path = _resolve_local_module(path.parent, alias.name, local_files)
                if module_path:
                    module_aliases[alias.asname or alias.name.split(".")[-1]] = module_path
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            base_dir = path.parent
            if node.level and node.level > 0:
                base_dir = _apply_relative(base_dir, node.level)
            module_path = _resolve_local_module(base_dir, node.module, local_files)
            if not module_path:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                function_aliases[alias.asname or alias.name] = (module_path, alias.name)
                class_aliases[alias.asname or alias.name] = (module_path, alias.name)

    return ImportMap(
        module_aliases=module_aliases,
        function_aliases=function_aliases,
        class_aliases=class_aliases,
    )


def _resolve_local_module(base_dir: Path, module_name: str, local_files: list[Path]) -> str | None:
    parts = module_name.split(".")
    candidates = [
        base_dir.joinpath(*parts).with_suffix(".py"),
        base_dir.joinpath(*parts, "__init__.py"),
    ]
    for candidate in candidates:
        if candidate in local_files:
            return str(candidate)
    # Conservative fallback: match any local file whose suffix path matches the module path.
    for file_path in local_files:
        if _matches_module_path(file_path, parts):
            return str(file_path)
    return None


def _matches_module_path(file_path: Path, parts: list[str]) -> bool:
    path_parts = list(file_path.parts)
    if file_path.name == "__init__.py":
        module_parts = [*parts, "__init__.py"]
    else:
        module_parts = [*parts[:-1], parts[-1] + ".py"]
    if len(path_parts) < len(module_parts):
        return False
    return path_parts[-len(module_parts) :] == module_parts


def _apply_relative(base_dir: Path, level: int) -> Path:
    target = base_dir
    for _ in range(level):
        target = target.parent
    return target


def _local_class_map(
    function: FunctionRef,
    class_names: set[str],
    factory_map: dict[str, str],
    imports: ImportMap,
    module_name_map: dict[str, str],
    module_factories: dict[str, dict[str, str]],
    module_classes: dict[str, set[str]],
) -> dict[str, tuple[str | None, str]]:
    try:
        tree = ast.parse(function.code)
    except SyntaxError:
        return {}
    class_map: dict[str, tuple[str | None, str]] = {}

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            resolved = _resolve_value_class(
                node.value,
                class_names,
                factory_map,
                imports,
                module_name_map,
                module_factories,
                module_classes,
            )
            if resolved:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        class_map[target.id] = resolved
            if isinstance(node.value, ast.Name) and node.value.id in class_map:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        class_map[target.id] = class_map[node.value.id]
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            resolved = _resolve_annotation_class(node.annotation, imports)
            if isinstance(node.target, ast.Name) and resolved:
                class_map[node.target.id] = resolved
            self.generic_visit(node)

    Visitor().visit(tree)
    return class_map


def _resolve_value_class(
    node: ast.AST,
    class_names: set[str],
    factory_map: dict[str, str],
    imports: ImportMap,
    module_name_map: dict[str, str],
    module_factories: dict[str, dict[str, str]],
    module_classes: dict[str, set[str]],
) -> tuple[str | None, str] | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in class_names:
            return (None, name)
        if name in imports.class_aliases:
            module_path, imported_class = imports.class_aliases[name]
            if _class_exists_in_module(
                module_path, imported_class, module_name_map, module_classes
            ):
                return (module_path, imported_class)
            return None
        if name in factory_map:
            return (None, factory_map[name])
        if name in imports.function_aliases:
            module_path, func_name = imports.function_aliases[name]
            cls_name = _resolve_factory_return(
                module_path, func_name, module_name_map, module_factories
            )
            if cls_name:
                return (module_path, cls_name)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = node.func.value
        if isinstance(base, ast.Name) and base.id in imports.module_aliases:
            module_path = imports.module_aliases[base.id]
            cls_name = _resolve_factory_return(
                module_path, node.func.attr, module_name_map, module_factories
            )
            if cls_name:
                return (module_path, cls_name)
    return None


def _resolve_annotation_class(node: ast.AST, imports: ImportMap) -> tuple[str | None, str] | None:
    if isinstance(node, ast.Name):
        if node.id in imports.class_aliases:
            module_path, imported_class = imports.class_aliases[node.id]
            return (module_path, imported_class)
        return (None, node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        base = node.value.id
        if base in imports.module_aliases:
            return (imports.module_aliases[base], node.attr)
        return (None, node.attr)
    return None


def _resolve_factory_return(
    module_path: str,
    func_name: str,
    module_name_map: dict[str, str],
    module_factories: dict[str, dict[str, str]],
) -> str | None:
    return cast(
        str | None,
        _resolve_from_module(module_path, func_name, module_name_map, module_factories),
    )


def _factory_map_for_functions(functions: list[FunctionRef]) -> dict[str, str]:
    factory_map: dict[str, str] = {}
    for fn in functions:
        name = fn.qualified_name.split(".")[-1]
        cls_name = _infer_return_class(fn.code)
        if cls_name:
            factory_map[name] = cls_name
    return factory_map


def _infer_return_class(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.found: str | None = None

        def visit_Return(self, node: ast.Return) -> None:
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                self.found = node.value.func.id
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.found


def _class_names(qualified_map: dict[str, FunctionRef]) -> set[str]:
    names: set[str] = set()
    for qname in qualified_map:
        parts = qname.split(".")
        if len(parts) >= 2:
            names.add(parts[-2])
    return names


def _module_class_names(functions: list[FunctionRef]) -> set[str]:
    qualified = {fn.qualified_name: fn for fn in functions}
    return _class_names(qualified)


def _class_exists_in_module(
    module_path: str,
    class_name: str,
    module_name_map: dict[str, str],
    module_classes: dict[str, set[str]],
) -> bool:
    classes = _resolve_map_for_module(module_path, module_name_map, module_classes)
    if classes is None:
        return False
    return class_name in cast(set[str], classes)


def _resolve_module_path(module_path: str, module_name_map: dict[str, str]) -> str | None:
    return module_name_map.get(Path(module_path).name) or module_name_map.get(
        Path(module_path).stem
    )


def _resolve_map_for_module(
    module_path: str,
    module_name_map: dict[str, str],
    module_values: Mapping[str, object],
) -> object | None:
    file_path = _resolve_module_path(module_path, module_name_map)
    if not file_path:
        return None
    return module_values.get(file_path)


def _resolve_from_module(
    module_path: str,
    key: str,
    module_name_map: dict[str, str],
    module_values: Mapping[str, Mapping[str, object]],
) -> object | None:
    raw = _resolve_map_for_module(module_path, module_name_map, module_values)
    if raw is None:
        return None
    values = cast(Mapping[str, object], raw)
    return values.get(key)
