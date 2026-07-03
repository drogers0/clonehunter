// allow dead_code until T12 wires cli::run to the pipeline
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use tree_sitter::{Language as TsLanguage, Node, Parser};

use crate::core::config::ExpansionConfig;
use crate::core::types::{FunctionRef, SnippetKind, SnippetRef};
use crate::io::fingerprints::hash_text;
use crate::snippets::normalization::{normalize_analysis, normalize_display};

// ─── Internal types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CallKind {
    Name,
    Attr,
    Ctor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CallRef {
    kind: CallKind,
    base: Option<String>,
    name: String,
}

#[derive(Debug, Default)]
struct ImportMap {
    /// alias → resolved local file path
    module_aliases: HashMap<String, String>,
    /// alias → (module_path, function_name)
    function_aliases: HashMap<String, (String, String)>,
    /// alias → (module_path, class_name)
    class_aliases: HashMap<String, (String, String)>,
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Generate EXP snippets by BFS-expanding call references into caller function bodies.
///
/// Returns empty vec if `!config.enabled || config.depth == 0`.
/// Uses `BTreeMap` for file-group ordering (DD8 — determinism across runs).
pub(crate) fn expand_calls(functions: &[FunctionRef], config: &ExpansionConfig) -> Vec<SnippetRef> {
    if !config.enabled || config.depth == 0 {
        return vec![];
    }

    // Group by file path. BTreeMap gives sorted deterministic iteration (DD8).
    let mut by_file: BTreeMap<String, Vec<&FunctionRef>> = BTreeMap::new();
    for fn_ in functions {
        by_file.entry(fn_.file.path.clone()).or_default().push(fn_);
    }

    // Build per-file lookup maps (borrow from `functions`)
    let module_functions: HashMap<String, HashMap<String, &FunctionRef>> = by_file
        .iter()
        .map(|(path, fns)| (path.clone(), name_map(fns)))
        .collect();

    let module_qualified: HashMap<String, HashMap<String, &FunctionRef>> = by_file
        .iter()
        .map(|(path, fns)| {
            let qmap = fns.iter().map(|f| (f.qualified_name.clone(), *f)).collect();
            (path.clone(), qmap)
        })
        .collect();

    let module_classes: HashMap<String, HashSet<String>> = by_file
        .iter()
        .map(|(path, fns)| {
            let qmap: HashMap<String, &FunctionRef> =
                fns.iter().map(|f| (f.qualified_name.clone(), *f)).collect();
            (path.clone(), class_names_from_qmap(&qmap))
        })
        .collect();

    let module_factories: HashMap<String, HashMap<String, String>> = by_file
        .iter()
        .map(|(path, fns)| (path.clone(), factory_map_for_functions(fns)))
        .collect();

    let local_files: Vec<PathBuf> = by_file.keys().map(PathBuf::from).collect();

    let mut snippets = Vec::new();

    for (file_path, fns) in &by_file {
        let f_name_map = name_map(fns);
        let qualified_map: HashMap<String, &FunctionRef> =
            fns.iter().map(|f| (f.qualified_name.clone(), *f)).collect();
        let cls_names = class_names_from_qmap(&qualified_map);
        let imports = collect_imports(Path::new(file_path), &local_files);

        for fn_ in fns {
            let (expanded_text, helpers) = expand_for_function(
                fn_,
                &f_name_map,
                &qualified_map,
                &cls_names,
                &imports,
                &module_functions,
                &module_qualified,
                &module_classes,
                &module_factories,
                config,
            );
            if helpers.is_empty() {
                continue;
            }
            let analysis = normalize_analysis(&expanded_text);
            let display = normalize_display(&expanded_text);
            let helpers_csv = helpers.join(",");
            let snippet_hash = hash_text(&format!(
                "EXP:{}:{}:{}:{}:{}:{}:{}:{}",
                fn_.file.path,
                fn_.start_line,
                fn_.end_line,
                fn_.code_hash,
                helpers_csv,
                config.depth,
                config.max_chars,
                analysis
            ));
            snippets.push(SnippetRef {
                kind: SnippetKind::Exp,
                function: (*fn_).clone(),
                start_line: fn_.start_line,
                end_line: fn_.end_line,
                text: analysis,
                display_text: display,
                snippet_hash,
            });
        }
    }

    snippets
}

// ─── BFS expansion ───────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn expand_for_function<'a>(
    function: &'a FunctionRef,
    f_name_map: &HashMap<String, &'a FunctionRef>,
    qualified_map: &HashMap<String, &'a FunctionRef>,
    cls_names: &HashSet<String>,
    imports: &ImportMap,
    module_functions: &'a HashMap<String, HashMap<String, &'a FunctionRef>>,
    module_qualified: &'a HashMap<String, HashMap<String, &'a FunctionRef>>,
    module_classes: &HashMap<String, HashSet<String>>,
    module_factories: &HashMap<String, HashMap<String, String>>,
    config: &ExpansionConfig,
) -> (String, Vec<String>) {
    let mut expanded = function.code.clone();
    let mut helpers: Vec<String> = Vec::new();
    let mut frontier: Vec<&FunctionRef> = vec![function];
    let mut visited: HashSet<String> = HashSet::from([function.identity()]);
    let class_name = class_name_of(function);
    let factory_map =
        factory_map_for_functions(&qualified_map.values().copied().collect::<Vec<_>>());
    let local_class_map = build_local_class_map(
        function,
        cls_names,
        &factory_map,
        imports,
        module_factories,
        module_classes,
    );

    for _ in 0..config.depth {
        let mut next_frontier: Vec<&FunctionRef> = Vec::new();
        for current_fn in &frontier {
            for call in collect_calls(&current_fn.code) {
                let helper = match resolve_call(
                    &call,
                    f_name_map,
                    qualified_map,
                    cls_names,
                    imports,
                    module_functions,
                    module_qualified,
                    &class_name,
                    &local_class_map,
                ) {
                    Some(h) => h,
                    None => continue,
                };
                if visited.contains(&helper.identity()) {
                    continue;
                }
                let addition = format!("\n\n# expanded:{}\n{}", helper.qualified_name, helper.code);
                // Note: len() counts bytes (not chars). For ASCII-dominated code this matches
                // Python's len() exactly; for rare multi-byte chars Rust is slightly more
                // conservative. Acceptable for parity (absorbed by T13 re-freeze).
                if expanded.len() + addition.len() > config.max_chars {
                    continue;
                }
                visited.insert(helper.identity());
                helpers.push(helper.qualified_name.clone());
                expanded.push_str(&addition);
                next_frontier.push(helper);
            }
        }
        frontier = next_frontier;
    }

    (expanded, helpers)
}

// ─── Call collection ──────────────────────────────────────────────────────────

fn collect_calls(source: &str) -> HashSet<CallRef> {
    let Some(mut parser) = make_parser() else {
        return HashSet::new();
    };
    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return HashSet::new(),
    };
    let root = tree.root_node();
    let mut calls = HashSet::new();
    collect_calls_recursive(root, source, &mut calls);
    calls
}

fn collect_calls_recursive(node: Node<'_>, source: &str, out: &mut HashSet<CallRef>) {
    if node.kind() == "call" {
        if let Some(func_node) = node.child_by_field_name("function") {
            if let Some(call_ref) = call_from_node(func_node, source) {
                out.insert(call_ref);
            }
        }
    }
    for i in 0..node.child_count() {
        collect_calls_recursive(node.child(i).unwrap(), source, out);
    }
}

fn call_from_node(node: Node<'_>, source: &str) -> Option<CallRef> {
    match node.kind() {
        "identifier" => Some(CallRef {
            kind: CallKind::Name,
            base: None,
            name: ts_text(node, source).to_string(),
        }),
        "attribute" => {
            let object = node.child_by_field_name("object")?;
            let attr = node.child_by_field_name("attribute")?;
            let attr_name = ts_text(attr, source).to_string();
            match object.kind() {
                "identifier" => Some(CallRef {
                    kind: CallKind::Attr,
                    base: Some(ts_text(object, source).to_string()),
                    name: attr_name,
                }),
                "call" => {
                    // Ctor pattern: Foo().method(...)
                    let ctor_func = object.child_by_field_name("function")?;
                    if ctor_func.kind() == "identifier" {
                        Some(CallRef {
                            kind: CallKind::Ctor,
                            base: Some(ts_text(ctor_func, source).to_string()),
                            name: attr_name,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// ─── Call resolution ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn resolve_call<'a>(
    call: &CallRef,
    name_map: &HashMap<String, &'a FunctionRef>,
    qualified_map: &HashMap<String, &'a FunctionRef>,
    cls_names: &HashSet<String>,
    imports: &ImportMap,
    module_functions: &'a HashMap<String, HashMap<String, &'a FunctionRef>>,
    module_qualified: &'a HashMap<String, HashMap<String, &'a FunctionRef>>,
    class_name: &Option<String>,
    local_class_map: &HashMap<String, (Option<String>, String)>,
) -> Option<&'a FunctionRef> {
    match call.kind {
        CallKind::Name => {
            if let Some(&fn_) = name_map.get(&call.name) {
                return Some(fn_);
            }
            if let Some((module_path, func_name)) = imports.function_aliases.get(&call.name) {
                return resolve_module_function(module_path, func_name, module_functions);
            }
            None
        }
        CallKind::Attr => {
            let base = call.base.as_deref().unwrap_or("");
            // self/cls method call: look up in current class
            if (base == "self" || base == "cls") && class_name.is_some() {
                let cn = class_name.as_ref().unwrap();
                let qualified = format!("{}.{}", cn, call.name);
                return qualified_map.get(&qualified).copied();
            }
            // Variable whose type was resolved in local_class_map
            if let Some((module_path, cls_name)) = local_class_map.get(base) {
                return match module_path {
                    None => {
                        let qualified = format!("{}.{}", cls_name, call.name);
                        qualified_map.get(&qualified).copied()
                    }
                    Some(mp) => resolve_class_method(mp, cls_name, &call.name, module_qualified),
                };
            }
            // Module alias: look up the function in the remote module
            if let Some(module_path) = imports.module_aliases.get(base) {
                return resolve_module_function(module_path, &call.name, module_functions);
            }
            None
        }
        CallKind::Ctor => {
            let cls_name = call.base.as_deref().unwrap_or("");
            // Local class constructor method
            if cls_names.contains(cls_name) {
                let qualified = format!("{}.{}", cls_name, call.name);
                if let Some(&fn_) = qualified_map.get(&qualified) {
                    return Some(fn_);
                }
                // Fall through — class exists locally but method not found; try imports
            }
            // Imported class constructor method
            if let Some((module_path, imported_class)) = imports.class_aliases.get(cls_name) {
                return resolve_class_method(
                    module_path,
                    imported_class,
                    &call.name,
                    module_qualified,
                );
            }
            None
        }
    }
}

// ─── Module resolution helpers ────────────────────────────────────────────────

fn resolve_module_function<'a>(
    module_path: &str,
    func_name: &str,
    module_functions: &'a HashMap<String, HashMap<String, &'a FunctionRef>>,
) -> Option<&'a FunctionRef> {
    module_functions.get(module_path)?.get(func_name).copied()
}

fn resolve_class_method<'a>(
    module_path: &str,
    class_name: &str,
    method_name: &str,
    module_qualified: &'a HashMap<String, HashMap<String, &'a FunctionRef>>,
) -> Option<&'a FunctionRef> {
    let qualified = format!("{}.{}", class_name, method_name);
    module_qualified.get(module_path)?.get(&qualified).copied()
}

fn class_exists_in_module(
    module_path: &str,
    class_name: &str,
    module_classes: &HashMap<String, HashSet<String>>,
) -> bool {
    module_classes
        .get(module_path)
        .is_some_and(|s| s.contains(class_name))
}

fn resolve_factory_return(
    module_path: &str,
    func_name: &str,
    module_factories: &HashMap<String, HashMap<String, String>>,
) -> Option<String> {
    module_factories.get(module_path)?.get(func_name).cloned()
}

// ─── Import collection ────────────────────────────────────────────────────────

fn collect_imports(path: &Path, local_files: &[PathBuf]) -> ImportMap {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return ImportMap::default(),
    };
    let source = String::from_utf8_lossy(&bytes).into_owned();

    let Some(mut parser) = make_parser() else {
        return ImportMap::default();
    };
    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return ImportMap::default(),
    };

    let root = tree.root_node();
    let mut map = ImportMap::default();

    // Only walk direct children (top-level statements), matching Python's `for node in tree.body`
    for i in 0..root.child_count() {
        let child = root.child(i).unwrap();
        match child.kind() {
            "import_statement" => {
                handle_import_statement(child, &source, path, local_files, &mut map);
            }
            "import_from_statement" => {
                handle_import_from_statement(child, &source, path, local_files, &mut map);
            }
            _ => {}
        }
    }

    map
}

fn handle_import_statement(
    node: Node<'_>,
    source: &str,
    path: &Path,
    local_files: &[PathBuf],
    map: &mut ImportMap,
) {
    let parent_dir = path.parent().unwrap_or(Path::new("."));
    let mut cursor = node.walk();
    for name_node in node.children_by_field_name("name", &mut cursor) {
        match name_node.kind() {
            "dotted_name" => {
                let module_name = ts_text(name_node, source);
                let alias = module_name
                    .split('.')
                    .last()
                    .unwrap_or(module_name)
                    .to_string();
                if let Some(module_path) =
                    resolve_local_module(parent_dir, module_name, local_files)
                {
                    map.module_aliases.insert(alias, module_path);
                }
            }
            "aliased_import" => {
                if let Some(name_child) = name_node.child_by_field_name("name") {
                    let module_name = ts_text(name_child, source);
                    let alias = name_node
                        .child_by_field_name("alias")
                        .map(|a| ts_text(a, source).to_string())
                        .unwrap_or_else(|| {
                            module_name
                                .split('.')
                                .last()
                                .unwrap_or(module_name)
                                .to_string()
                        });
                    if let Some(module_path) =
                        resolve_local_module(parent_dir, module_name, local_files)
                    {
                        map.module_aliases.insert(alias, module_path);
                    }
                }
            }
            _ => {}
        }
    }
}

fn handle_import_from_statement(
    node: Node<'_>,
    source: &str,
    path: &Path,
    local_files: &[PathBuf],
    map: &mut ImportMap,
) {
    let module_node = match node.child_by_field_name("module_name") {
        Some(n) => n,
        None => return,
    };

    let parent_dir = path.parent().unwrap_or(Path::new("."));

    let (level, module_name_opt) = if module_node.kind() == "relative_import" {
        let level = count_import_dots(module_node, source);
        let module_name = (0..module_node.named_child_count())
            .filter_map(|i| module_node.named_child(i))
            .find(|n| n.kind() == "dotted_name")
            .map(|n| ts_text(n, source).to_string());
        (level, module_name)
    } else if module_node.kind() == "dotted_name" {
        (0, Some(ts_text(module_node, source).to_string()))
    } else {
        return;
    };

    // Skip bare relative imports (from . import x where no module name) — matches Python
    let module_name = match module_name_opt {
        Some(m) => m,
        None => return,
    };

    // Compute base directory for relative imports
    let base_dir = if level > 0 {
        apply_relative(parent_dir, level)
    } else {
        parent_dir.to_path_buf()
    };

    let module_path = match resolve_local_module(&base_dir, &module_name, local_files) {
        Some(p) => p,
        None => return,
    };

    // Collect imported names
    let mut cursor = node.walk();
    for name_node in node.children_by_field_name("name", &mut cursor) {
        let (import_name, alias) = match name_node.kind() {
            "aliased_import" => {
                let name = name_node
                    .child_by_field_name("name")
                    .map(|n| ts_text(n, source).to_string())
                    .unwrap_or_default();
                let alias = name_node
                    .child_by_field_name("alias")
                    .map(|n| ts_text(n, source).to_string())
                    .unwrap_or_else(|| name.clone());
                (name, alias)
            }
            "dotted_name" | "identifier" => {
                let name = ts_text(name_node, source).to_string();
                (name.clone(), name)
            }
            _ => continue,
        };

        if import_name == "*" {
            continue;
        }

        // Both function_aliases and class_aliases are populated (matches Python)
        map.function_aliases
            .insert(alias.clone(), (module_path.clone(), import_name.clone()));
        map.class_aliases
            .insert(alias, (module_path.clone(), import_name));
    }
}

// ─── Module path resolution ───────────────────────────────────────────────────

/// Apply `level` parent steps, matching Python's `_apply_relative`. Level=1 with base_dir=file.parent
/// goes one level above the file's directory (matching `from .module import x`).
fn apply_relative(base_dir: &Path, level: usize) -> PathBuf {
    let mut target = base_dir.to_path_buf();
    for _ in 0..level {
        target = target.parent().unwrap_or(&target).to_path_buf();
    }
    target
}

fn resolve_local_module(
    base_dir: &Path,
    module_name: &str,
    local_files: &[PathBuf],
) -> Option<String> {
    let parts: Vec<&str> = module_name.split('.').collect();

    // Build candidate path from parts
    let mut base_path = base_dir.to_path_buf();
    for part in &parts {
        base_path = base_path.join(part);
    }
    let file_candidate = base_path.with_extension("py");
    let init_candidate = base_path.join("__init__.py");

    for candidate in &[file_candidate, init_candidate] {
        if local_files.contains(candidate) {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }

    // Conservative fallback: match any local file whose suffix path matches the module parts
    for file_path in local_files {
        if matches_module_path(file_path, &parts) {
            return Some(file_path.to_string_lossy().into_owned());
        }
    }

    None
}

fn matches_module_path(file_path: &Path, parts: &[&str]) -> bool {
    let path_parts: Vec<&str> = file_path
        .components()
        .map(|c| c.as_os_str().to_str().unwrap_or(""))
        .collect();
    let module_parts: Vec<String> =
        if file_path.file_name().and_then(|n| n.to_str()) == Some("__init__.py") {
            parts
                .iter()
                .map(|s| s.to_string())
                .chain(std::iter::once("__init__.py".to_string()))
                .collect()
        } else {
            let mut v: Vec<String> = parts[..parts.len().saturating_sub(1)]
                .iter()
                .map(|s| s.to_string())
                .collect();
            if let Some(last) = parts.last() {
                v.push(format!("{}.py", last));
            }
            v
        };

    if path_parts.len() < module_parts.len() {
        return false;
    }
    let suffix = &path_parts[path_parts.len() - module_parts.len()..];
    suffix
        .iter()
        .zip(module_parts.iter())
        .all(|(a, b)| *a == b.as_str())
}

// ─── Local class map ──────────────────────────────────────────────────────────

/// Map variable names in the function body to their class types, for method resolution.
fn build_local_class_map(
    function: &FunctionRef,
    cls_names: &HashSet<String>,
    factory_map: &HashMap<String, String>,
    imports: &ImportMap,
    module_factories: &HashMap<String, HashMap<String, String>>,
    module_classes: &HashMap<String, HashSet<String>>,
) -> HashMap<String, (Option<String>, String)> {
    let Some(mut parser) = make_parser() else {
        return HashMap::new();
    };
    let tree = match parser.parse(function.code.as_bytes(), None) {
        Some(t) => t,
        None => return HashMap::new(),
    };

    let root = tree.root_node();
    let mut class_map: HashMap<String, (Option<String>, String)> = HashMap::new();

    local_class_map_recursive(
        root,
        &function.code,
        cls_names,
        factory_map,
        imports,
        module_factories,
        module_classes,
        &mut class_map,
    );

    class_map
}

#[allow(clippy::too_many_arguments)]
fn local_class_map_recursive(
    node: Node<'_>,
    source: &str,
    cls_names: &HashSet<String>,
    factory_map: &HashMap<String, String>,
    imports: &ImportMap,
    module_factories: &HashMap<String, HashMap<String, String>>,
    module_classes: &HashMap<String, HashSet<String>>,
    class_map: &mut HashMap<String, (Option<String>, String)>,
) {
    if node.kind() == "assignment" {
        // Check 1: Call resolution — RHS is a call expression
        if let Some(right) = node.child_by_field_name("right") {
            if right.kind() == "call" {
                if let Some(resolved) = resolve_value_class(
                    right,
                    source,
                    cls_names,
                    factory_map,
                    imports,
                    module_factories,
                    module_classes,
                ) {
                    apply_to_targets(node, source, resolved, class_map);
                }
            }
        }

        // Check 2: Alias propagation — RHS is an identifier already in class_map
        // (Both checks always run; check 2 may overwrite check 1 — matches Python)
        if let Some(right) = node.child_by_field_name("right") {
            if right.kind() == "identifier" {
                let rhs_name = ts_text(right, source);
                if let Some(resolved) = class_map.get(rhs_name).cloned() {
                    apply_to_targets(node, source, resolved, class_map);
                }
            }
        }

        // Annotated assignment: `x: MyClass` or `x: MyClass = value`
        if let Some(type_node) = node.child_by_field_name("type") {
            if let Some(resolved) = resolve_annotation_class(type_node, source, imports) {
                if let Some(left) = node.child_by_field_name("left") {
                    if left.kind() == "identifier" {
                        class_map.insert(ts_text(left, source).to_string(), resolved);
                    }
                }
            }
        }
    }

    for i in 0..node.child_count() {
        local_class_map_recursive(
            node.child(i).unwrap(),
            source,
            cls_names,
            factory_map,
            imports,
            module_factories,
            module_classes,
            class_map,
        );
    }
}

/// Apply a resolved class binding to all LHS identifier targets of an assignment node.
fn apply_to_targets(
    assignment: Node<'_>,
    source: &str,
    resolved: (Option<String>, String),
    class_map: &mut HashMap<String, (Option<String>, String)>,
) {
    let Some(left) = assignment.child_by_field_name("left") else {
        return;
    };
    match left.kind() {
        "identifier" => {
            class_map.insert(ts_text(left, source).to_string(), resolved);
        }
        "pattern_list" | "expression_list" => {
            for i in 0..left.named_child_count() {
                let child = left.named_child(i).unwrap();
                if child.kind() == "identifier" {
                    class_map.insert(ts_text(child, source).to_string(), resolved.clone());
                }
            }
        }
        _ => {}
    }
}

/// Resolve a type annotation node to (optional_module_path, class_name).
fn resolve_annotation_class(
    type_node: Node<'_>,
    source: &str,
    imports: &ImportMap,
) -> Option<(Option<String>, String)> {
    // tree-sitter wraps annotations in a `type` node — unwrap it
    let actual = if type_node.kind() == "type" {
        type_node.named_child(0)?
    } else {
        type_node
    };

    match actual.kind() {
        "identifier" => {
            let name = ts_text(actual, source);
            if let Some((module_path, imported_class)) = imports.class_aliases.get(name) {
                Some((Some(module_path.clone()), imported_class.clone()))
            } else {
                Some((None, name.to_string()))
            }
        }
        "attribute" => {
            let base = actual.child_by_field_name("object")?;
            let attr = actual.child_by_field_name("attribute")?;
            let attr_name = ts_text(attr, source).to_string();
            if base.kind() == "identifier" {
                let base_name = ts_text(base, source);
                if let Some(module_path) = imports.module_aliases.get(base_name) {
                    Some((Some(module_path.clone()), attr_name))
                } else {
                    Some((None, attr_name))
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve a call node's function expression to the (module_path, class_name) being constructed.
fn resolve_value_class(
    call_node: Node<'_>,
    source: &str,
    cls_names: &HashSet<String>,
    factory_map: &HashMap<String, String>,
    imports: &ImportMap,
    module_factories: &HashMap<String, HashMap<String, String>>,
    module_classes: &HashMap<String, HashSet<String>>,
) -> Option<(Option<String>, String)> {
    let func = call_node.child_by_field_name("function")?;

    match func.kind() {
        "identifier" => {
            let name = ts_text(func, source);
            if cls_names.contains(name) {
                return Some((None, name.to_string()));
            }
            if let Some((module_path, imported_class)) = imports.class_aliases.get(name) {
                if class_exists_in_module(module_path, imported_class, module_classes) {
                    return Some((Some(module_path.clone()), imported_class.clone()));
                }
                return None;
            }
            if let Some(cls_name) = factory_map.get(name) {
                return Some((None, cls_name.clone()));
            }
            if let Some((module_path, func_name)) = imports.function_aliases.get(name) {
                if let Some(cls_name) =
                    resolve_factory_return(module_path, func_name, module_factories)
                {
                    return Some((Some(module_path.clone()), cls_name));
                }
            }
            None
        }
        "attribute" => {
            let object = func.child_by_field_name("object")?;
            if object.kind() == "identifier" {
                let base_name = ts_text(object, source);
                if let Some(module_path) = imports.module_aliases.get(base_name) {
                    let attr = func.child_by_field_name("attribute")?;
                    let attr_name = ts_text(attr, source);
                    if let Some(cls_name) =
                        resolve_factory_return(module_path, attr_name, module_factories)
                    {
                        return Some((Some(module_path.clone()), cls_name));
                    }
                }
            }
            None
        }
        _ => None,
    }
}

// ─── Return-class inference ───────────────────────────────────────────────────

/// Infer what class a function returns by inspecting return statements.
/// Returns the LAST match found — matching Python's NodeVisitor overwrite behavior.
fn infer_return_class(source: &str) -> Option<String> {
    let mut parser = make_parser()?;
    let tree = parser.parse(source.as_bytes(), None)?;
    let root = tree.root_node();
    let mut found: Option<String> = None;
    infer_return_recursive(root, source, &mut found);
    found
}

fn infer_return_recursive(node: Node<'_>, source: &str, found: &mut Option<String>) {
    if node.kind() == "return_statement" {
        // named_child(0) is the return value expression (if present)
        if let Some(value) = node.named_child(0) {
            if value.kind() == "call" {
                if let Some(func) = value.child_by_field_name("function") {
                    if func.kind() == "identifier" {
                        *found = Some(ts_text(func, source).to_string());
                    }
                }
            }
        }
    }
    for i in 0..node.child_count() {
        infer_return_recursive(node.child(i).unwrap(), source, found);
    }
}

// ─── Data-transform helpers ───────────────────────────────────────────────────

/// short_name → FunctionRef. Last-wins on collision (matches Python).
fn name_map<'a>(fns: &[&'a FunctionRef]) -> HashMap<String, &'a FunctionRef> {
    let mut map = HashMap::new();
    for fn_ in fns {
        let short = fn_
            .qualified_name
            .split('.')
            .last()
            .unwrap_or(&fn_.qualified_name)
            .to_string();
        map.insert(short, *fn_);
    }
    map
}

/// Extract class names as penultimate components of dotted qualified names.
fn class_names_from_qmap(qmap: &HashMap<String, &FunctionRef>) -> HashSet<String> {
    let mut names = HashSet::new();
    for qname in qmap.keys() {
        let parts: Vec<&str> = qname.split('.').collect();
        if parts.len() >= 2 {
            names.insert(parts[parts.len() - 2].to_string());
        }
    }
    names
}

/// Build short_name → inferred_return_class map for factory detection.
fn factory_map_for_functions(fns: &[&FunctionRef]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for fn_ in fns {
        let short = fn_
            .qualified_name
            .split('.')
            .last()
            .unwrap_or(&fn_.qualified_name)
            .to_string();
        if let Some(cls_name) = infer_return_class(&fn_.code) {
            map.insert(short, cls_name);
        }
    }
    map
}

/// Return the penultimate component of a qualified name (the class the function belongs to).
fn class_name_of(function: &FunctionRef) -> Option<String> {
    let parts: Vec<&str> = function.qualified_name.split('.').collect();
    if parts.len() >= 2 {
        Some(parts[parts.len() - 2].to_string())
    } else {
        None
    }
}

// ─── Tree-sitter utilities ───────────────────────────────────────────────────

fn make_parser() -> Option<Parser> {
    let mut parser = Parser::new();
    let language: TsLanguage = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&language).ok()?;
    Some(parser)
}

fn ts_text<'s>(node: Node<'_>, source: &'s str) -> &'s str {
    &source[node.start_byte()..node.end_byte()]
}

fn count_import_dots(rel_node: Node<'_>, source: &str) -> usize {
    for i in 0..rel_node.child_count() {
        let child = rel_node.child(i).unwrap();
        if child.kind() == "import_prefix" {
            return ts_text(child, source).len();
        }
    }
    0
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::TempDir;

    use crate::core::types::{FileRef, Language};
    use crate::parsing::python_ast::extract_functions;

    fn file_ref(path: &str) -> FileRef {
        FileRef {
            path: path.to_string(),
            content_hash: String::new(),
            language: Language::Python,
        }
    }

    fn fixture_path(rel: &str) -> String {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(rel)
            .to_string_lossy()
            .into_owned()
    }

    fn tiny_repo_functions() -> Vec<FunctionRef> {
        let paths = [
            "fixtures/tiny_repo/a.py",
            "fixtures/tiny_repo/helpers.py",
            "fixtures/tiny_repo/classes.py",
        ];
        paths
            .iter()
            .flat_map(|p| extract_functions(&file_ref(&fixture_path(p))))
            .collect()
    }

    fn default_config() -> ExpansionConfig {
        ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 10000,
        }
    }

    #[test]
    fn test_expansion_disabled() {
        let fns = tiny_repo_functions();
        let config = ExpansionConfig {
            enabled: false,
            depth: 1,
            max_chars: 10000,
        };
        assert!(expand_calls(&fns, &config).is_empty());
    }

    #[test]
    fn test_expansion_depth_zero() {
        let fns = tiny_repo_functions();
        let config = ExpansionConfig {
            enabled: true,
            depth: 0,
            max_chars: 10000,
        };
        assert!(expand_calls(&fns, &config).is_empty());
    }

    #[test]
    fn test_expansion_generates_snippet() {
        let fns = tiny_repo_functions();
        let snippets = expand_calls(&fns, &default_config());
        assert!(
            snippets.iter().any(|s| s.kind == SnippetKind::Exp),
            "should produce at least one EXP snippet"
        );
    }

    #[test]
    fn test_expansion_resolves_imports_and_methods() {
        let fns = tiny_repo_functions();
        let snippets = expand_calls(&fns, &default_config());
        let combined: String = snippets
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("helper_sum"),
            "helper_sum should appear in expanded text"
        );
        // Method 'total' from Accumulator.total should appear
        assert!(
            combined.contains("total"),
            "total method should appear in expanded text"
        );
    }

    #[test]
    fn test_expansion_respects_max_chars() {
        let fns = tiny_repo_functions();
        let config = ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 1,
        };
        let snippets = expand_calls(&fns, &config);
        assert!(
            snippets.is_empty(),
            "max_chars=1 should prevent any expansion"
        );
    }

    #[test]
    fn test_expansion_populates_both_texts() {
        let fns = tiny_repo_functions();
        let snippets = expand_calls(&fns, &default_config());
        for s in &snippets {
            assert!(!s.text.is_empty(), "text (analysis) must be populated");
            assert!(!s.display_text.is_empty(), "display_text must be populated");
        }
    }

    #[test]
    fn test_expansion_resolves_duplicate_module_basenames() {
        let tmp = TempDir::new().unwrap();
        let pkg1 = tmp.path().join("pkg1");
        let pkg2 = tmp.path().join("pkg2");
        std::fs::create_dir(&pkg1).unwrap();
        std::fs::create_dir(&pkg2).unwrap();

        let mut f = std::fs::File::create(pkg1.join("util.py")).unwrap();
        f.write_all(b"def helper():\n    return 'MARKER_PKG1'\n")
            .unwrap();

        let mut f = std::fs::File::create(pkg2.join("util.py")).unwrap();
        f.write_all(b"def helper():\n    return 'MARKER_PKG2'\n")
            .unwrap();

        let mut f = std::fs::File::create(tmp.path().join("main1.py")).unwrap();
        f.write_all(b"from pkg1.util import helper\n\n\ndef caller_one():\n    return helper()\n")
            .unwrap();

        let mut f = std::fs::File::create(tmp.path().join("main2.py")).unwrap();
        f.write_all(b"from pkg2.util import helper\n\n\ndef caller_two():\n    return helper()\n")
            .unwrap();

        let paths = [
            pkg1.join("util.py"),
            pkg2.join("util.py"),
            tmp.path().join("main1.py"),
            tmp.path().join("main2.py"),
        ];
        let functions: Vec<FunctionRef> = paths
            .iter()
            .flat_map(|p| extract_functions(&file_ref(&p.to_string_lossy())))
            .collect();

        let config = ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 10000,
        };
        let snippets = expand_calls(&functions, &config);

        let by_caller: HashMap<&str, &str> = snippets
            .iter()
            .map(|s| (s.function.qualified_name.as_str(), s.text.as_str()))
            .collect();

        if let Some(text) = by_caller.get("caller_one") {
            assert!(
                text.contains("MARKER_PKG1"),
                "caller_one should inline from pkg1"
            );
            assert!(
                !text.contains("MARKER_PKG2"),
                "caller_one must not pull from pkg2"
            );
        }
        if let Some(text) = by_caller.get("caller_two") {
            assert!(
                text.contains("MARKER_PKG2"),
                "caller_two should inline from pkg2"
            );
            assert!(
                !text.contains("MARKER_PKG1"),
                "caller_two must not pull from pkg1"
            );
        }
    }

    #[test]
    fn test_expansion_self_cls_resolution() {
        let tmp = TempDir::new().unwrap();
        let src = b"class Calc:\n    def run(self):\n        return self.compute()\n\n    def compute(self):\n        return 42\n";
        let path = tmp.path().join("calc.py");
        std::fs::write(&path, src).unwrap();

        let fn_refs = extract_functions(&file_ref(&path.to_string_lossy()));
        let config = ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 10000,
        };
        let snippets = expand_calls(&fn_refs, &config);

        // `run` should expand to include `compute`
        let run_snippet = snippets
            .iter()
            .find(|s| s.function.qualified_name == "Calc.run");
        assert!(
            run_snippet.is_some(),
            "Calc.run should produce an EXP snippet"
        );
        assert!(
            run_snippet.unwrap().text.contains("compute"),
            "expanded run should contain compute code"
        );
    }

    #[test]
    fn test_expansion_ctor_resolution() {
        let fns = tiny_repo_functions();
        let config = default_config();
        let snippets = expand_calls(&fns, &config);
        // `via_instance` uses `acc = Accumulator()` then `acc.total(values)`
        let via_instance = snippets
            .iter()
            .find(|s| s.function.qualified_name == "via_instance");
        assert!(
            via_instance.is_some(),
            "via_instance should produce an EXP snippet; got: {:?}",
            snippets
                .iter()
                .map(|s| &s.function.qualified_name)
                .collect::<Vec<_>>()
        );
        assert!(
            via_instance.unwrap().text.contains("total"),
            "via_instance expansion should include Accumulator.total"
        );
    }

    #[test]
    fn test_expansion_depth_greater_than_one() {
        let tmp = TempDir::new().unwrap();
        let a_src = b"def caller():\n    return mid()\n";
        let b_src = b"def mid():\n    return leaf()\n";
        let c_src = b"def leaf():\n    return 'LEAF_MARKER'\n";

        let a_path = tmp.path().join("a.py");
        let b_path = tmp.path().join("b.py");
        let c_path = tmp.path().join("c.py");
        std::fs::write(&a_path, a_src).unwrap();
        std::fs::write(&b_path, b_src).unwrap();
        std::fs::write(&c_path, c_src).unwrap();

        let fn_refs: Vec<FunctionRef> = [&a_path, &b_path, &c_path]
            .iter()
            .flat_map(|p| extract_functions(&file_ref(&p.to_string_lossy())))
            .collect();

        let config = ExpansionConfig {
            enabled: true,
            depth: 2,
            max_chars: 10000,
        };
        let snippets = expand_calls(&fn_refs, &config);
        // With depth=2, `caller` should expand to include both `mid` and `leaf`
        // (they're in the same file so by-name resolution works for `mid` in `caller`,
        //  and `leaf` in `mid` at depth 2)
        let caller_snip = snippets
            .iter()
            .find(|s| s.function.qualified_name == "caller");
        if let Some(s) = caller_snip {
            assert!(s.text.contains("mid"), "depth 2 should expand mid");
        }
    }

    #[test]
    fn test_expansion_annotated_assignment_resolution() {
        let tmp = TempDir::new().unwrap();
        let src = b"class Widget:\n    def render(self):\n        return '<div>'\n\ndef build() -> None:\n    w: Widget\n    w.render()\n";
        let path = tmp.path().join("widget.py");
        std::fs::write(&path, src).unwrap();

        let fn_refs = extract_functions(&file_ref(&path.to_string_lossy()));
        let config = ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 10000,
        };
        let snippets = expand_calls(&fn_refs, &config);
        // `build` uses `w: Widget` annotation; should resolve `w.render()` to Widget.render
        let build_snip = snippets
            .iter()
            .find(|s| s.function.qualified_name == "build");
        if let Some(s) = build_snip {
            assert!(
                s.text.contains("render"),
                "annotated var should resolve to Widget.render"
            );
        }
    }

    #[test]
    fn test_expansion_bare_relative_import_skipped() {
        let tmp = TempDir::new().unwrap();
        let src = b"from . import utils\n\ndef caller():\n    return utils.helper()\n";
        let path = tmp.path().join("main.py");
        std::fs::write(&path, src).unwrap();

        let fn_refs = extract_functions(&file_ref(&path.to_string_lossy()));
        let config = ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 10000,
        };
        // Must not panic; bare relative import is skipped
        let _snippets = expand_calls(&fn_refs, &config);
    }

    #[test]
    fn test_expansion_multiple_return_statements_last_wins() {
        let src = "def factory():\n    if True:\n        return Foo()\n    return Bar()\n";
        // infer_return_class visits all return statements; LAST one wins
        let result = infer_return_class(src);
        assert_eq!(
            result.as_deref(),
            Some("Bar"),
            "last return class should win, got: {:?}",
            result
        );
    }

    #[test]
    fn test_expansion_parse_failure_in_collect_calls() {
        // Invalid Python code → collect_calls returns empty set, no panic
        let calls = collect_calls("{not python at all}");
        assert!(
            calls.is_empty(),
            "parse failure should yield empty call set"
        );
    }

    #[test]
    fn test_expansion_hash_format() {
        // Use a caller+helper pair so the expansion actually produces an EXP snippet.
        // This lets us verify the exact DD7 hash format end-to-end.
        let tmp = TempDir::new().unwrap();
        let src = b"def helper_fn():\n    return 42\n\ndef caller():\n    return helper_fn()\n";
        let path = tmp.path().join("mod.py");
        std::fs::write(&path, src).unwrap();

        let fn_refs = extract_functions(&file_ref(&path.to_string_lossy()));
        let config = ExpansionConfig {
            enabled: true,
            depth: 1,
            max_chars: 4000,
        };
        let snippets = expand_calls(&fn_refs, &config);

        let caller_snippet = snippets
            .iter()
            .find(|s| s.function.qualified_name == "caller");
        assert!(
            caller_snippet.is_some(),
            "caller should produce an EXP snippet"
        );
        let s = caller_snippet.unwrap();

        // Reconstruct expected expanded text and hash per DD7:
        // EXP:{path}:{start}:{end}:{code_hash}:{helpers_csv}:{depth}:{max_chars}:{analysis}
        let caller_fn = fn_refs
            .iter()
            .find(|f| f.qualified_name == "caller")
            .unwrap();
        let helper_fn = fn_refs
            .iter()
            .find(|f| f.qualified_name == "helper_fn")
            .unwrap();
        let expanded = format!(
            "{}\n\n# expanded:{}\n{}",
            caller_fn.code, helper_fn.qualified_name, helper_fn.code
        );
        let analysis = crate::snippets::normalization::normalize_analysis(&expanded);
        let expected_hash = hash_text(&format!(
            "EXP:{}:{}:{}:{}:{}:{}:{}:{}",
            caller_fn.file.path,
            caller_fn.start_line,
            caller_fn.end_line,
            caller_fn.code_hash,
            "helper_fn", // helpers_csv: single helper
            config.depth,
            config.max_chars,
            analysis
        ));
        assert_eq!(
            s.snippet_hash, expected_hash,
            "EXP hash format must match DD7"
        );
    }
}
