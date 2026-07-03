//! T1b: tree-sitter vs Python ast function extraction parity check.
#![allow(dead_code)]
//!
//! In tree-sitter-python 0.23.x:
//! - Regular and async functions share the `function_definition` node type.
//!   Async is detected by "async" appearing first in the node's source text.
//! - Decorated functions are wrapped in `decorated_definition` with a `definition`
//!   field pointing to the inner `function_definition` (or `class_definition`).
//! - There is NO `async_function_statement` node type.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;
use tree_sitter::{Language, Node, Parser};

/// Reference function datum from python_functions.json.
#[derive(Deserialize, Debug)]
pub struct RefFunction {
    pub file: String,
    pub qualified_name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub is_async: bool,
    pub code: String,
}

/// Function extracted by tree-sitter.
#[derive(Debug, Clone)]
pub struct TsFunction {
    pub file: String,
    pub qualified_name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub is_async: bool,
    pub code: String,
}

fn make_parser() -> Result<Parser> {
    let mut parser = Parser::new();
    let language: Language = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&language)?;
    Ok(parser)
}

/// Detect whether a `function_definition` node is async by inspecting source text.
/// In tree-sitter-python 0.23, async functions start with "async" before "def".
fn is_async_function(node: &Node, source: &str) -> bool {
    let start = node.start_byte();
    let snippet = &source[start..(start + 10).min(source.len())];
    snippet.trim_start().starts_with("async")
}

/// Emit one function from its `function_definition` node.
fn emit_function<'a>(
    func_node: Node<'a>,
    source: &str,
    lines: &[&str],
    name_stack: &mut Vec<String>,
    results: &mut Vec<TsFunction>,
    file: &str,
) {
    let func_name = func_node
        .child_by_field_name("name")
        .map(|n| &source[n.start_byte()..n.end_byte()])
        .unwrap_or("<unknown>")
        .to_string();

    let is_async = is_async_function(&func_node, source);
    let start_row = func_node.start_position().row;
    let end_row = func_node.end_position().row;
    let start_line = (start_row + 1) as u32;
    let end_line = (end_row + 1) as u32;

    let code_end = (end_row + 1).min(lines.len());
    let code = lines[start_row..code_end].join("\n");

    let qname = if name_stack.is_empty() {
        func_name.clone()
    } else {
        format!("{}.{}", name_stack.join("."), func_name)
    };

    results.push(TsFunction {
        file: file.to_string(),
        qualified_name: qname,
        start_line,
        end_line,
        is_async,
        code,
    });

    // Recurse into body for nested functions/classes
    name_stack.push(func_name);
    if let Some(body) = func_node.child_by_field_name("body") {
        for i in 0..body.child_count() {
            walk_node(body.child(i).unwrap(), source, lines, name_stack, results, file);
        }
    }
    name_stack.pop();
}

/// Walk the CST and collect function definitions.
fn walk_node<'a>(
    node: Node<'a>,
    source: &str,
    lines: &[&str],
    name_stack: &mut Vec<String>,
    results: &mut Vec<TsFunction>,
    file: &str,
) {
    match node.kind() {
        "function_definition" => {
            emit_function(node, source, lines, name_stack, results, file);
        }
        "decorated_definition" => {
            // The `definition` field is the function_definition or class_definition.
            if let Some(def_node) = node.child_by_field_name("definition") {
                match def_node.kind() {
                    "function_definition" => {
                        emit_function(def_node, source, lines, name_stack, results, file);
                    }
                    "class_definition" => {
                        let class_name = def_node
                            .child_by_field_name("name")
                            .map(|n| &source[n.start_byte()..n.end_byte()])
                            .unwrap_or("<unknown>")
                            .to_string();
                        name_stack.push(class_name);
                        if let Some(body) = def_node.child_by_field_name("body") {
                            for i in 0..body.child_count() {
                                walk_node(
                                    body.child(i).unwrap(),
                                    source,
                                    lines,
                                    name_stack,
                                    results,
                                    file,
                                );
                            }
                        }
                        name_stack.pop();
                    }
                    _ => {
                        // Recurse into unknown node kinds
                        for i in 0..def_node.child_count() {
                            walk_node(
                                def_node.child(i).unwrap(),
                                source,
                                lines,
                                name_stack,
                                results,
                                file,
                            );
                        }
                    }
                }
            }
        }
        "class_definition" => {
            let class_name = node
                .child_by_field_name("name")
                .map(|n| &source[n.start_byte()..n.end_byte()])
                .unwrap_or("<unknown>")
                .to_string();
            name_stack.push(class_name);
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.child_count() {
                    walk_node(body.child(i).unwrap(), source, lines, name_stack, results, file);
                }
            }
            name_stack.pop();
        }
        _ => {
            // Recurse into all other node types (module, if_statement, etc.)
            for i in 0..node.child_count() {
                walk_node(node.child(i).unwrap(), source, lines, name_stack, results, file);
            }
        }
    }
}

/// Extract functions from a Python source file using tree-sitter.
/// Returns [] on parse error or if the tree has syntax errors — matching Python's behavior.
pub fn extract_functions_ts(path: &Path) -> Result<Vec<TsFunction>> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Ok(vec![]),
    };

    let mut parser = make_parser()?;
    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return Ok(vec![]),
    };

    let root = tree.root_node();
    // If tree has syntax errors, return [] matching Python's ast.parse SyntaxError handling
    if root.has_error() {
        return Ok(vec![]);
    }

    let lines: Vec<&str> = source.lines().collect();
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let mut results = Vec::new();
    let mut name_stack = Vec::new();
    walk_node(root, &source, &lines, &mut name_stack, &mut results, file_name);

    Ok(results)
}

/// T1b comparison result.
pub struct T1bResult {
    pub total_ref: usize,
    pub total_rust: usize,
    pub qname_matches: usize,
    pub code_text_matches: usize,
    pub span_diffs: Vec<String>,
    pub adjustments_applied: Vec<String>,
    pub lambda_exclusion_ok: bool,
}

pub fn run_t1b(parse_targets_dir: &Path, ref_functions: &[RefFunction]) -> T1bResult {
    let mut all_rust: Vec<TsFunction> = Vec::new();

    let mut py_files: Vec<_> = match std::fs::read_dir(parse_targets_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("py"))
            .map(|e| e.path())
            .collect(),
        Err(e) => {
            eprintln!("[T1b] cannot read parse_targets dir: {e}");
            return T1bResult {
                total_ref: ref_functions.len(),
                total_rust: 0,
                qname_matches: 0,
                code_text_matches: 0,
                span_diffs: vec![],
                adjustments_applied: vec![],
                lambda_exclusion_ok: true,
            };
        }
    };
    py_files.sort();

    for path in &py_files {
        match extract_functions_ts(path) {
            Ok(fns) => all_rust.extend(fns),
            Err(e) => eprintln!("[T1b] parse error for {:?}: {e}", path),
        }
    }

    let total_ref = ref_functions.len();
    let total_rust = all_rust.len();
    let mut qname_matches = 0usize;
    let mut code_text_matches = 0usize;
    let mut span_diffs = Vec::new();
    let adjustments_applied = Vec::new();

    for ref_fn in ref_functions {
        // Find matching rust extraction by (file, qname)
        let rust_match = all_rust
            .iter()
            .find(|r| r.file == ref_fn.file && r.qualified_name == ref_fn.qualified_name);

        match rust_match {
            Some(rm) => {
                qname_matches += 1;

                // Compare line spans
                if rm.start_line != ref_fn.start_line || rm.end_line != ref_fn.end_line {
                    span_diffs.push(format!(
                        "{}/{}: py=[{},{}] rust=[{},{}]",
                        ref_fn.file,
                        ref_fn.qualified_name,
                        ref_fn.start_line,
                        ref_fn.end_line,
                        rm.start_line,
                        rm.end_line,
                    ));
                }

                // Compare code text byte-for-byte
                if rm.code == ref_fn.code {
                    code_text_matches += 1;
                } else {
                    eprintln!(
                        "[T1b] code mismatch: {}/{} py_len={} rust_len={}",
                        ref_fn.file,
                        ref_fn.qualified_name,
                        ref_fn.code.len(),
                        rm.code.len()
                    );
                    // Show the first differing line
                    for (li, (pl, rl)) in
                        ref_fn.code.lines().zip(rm.code.lines()).enumerate()
                    {
                        if pl != rl {
                            eprintln!("    line {li}: py={pl:?}");
                            eprintln!("    line {li}: rs={rl:?}");
                            break;
                        }
                    }
                }
            }
            None => {
                eprintln!(
                    "[T1b] NO MATCH for ref: {}/{}",
                    ref_fn.file, ref_fn.qualified_name
                );
            }
        }
    }

    // Lambda exclusion: none of the rust results should have <lambda> in their qname
    let lambda_exclusion_ok = !all_rust.iter().any(|f| f.qualified_name.contains("<lambda>"));
    if !lambda_exclusion_ok {
        eprintln!("[T1b] FAIL: lambda found in extracted functions");
    }

    T1bResult {
        total_ref,
        total_rust,
        qname_matches,
        code_text_matches,
        span_diffs,
        adjustments_applied,
        lambda_exclusion_ok,
    }
}
