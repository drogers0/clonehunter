// allow dead_code until T12 wires cli::run to the pipeline
#![allow(dead_code)]

use tree_sitter::{Language as TsLanguage, Node, Parser};

use crate::core::types::{FileRef, FunctionRef};
use crate::io::fingerprints::hash_text;

/// Extract functions from a Python source file via tree-sitter.
/// Returns empty Vec on any error (IO, parse failure, syntax errors). Never panics.
pub(crate) fn extract_functions(file: &FileRef) -> Vec<FunctionRef> {
    let bytes = match std::fs::read(&file.path) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let source = String::from_utf8_lossy(&bytes).into_owned();

    let mut parser = Parser::new();
    let language: TsLanguage = tree_sitter_python::LANGUAGE.into();
    if parser.set_language(&language).is_err() {
        return vec![];
    }

    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return vec![],
    };

    let root = tree.root_node();
    if root.has_error() {
        return vec![];
    }

    let lines: Vec<&str> = source.lines().collect();
    let mut results = Vec::new();
    let mut name_stack: Vec<String> = Vec::new();
    walk_node(root, &source, &lines, &mut name_stack, &mut results, file);
    results
}

/// Emit one FunctionRef from a `function_definition` node and recurse into its body.
fn emit_function(
    func_node: Node<'_>,
    source: &str,
    lines: &[&str],
    name_stack: &mut Vec<String>,
    results: &mut Vec<FunctionRef>,
    file: &FileRef,
) {
    let func_name = func_node
        .child_by_field_name("name")
        .map(|n| source[n.start_byte()..n.end_byte()].to_string())
        .unwrap_or_else(|| "<unknown>".to_string());

    let qualified_name = if name_stack.is_empty() {
        func_name.clone()
    } else {
        format!("{}.{}", name_stack.join("."), func_name)
    };

    let start_row = func_node.start_position().row;
    let end_row = func_node.end_position().row;
    let start_line = start_row + 1;
    let end_line = end_row + 1;

    let code_end = (end_row + 1).min(lines.len());
    let code = lines[start_row..code_end].join("\n");
    let code_hash = hash_text(&code);

    results.push(FunctionRef {
        file: file.clone(),
        qualified_name,
        start_line,
        end_line,
        code,
        code_hash,
    });

    // Recurse into body for nested functions/classes
    name_stack.push(func_name);
    if let Some(body) = func_node.child_by_field_name("body") {
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

/// Walk the CST and collect all function definitions recursively.
/// Lambdas are never matched (no "lambda" case), so they are excluded.
fn walk_node(
    node: Node<'_>,
    source: &str,
    lines: &[&str],
    name_stack: &mut Vec<String>,
    results: &mut Vec<FunctionRef>,
    file: &FileRef,
) {
    match node.kind() {
        "function_definition" => {
            emit_function(node, source, lines, name_stack, results, file);
        }
        "decorated_definition" => {
            // `definition` field points to the inner function_definition or class_definition.
            if let Some(def_node) = node.child_by_field_name("definition") {
                match def_node.kind() {
                    "function_definition" => {
                        // Span starts at `def`, not at decorator — matches Python's ast.FunctionDef.lineno
                        emit_function(def_node, source, lines, name_stack, results, file);
                    }
                    "class_definition" => {
                        let class_name = def_node
                            .child_by_field_name("name")
                            .map(|n| source[n.start_byte()..n.end_byte()].to_string())
                            .unwrap_or_else(|| "<unknown>".to_string());
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
                .map(|n| source[n.start_byte()..n.end_byte()].to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            name_stack.push(class_name);
            if let Some(body) = node.child_by_field_name("body") {
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
            for i in 0..node.child_count() {
                walk_node(
                    node.child(i).unwrap(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::path::Path;
    use tempfile::NamedTempFile;

    use crate::core::types::Language;

    fn make_file_ref(path: &str, language: Language) -> FileRef {
        FileRef {
            path: path.to_string(),
            content_hash: String::new(),
            language,
        }
    }

    fn fixture(rel_path: &str) -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(rel_path)
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_extract_functions_from_fixture() {
        let path = fixture("fixtures/tiny_repo/a.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        let names: Vec<&str> = fns.iter().map(|f| f.qualified_name.as_str()).collect();
        assert!(
            names.contains(&"add_numbers"),
            "missing add_numbers, got: {:?}",
            names
        );
        assert!(
            names.contains(&"sum_list"),
            "missing sum_list, got: {:?}",
            names
        );
        assert!(
            names.contains(&"wrapper"),
            "missing wrapper, got: {:?}",
            names
        );
    }

    #[test]
    fn test_extract_functions_skips_invalid_syntax() {
        let mut tmp = NamedTempFile::with_suffix(".py").unwrap();
        tmp.write_all(b"def oops(:\n  pass\n").unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        assert!(
            fns.is_empty(),
            "expected empty result for invalid syntax, got {} fns",
            fns.len()
        );
    }

    #[test]
    fn test_qualified_names_nested() {
        let path = fixture("spike/fixtures/parse_targets/05_nested.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        let names: Vec<&str> = fns.iter().map(|f| f.qualified_name.as_str()).collect();
        assert!(
            names.contains(&"Processor.process"),
            "missing Processor.process, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Processor.process._validate"),
            "missing Processor.process._validate, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Processor.process._transform"),
            "missing Processor.process._transform, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Processor.pipeline"),
            "missing Processor.pipeline, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Processor.pipeline.apply_step"),
            "missing Processor.pipeline.apply_step, got: {:?}",
            names
        );
    }

    #[test]
    fn test_async_functions() {
        let path = fixture("spike/fixtures/parse_targets/04_async_func.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        let names: Vec<&str> = fns.iter().map(|f| f.qualified_name.as_str()).collect();
        assert!(
            names.contains(&"fetch_data"),
            "missing fetch_data, got: {:?}",
            names
        );
        assert!(
            names.contains(&"process"),
            "missing process, got: {:?}",
            names
        );
        assert!(
            names.contains(&"gather_results"),
            "missing gather_results, got: {:?}",
            names
        );
        // Verify async functions have code text starting with "async def"
        let fetch = fns
            .iter()
            .find(|f| f.qualified_name == "fetch_data")
            .unwrap();
        assert!(
            fetch.code.trim_start().starts_with("async def"),
            "expected async def in code"
        );
    }

    #[test]
    fn test_decorated_function_spans() {
        let path = fixture("spike/fixtures/parse_targets/06_decorated.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);

        // my_function: @decorator on line 8, def on line 9 → start_line = 9
        let my_fn = fns
            .iter()
            .find(|f| f.qualified_name == "my_function")
            .unwrap();
        assert_eq!(
            my_fn.start_line, 9,
            "my_function start_line should be at def, not decorator"
        );
        // Code should start with def, not @
        assert!(
            my_fn.code.trim_start().starts_with("def "),
            "my_function code should start with def, got: {:?}",
            &my_fn.code[..my_fn.code.len().min(30)]
        );

        // another_function: @decorator on line 13, def on line 14 → start_line = 14
        let another = fns
            .iter()
            .find(|f| f.qualified_name == "another_function")
            .unwrap();
        assert_eq!(
            another.start_line, 14,
            "another_function start_line should be at def"
        );
    }

    #[test]
    fn test_lambda_exclusion() {
        let path = fixture("spike/fixtures/parse_targets/09_lambda_default.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        let lambda_found = fns.iter().any(|f| f.qualified_name.contains("<lambda>"));
        assert!(
            !lambda_found,
            "lambda should be excluded, got: {:?}",
            fns.iter().map(|f| &f.qualified_name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_class_methods() {
        let path = fixture("spike/fixtures/parse_targets/03_class_methods.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        let names: Vec<&str> = fns.iter().map(|f| f.qualified_name.as_str()).collect();
        assert!(
            names.contains(&"Calculator.add"),
            "missing Calculator.add, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Calculator.subtract"),
            "missing Calculator.subtract"
        );
        assert!(
            names.contains(&"InheritedCalc.multiply"),
            "missing InheritedCalc.multiply"
        );
    }

    #[test]
    fn test_empty_file() {
        let mut tmp = NamedTempFile::with_suffix(".py").unwrap();
        tmp.write_all(b"").unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        assert!(fns.is_empty(), "expected empty result for empty file");
    }

    #[test]
    fn test_non_utf8_file() {
        let mut tmp = NamedTempFile::with_suffix(".py").unwrap();
        tmp.write_all(&[0xff, 0xfe]).unwrap();
        let path = tmp.path().to_string_lossy().into_owned();
        let file = make_file_ref(&path, Language::Python);
        // Must not panic; lossy read + parse failure → empty
        let fns = extract_functions(&file);
        // Could be empty (parse error) or contain some result — just must not panic
        let _ = fns;
    }

    #[test]
    fn test_code_hash_matches_hash_text() {
        let path = fixture("fixtures/tiny_repo/helpers.py");
        let file = make_file_ref(&path, Language::Python);
        let fns = extract_functions(&file);
        assert!(!fns.is_empty());
        for fn_ in &fns {
            let expected = hash_text(&fn_.code);
            assert_eq!(
                fn_.code_hash, expected,
                "code_hash mismatch for {}",
                fn_.qualified_name
            );
        }
    }
}
