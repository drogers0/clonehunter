use tree_sitter::{Language as TsLanguage, Node, Parser};

/// A pending replacement over a byte range in the source string.
struct Replacement {
    /// Byte offset of the start of the docstring's line (before indentation).
    start: usize,
    /// Byte offset just past the line's trailing newline (or source.len() at EOF).
    end: usize,
    /// Replacement text, e.g. `"    pass\n"`.
    text: String,
}

/// Analysis text: docstrings → pass, comments stripped, source passthrough otherwise.
/// Drives embeddings, cache keys, and lexical scoring.
pub(crate) fn normalize_analysis(source: &str) -> String {
    let Some((tree, src)) = parse(source) else {
        return source.to_string();
    };
    let root = tree.root_node();
    let docstrings = find_docstring_replacements(&root, src);
    let comments = find_comment_ranges(&root);
    apply_replacements(src, &docstrings, Some(&comments))
}

/// Display text: docstrings → pass, comments preserved.
/// Used only by reporters for rendered diffs.
pub(crate) fn normalize_display(source: &str) -> String {
    let Some((tree, src)) = parse(source) else {
        return source.to_string();
    };
    let root = tree.root_node();
    let docstrings = find_docstring_replacements(&root, src);
    apply_replacements(src, &docstrings, None)
}

/// Parse `source` with tree-sitter-python. Returns `None` on parse failure or if the
/// tree contains error nodes (raw-source fallback triggers in callers).
fn parse(source: &str) -> Option<(tree_sitter::Tree, &str)> {
    let mut parser = Parser::new();
    let language: TsLanguage = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(source.as_bytes(), None)?;
    if tree.root_node().has_error() {
        return None;
    }
    Some((tree, source))
}

/// Recursively find docstring nodes to replace per the locked normalization contract (DD3):
/// - module: treat root as its own body
/// - function_definition: strip first string statement from body
/// - class_definition: do NOT strip class docstrings; recurse into body for methods
fn find_docstring_replacements(node: &Node<'_>, source: &str) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    find_docstrings_recursive(node, source, &mut replacements);
    replacements
}

fn find_docstrings_recursive(node: &Node<'_>, source: &str, out: &mut Vec<Replacement>) {
    match node.kind() {
        "module" => {
            // Module root acts as its own body — look for module-level docstring
            try_replace_docstring(node, source, out);
            for i in 0..node.child_count() {
                find_docstrings_recursive(&node.child(i).unwrap(), source, out);
            }
        }
        "function_definition" => {
            if let Some(body) = node.child_by_field_name("body") {
                try_replace_docstring(&body, source, out);
            }
            // Recurse to find nested functions/classes
            for i in 0..node.child_count() {
                find_docstrings_recursive(&node.child(i).unwrap(), source, out);
            }
        }
        "class_definition" => {
            // DD3: do NOT strip class docstrings. Only recurse into body to find methods.
            if let Some(body) = node.child_by_field_name("body") {
                for i in 0..body.child_count() {
                    find_docstrings_recursive(&body.child(i).unwrap(), source, out);
                }
            }
        }
        _ => {
            for i in 0..node.child_count() {
                find_docstrings_recursive(&node.child(i).unwrap(), source, out);
            }
        }
    }
}

/// Check if the body's first non-comment/non-newline child is a string literal (docstring).
/// If so, record the replacement: the entire line range → `{indent}pass\n`.
fn try_replace_docstring(body: &Node<'_>, source: &str, out: &mut Vec<Replacement>) {
    let first_stmt = (0..body.child_count())
        .filter_map(|i| body.child(i))
        .find(|c| !matches!(c.kind(), "comment" | "\n" | "newline"));

    let stmt = match first_stmt {
        Some(s) => s,
        None => return,
    };

    if stmt.kind() != "expression_statement" {
        return;
    }

    // The expression_statement should contain a string node as its first meaningful child
    let is_string = (0..stmt.child_count())
        .filter_map(|i| stmt.child(i))
        .find(|c| !matches!(c.kind(), "comment" | "\n"))
        .map(|c| matches!(c.kind(), "string" | "concatenated_string"))
        .unwrap_or(false);

    if !is_string {
        return;
    }

    let start_byte = stmt.start_byte();
    let end_byte = stmt.end_byte();

    // Walk back to start of the docstring's line (captures any leading indentation)
    let line_start = source[..start_byte].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let indent = &source[line_start..start_byte];

    // Walk forward to consume the trailing newline
    let line_end = source[end_byte..]
        .find('\n')
        .map(|i| end_byte + i + 1)
        .unwrap_or(source.len());

    out.push(Replacement {
        start: line_start,
        end: line_end,
        text: format!("{}pass\n", indent),
    });
}

/// Collect the byte ranges of all `comment` nodes in the tree (DD4).
fn find_comment_ranges(root: &Node<'_>) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    collect_comments(root, &mut ranges);
    ranges
}

fn collect_comments(node: &Node<'_>, out: &mut Vec<(usize, usize)>) {
    if node.kind() == "comment" {
        out.push((node.start_byte(), node.end_byte()));
    }
    for i in 0..node.child_count() {
        collect_comments(&node.child(i).unwrap(), out);
    }
}

/// Apply docstring and optional comment replacements to `source`.
///
/// Replacements are applied in reverse byte-offset order so earlier offsets stay valid.
/// Comment ranges that fall entirely within a docstring replacement range are filtered
/// out to avoid double-replacement corruption.
fn apply_replacements(
    source: &str,
    docstrings: &[Replacement],
    comments: Option<&[(usize, usize)]>,
) -> String {
    // Build unified list: (start, end, replacement_text)
    let mut ops: Vec<(usize, usize, String)> = docstrings
        .iter()
        .map(|r| (r.start, r.end, r.text.clone()))
        .collect();

    if let Some(comment_ranges) = comments {
        for &(cs, ce) in comment_ranges {
            // Skip comment ranges entirely enclosed by a docstring replacement (DD4 overlap)
            let inside_docstring = docstrings.iter().any(|d| cs >= d.start && ce <= d.end);
            if !inside_docstring {
                ops.push((cs, ce, String::new()));
            }
        }
    }

    // Apply in reverse order to preserve byte offsets of earlier operations
    ops.sort_by(|a, b| b.0.cmp(&a.0));

    let mut result = source.to_string();
    for (start, end, text) in ops {
        result.replace_range(start..end, &text);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_analysis_strips_docstring_and_comments() {
        let src = "def foo():\n    \"\"\"docstring\"\"\"\n    # comment\n    return 1\n";
        let analysis = normalize_analysis(src);
        let display = normalize_display(src);

        // Analysis: docstring → pass, comment stripped
        assert!(
            analysis.contains("pass"),
            "analysis should contain pass: {analysis:?}"
        );
        assert!(
            !analysis.contains("docstring"),
            "analysis should strip docstring: {analysis:?}"
        );
        assert!(
            !analysis.contains("# comment"),
            "analysis should strip comment: {analysis:?}"
        );

        // Display: docstring → pass, comment preserved
        assert!(
            display.contains("pass"),
            "display should contain pass: {display:?}"
        );
        assert!(
            !display.contains("docstring"),
            "display should strip docstring: {display:?}"
        );
        assert!(
            display.contains("# comment"),
            "display should preserve comment: {display:?}"
        );
    }

    #[test]
    fn test_normalize_multiline_docstring() {
        let src = "def foo():\n    \"\"\"\n    multi\n    line\n    \"\"\"\n    return 1\n";
        let analysis = normalize_analysis(src);
        assert!(
            analysis.contains("pass"),
            "should replace multi-line docstring with pass"
        );
        assert!(
            !analysis.contains("multi"),
            "multi-line docstring body should be gone"
        );
        assert!(
            !analysis.contains("line"),
            "multi-line docstring body should be gone"
        );
    }

    #[test]
    fn test_normalize_inline_comment() {
        let src = "x = 1  # inline comment\n";
        let analysis = normalize_analysis(src);
        let display = normalize_display(src);
        assert!(
            !analysis.contains("# inline comment"),
            "analysis should strip inline comment"
        );
        assert!(analysis.contains("x = 1"), "analysis should keep code");
        assert!(
            display.contains("# inline comment"),
            "display should keep inline comment"
        );
    }

    #[test]
    fn test_normalize_no_docstring_no_comment() {
        let src = "x = 1\ny = 2\n";
        let analysis = normalize_analysis(src);
        let display = normalize_display(src);
        assert_eq!(analysis, src, "no changes expected for plain code");
        assert_eq!(display, src, "no changes expected for plain code");
    }

    #[test]
    fn test_normalize_parse_failure_fallback() {
        // Non-Python code or syntax error → raw source returned
        let src = "{not python}";
        assert_eq!(normalize_analysis(src), src);
        assert_eq!(normalize_display(src), src);
    }

    #[test]
    fn test_normalize_empty_source() {
        assert_eq!(normalize_analysis(""), "");
        assert_eq!(normalize_display(""), "");
    }

    #[test]
    fn test_normalize_module_docstring() {
        // A top-level string should be treated as a module docstring and stripped
        let src = "\"\"\"Module docstring.\"\"\"\n\nx = 1\n";
        let analysis = normalize_analysis(src);
        assert!(
            analysis.contains("pass"),
            "module docstring should be replaced: {analysis:?}"
        );
        assert!(
            !analysis.contains("Module docstring"),
            "module docstring should be gone"
        );
        assert!(analysis.contains("x = 1"), "rest of code should remain");
    }

    #[test]
    fn test_normalize_comment_before_docstring() {
        // A comment before the docstring should be skipped; docstring still detected and stripped
        let src = "def foo():\n    # comment first\n    \"\"\"real docstring\"\"\"\n    return 1\n";
        let analysis = normalize_analysis(src);
        assert!(
            analysis.contains("pass"),
            "docstring should still be stripped: {analysis:?}"
        );
        assert!(
            !analysis.contains("real docstring"),
            "docstring text should be gone"
        );
    }

    #[test]
    fn test_normalize_async_function_docstring() {
        let src = "async def foo():\n    \"\"\"doc\"\"\"\n    pass\n";
        let analysis = normalize_analysis(src);
        let display = normalize_display(src);
        assert!(
            analysis.contains("pass"),
            "async function docstring should be stripped"
        );
        assert!(
            !analysis.contains("doc"),
            "docstring text should be gone in analysis"
        );
        assert!(
            !display.contains("doc"),
            "docstring text should be gone in display"
        );
    }

    #[test]
    fn test_normalize_class_docstring_preserved() {
        // DD3: class docstrings are NOT stripped (only module + function)
        let src =
            "class Foo:\n    \"\"\"class docstring\"\"\"\n    def method(self):\n        pass\n";
        let analysis = normalize_analysis(src);
        // Class docstring must remain as-is
        assert!(
            analysis.contains("class docstring"),
            "class docstring must NOT be stripped (DD3): {analysis:?}"
        );
    }

    #[test]
    fn test_normalize_nested_function_docstring() {
        let src = "def outer():\n    def inner():\n        \"\"\"inner doc\"\"\"\n        return 1\n    return inner\n";
        let analysis = normalize_analysis(src);
        assert!(
            analysis.contains("pass"),
            "inner function docstring should be stripped"
        );
        assert!(
            !analysis.contains("inner doc"),
            "inner docstring text should be gone"
        );
    }
}
