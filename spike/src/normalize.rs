//! T1c: Normalization definition and impact measurement.
#![allow(dead_code)]
//!
//! Implements DD7: tree-sitter docstring strip (→ pass) + source-preserving passthrough.
//! Compares against Python's `ast.unparse(strip_docstrings(...))` reference.

use regex::Regex;
use std::sync::OnceLock;
use tree_sitter::{Language, Node, Parser};

/// Normalize Python source per DD7:
/// - Identify docstrings as `expression_statement > string` at body position 0
///   of `function_definition`, `class_definition`, or module.
/// - Replace the entire expression_statement's line range with `{indent}pass\n`.
/// - Everything else is preserved as-is (whitespace, comments, quotes).
pub fn normalize_python(source: &str) -> String {
    let mut parser = Parser::new();
    let language: Language = tree_sitter_python::LANGUAGE.into();
    if parser.set_language(&language).is_err() {
        return source.to_string();
    }

    let tree = match parser.parse(source.as_bytes(), None) {
        Some(t) => t,
        None => return source.to_string(),
    };

    // Collect replacement ranges: (line_start_byte, line_end_byte, replacement_string)
    // Collected in ascending order; applied in reverse so byte offsets stay valid.
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    find_docstrings(&tree.root_node(), source, &mut replacements);

    // Apply replacements in reverse order
    let mut result = source.to_string();
    replacements.sort_by_key(|&(start, _, _)| start);
    for (line_start, line_end, replacement) in replacements.iter().rev() {
        result.replace_range(*line_start..*line_end, replacement);
    }

    result
}

/// Recursively find docstring nodes and record their replacement ranges.
fn find_docstrings(node: &Node, source: &str, replacements: &mut Vec<(usize, usize, String)>) {
    match node.kind() {
        "module" | "function_definition" | "async_function_statement" => {
            // Check body for first expression_statement that is a docstring
            let body_node = if node.kind() == "module" {
                Some(*node)
            } else {
                node.child_by_field_name("body")
            };

            if let Some(body) = body_node {
                try_replace_docstring(&body, source, replacements);
            }

            // Recurse into children
            for i in 0..node.child_count() {
                find_docstrings(&node.child(i).unwrap(), source, replacements);
            }
        }
        "class_definition" => {
            if let Some(body) = node.child_by_field_name("body") {
                try_replace_docstring(&body, source, replacements);
                // Recurse into body children
                for i in 0..body.child_count() {
                    find_docstrings(&body.child(i).unwrap(), source, replacements);
                }
            }
        }
        _ => {
            for i in 0..node.child_count() {
                find_docstrings(&node.child(i).unwrap(), source, replacements);
            }
        }
    }
}

/// Check if the body's first non-comment statement is a string literal (docstring).
/// If so, record the replacement range.
fn try_replace_docstring(
    body: &Node,
    source: &str,
    replacements: &mut Vec<(usize, usize, String)>,
) {
    // Find the first non-comment child statement
    let first_stmt = (0..body.child_count())
        .filter_map(|i| body.child(i))
        .find(|c| !matches!(c.kind(), "comment" | "\n" | "newline"));

    let stmt = match first_stmt {
        Some(s) => s,
        None => return,
    };

    // Must be expression_statement containing a string
    if stmt.kind() != "expression_statement" {
        return;
    }
    let inner = (0..stmt.child_count())
        .filter_map(|i| stmt.child(i))
        .find(|c| !matches!(c.kind(), "comment" | "\n"));
    let is_string = inner.map(|c| c.kind() == "string").unwrap_or(false);
    if !is_string {
        return;
    }

    // Compute replacement: from start of line to end of line (inclusive of trailing newline)
    let start_byte = stmt.start_byte();
    let end_byte = stmt.end_byte();

    // Walk back to find start of the line (past any leading indentation)
    let line_start = source[..start_byte].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let indent = &source[line_start..start_byte];

    // Walk forward to consume the trailing newline
    let line_end = source[end_byte..]
        .find('\n')
        .map(|i| end_byte + i + 1)
        .unwrap_or(end_byte);

    replacements.push((line_start, line_end, format!("{}pass\n", indent)));
}

/// Lexical Jaccard similarity on identifier tokens.
/// Mirrors Python's `similarity.lexical.lexical_similarity`.
pub fn lexical_similarity(a: &str, b: &str) -> f64 {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[A-Za-z0-9_]+").unwrap());

    let tokens_a: std::collections::HashSet<String> = re
        .find_iter(&a.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    let tokens_b: std::collections::HashSet<String> = re
        .find_iter(&b.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();

    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();

    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// T1c result.
pub struct T1cResult {
    pub embedding_cosines: Vec<f32>,
    pub min_cosine: f32,
    pub mean_cosine: f32,
    pub max_cosine: f32,
    pub max_lexical_diff: f64,
    pub mean_lexical_diff: f64,
    pub near_threshold_flips: Vec<String>,
    pub diff_categories: Vec<String>,
}

pub fn run_t1c(
    original_texts: &[String],
    py_normalized: &[String],
    py_lexical: &[Vec<f64>],
    embed_fn: impl Fn(&[&str]) -> anyhow::Result<Vec<Vec<f32>>>,
) -> T1cResult {
    let n = original_texts.len();
    let rust_normalized: Vec<String> =
        original_texts.iter().map(|t| normalize_python(t)).collect();

    // Categorize differences between rust and python normalization
    let mut has_comment_diff = false;
    let mut has_whitespace_diff = false;
    let mut has_quote_diff = false;
    let mut has_pass_diff = false;
    let mut has_other_diff = false;

    for (rust_norm, py_norm) in rust_normalized.iter().zip(py_normalized.iter()) {
        if rust_norm != py_norm {
            // Check what kind of difference
            let r_no_comments = strip_comments_for_comparison(rust_norm);
            let p_no_comments = strip_comments_for_comparison(py_norm);
            if r_no_comments != p_no_comments {
                // More than just comment differences
                let r_normalized_ws = normalize_whitespace(&r_no_comments);
                let p_normalized_ws = normalize_whitespace(&p_no_comments);
                if r_normalized_ws != p_normalized_ws {
                    has_quote_diff = true; // quote/syntax diff
                    has_other_diff = true;
                } else {
                    has_whitespace_diff = true;
                }
            } else {
                has_comment_diff = true;
            }
            if rust_norm.contains("pass") != py_norm.contains("pass") {
                has_pass_diff = true;
            }
        }
    }

    let mut diff_categories = Vec::new();
    if has_comment_diff { diff_categories.push("comment preservation (rust keeps, python strips)".to_string()); }
    if has_whitespace_diff { diff_categories.push("whitespace/indentation differences".to_string()); }
    if has_quote_diff { diff_categories.push("quote style normalization (ast.unparse normalizes)".to_string()); }
    if has_pass_diff { diff_categories.push("pass insertion differences".to_string()); }
    if has_other_diff { diff_categories.push("other ast.unparse reformatting".to_string()); }
    if diff_categories.is_empty() { diff_categories.push("no differences (texts match)".to_string()); }

    // Embed rust-normalized and python-normalized texts using candle
    let rust_norm_refs: Vec<&str> = rust_normalized.iter().map(|s| s.as_str()).collect();
    let py_norm_refs: Vec<&str> = py_normalized.iter().map(|s| s.as_str()).collect();

    let rust_embeddings = match embed_fn(&rust_norm_refs) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[T1c] embed rust_normalized failed: {e}");
            return T1cResult {
                embedding_cosines: vec![],
                min_cosine: 0.0,
                mean_cosine: 0.0,
                max_cosine: 0.0,
                max_lexical_diff: f64::MAX,
                mean_lexical_diff: f64::MAX,
                near_threshold_flips: vec![format!("embed error: {e}")],
                diff_categories,
            };
        }
    };

    let py_embeddings = match embed_fn(&py_norm_refs) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[T1c] embed py_normalized failed: {e}");
            return T1cResult {
                embedding_cosines: vec![],
                min_cosine: 0.0,
                mean_cosine: 0.0,
                max_cosine: 0.0,
                max_lexical_diff: f64::MAX,
                mean_lexical_diff: f64::MAX,
                near_threshold_flips: vec![format!("embed error: {e}")],
                diff_categories,
            };
        }
    };

    // Cosine between rust-norm and python-norm embeddings per snippet
    let embedding_cosines: Vec<f32> = rust_embeddings
        .iter()
        .zip(py_embeddings.iter())
        .map(|(r, p)| crate::candle_embed::cosine(r, p))
        .collect();

    let min_cosine = embedding_cosines.iter().cloned().fold(f32::MAX, f32::min);
    let mean_cosine = embedding_cosines.iter().sum::<f32>() / embedding_cosines.len() as f32;
    let max_cosine = embedding_cosines.iter().cloned().fold(f32::MIN, f32::max);

    // Lexical similarity on rust-normalized vs python-normalized
    let mut max_lexical_diff = 0f64;
    let mut total_lexical_diff = 0f64;
    let mut lexical_count = 0usize;
    let mut near_threshold_flips = Vec::new();
    let thresholds = [0.5f64, 0.90f64, 0.92f64];

    for i in 0..n {
        for j in (i + 1)..n {
            let rust_score = lexical_similarity(&rust_normalized[i], &rust_normalized[j]);
            let py_score = py_lexical[i][j];
            let diff = (rust_score - py_score).abs();
            max_lexical_diff = max_lexical_diff.max(diff);
            total_lexical_diff += diff;
            lexical_count += 1;

            for &thresh in &thresholds {
                let rust_side = rust_score >= thresh;
                let py_side = py_score >= thresh;
                let near = (rust_score - thresh).abs() < 0.02 || (py_score - thresh).abs() < 0.02;
                if near && rust_side != py_side {
                    near_threshold_flips.push(format!(
                        "lexical pair ({i},{j}) flips at thresh={thresh:.2}: \
                         py={py_score:.4} rust={rust_score:.4}"
                    ));
                }
            }
        }
    }
    let mean_lexical_diff =
        if lexical_count > 0 { total_lexical_diff / lexical_count as f64 } else { 0.0 };

    // Check near-threshold embedding cosine pairs
    for (i, emb_cos) in embedding_cosines.iter().enumerate() {
        if *emb_cos < 0.95 {
            eprintln!(
                "[T1c] Low normalization cosine at snippet {i}: {emb_cos:.4}"
            );
        }
    }

    T1cResult {
        embedding_cosines,
        min_cosine,
        mean_cosine,
        max_cosine,
        max_lexical_diff,
        mean_lexical_diff,
        near_threshold_flips,
        diff_categories,
    }
}

fn strip_comments_for_comparison(s: &str) -> String {
    s.lines()
        .map(|line| {
            if let Some(idx) = line.find('#') {
                // Don't strip # inside strings — simple heuristic: only strip if # is not quoted
                let before = &line[..idx];
                let quote_count = before.matches('"').count() + before.matches('\'').count();
                if quote_count % 2 == 0 {
                    return before.trim_end().to_string();
                }
            }
            line.to_string()
        })
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
