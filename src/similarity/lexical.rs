#![allow(dead_code)] // T10 (pipeline) will wire these up

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;

static TOKEN_RE: OnceLock<Regex> = OnceLock::new();

fn token_regex() -> &'static Regex {
    TOKEN_RE.get_or_init(|| Regex::new(r"[A-Za-z0-9_]+").expect("valid regex"))
}

fn tokenize(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    token_regex()
        .find_iter(&lower)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Jaccard similarity over lowercased identifier tokens.
///
/// Direct port of Python's `lexical_similarity`: `re.findall(r"[A-Za-z0-9_]+", text.lower())`
/// then set-intersection / set-union. Returns 0.0 for empty inputs (DD6).
pub(crate) fn lexical_similarity(text_a: &str, text_b: &str) -> f64 {
    let tokens_a = tokenize(text_a);
    let tokens_b = tokenize(text_b);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_identical_text() {
        assert!((lexical_similarity("def foo(): pass", "def foo(): pass") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lexical_disjoint_tokens() {
        assert!((lexical_similarity("abc def", "xyz uvw") - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lexical_empty_text() {
        assert_eq!(lexical_similarity("", "def foo(): pass"), 0.0);
        assert_eq!(lexical_similarity("def foo(): pass", ""), 0.0);
        assert_eq!(lexical_similarity("", ""), 0.0);
    }

    #[test]
    fn test_lexical_partial_overlap() {
        // tokens_a = {a, b, c}, tokens_b = {b, c, d}
        // intersection = {b, c} = 2, union = {a, b, c, d} = 4 → 0.5
        let sim = lexical_similarity("a b c", "b c d");
        assert!((sim - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lexical_case_insensitive() {
        // "Foo" and "foo" should be the same token
        assert!((lexical_similarity("Foo", "foo") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lexical_numbers_included() {
        // Numbers are valid tokens per [A-Za-z0-9_]+
        let sim = lexical_similarity("val1 val2", "val1 val3");
        // intersection = {val1}, union = {val1, val2, val3} → 1/3
        assert!((sim - 1.0 / 3.0).abs() < 1e-9);
    }
}
