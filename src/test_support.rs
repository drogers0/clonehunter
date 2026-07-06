//! Shared `#[cfg(test)]` builders for constructing test fixtures without per-module duplication.

use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind, SnippetRef};
use crate::io::fingerprints::hash_text;

/// A minimal Python `FunctionRef` wrapping `code` (lines 1-3, deterministic hashes).
pub(crate) fn make_function(code: &str) -> FunctionRef {
    FunctionRef {
        file: FileRef {
            path: "test.py".into(),
            content_hash: "abc".into(),
            language: Language::Python,
        },
        qualified_name: "f".into(),
        start_line: 1,
        end_line: 3,
        code: code.into(),
        code_hash: hash_text(code),
    }
}

/// A minimal FUNC `SnippetRef` whose `text`/`display_text` are `text` and whose
/// `snippet_hash` is `hash_text(text)` (so identical text → identical cache key).
pub(crate) fn make_snippet(text: &str) -> SnippetRef {
    SnippetRef {
        kind: SnippetKind::Func,
        function: make_function(text),
        start_line: 1,
        end_line: 3,
        text: text.into(),
        display_text: text.into(),
        snippet_hash: hash_text(text),
    }
}
