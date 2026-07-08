//! Shared `#[cfg(test)]` builders for constructing test fixtures without per-module duplication.
//!
//! These are the common cases. Some modules keep bespoke local builders where the fixture needs
//! precise control the shared helpers deliberately don't expose — chiefly explicit `snippet_hash`
//! values (the shared builders derive `snippet_hash = hash_text(text)`, which would collapse
//! same-text fixtures onto one hash). That applies to `similarity::candidates` (hashes double as
//! index IDs), `similarity::{rollup, ranking, occurrences}`, and the self-clone tests in
//! `reporting::html` (same-text windows that must stay distinguishable).

use std::collections::BTreeMap;

use crate::core::types::{
    CandidateMatch, Embedding, FileRef, Finding, FunctionRef, Language, ScanResult, ScanStats,
    SnippetKind, SnippetRef,
};
use crate::io::fingerprints::hash_text;

/// A Python `FileRef` with a fixed content hash.
pub(crate) fn make_file(path: &str) -> FileRef {
    FileRef {
        path: path.into(),
        content_hash: "h".into(),
        language: Language::Python,
        content: "".into(),
    }
}

/// A `FunctionRef` at `path` with a deterministic `code_hash`.
pub(crate) fn make_function(
    path: &str,
    qname: &str,
    start: usize,
    end: usize,
    code: &str,
) -> FunctionRef {
    FunctionRef {
        file: make_file(path),
        qualified_name: qname.into(),
        start_line: start,
        end_line: end,
        code: code.into(),
        code_hash: hash_text(code),
    }
}

/// A fully-specified `SnippetRef` (kind, span, text vs display) with `snippet_hash = hash_text(text)`.
pub(crate) fn make_snippet_kind(
    kind: SnippetKind,
    function: FunctionRef,
    start: usize,
    end: usize,
    text: &str,
    display: &str,
) -> SnippetRef {
    SnippetRef {
        kind,
        function,
        start_line: start,
        end_line: end,
        text: text.into(),
        display_text: display.into(),
        snippet_hash: hash_text(text),
    }
}

/// A FUNC `SnippetRef` spanning `function`, with `text == display_text`.
pub(crate) fn make_snippet_for(function: &FunctionRef, text: &str) -> SnippetRef {
    make_snippet_kind(
        SnippetKind::Func,
        function.clone(),
        function.start_line,
        function.end_line,
        text,
        text,
    )
}

/// A minimal FUNC `SnippetRef` from `text` alone (function `f` @ `test.py` lines 1-3).
/// Identical text → identical `snippet_hash`, so it doubles as a stable cache key.
pub(crate) fn make_snippet(text: &str) -> SnippetRef {
    make_snippet_for(&make_function("test.py", "f", 1, 3, text), text)
}

/// A `CandidateMatch` with empty evidence.
pub(crate) fn make_match(
    snippet_a: SnippetRef,
    snippet_b: SnippetRef,
    similarity: f64,
) -> CandidateMatch {
    CandidateMatch {
        snippet_a,
        snippet_b,
        similarity,
        evidence: String::new(),
    }
}

/// An `Embedding` from a raw vector.
pub(crate) fn make_embedding(vector: Vec<f32>) -> Embedding {
    Embedding { vector }
}

/// A `Finding` with empty metadata; `reasons` are string-copied from the slice.
pub(crate) fn make_finding(
    function_a: FunctionRef,
    function_b: FunctionRef,
    score: f64,
    duplicated_lines: usize,
    evidence: Vec<CandidateMatch>,
    reasons: &[&str],
) -> Finding {
    Finding {
        function_a,
        function_b,
        score,
        duplicated_lines,
        evidence,
        reasons: reasons.iter().map(|s| s.to_string()).collect(),
        metadata: BTreeMap::new(),
    }
}

/// A `ScanResult` wrapping `findings` with zeroed stats (except `finding_count`), empty
/// config/timing/degradations. Tests that assert on stats set the fields they care about.
pub(crate) fn make_scan_result(findings: Vec<Finding>) -> ScanResult {
    let finding_count = findings.len();
    ScanResult {
        findings,
        stats: ScanStats {
            file_count: 0,
            function_count: 0,
            snippet_count: 0,
            candidate_count: 0,
            finding_count,
            group_count: 0,
            grouped_function_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        },
        config_snapshot: serde_json::json!({}),
        timing: BTreeMap::new(),
        degradations: vec![],
    }
}
