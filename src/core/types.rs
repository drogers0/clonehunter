use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Language {
    Python,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub(crate) enum SnippetKind {
    Func,
    Win,
    Exp,
}

/// Represents a single collected source file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct FileRef {
    pub path: String,
    pub content_hash: String,
    pub language: Language,
}

/// Represents an extracted function (or a non-Python file treated as one whole-file unit).
///
/// `identity()` is the pervasive grouping/dedupe key used in rollup, dedupe,
/// and orientation normalization. Its format is locked: `"{path}:{qname}:{start}:{end}"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FunctionRef {
    pub file: FileRef,
    pub qualified_name: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Original source text (not normalized). Used for rendered diffs in reporters.
    pub code: String,
    pub code_hash: String,
}

impl FunctionRef {
    /// The pervasive grouping/dedupe key. Format: `"{path}:{qname}:{start}:{end}"`.
    /// Used as HashMap/BTreeMap key in rollup, dedupe, and orientation normalization.
    pub fn identity(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.file.path, self.qualified_name, self.start_line, self.end_line
        )
    }
}

/// A code snippet extracted from a `FunctionRef` (FUNC, WIN, or EXP kind).
///
/// Memory note for T9/T10: `SnippetRef` owns a full `FunctionRef` (including `code: String`).
/// With thousands of snippets derived from the same function, this deep-copies the code string
/// for each snippet. If memory profiling in T9 shows excessive usage, consider `Arc<FunctionRef>`.
///
/// Carries both normalized text forms per the locked normalization contract (Phase 0 DD7):
/// - `text`: analysis text (docstrings → pass, comments stripped, source passthrough).
///   Drives embeddings, cache keys, and lexical scoring.
/// - `display_text`: display text (docstrings → pass, comments preserved).
///   Used only by reporters for rendered diffs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnippetRef {
    pub kind: SnippetKind,
    pub function: FunctionRef,
    pub start_line: usize,
    pub end_line: usize,
    /// Analysis text: docstrings stripped (→ pass), comments stripped, source passthrough.
    /// Drives embeddings, cache keys, lexical scoring.
    pub text: String,
    /// Display text: docstrings stripped (→ pass), comments preserved.
    /// Used only by reporters for rendered diffs.
    pub display_text: String,
    /// SHA-256 of `text` (analysis text, not display_text).
    pub snippet_hash: String,
}

/// An embedding vector for a snippet.
///
/// Uses `f32` (not `f64`) — candle tensors are f32, matching PyTorch's actual precision.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Embedding {
    pub vector: Vec<f32>,
    pub dim: usize,
}

/// A candidate clone pair produced by the index before rollup.
#[derive(Debug, Clone)]
pub(crate) struct CandidateMatch {
    pub snippet_a: SnippetRef,
    pub snippet_b: SnippetRef,
    pub similarity: f64,
    pub evidence: String,
}

/// A rolled-up clone finding (one per function pair).
#[derive(Debug, Clone)]
pub(crate) struct Finding {
    pub function_a: FunctionRef,
    pub function_b: FunctionRef,
    pub score: f64,
    pub duplicated_lines: usize,
    pub evidence: Vec<CandidateMatch>,
    pub reasons: Vec<String>,
    /// BTreeMap for deterministic JSON serialization (DD7).
    pub metadata: BTreeMap<String, String>,
}

/// Aggregate statistics for a scan run.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ScanStats {
    pub file_count: usize,
    pub function_count: usize,
    pub snippet_count: usize,
    pub candidate_count: usize,
    pub finding_count: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// The complete result of a scan run.
#[derive(Debug, Clone)]
pub(crate) struct ScanResult {
    pub findings: Vec<Finding>,
    pub stats: ScanStats,
    /// Nested config snapshot. Populated by serializing `CloneHunterConfig` to
    /// `serde_json::Value` in T10 (pipeline). `BTreeMap` key order in serde_json
    /// ensures deterministic JSON output (DD7).
    pub config_snapshot: serde_json::Value,
    /// Per-stage timing in seconds. BTreeMap for deterministic JSON key order (DD7).
    pub timing: BTreeMap<String, f64>,
    /// Graceful degradation diagnostics (Phase 0 DD3).
    pub degradations: Vec<Degradation>,
}

/// The input to a scan run.
#[derive(Debug, Clone)]
pub(crate) struct ScanRequest {
    pub paths: Vec<String>,
    pub config: crate::core::config::CloneHunterConfig,
}

/// Classification of a graceful degradation event (Phase 0 DD3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // ParseSkip/CacheSelfHeal reserved for T14 test port
pub(crate) enum DegradationKind {
    DeviceFallback,
    IndexFallback,
    ParseSkip,
    CacheSelfHeal,
}

/// A graceful degradation event (Phase 0 DD3).
/// Logged via `tracing::warn!` and surfaced in stats/reports.
#[derive(Debug, Clone)]
pub(crate) struct Degradation {
    #[allow(dead_code)] // consumed by T14 test port
    pub kind: DegradationKind,
    #[allow(dead_code)] // consumed by T14 test port
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_ref_identity_format() {
        let file = FileRef {
            path: "src/main.py".into(),
            content_hash: "abc123".into(),
            language: Language::Python,
        };
        let func = FunctionRef {
            file,
            qualified_name: "MyClass.my_method".into(),
            start_line: 10,
            end_line: 20,
            code: "def my_method(self): pass".into(),
            code_hash: "def456".into(),
        };
        assert_eq!(func.identity(), "src/main.py:MyClass.my_method:10:20");
    }

    #[test]
    fn snippet_kind_serializes_uppercase() {
        assert_eq!(
            serde_json::to_string(&SnippetKind::Func).unwrap(),
            "\"FUNC\""
        );
        assert_eq!(serde_json::to_string(&SnippetKind::Win).unwrap(), "\"WIN\"");
        assert_eq!(serde_json::to_string(&SnippetKind::Exp).unwrap(), "\"EXP\"");
    }

    #[test]
    fn language_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&Language::Python).unwrap(),
            "\"python\""
        );
        assert_eq!(serde_json::to_string(&Language::Text).unwrap(), "\"text\"");
    }

    #[test]
    fn file_ref_json_matches_python_schema() {
        let f = FileRef {
            path: "src/lib.py".into(),
            content_hash: "deadbeef".into(),
            language: Language::Python,
        };
        let json: serde_json::Value = serde_json::to_value(&f).unwrap();
        assert_eq!(json["path"], "src/lib.py");
        assert_eq!(json["content_hash"], "deadbeef");
        assert_eq!(json["language"], "python");
    }

    #[test]
    fn degradation_variants_are_constructible() {
        let d = Degradation {
            kind: DegradationKind::DeviceFallback,
            message: "fell back".into(),
        };
        assert_eq!(d.kind, DegradationKind::DeviceFallback);
        let _ = format!("{:?}", d); // exercises Debug
    }

    #[test]
    fn scan_stats_serializes_all_fields() {
        let stats = ScanStats {
            file_count: 10,
            function_count: 50,
            snippet_count: 200,
            candidate_count: 30,
            finding_count: 5,
            cache_hits: 180,
            cache_misses: 20,
        };
        let json: serde_json::Value = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["file_count"], 10);
        assert_eq!(json["cache_hits"], 180);
    }
}
