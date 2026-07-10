use std::collections::BTreeMap;

use thiserror::Error;

use crate::core::types::{
    CandidateMatch, FileRef, Finding, FunctionRef, Language, ScanRequest, ScanResult, ScanStats,
    SnippetKind, SnippetRef,
};
use crate::io::fingerprints::hash_text;

use super::{Engine, PipelineError};

#[derive(Debug, Error)]
pub(crate) enum SonarError {
    #[error("CLONEHUNTER_SONAR_REPORT env var is not set or empty")]
    EnvNotSet,
    #[error("report file not found: {path}")]
    FileNotFound { path: String },
    #[error("failed to parse report JSON: {0}")]
    Parse(#[from] serde_json::Error),
}

pub(crate) struct SonarQubeEngine;

impl Engine for SonarQubeEngine {
    fn scan(&self, _request: &ScanRequest) -> Result<ScanResult, PipelineError> {
        // The sonarqube engine sources findings entirely from CLONEHUNTER_SONAR_REPORT;
        // scan paths and every tuning flag on the request are ignored. Warn so a user who
        // passed `scan ./src --engine sonarqube ...` isn't misled into thinking they took effect.
        tracing::warn!(
            "sonarqube engine reads findings from CLONEHUNTER_SONAR_REPORT; scan paths and tuning flags are ignored"
        );
        Ok(scan_sonarqube(None)?)
    }
}

/// Internal helper — accepts an optional path override for testability.
/// When `path_override` is None, reads from `CLONEHUNTER_SONAR_REPORT` env var.
/// NOTE: In Rust edition 2024, `std::env::set_var` is unsafe — tests must use
/// `path_override` directly rather than mutating env vars.
pub(crate) fn scan_sonarqube(path_override: Option<&str>) -> Result<ScanResult, SonarError> {
    let report_path = match path_override {
        Some(p) => p.to_string(),
        None => {
            let v = std::env::var("CLONEHUNTER_SONAR_REPORT").unwrap_or_default();
            let v = v.trim().to_string();
            if v.is_empty() {
                return Err(SonarError::EnvNotSet);
            }
            v
        }
    };

    let path = std::path::Path::new(&report_path);
    if !path.exists() {
        return Err(SonarError::FileNotFound {
            path: report_path.clone(),
        });
    }

    let content = std::fs::read_to_string(path).map_err(|_| SonarError::FileNotFound {
        path: report_path.clone(),
    })?;
    let payload: serde_json::Value = serde_json::from_str(&content)?;

    let mut findings: Vec<Finding> = Vec::new();
    if let Some(dups) = payload.get("duplications").and_then(|v| v.as_array()) {
        for issue in dups {
            let a = match to_function(issue.get("a")) {
                Some(f) => f,
                None => continue,
            };
            let b = match to_function(issue.get("b")) {
                Some(f) => f,
                None => continue,
            };
            let snip_a = make_snippet(&a);
            let snip_b = make_snippet(&b);
            let dup_lines = span_len(&a).min(span_len(&b));
            let evidence = CandidateMatch {
                snippet_a: snip_a,
                snippet_b: snip_b,
                similarity: 1.0,
                evidence: "sonarqube".into(),
            };
            findings.push(Finding {
                function_a: a,
                function_b: b,
                score: 1.0,
                duplicated_lines: dup_lines,
                evidence: vec![evidence],
                reasons: vec!["sonarqube".into()],
            });
        }
    }

    // SonarQube findings flow through the same unconditional grouping/stats path as semantic runs.
    let (group_count, grouped_function_count) = crate::similarity::group_stats(&findings);
    let stats = ScanStats {
        file_count: 0,
        function_count: 0,
        snippet_count: 0,
        candidate_count: 0,
        finding_count: findings.len(),
        group_count,
        grouped_function_count,
        cache_hits: 0,
        cache_misses: 0,
    };
    Ok(ScanResult {
        findings,
        stats,
        config_snapshot: serde_json::Value::Object(Default::default()),
        timing: BTreeMap::new(),
        degradations: Vec::new(),
    })
}

/// Port of Python's `_to_function`.
/// Python uses `str(data.get("path", ""))` — string-coerces any JSON value, defaults missing
/// keys to "". Do the same to avoid skipping findings Python would keep.
fn to_function(data: Option<&serde_json::Value>) -> Option<FunctionRef> {
    let data = data?.as_object()?;
    let file_path = json_to_string(data.get("path"), "");
    let start = to_int(data.get("start"), 1);
    let end = to_int(data.get("end"), start);
    let code = json_to_string(data.get("code"), "");
    let name = {
        let n = json_to_string(data.get("name"), "");
        if n.is_empty() { file_path.clone() } else { n }
    };
    let file_ref = FileRef {
        path: file_path,
        content_hash: String::new(),
        language: Language::Python, // always Python (DD10)
        content: "".into(),
    };
    Some(FunctionRef {
        file: file_ref,
        qualified_name: name,
        start_line: start,
        end_line: end,
        code_hash: hash_text(&code),
        code,
    })
}

/// Coerce a JSON value to String, approximating Python's `str(v)`.
/// Returns `default` when key is absent or null.
fn json_to_string(value: Option<&serde_json::Value>, default: &str) -> String {
    match value {
        None | Some(serde_json::Value::Null) => default.to_string(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        Some(serde_json::Value::Bool(b)) => b.to_string(),
        Some(v) => v.to_string(),
    }
}

fn to_int(value: Option<&serde_json::Value>, default: usize) -> usize {
    match value {
        Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(default as u64) as usize,
        Some(serde_json::Value::String(s)) => s.parse().unwrap_or(default),
        _ => default,
    }
}

/// Returns 0 when end < start (matches Python's `max(0, end - start + 1)`).
fn span_len(func: &FunctionRef) -> usize {
    if func.end_line >= func.start_line {
        func.end_line - func.start_line + 1
    } else {
        0
    }
}

fn make_snippet(func: &FunctionRef) -> SnippetRef {
    SnippetRef {
        kind: SnippetKind::Func,
        function: func.clone(),
        start_line: func.start_line,
        end_line: func.end_line,
        text: func.code.clone(),         // raw code (DD14)
        display_text: func.code.clone(), // same as text for adapter (DD14)
        snippet_hash: hash_text(&func.code),
    }
}

#[cfg(test)]
mod tests {
    use super::{SonarError, scan_sonarqube};
    use tempfile::TempDir;

    #[test]
    fn test_sonarqube_engine_reads_report() {
        let dir = TempDir::new().unwrap();
        let report = serde_json::json!({
            "duplications": [{
                "a": {"path": "a.py", "start": 1, "end": 2, "code": "pass", "name": "a"},
                "b": {"path": "b.py", "start": 1, "end": 2, "code": "pass", "name": "b"}
            }]
        });
        let path = dir.path().join("report.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let result = scan_sonarqube(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(result.findings.len(), 1);
        assert_eq!(result.findings[0].score, 1.0);
        assert_eq!(result.stats.finding_count, 1);
    }

    #[test]
    fn test_sonarqube_engine_requires_env() {
        // Reading env is safe; only *writing* is unsafe in Rust 2024.
        if std::env::var("CLONEHUNTER_SONAR_REPORT").is_ok() {
            eprintln!(
                "SKIP: CLONEHUNTER_SONAR_REPORT is set; skipping test_sonarqube_engine_requires_env"
            );
            return;
        }
        let result = scan_sonarqube(None);
        assert!(matches!(result, Err(SonarError::EnvNotSet)));
    }

    #[test]
    fn test_sonarqube_engine_evidence_fields() {
        let dir = TempDir::new().unwrap();
        let report = serde_json::json!({
            "duplications": [{
                "a": {"path": "a.py", "start": 1, "end": 2, "code": "code_a", "name": "func_a"},
                "b": {"path": "b.py", "start": 10, "end": 12, "code": "code_b", "name": "func_b"}
            }]
        });
        let path = dir.path().join("report.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let result = scan_sonarqube(Some(path.to_str().unwrap())).unwrap();
        let m = &result.findings[0].evidence[0];
        assert_eq!(m.snippet_a.function.file.path, "a.py");
        assert_eq!(m.snippet_a.text, "code_a");
        assert_eq!(m.snippet_b.function.file.path, "b.py");
        assert_eq!(m.snippet_b.text, "code_b");
        assert_eq!(m.snippet_b.start_line, 10);
        assert_eq!(m.snippet_b.end_line, 12);
    }

    #[test]
    fn test_sonarqube_file_not_found() {
        let result = scan_sonarqube(Some("/nonexistent/path/report.json"));
        assert!(matches!(result, Err(SonarError::FileNotFound { .. })));
    }

    #[test]
    fn test_sonarqube_invalid_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "not json at all").unwrap();
        let result = scan_sonarqube(Some(path.to_str().unwrap()));
        assert!(matches!(result, Err(SonarError::Parse(_))));
    }

    #[test]
    fn test_sonarqube_empty_duplications() {
        let dir = TempDir::new().unwrap();
        let report = serde_json::json!({ "duplications": [] });
        let path = dir.path().join("report.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let result = scan_sonarqube(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(result.findings.len(), 0);
    }

    #[test]
    fn test_sonarqube_span_len_and_dup_lines() {
        let dir = TempDir::new().unwrap();
        // a spans lines 5-10 (6 lines), b spans lines 1-3 (3 lines) → dup_lines = min(6,3) = 3
        let report = serde_json::json!({
            "duplications": [{
                "a": {"path": "a.py", "start": 5, "end": 10, "code": "x", "name": "fa"},
                "b": {"path": "b.py", "start": 1, "end": 3, "code": "x", "name": "fb"}
            }]
        });
        let path = dir.path().join("report.json");
        std::fs::write(&path, report.to_string()).unwrap();
        let result = scan_sonarqube(Some(path.to_str().unwrap())).unwrap();
        assert_eq!(result.findings[0].duplicated_lines, 3);
    }
}
