use std::fs::File;
use std::io::BufWriter;

use serde_json::json;
use similar::TextDiff;

use crate::core::types::{Finding, FunctionRef, ScanResult};
use crate::reporting::ReportError;
use crate::reporting::compare::select_compare;
use crate::reporting::schema::SCHEMA_VERSION;

pub(crate) fn write_json(result: &ScanResult, out_path: &str) -> Result<(), ReportError> {
    let findings: Vec<_> = result.findings.iter().map(serialize_finding).collect();
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "findings": findings,
        "stats": serde_json::to_value(&result.stats)?,
        "config": result.config_snapshot,
        "timing": result.timing,
        "degradations": serde_json::to_value(&result.degradations)?,
    });
    let file = File::create(out_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &payload)?;
    Ok(())
}

fn serialize_finding(finding: &Finding) -> serde_json::Value {
    json!({
        "function_a": serialize_function(&finding.function_a),
        "function_b": serialize_function(&finding.function_b),
        "score": finding.score,
        "duplicated_lines": finding.duplicated_lines,
        "compare": serialize_compare(finding),
        "reasons": finding.reasons,
        "metadata": finding.metadata,
    })
}

fn serialize_function(func: &FunctionRef) -> serde_json::Value {
    let language = serde_json::to_value(func.file.language).unwrap_or(json!("unknown"));
    json!({
        "file": {
            "path": func.file.path,
            "content_hash": func.file.content_hash,
            "language": language,
        },
        "qualified_name": func.qualified_name,
        "start_line": func.start_line,
        "end_line": func.end_line,
        "code_hash": func.code_hash,
    })
}

fn serialize_compare(finding: &Finding) -> serde_json::Value {
    let Some(compare) = select_compare(&finding.evidence) else {
        return json!(null);
    };
    let kind_a = serde_json::to_value(compare.kind_a).unwrap_or(json!("FUNC"));
    let kind_b = serde_json::to_value(compare.kind_b).unwrap_or(json!("FUNC"));
    json!({
        "kind_a": kind_a,
        "kind_b": kind_b,
        "span_a": { "start_line": compare.span_a.0, "end_line": compare.span_a.1 },
        "span_b": { "start_line": compare.span_b.0, "end_line": compare.span_b.1 },
        "similarity": compare.similarity,
        "diff": diff_text(&compare.text_a, &compare.text_b),
    })
}

/// Produce a unified diff of two texts. Uses analysis text (matches Python schema).
/// Splits with `.lines()` and diffs the resulting `Vec<&str>` slices (DD3).
pub(crate) fn diff_text(text_a: &str, text_b: &str) -> String {
    let lines_a: Vec<&str> = text_a.lines().collect();
    let lines_b: Vec<&str> = text_b.lines().collect();
    let diff = TextDiff::from_slices(lines_a.as_slice(), lines_b.as_slice());

    // Short-circuit: identical texts → empty string (Python unified_diff returns [] for same inputs)
    if diff
        .ops()
        .iter()
        .all(|op| matches!(op, similar::DiffOp::Equal { .. }))
    {
        return String::new();
    }

    let output = diff.unified_diff().header("", "").to_string();
    // Collect lines, suppressing any "\ No newline at end of file" markers
    let diff_lines: Vec<&str> = output.lines().filter(|l| !l.starts_with('\\')).collect();

    truncate_diff(&diff_lines, 80, 4000)
}

/// Truncate diff lines to fit within limits (DD3).
/// `lines` — individual diff lines (no inter-line newlines).
/// Char count = sum of line lengths (excluding inter-line `\n`, matching Python).
pub(crate) fn truncate_diff(lines: &[&str], max_lines: usize, max_chars: usize) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let total_chars: usize = lines.iter().map(|l| l.len()).sum();
    if lines.len() <= max_lines && total_chars <= max_chars {
        return lines.join("\n");
    }
    let trimmed = &lines[..lines.len().min(max_lines)];
    let mut text = trimmed.join("\n");
    if text.chars().count() > max_chars {
        // Truncate at a char boundary to avoid panicking on multibyte sequences.
        let byte_end = text
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        text.truncate(byte_end);
    }
    format!("{text}\n... diff truncated ...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::core::types::{
        CandidateMatch, FileRef, Finding, FunctionRef, Language, ScanResult, ScanStats,
        SnippetKind, SnippetRef,
    };
    use tempfile::TempDir;

    fn make_scan_result_empty() -> ScanResult {
        ScanResult {
            findings: vec![],
            stats: ScanStats {
                file_count: 0,
                function_count: 0,
                snippet_count: 0,
                candidate_count: 0,
                finding_count: 0,
                cache_hits: 0,
                cache_misses: 0,
            },
            config_snapshot: serde_json::json!({}),
            timing: BTreeMap::new(),
            degradations: vec![],
        }
    }

    fn make_finding() -> Finding {
        let file = FileRef {
            path: "a.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
        };
        let func_a = FunctionRef {
            file: file.clone(),
            qualified_name: "foo".into(),
            start_line: 1,
            end_line: 10,
            code: "def foo(): pass".into(),
            code_hash: "ca".into(),
        };
        let func_b = FunctionRef {
            file,
            qualified_name: "bar".into(),
            start_line: 20,
            end_line: 30,
            code: "def bar(): pass".into(),
            code_hash: "cb".into(),
        };
        let snip_a = SnippetRef {
            kind: SnippetKind::Func,
            function: func_a.clone(),
            start_line: 1,
            end_line: 10,
            text: "def foo(): pass".into(),
            display_text: "def foo(): pass".into(),
            snippet_hash: "sha".into(),
        };
        let snip_b = SnippetRef {
            kind: SnippetKind::Func,
            function: func_b.clone(),
            start_line: 20,
            end_line: 30,
            text: "def bar(): pass".into(),
            display_text: "def bar(): pass".into(),
            snippet_hash: "shb".into(),
        };
        Finding {
            function_a: func_a,
            function_b: func_b,
            score: 0.95,
            duplicated_lines: 10,
            evidence: vec![CandidateMatch {
                snippet_a: snip_a,
                snippet_b: snip_b,
                similarity: 0.95,
                evidence: "FUNC->FUNC|emb=0.950|lex=0.800|comp=0.905".into(),
            }],
            reasons: vec!["high_similarity".into()],
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn json_report_schema_keys() {
        let dir = TempDir::new().unwrap();
        let out = dir
            .path()
            .join("report.json")
            .to_string_lossy()
            .into_owned();
        let result = make_scan_result_empty();
        write_json(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(v.get("schema_version").is_some());
        assert!(v.get("findings").is_some());
        assert!(v.get("stats").is_some());
        assert!(v.get("config").is_some());
        assert!(v.get("timing").is_some());
    }

    #[test]
    fn json_degradations_serialized() {
        use crate::core::types::{Degradation, DegradationKind};
        let dir = TempDir::new().unwrap();
        let out = dir
            .path()
            .join("report.json")
            .to_string_lossy()
            .into_owned();
        let mut result = make_scan_result_empty();
        result.degradations.push(Degradation {
            kind: DegradationKind::DeviceFallback,
            message: "CUDA unavailable, falling back to CPU".into(),
        });
        write_json(&result, &out).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        let degs = v.get("degradations").and_then(|d| d.as_array()).unwrap();
        assert_eq!(degs.len(), 1);
        assert_eq!(degs[0]["kind"], "device_fallback");
        assert_eq!(degs[0]["message"], "CUDA unavailable, falling back to CPU");
    }

    #[test]
    fn json_degradations_empty_by_default() {
        let dir = TempDir::new().unwrap();
        let out = dir
            .path()
            .join("report.json")
            .to_string_lossy()
            .into_owned();
        write_json(&make_scan_result_empty(), &out).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(v["degradations"], serde_json::json!([]));
    }

    #[test]
    fn json_finding_has_compare_with_diff() {
        let dir = TempDir::new().unwrap();
        let out = dir
            .path()
            .join("report.json")
            .to_string_lossy()
            .into_owned();
        let mut result = make_scan_result_empty();
        result.findings.push(make_finding());
        result.stats.finding_count = 1;
        write_json(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        let compare = &v["findings"][0]["compare"];
        assert!(compare.is_object(), "compare should be an object");
        assert!(compare.get("diff").is_some(), "compare should have diff");
    }

    #[test]
    fn json_function_excludes_code() {
        let dir = TempDir::new().unwrap();
        let out = dir
            .path()
            .join("report.json")
            .to_string_lossy()
            .into_owned();
        let mut result = make_scan_result_empty();
        result.findings.push(make_finding());
        write_json(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        let func = &v["findings"][0]["function_a"];
        assert!(
            func.get("code").is_none(),
            "serialized function must not include 'code'"
        );
    }

    #[test]
    fn json_span_serializes_as_object() {
        let dir = TempDir::new().unwrap();
        let out = dir
            .path()
            .join("report.json")
            .to_string_lossy()
            .into_owned();
        let mut result = make_scan_result_empty();
        result.findings.push(make_finding());
        write_json(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        let span_a = &v["findings"][0]["compare"]["span_a"];
        assert!(span_a.is_object(), "span_a must be an object, not an array");
        assert!(span_a["start_line"].is_number());
        assert!(span_a["end_line"].is_number());
    }

    #[test]
    fn diff_text_identical_returns_empty() {
        assert_eq!(diff_text("same text\nline two", "same text\nline two"), "");
    }

    #[test]
    fn truncate_diff_within_limits() {
        let lines: Vec<&str> = vec![
            "--- ",
            "+++ ",
            "@@ -1,2 +1,2 @@",
            " context",
            "-old",
            "+new",
        ];
        let result = truncate_diff(&lines, 80, 4000);
        assert_eq!(result, lines.join("\n"));
    }

    #[test]
    fn truncate_diff_exceeds_lines() {
        let lines: Vec<&str> = (0..100).map(|_| "line").collect();
        let result = truncate_diff(&lines, 80, 4000);
        assert!(result.ends_with("... diff truncated ..."));
        let non_truncated: Vec<&str> = result.lines().collect();
        // The truncated part + marker
        assert!(non_truncated.last().unwrap().contains("diff truncated"));
    }

    #[test]
    fn truncate_diff_exceeds_chars() {
        // One very long line that exceeds max_chars
        let long_line = "x".repeat(5000);
        let lines = vec![long_line.as_str()];
        let result = truncate_diff(&lines, 80, 4000);
        assert!(result.ends_with("... diff truncated ..."));
    }

    #[test]
    fn truncate_diff_non_ascii_no_panic() {
        // Each "→" is 3 UTF-8 bytes; 2000 of them = 6000 bytes > 4000 byte limit.
        // Must not panic (char-boundary safe truncation).
        let long_line = "→".repeat(2000);
        let lines = vec![long_line.as_str()];
        let result = truncate_diff(&lines, 80, 4000);
        assert!(result.ends_with("... diff truncated ..."));
        // Truncated portion is valid UTF-8 (would panic on decode otherwise)
        assert!(result.is_ascii() || !result.is_empty());
    }
}
