use std::fs::File;
use std::io::BufWriter;

use serde_json::json;

use crate::core::types::{Finding, FunctionRef, ScanResult};
use crate::reporting::ReportError;
use crate::reporting::schema::SCHEMA_VERSION;

pub(crate) fn write_sarif(result: &ScanResult, out_path: &str) -> Result<(), ReportError> {
    let results: Vec<_> = result.findings.iter().map(to_sarif_result).collect();
    let payload = json!({
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
        "version": "2.1.0",
        "properties": { "schema_version": SCHEMA_VERSION },
        "runs": [{
            "tool": {
                "driver": {
                    "name": "CloneHunter",
                    "informationUri": "https://example.com/clonehunter",
                    "rules": [{
                        "id": "clonehunter",
                        "name": "SemanticClone",
                        "shortDescription": { "text": "Semantic code clone" },
                    }],
                }
            },
            "results": results,
        }],
    });
    let file = File::create(out_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &payload)?;
    Ok(())
}

fn to_sarif_result(finding: &Finding) -> serde_json::Value {
    json!({
        "ruleId": "clonehunter",
        "level": "note",
        "message": { "text": "Potential semantic clone" },
        "locations": [
            sarif_location(&finding.function_a),
            sarif_location(&finding.function_b),
        ],
        "properties": {
            "score": finding.score,
            "duplicated_lines": finding.duplicated_lines,
            "reasons": finding.reasons,
            "metadata": finding.metadata,
        },
    })
}

fn sarif_location(func: &FunctionRef) -> serde_json::Value {
    json!({
        "physicalLocation": {
            "artifactLocation": { "uri": func.file.path },
            "region": { "startLine": func.start_line, "endLine": func.end_line },
        }
    })
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

    fn make_scan_result_with_finding() -> ScanResult {
        let file = FileRef {
            path: "src/a.py".into(),
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
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: "sha".into(),
        };
        let snip_b = SnippetRef {
            kind: SnippetKind::Func,
            function: func_b.clone(),
            start_line: 20,
            end_line: 30,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: "shb".into(),
        };
        ScanResult {
            findings: vec![Finding {
                function_a: func_a,
                function_b: func_b,
                score: 0.95,
                duplicated_lines: 10,
                evidence: vec![CandidateMatch {
                    snippet_a: snip_a,
                    snippet_b: snip_b,
                    similarity: 0.95,
                    evidence: "".into(),
                }],
                reasons: vec!["high_similarity".into()],
                metadata: BTreeMap::new(),
            }],
            stats: ScanStats {
                file_count: 1,
                function_count: 2,
                snippet_count: 2,
                candidate_count: 1,
                finding_count: 1,
                cache_hits: 0,
                cache_misses: 2,
            },
            config_snapshot: serde_json::json!({}),
            timing: BTreeMap::new(),
            degradations: vec![],
        }
    }

    #[test]
    fn sarif_structure_valid() {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.sarif").to_string_lossy().into_owned();
        let result = make_scan_result_with_finding();
        write_sarif(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(v["version"], "2.1.0");
        assert!(v.get("$schema").is_some());
        let results = &v["runs"][0]["results"];
        assert_eq!(results.as_array().unwrap().len(), 1);
    }

    #[test]
    fn sarif_result_has_two_locations() {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.sarif").to_string_lossy().into_owned();
        let result = make_scan_result_with_finding();
        write_sarif(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        let locs = &v["runs"][0]["results"][0]["locations"];
        assert_eq!(locs.as_array().unwrap().len(), 2);
    }

    #[test]
    fn sarif_properties_include_score() {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("r.sarif").to_string_lossy().into_owned();
        let result = make_scan_result_with_finding();
        write_sarif(&result, &out).unwrap();
        let content = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        let props = &v["runs"][0]["results"][0]["properties"];
        assert!(props.get("score").is_some());
        assert!(props.get("duplicated_lines").is_some());
        assert!(props.get("reasons").is_some());
    }
}
