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
                    "informationUri": "https://github.com/drogers0/clonehunter",
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
    use crate::test_support::{
        make_finding, make_function, make_match, make_scan_result, make_snippet_for,
    };
    use tempfile::TempDir;

    fn make_scan_result_with_finding() -> ScanResult {
        let func_a = make_function("src/a.py", "foo", 1, 10, "def foo(): pass");
        let func_b = make_function("src/a.py", "bar", 20, 30, "def bar(): pass");
        let m = make_match(
            make_snippet_for(&func_a, "t"),
            make_snippet_for(&func_b, "t"),
            0.95,
        );
        make_scan_result(vec![make_finding(
            func_a,
            func_b,
            0.95,
            10,
            vec![m],
            &["high_similarity"],
        )])
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
