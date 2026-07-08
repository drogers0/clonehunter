// Golden snapshot tests that lock the JSON, SARIF, and HTML report schemas.
// Uses CLONEHUNTER_EMBEDDER=stub to avoid model weights.
//
// To regenerate snapshots after an intentional schema change:
//   INSTA_UPDATE=new cargo test --test golden_fixtures
//   cargo insta review

use assert_cmd::Command;
use std::fs;
use tempfile::TempDir;

// ── Fixture ───────────────────────────────────────────────────────────────────

/// Two Python files with identical functions. `snippet_hash` folds in the file path, so the
/// self-hash filter in retrieve_candidates skips only a snippet matching *itself* — byte-identical
/// code in different files IS reported. The two functions (`compute`, `helper`) therefore yield
/// two cross-file findings, each an unclustered singleton group (2 locations, 1 finding). The
/// snapshot locks the top-level schema keys (schema_version, groups, stats, config, timing,
/// degradations) and the grouped finding shape.
fn make_dup_fixture() -> TempDir {
    let dir = TempDir::new().unwrap();
    let code = "def compute(x, y):\n    result = x + y\n    result = result * 2\n    result = result - 1\n    return result\n\n\ndef helper(items):\n    output = []\n    for item in items:\n        if item > 0:\n            output.append(item)\n    return output\n";
    fs::write(dir.path().join("file_a.py"), code).unwrap();
    fs::write(dir.path().join("file_b.py"), code).unwrap();
    dir
}

fn ch() -> Command {
    let mut cmd = Command::cargo_bin("clonehunter").unwrap();
    cmd.env("CLONEHUNTER_EMBEDDER", "stub");
    cmd
}

// ── JSON golden ───────────────────────────────────────────────────────────────

#[test]
fn json_schema_golden() {
    let fixture = make_dup_fixture();
    let out = fixture.path().join("report.json");
    ch().args([
        "scan",
        fixture.path().to_str().unwrap(),
        "--format",
        "json",
        "--out",
        out.to_str().unwrap(),
    ])
    .assert()
    .success();

    let content = fs::read_to_string(&out).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();

    let mut settings = insta::Settings::clone_current();
    // Volatile fields — replaced with stable placeholders in the snapshot.
    settings.add_redaction(".schema_version", "[VERSION]");
    settings.add_redaction(".timing", "[TIMING]");
    settings.add_redaction(".config.embedder.revision", "[REVISION]");
    // Temp-dir absolute paths differ per run. Findings are nested under groups (#5), and each
    // group also carries its member functions under `locations`.
    settings.add_redaction(".groups[].locations[].file.path", "[PATH]");
    settings.add_redaction(".groups[].findings[].function_a.file.path", "[PATH]");
    settings.add_redaction(".groups[].findings[].function_b.file.path", "[PATH]");
    settings.bind(|| {
        insta::assert_json_snapshot!("json_schema_golden", &value);
    });
}

// ── SARIF golden ──────────────────────────────────────────────────────────────

#[test]
fn sarif_schema_golden() {
    let fixture = make_dup_fixture();
    let out = fixture.path().join("report.sarif");
    ch().args([
        "scan",
        fixture.path().to_str().unwrap(),
        "--format",
        "sarif",
        "--out",
        out.to_str().unwrap(),
    ])
    .assert()
    .success();

    let content = fs::read_to_string(&out).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();

    let mut settings = insta::Settings::clone_current();
    // SARIF stores schema_version under `.properties`, not at top level.
    settings.add_redaction(".properties.schema_version", "[VERSION]");
    // Temp-dir paths in finding locations.
    settings.add_redaction(
        ".runs[].results[].locations[].physicalLocation.artifactLocation.uri",
        "[PATH]",
    );
    settings.add_redaction(
        ".runs[].results[].relatedLocations[].physicalLocation.artifactLocation.uri",
        "[PATH]",
    );
    settings.bind(|| {
        insta::assert_json_snapshot!("sarif_schema_golden", &value);
    });
}

// ── HTML structural smoke ─────────────────────────────────────────────────────

/// Structural smoke test — HTML is too large and CSS/JS-noisy for a golden snapshot.
/// This test checks that the report is well-formed HTML with the expected markers.
/// The JSON/SARIF goldens above ARE the schema contract.
#[test]
fn html_smoke_test() {
    let fixture = make_dup_fixture();
    let out = fixture.path().join("report.html");
    ch().args([
        "scan",
        fixture.path().to_str().unwrap(),
        "--format",
        "html",
        "--out",
        out.to_str().unwrap(),
    ])
    .assert()
    .success();

    let content = fs::read_to_string(&out).unwrap();
    assert!(
        content.contains("CloneHunter Report"),
        "HTML must have page title"
    );
    assert!(content.contains("Schema:"), "HTML must show schema version");
    assert!(
        content.contains("Findings:"),
        "HTML must show findings count"
    );
    assert!(
        content.contains("sort-findings"),
        "HTML must have sort control"
    );
}
