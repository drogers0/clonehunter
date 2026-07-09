// Integration tests for the `clonehunter` binary.
// All tests use CLONEHUNTER_EMBEDDER=stub to avoid downloading model weights.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ── Fixture helpers ───────────────────────────────────────────────────────────

/// Two Python files containing identical functions so the stub embedder detects them.
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

// ── JSON output ───────────────────────────────────────────────────────────────

#[test]
fn test_scan_produces_json_report() {
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
    assert!(out.exists(), "report.json must exist");
    let content = fs::read_to_string(&out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");
    assert!(v.get("groups").is_some(), "JSON must have 'groups' key");
    assert!(
        v.get("schema_version").is_some(),
        "JSON must have 'schema_version'"
    );
    assert!(v.get("stats").is_some(), "JSON must have 'stats'");
}

/// The same function duplicated across three files → one clone group of 3 locations.
fn make_triple_fixture() -> TempDir {
    let dir = TempDir::new().unwrap();
    let code = "def compute(x, y):\n    result = x + y\n    result = result * 2\n    result = result - 1\n    return result\n";
    for name in ["file_a.py", "file_b.py", "file_c.py"] {
        fs::write(dir.path().join(name), code).unwrap();
    }
    dir
}

#[test]
fn test_scan_produces_one_group_json_for_shared_function_family() {
    let fixture = make_triple_fixture();
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
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&out).unwrap()).expect("valid JSON");
    assert_eq!(v["stats"]["group_count"], 1, "3 files → 1 clone group");
    assert_eq!(
        v["stats"]["grouped_function_count"], 3,
        "3 unique duplicated functions"
    );
    let groups = v["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0]["locations"].as_array().unwrap().len(), 3);
    assert_eq!(groups[0]["findings"].as_array().unwrap().len(), 3);
}

#[test]
fn test_scan_produces_group_chrome_html_for_shared_function_family() {
    let fixture = make_triple_fixture();
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
        content.contains("clone group across"),
        "summary line present"
    );
    assert!(content.contains("Clone group #1"), "group header present");
}

// ── HTML output ───────────────────────────────────────────────────────────────

#[test]
fn test_scan_produces_html_report() {
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
    assert!(out.exists(), "report.html must exist");
    let content = fs::read_to_string(&out).unwrap();
    assert!(
        content.contains("CloneHunter Report"),
        "HTML must contain page header"
    );
}

// ── SARIF output ──────────────────────────────────────────────────────────────

#[test]
fn test_scan_produces_sarif_report() {
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
    assert!(out.exists(), "report.sarif must exist");
    let content = fs::read_to_string(&out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).expect("valid SARIF JSON");
    assert_eq!(v["version"], "2.1.0", "SARIF version must be 2.1.0");
}

// ── Default format / output path ─────────────────────────────────────────────

#[test]
fn test_scan_default_format_is_html() {
    let fixture = make_dup_fixture();
    // No --format or --out → should write clonehunter_report.html in CWD
    ch().current_dir(fixture.path())
        .arg("scan")
        .arg(".")
        .assert()
        .success();
    assert!(
        fixture.path().join("clonehunter_report.html").exists(),
        "default output must be clonehunter_report.html"
    );
}

#[test]
fn test_scan_default_out_derives_from_format() {
    let fixture = make_dup_fixture();
    ch().current_dir(fixture.path())
        .args(["scan", ".", "--format", "json"])
        .assert()
        .success();
    assert!(
        fixture.path().join("clonehunter_report.json").exists(),
        "--format json without --out must write clonehunter_report.json"
    );
}

// ── Repotype filtering ────────────────────────────────────────────────────────

#[test]
fn test_scan_repotype_python() {
    let fixture = make_dup_fixture();
    // Add a non-Python file
    fs::write(fixture.path().join("lib.rs"), "fn main() {}").unwrap();
    let out = fixture.path().join("report.json");
    ch().args([
        "scan",
        fixture.path().to_str().unwrap(),
        "--repotype",
        "python",
        "--format",
        "json",
        "--out",
        out.to_str().unwrap(),
    ])
    .assert()
    .success();
    let content = fs::read_to_string(&out).unwrap();
    // Rust file should not appear in findings
    assert!(
        !content.contains("lib.rs"),
        "--repotype python must exclude .rs files"
    );
}

#[test]
fn test_scan_monorepo_default() {
    let fixture = make_dup_fixture();
    fs::write(
        fixture.path().join("script.js"),
        "function foo() { return 1; }\n",
    )
    .unwrap();
    let out = fixture.path().join("report.json");
    // No --repotype → monorepo default, scans both .py and .js
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
    // Stats must reference > 0 files (both .py and .js were collected)
    let v: serde_json::Value = serde_json::from_str(&content).unwrap();
    let file_count = v["stats"]["file_count"].as_u64().unwrap_or(0);
    assert!(
        file_count >= 3,
        "monorepo default must collect .py and .js files (got {file_count})"
    );
}

// ── Engine errors ─────────────────────────────────────────────────────────────

#[test]
fn test_scan_sonarqube_error() {
    let fixture = make_dup_fixture();
    let out = fixture.path().join("report.json");
    // No CLONEHUNTER_SONAR_REPORT → SonarQube engine should fail
    Command::cargo_bin("clonehunter")
        .unwrap()
        .env_remove("CLONEHUNTER_SONAR_REPORT")
        .args([
            "scan",
            fixture.path().to_str().unwrap(),
            "--engine",
            "sonarqube",
            "--format",
            "json",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .failure();
}

// ── Diff command ──────────────────────────────────────────────────────────────

fn init_git_repo(dir: &std::path::Path) {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git command");
    };
    run(&["init"]);
    run(&["config", "user.email", "test@test.com"]);
    run(&["config", "user.name", "Test"]);
}

#[test]
fn test_diff_end_to_end() {
    let dir = TempDir::new().unwrap();
    init_git_repo(dir.path());

    let code =
        "def compute(x, y):\n    result = x + y\n    result = result * 2\n    return result\n";
    let file_a = dir.path().join("a.py");
    let file_b = dir.path().join("b.py");
    fs::write(&file_a, code).unwrap();
    fs::write(&file_b, code).unwrap();

    let run_git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .output()
            .expect("git command");
    };
    run_git(&["add", "."]);
    run_git(&["commit", "-m", "init"]);

    // Modify b.py (tracked changed file)
    fs::write(&file_b, format!("{code}\n# modified\n")).unwrap();

    let out = dir.path().join("diff.json");
    ch().current_dir(dir.path())
        .env("CLONEHUNTER_EMBEDDER", "stub")
        .args([
            "diff",
            "--base",
            "HEAD",
            "--format",
            "json",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(out.exists(), "diff report must be written");
    let content = fs::read_to_string(&out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");
    assert!(
        v.get("groups").is_some(),
        "diff report must have groups key"
    );
}

#[test]
fn test_diff_config_root_walk_up() {
    // Regression: `diff` run from a subdirectory must discover clonehunter.toml at the repo
    // root (previously it used cwd directly and silently fell back to defaults, unlike `scan`).
    let dir = TempDir::new().unwrap();
    init_git_repo(dir.path());
    // Root config with a non-default func threshold so we can detect it took effect.
    fs::write(
        dir.path().join("clonehunter.toml"),
        "[thresholds]\nfunc = 0.0\nwin = 0.0\nexp = 0.0\nlexical_min_ratio = 0.0\n",
    )
    .unwrap();
    let sub = dir.path().join("src").join("pkg");
    fs::create_dir_all(&sub).unwrap();
    let f = sub.join("a.py");
    fs::write(&f, "def foo(x):\n    return x + 1\n").unwrap();

    let run_git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .output()
            .expect("git command");
    };
    run_git(&["add", "."]);
    run_git(&["commit", "-m", "init"]);
    fs::write(&f, "def foo(x):\n    return x + 1\n# changed\n").unwrap();

    let out = dir.path().join("diff.json");
    ch().current_dir(&sub) // run from the SUBDIRECTORY
        .args([
            "diff",
            "--base",
            "HEAD",
            "--format",
            "json",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&out).unwrap()).expect("valid JSON");
    // The root config's func threshold (0.0) must be in effect, not the default (0.92).
    assert_eq!(
        v["config"]["thresholds"]["func"].as_f64(),
        Some(0.0),
        "diff must discover clonehunter.toml via walk-up from a subdirectory"
    );
}

#[test]
fn test_diff_includes_untracked() {
    let dir = TempDir::new().unwrap();
    init_git_repo(dir.path());

    let code = "def foo(x):\n    return x + 1\n\n\ndef bar(x):\n    return x + 1\n";
    let committed = dir.path().join("committed.py");
    fs::write(&committed, code).unwrap();

    let run_git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .output()
            .expect("git command");
    };
    run_git(&["add", "."]);
    run_git(&["commit", "-m", "init"]);

    // Untracked new file
    let untracked = dir.path().join("untracked.py");
    fs::write(&untracked, code).unwrap();

    let out = dir.path().join("diff.json");
    ch().current_dir(dir.path())
        .args([
            "diff",
            "--base",
            "HEAD",
            "--format",
            "json",
            "--out",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(out.exists());
}

#[test]
fn test_diff_outside_git_fails() {
    let dir = TempDir::new().unwrap(); // plain dir, no git repo
    ch().current_dir(dir.path())
        .args(["diff", "--format", "json", "--out", "out.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("changed files").or(predicate::str::contains("git")));
}

// ── Config-root walk-up ───────────────────────────────────────────────────────

/// Port of Python test_run_scan_loads_config_from_repo_root_when_path_is_nested.
///
/// Proves that config-root walk-up finds `clonehunter.toml` at the repo root when
/// the scan path points to a nested sub-directory.
///
/// Mechanism: a root-level `clonehunter.toml` sets all thresholds to 0.0 and
/// lexical_min_ratio=0.0.  With those settings, the stub embedder produces findings
/// even for snippets with random similarity (any composite ≥ 0.0 passes).
/// Without walk-up, the default threshold 0.92 would suppress all stub findings.
///
/// The two Python files intentionally differ (different function names) so their
/// snippet_hashes differ and the self-hash filter does not skip them.
#[test]
fn test_scan_config_root_walk_up() {
    // Build temp tree:
    //   root/
    //     clonehunter.toml   ← all thresholds 0.0
    //     src/pkg/
    //       file_a.py
    //       file_b.py
    let root = TempDir::new().unwrap();
    let pkg = root.path().join("src").join("pkg");
    fs::create_dir_all(&pkg).unwrap();

    // Config: thresholds all 0.0 so stub finds anything with any similarity.
    fs::write(
        root.path().join("clonehunter.toml"),
        "[thresholds]\nfunc = 0.0\nwin = 0.0\nexp = 0.0\nlexical_min_ratio = 0.0\n",
    )
    .unwrap();

    // Two files with nearly-identical bodies but different function names
    // → different snippet_hashes → not filtered by self-hash filter.
    let body =
        "    result = x + y\n    result = result * 2\n    result = result - 1\n    return result\n";
    fs::write(pkg.join("file_a.py"), format!("def compute(x, y):\n{body}")).unwrap();
    fs::write(
        pkg.join("file_b.py"),
        format!("def calculate(x, y):\n{body}"),
    )
    .unwrap();

    let out = root.path().join("report.json");
    ch().args([
        "scan",
        pkg.to_str().unwrap(), // scan the nested dir, not root
        "--repotype",
        "python",
        "--format",
        "json",
        "--out",
        out.to_str().unwrap(),
    ])
    .assert()
    .success();

    let content = fs::read_to_string(&out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");
    let finding_count = v["stats"]["finding_count"].as_u64().unwrap_or(0);
    assert!(
        finding_count > 0,
        "config-root walk-up must load threshold=0.0 and produce findings (got {finding_count})"
    );
}

// ── --repotype none ───────────────────────────────────────────────────────────

/// Port of Python test_cli_glob_merge.py's repotype=none behaviour.
///
/// `--repotype none` emits an empty preset list, which collapses to no include globs.
/// With no globs, collect_files returns 0 files → finding_count=0 and file_count=0.
#[test]
fn test_repotype_none_collects_no_files() {
    let fixture = make_dup_fixture();
    let out = fixture.path().join("report.json");
    ch().args([
        "scan",
        fixture.path().to_str().unwrap(),
        "--repotype",
        "none",
        "--format",
        "json",
        "--out",
        out.to_str().unwrap(),
    ])
    .assert()
    .success();

    let content = fs::read_to_string(&out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&content).expect("valid JSON");
    let file_count = v["stats"]["file_count"].as_u64().unwrap_or(999);
    assert_eq!(
        file_count, 0,
        "--repotype none must collect 0 files (got {file_count})"
    );
}
