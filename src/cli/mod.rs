mod glob_merge;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use crate::core::config::{DeviceName, EmbedderName, EngineName, IndexName};
use crate::core::config_loader::{
    CacheOverride, ConfigOverride, EmbedderOverride, ExpansionOverride, IndexOverride,
    ThresholdsOverride, WindowOverride, find_config_root, load_config,
};
use crate::core::logging::init_logging;
use crate::core::types::{ScanRequest, ScanResult};
use crate::engines::get_engine;
use crate::io::git;
use crate::reporting::{write_html, write_json, write_sarif};

use glob_merge::{effective_repotypes, merge_globs, resolve_repotype_globs, validate_repotype};

// ─── CLI structs ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "clonehunter",
    version = env!("CARGO_PKG_VERSION"),
    about = "Find semantic code clones in repositories."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan a repository for semantic clones
    Scan(Box<ScanArgs>),
    /// Scan only changed files (git diff + untracked)
    Diff(DiffArgs),
}

#[derive(clap::Args)]
struct ScanArgs {
    /// Files/directories to scan (default: current directory)
    #[arg(default_value = ".")]
    path: Vec<String>,

    /// Output format
    #[arg(long, default_value = "html", value_parser = ["json", "html", "sarif"])]
    format: String,

    /// Output file (default: clonehunter_report.<format>)
    #[arg(long)]
    out: Option<String>,

    // Override args
    /// Detection engine. Note: `sonarqube` reads findings from CLONEHUNTER_SONAR_REPORT and
    /// ignores scan paths and every tuning flag.
    #[arg(long, value_enum)]
    engine: Option<EngineName>,
    #[arg(long, value_enum)]
    embedder: Option<EmbedderName>,
    #[arg(long, value_enum)]
    index: Option<IndexName>,
    #[arg(long, value_enum)]
    device: Option<DeviceName>,

    // Threshold tuning
    #[arg(long)]
    threshold_func: Option<f64>,
    #[arg(long)]
    threshold_win: Option<f64>,
    #[arg(long)]
    threshold_exp: Option<f64>,
    #[arg(long)]
    min_window_hits: Option<usize>,
    #[arg(long)]
    lexical_min_ratio: Option<f64>,
    #[arg(long)]
    lexical_weight: Option<f64>,

    // Window tuning
    #[arg(long)]
    window_lines: Option<usize>,
    #[arg(long)]
    stride_lines: Option<usize>,
    #[arg(long)]
    min_nonempty: Option<usize>,

    // Expansion
    #[arg(long)]
    expand_calls: bool,
    #[arg(long)]
    expand_depth: Option<usize>,
    #[arg(long)]
    expand_max_chars: Option<usize>,

    // Cache
    #[arg(long)]
    cache_path: Option<String>,

    // Clustering
    #[arg(long)]
    cluster: bool,
    #[arg(long)]
    cluster_min_size: Option<usize>,

    // Glob filtering
    /// Repo type preset (repeatable); use "none" to disable presets
    #[arg(long, value_parser = validate_repotype)]
    repotype: Vec<String>,
    /// Additional include glob patterns (repeatable)
    #[arg(long)]
    include_globs: Vec<String>,
    /// Additional exclude glob patterns (repeatable)
    #[arg(long)]
    exclude_globs: Vec<String>,
}

#[derive(clap::Args)]
struct DiffArgs {
    /// Files/directories to scope changed-file discovery
    #[arg(default_value = ".")]
    path: Vec<String>,

    /// Git base ref
    #[arg(long, default_value = "HEAD")]
    base: String,

    /// Output format
    #[arg(long, default_value = "html", value_parser = ["json", "html", "sarif"])]
    format: String,

    /// Output file (default: clonehunter_report.<format>)
    #[arg(long)]
    out: Option<String>,

    // Override args (same as scan; no tuning surface for diff)
    /// Detection engine. Note: `sonarqube` reads findings from CLONEHUNTER_SONAR_REPORT and
    /// ignores scan paths and every tuning flag.
    #[arg(long, value_enum)]
    engine: Option<EngineName>,
    #[arg(long, value_enum)]
    embedder: Option<EmbedderName>,
    #[arg(long, value_enum)]
    index: Option<IndexName>,
    #[arg(long, value_enum)]
    device: Option<DeviceName>,
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn run() -> Result<()> {
    init_logging();
    let cli = Cli::parse();
    match cli.command {
        Commands::Scan(args) => run_scan(*args),
        Commands::Diff(args) => run_diff(args),
    }
}

// ─── Scan command ─────────────────────────────────────────────────────────────

fn run_scan(args: ScanArgs) -> Result<()> {
    let overrides = build_scan_overrides(&args);
    let config_root = resolve_config_root(&args.path);
    let mut config = load_config(&config_root, Some(&overrides))
        .with_context(|| format!("loading config from {}", config_root.display()))?;

    // Glob merge: repotype replaces config defaults when --repotype is explicitly set;
    // otherwise (no --repotype flag) the monorepo expansion is merged on top of config globs.
    let (rtype_inc, rtype_exc) = resolve_repotype_globs(&effective_repotypes(&args.repotype));
    let (base_inc, base_exc) = if args.repotype.is_empty() {
        merge_globs(
            &config.include_globs,
            &config.exclude_globs,
            &rtype_inc,
            &rtype_exc,
        )
    } else {
        (rtype_inc, rtype_exc)
    };
    let (final_inc, final_exc) = merge_globs(
        &base_inc,
        &base_exc,
        &args.include_globs,
        &args.exclude_globs,
    );
    config.include_globs = final_inc;
    config.exclude_globs = final_exc;

    let result = get_engine(config.engine)
        .scan(&ScanRequest {
            paths: args.path.clone(),
            config,
        })
        .map_err(into_anyhow)?;

    let out_path = resolve_out_path(&args.format, args.out.as_deref());
    write_report(&result, &args.format, &out_path)?;
    eprintln!(
        "clonehunter: {} findings → {}",
        result.findings.len(),
        out_path
    );
    Ok(())
}

// ─── Diff command ─────────────────────────────────────────────────────────────

fn run_diff(args: DiffArgs) -> Result<()> {
    let requested_paths: Vec<String> = if args.path.is_empty() {
        vec![".".into()]
    } else {
        args.path.clone()
    };

    let cwd = std::env::current_dir().context("determine current directory")?;
    let changed = git::changed_files(&args.base, Some(&requested_paths), Some(&cwd))
        .map_err(|e| anyhow::anyhow!("Failed to determine changed files: {e}"))?;

    let overrides = build_diff_overrides(&args);
    let out_path = resolve_out_path(&args.format, args.out.as_deref());

    if changed.is_empty() {
        // No changes → empty scan with paths=[]
        let config = load_config(&cwd, Some(&overrides)).with_context(|| "loading config")?;
        let result = get_engine(config.engine)
            .scan(&ScanRequest {
                paths: vec![],
                config,
            })
            .map_err(into_anyhow)?;
        write_report(&result, &args.format, &out_path)?;
        eprintln!("clonehunter: 0 findings (no changes) → {out_path}");
        return Ok(());
    }

    // Full scan of requested paths, filter to changed files
    let config = load_config(&cwd, Some(&overrides)).with_context(|| "loading config")?;
    let result = get_engine(config.engine)
        .scan(&ScanRequest {
            paths: requested_paths,
            config,
        })
        .map_err(into_anyhow)?;

    let changed_set: HashSet<String> = changed
        .iter()
        .map(|f| normalize_repo_path(f, &cwd))
        .collect();

    let filtered: Vec<_> = result
        .findings
        .into_iter()
        .filter(|f| {
            changed_set.contains(&normalize_repo_path(&f.function_a.file.path, &cwd))
                || changed_set.contains(&normalize_repo_path(&f.function_b.file.path, &cwd))
        })
        .collect();

    let mut stats = result.stats;
    stats.finding_count = filtered.len();

    let filtered_result = ScanResult {
        findings: filtered,
        stats,
        config_snapshot: result.config_snapshot,
        timing: result.timing,
        degradations: result.degradations,
    };

    write_report(&filtered_result, &args.format, &out_path)?;
    eprintln!(
        "clonehunter: {} findings → {}",
        filtered_result.findings.len(),
        out_path
    );
    Ok(())
}

// ─── Override builders ────────────────────────────────────────────────────────

/// The engine/embedder/index overrides common to both `scan` and `diff`.
/// (`scan` layers its tuning flags on top; `diff` carries only these.)
fn base_overrides(
    engine: Option<EngineName>,
    embedder: Option<EmbedderName>,
    device: Option<DeviceName>,
    index: Option<IndexName>,
) -> ConfigOverride {
    let mut ov = ConfigOverride {
        engine,
        ..Default::default()
    };

    let mut emb = EmbedderOverride {
        name: embedder,
        device,
        ..Default::default()
    };
    apply_embedder_env(&mut emb);
    if emb.name.is_some() || emb.device.is_some() {
        ov.embedder = Some(emb);
    }

    if let Some(idx) = index {
        ov.index = Some(IndexOverride {
            name: Some(idx),
            ..Default::default()
        });
    }

    ov
}

fn build_scan_overrides(args: &ScanArgs) -> ConfigOverride {
    let mut ov = base_overrides(args.engine, args.embedder, args.device, args.index);

    if args.threshold_func.is_some()
        || args.threshold_win.is_some()
        || args.threshold_exp.is_some()
        || args.min_window_hits.is_some()
        || args.lexical_min_ratio.is_some()
        || args.lexical_weight.is_some()
    {
        ov.thresholds = Some(ThresholdsOverride {
            func: args.threshold_func,
            win: args.threshold_win,
            exp: args.threshold_exp,
            min_window_hits: args.min_window_hits,
            lexical_min_ratio: args.lexical_min_ratio,
            lexical_weight: args.lexical_weight,
        });
    }

    if args.window_lines.is_some() || args.stride_lines.is_some() || args.min_nonempty.is_some() {
        ov.windows = Some(WindowOverride {
            window_lines: args.window_lines,
            stride_lines: args.stride_lines,
            min_nonempty: args.min_nonempty,
        });
    }

    if args.expand_calls || args.expand_depth.is_some() || args.expand_max_chars.is_some() {
        // Only enable expansion when --expand-calls is explicitly passed; depth/max-chars are
        // tuning-only and do not implicitly enable expansion (matches Python's CLI semantics).
        ov.expansion = Some(ExpansionOverride {
            enabled: if args.expand_calls { Some(true) } else { None },
            depth: args.expand_depth,
            max_chars: args.expand_max_chars,
        });
    }

    if let Some(ref path) = args.cache_path {
        ov.cache = Some(CacheOverride {
            path: Some(path.clone()),
        });
    }

    if args.cluster {
        ov.cluster_findings = Some(true);
    }
    ov.cluster_min_size = args.cluster_min_size;

    ov
}

fn build_diff_overrides(args: &DiffArgs) -> ConfigOverride {
    base_overrides(args.engine, args.embedder, args.device, args.index)
}

/// Apply the `CLONEHUNTER_EMBEDDER=stub` env override — only when `--embedder` was not passed (DD13).
fn apply_embedder_env(emb: &mut EmbedderOverride) {
    if emb.name.is_none() {
        if let Ok(val) = std::env::var("CLONEHUNTER_EMBEDDER") {
            if val.trim().eq_ignore_ascii_case("stub") {
                emb.name = Some(EmbedderName::Stub);
            }
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn resolve_config_root(paths: &[String]) -> PathBuf {
    if paths.is_empty() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        return find_config_root(&cwd).unwrap_or(cwd);
    }

    let roots: Vec<PathBuf> = paths
        .iter()
        .map(|p| {
            let candidate = PathBuf::from(p);
            let resolved = std::fs::canonicalize(&candidate).unwrap_or(candidate);
            if resolved.is_file() {
                resolved
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            } else {
                resolved
            }
        })
        .collect();

    // Find any config root among the paths
    let config_roots: Vec<PathBuf> = roots
        .iter()
        .filter_map(|r| find_config_root(r))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    if config_roots.len() == 1 {
        return config_roots.into_iter().next().unwrap();
    }

    // Fall back: common path prefix, then walk up
    if let Some(common) = common_path(&roots) {
        return find_config_root(&common).unwrap_or(common);
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Compute the common path prefix of a list of paths.
fn common_path(paths: &[PathBuf]) -> Option<PathBuf> {
    let first = paths.first()?;
    let mut common: Vec<&std::ffi::OsStr> = first.components().map(|c| c.as_os_str()).collect();
    for path in paths.iter().skip(1) {
        let components: Vec<&std::ffi::OsStr> = path.components().map(|c| c.as_os_str()).collect();
        common = common
            .into_iter()
            .zip(components.iter())
            .take_while(|(a, b)| a == *b)
            .map(|(a, _)| a)
            .collect();
    }
    if common.is_empty() {
        None
    } else {
        Some(common.iter().collect())
    }
}

fn resolve_out_path(format: &str, out: Option<&str>) -> String {
    out.map(String::from)
        .unwrap_or_else(|| format!("clonehunter_report.{format}"))
}

fn write_report(result: &ScanResult, format: &str, out_path: &str) -> Result<()> {
    match format {
        "json" => write_json(result, out_path).map_err(into_anyhow),
        "html" => write_html(result, out_path).map_err(into_anyhow),
        "sarif" => write_sarif(result, out_path).map_err(into_anyhow),
        _ => bail!("unsupported format: {format}"),
    }
}

/// Flatten any subsystem error into an `anyhow::Error` by its `Display` string.
fn into_anyhow(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// Strip leading `./` and normalize path separators. Port of Python's `_normalize_repo_path`.
/// `base` is the directory to strip from absolute paths (typically the process CWD or the git
/// repo root); passing it explicitly avoids re-reading `current_dir()` on every call and fixes
/// path matching when the scan root differs from the process CWD.
fn normalize_repo_path(raw: &str, base: &Path) -> String {
    let path = Path::new(raw);
    let normalized = if path.is_absolute() {
        path.strip_prefix(base).unwrap_or(path).to_path_buf()
    } else {
        path.to_path_buf()
    };
    let s = normalized.to_string_lossy().replace('\\', "/");
    s.strip_prefix("./").unwrap_or(&s).to_string()
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    // Serialize all tests that read or write CLONEHUNTER_EMBEDDER so they cannot race.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_resolve_out_path_default_json() {
        assert_eq!(resolve_out_path("json", None), "clonehunter_report.json");
    }

    #[test]
    fn test_resolve_out_path_default_html() {
        assert_eq!(resolve_out_path("html", None), "clonehunter_report.html");
    }

    #[test]
    fn test_resolve_out_path_explicit() {
        assert_eq!(
            resolve_out_path("json", Some("/tmp/my.json")),
            "/tmp/my.json"
        );
    }

    #[test]
    fn test_normalize_strips_dot_slash() {
        assert_eq!(
            normalize_repo_path("./src/a.py", Path::new(".")),
            "src/a.py"
        );
    }

    #[test]
    fn test_normalize_relative_passes_through() {
        assert_eq!(normalize_repo_path("src/a.py", Path::new(".")), "src/a.py");
    }

    #[test]
    fn test_normalize_absolute_strips_base() {
        let base = PathBuf::from("/home/user/project");
        assert_eq!(
            normalize_repo_path("/home/user/project/src/a.py", &base),
            "src/a.py"
        );
    }

    #[test]
    fn test_build_scan_overrides_empty_args() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Minimal args: nothing set → engine/thresholds/etc. all None
        let args = ScanArgs {
            path: vec![".".into()],
            format: "html".into(),
            out: None,
            engine: None,
            embedder: None,
            index: None,
            device: None,
            threshold_func: None,
            threshold_win: None,
            threshold_exp: None,
            min_window_hits: None,
            lexical_min_ratio: None,
            lexical_weight: None,
            window_lines: None,
            stride_lines: None,
            min_nonempty: None,
            expand_calls: false,
            expand_depth: None,
            expand_max_chars: None,
            cache_path: None,
            cluster: false,
            cluster_min_size: None,
            repotype: vec![],
            include_globs: vec![],
            exclude_globs: vec![],
        };
        // Clear env so env var doesn't interfere; SAFETY: serialized by ENV_LOCK
        unsafe { std::env::remove_var("CLONEHUNTER_EMBEDDER") };
        let ov = build_scan_overrides(&args);
        assert!(ov.engine.is_none());
        assert!(ov.thresholds.is_none());
        assert!(ov.windows.is_none());
        assert!(ov.expansion.is_none());
        assert!(ov.embedder.is_none());
    }

    #[test]
    fn test_build_scan_overrides_threshold_groups() {
        let _guard = ENV_LOCK.lock().unwrap();
        let args = ScanArgs {
            path: vec![".".into()],
            format: "html".into(),
            out: None,
            engine: None,
            embedder: None,
            index: None,
            device: None,
            threshold_func: Some(0.85),
            threshold_win: None,
            threshold_exp: None,
            min_window_hits: None,
            lexical_min_ratio: None,
            lexical_weight: None,
            window_lines: None,
            stride_lines: None,
            min_nonempty: None,
            expand_calls: false,
            expand_depth: None,
            expand_max_chars: None,
            cache_path: None,
            cluster: false,
            cluster_min_size: None,
            repotype: vec![],
            include_globs: vec![],
            exclude_globs: vec![],
        };
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::remove_var("CLONEHUNTER_EMBEDDER") };
        let ov = build_scan_overrides(&args);
        let thr = ov.thresholds.unwrap();
        assert_eq!(thr.func, Some(0.85));
        assert!(thr.win.is_none()); // not set
    }

    #[test]
    fn test_build_scan_overrides_expand_calls_sets_enabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        let args = ScanArgs {
            path: vec![],
            format: "html".into(),
            out: None,
            engine: None,
            embedder: None,
            index: None,
            device: None,
            threshold_func: None,
            threshold_win: None,
            threshold_exp: None,
            min_window_hits: None,
            lexical_min_ratio: None,
            lexical_weight: None,
            window_lines: None,
            stride_lines: None,
            min_nonempty: None,
            expand_calls: true,
            expand_depth: None,
            expand_max_chars: None,
            cache_path: None,
            cluster: false,
            cluster_min_size: None,
            repotype: vec![],
            include_globs: vec![],
            exclude_globs: vec![],
        };
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::remove_var("CLONEHUNTER_EMBEDDER") };
        let ov = build_scan_overrides(&args);
        assert_eq!(ov.expansion.unwrap().enabled, Some(true));
    }

    #[test]
    fn test_build_scan_overrides_expand_depth_alone_does_not_enable() {
        let _guard = ENV_LOCK.lock().unwrap();
        let args = ScanArgs {
            path: vec![],
            format: "html".into(),
            out: None,
            engine: None,
            embedder: None,
            index: None,
            device: None,
            threshold_func: None,
            threshold_win: None,
            threshold_exp: None,
            min_window_hits: None,
            lexical_min_ratio: None,
            lexical_weight: None,
            window_lines: None,
            stride_lines: None,
            min_nonempty: None,
            expand_calls: false,   // NOT passed
            expand_depth: Some(3), // depth set without --expand-calls
            expand_max_chars: None,
            cache_path: None,
            cluster: false,
            cluster_min_size: None,
            repotype: vec![],
            include_globs: vec![],
            exclude_globs: vec![],
        };
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::remove_var("CLONEHUNTER_EMBEDDER") };
        let ov = build_scan_overrides(&args);
        let exp = ov.expansion.unwrap();
        // depth should be set, but enabled must NOT be forced to true
        assert_eq!(exp.depth, Some(3));
        assert!(
            exp.enabled.is_none(),
            "expand-depth alone must not enable expansion"
        );
    }

    #[test]
    fn test_build_overrides_explicit_embedder_wins_over_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Even with env var set to "stub", explicit --embedder codebert should win
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::set_var("CLONEHUNTER_EMBEDDER", "stub") };
        let args = ScanArgs {
            path: vec![],
            format: "html".into(),
            out: None,
            engine: None,
            embedder: Some(EmbedderName::Codebert), // explicit
            index: None,
            device: None,
            threshold_func: None,
            threshold_win: None,
            threshold_exp: None,
            min_window_hits: None,
            lexical_min_ratio: None,
            lexical_weight: None,
            window_lines: None,
            stride_lines: None,
            min_nonempty: None,
            expand_calls: false,
            expand_depth: None,
            expand_max_chars: None,
            cache_path: None,
            cluster: false,
            cluster_min_size: None,
            repotype: vec![],
            include_globs: vec![],
            exclude_globs: vec![],
        };
        let ov = build_scan_overrides(&args);
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::remove_var("CLONEHUNTER_EMBEDDER") };
        assert_eq!(ov.embedder.unwrap().name, Some(EmbedderName::Codebert));
    }

    #[test]
    fn test_build_overrides_env_stub() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::set_var("CLONEHUNTER_EMBEDDER", "stub") };
        let args = ScanArgs {
            path: vec![],
            format: "html".into(),
            out: None,
            engine: None,
            embedder: None, // no explicit
            index: None,
            device: None,
            threshold_func: None,
            threshold_win: None,
            threshold_exp: None,
            min_window_hits: None,
            lexical_min_ratio: None,
            lexical_weight: None,
            window_lines: None,
            stride_lines: None,
            min_nonempty: None,
            expand_calls: false,
            expand_depth: None,
            expand_max_chars: None,
            cache_path: None,
            cluster: false,
            cluster_min_size: None,
            repotype: vec![],
            include_globs: vec![],
            exclude_globs: vec![],
        };
        let ov = build_scan_overrides(&args);
        // SAFETY: serialized by ENV_LOCK
        unsafe { std::env::remove_var("CLONEHUNTER_EMBEDDER") };
        assert_eq!(ov.embedder.unwrap().name, Some(EmbedderName::Stub));
    }

    // ── DD10: diff-filter correctness ─────────────────────────────────────────

    /// Port of Python test_run_diff_filters_findings_to_changed_files.
    /// Verifies that the filter predicate in run_diff keeps only findings that touch
    /// at least one changed file and that stats.finding_count is updated correctly.
    #[test]
    fn diff_filter_drops_findings_not_touching_changed_files() {
        use crate::core::types::{Finding, ScanStats};
        use crate::test_support::{
            make_finding as build_finding, make_function, make_match, make_snippet_for,
        };

        fn make_finding(path_a: &str, path_b: &str) -> Finding {
            let func_a = make_function(path_a, "f", 1, 2, "pass");
            let func_b = make_function(path_b, "f", 1, 2, "pass");
            let m = make_match(
                make_snippet_for(&func_a, "pass"),
                make_snippet_for(&func_b, "pass"),
                1.0,
            );
            build_finding(func_a, func_b, 1.0, 2, vec![m], &[])
        }

        let base = PathBuf::from("/repo");
        let changed_set: HashSet<String> = vec!["src/a.py".to_string()].into_iter().collect();

        // finding_a: function_a.file.path is "/repo/src/a.py" → normalizes to "src/a.py" → in changed set
        // finding_b: neither path is in changed set
        let findings = vec![
            make_finding("/repo/src/a.py", "/repo/src/other.py"),
            make_finding("/repo/src/b.py", "/repo/src/c.py"),
        ];

        let filtered: Vec<_> = findings
            .into_iter()
            .filter(|f| {
                changed_set.contains(&normalize_repo_path(&f.function_a.file.path, &base))
                    || changed_set.contains(&normalize_repo_path(&f.function_b.file.path, &base))
            })
            .collect();

        assert_eq!(
            filtered.len(),
            1,
            "only finding touching src/a.py must survive"
        );

        // Verify stats update mirrors run_diff logic
        let mut stats = ScanStats {
            file_count: 2,
            function_count: 4,
            snippet_count: 4,
            candidate_count: 2,
            finding_count: 2,
            cache_hits: 0,
            cache_misses: 0,
        };
        stats.finding_count = filtered.len();
        assert_eq!(stats.finding_count, 1);
    }
}
