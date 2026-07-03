use std::collections::BTreeMap;
use std::time::Instant;

use tracing::info;

use crate::core::config::{CloneHunterConfig, EmbedderConfig, EmbedderName};
use crate::core::types::{Degradation, Language, ScanResult, ScanStats};
use crate::embedding::{EmbeddingCache, EmbeddingError, create_embedder, embed_with_cache};
use crate::index::create_index;
use crate::io::fs::collect_files;
use crate::parsing::python_ast::extract_functions;
use crate::parsing::text_units::extract_file_unit;
use crate::similarity::{cluster_findings, filter_clusters, retrieve_candidates, rollup_findings};
use crate::snippets::{
    expansion::expand_calls,
    generators::{generate_function_snippets, generate_window_snippets},
};

use super::PipelineError;

pub(crate) fn run_pipeline(
    paths: &[String],
    config: &CloneHunterConfig,
) -> Result<ScanResult, PipelineError> {
    let mut timing: BTreeMap<String, f64> = BTreeMap::new();
    let mut degradations: Vec<Degradation> = Vec::new();

    // ── Stage 1: collect files ────────────────────────────────────────────────
    let t = Instant::now();
    info!("Stage 1: collecting files from {} paths", paths.len());
    let files = collect_files(paths, &config.include_globs, &config.exclude_globs)?;
    info!("  → {} files collected", files.len());
    timing.insert("collect_files".into(), t.elapsed().as_secs_f64());

    // ── Stage 2: extract units ────────────────────────────────────────────────
    let t = Instant::now();
    info!("Stage 2: extracting units from {} files", files.len());
    let mut python_functions = Vec::new();
    let mut window_units = Vec::new();
    for file in &files {
        match file.language {
            Language::Python => {
                let extracted = extract_functions(file);
                python_functions.extend(extracted.iter().cloned());
                window_units.extend(extracted);
            }
            Language::Text => {
                window_units.extend(extract_file_unit(file));
            }
        }
    }
    info!(
        "  → {} python functions, {} window units",
        python_functions.len(),
        window_units.len()
    );
    timing.insert("extract_functions".into(), t.elapsed().as_secs_f64());

    // ── Stage 3: generate snippets ────────────────────────────────────────────
    let t = Instant::now();
    info!("Stage 3: generating snippets");
    let mut snippets = Vec::new();
    snippets.extend(generate_function_snippets(&python_functions));
    snippets.extend(generate_window_snippets(&window_units, &config.windows));
    // expand_calls checks config.enabled internally (mirrors Python); no outer gate needed.
    snippets.extend(expand_calls(&python_functions, &config.expansion));
    info!("  → {} snippets (FUNC+WIN+EXP)", snippets.len());
    timing.insert("generate_snippets".into(), t.elapsed().as_secs_f64());

    // ── Stage 4: embed ────────────────────────────────────────────────────────
    let t = Instant::now();
    info!("Stage 4: embedding {} snippets", snippets.len());
    let embedder = create_embedder(&config.embedder)?;
    let mut cache = EmbeddingCache::new(&config.cache.path)
        .map_err(|e| EmbeddingError::Cache(e.to_string()))?;
    // DD15: stub embedder gets its own cache namespace (matches Python "stub"/"0"/0 override)
    let cache_config = if config.embedder.name == EmbedderName::Stub {
        EmbedderConfig {
            model_name: "stub".into(),
            revision: "0".into(),
            max_length: 0,
            ..config.embedder.clone()
        }
    } else {
        config.embedder.clone()
    };
    let snippet_refs: Vec<&_> = snippets.iter().collect();
    let (embeddings, cache_hits, cache_misses) =
        embed_with_cache(&snippet_refs, embedder.as_ref(), &cache, &cache_config)?;
    degradations.extend(cache.take_degradations());
    info!(
        "  → embedded ({} hits, {} misses)",
        cache_hits, cache_misses
    );
    timing.insert("embed".into(), t.elapsed().as_secs_f64());

    // ── Stage 5: similarity ───────────────────────────────────────────────────
    let t = Instant::now();
    info!("Stage 5: building index and retrieving candidates");
    let (mut index, index_degradations) = create_index(config.index.name);
    degradations.extend(index_degradations);
    let ids: Vec<String> = snippets.iter().map(|s| s.snippet_hash.clone()).collect();
    index.build(&embeddings, &ids);
    let candidates = retrieve_candidates(
        &snippets,
        &embeddings,
        index.as_ref(),
        &config.thresholds,
        config.index.top_k,
    );
    let candidate_count = candidates.len(); // capture BEFORE rollup consumes the vec (DD12)
    let mut findings = rollup_findings(candidates, &config.thresholds);
    if config.cluster_findings {
        findings = cluster_findings(&findings);
        findings = filter_clusters(&findings, config.cluster_min_size);
    }
    info!(
        "  → {} candidates → {} findings",
        candidate_count,
        findings.len()
    );
    timing.insert("similarity".into(), t.elapsed().as_secs_f64());

    // ── Stage 6: assemble ─────────────────────────────────────────────────────
    let stats = ScanStats {
        file_count: files.len(),
        function_count: python_functions.len(), // Python functions only (DD12)
        snippet_count: snippets.len(),
        candidate_count,
        finding_count: findings.len(),
        cache_hits,
        cache_misses,
    };
    let config_snapshot = serde_json::to_value(config)?; // DD7: full config

    Ok(ScanResult {
        findings,
        stats,
        config_snapshot,
        timing,
        degradations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{
        CacheConfig, CloneHunterConfig, EmbedderConfig, EmbedderName, Thresholds,
    };
    use tempfile::TempDir;

    fn stub_config(cache_dir: &TempDir) -> CloneHunterConfig {
        CloneHunterConfig {
            embedder: EmbedderConfig {
                name: EmbedderName::Stub,
                ..Default::default()
            },
            thresholds: Thresholds {
                func: 0.0,
                win: 0.0,
                exp: 0.0,
                min_window_hits: 1,
                lexical_min_ratio: 0.0,
                lexical_weight: 0.0,
            },
            cache: CacheConfig {
                path: cache_dir.path().join("cache").to_str().unwrap().into(),
            },
            include_globs: vec!["**/*.py".into()],
            exclude_globs: vec![],
            ..CloneHunterConfig::default()
        }
    }

    // ── Test A: end-to-end clone detection ────────────────────────────────────

    #[test]
    fn test_pipeline_detects_clones_e2e() {
        let dir = TempDir::new().unwrap();
        let dup = "def compute(x, y):\n    result = x * 2 + y\n    return result\n";
        std::fs::write(dir.path().join("a.py"), dup).unwrap();
        std::fs::write(dir.path().join("b.py"), dup).unwrap();

        let config = stub_config(&dir);
        let result = run_pipeline(&[dir.path().to_str().unwrap().into()], &config).unwrap();

        assert_eq!(result.stats.file_count, 2);
        assert_eq!(result.stats.function_count, 2); // python_functions only
        assert!(result.stats.snippet_count > 0);
        assert!(
            !result.findings.is_empty(),
            "identical functions must produce at least one finding; got 0"
        );
        let f = &result.findings[0];
        // Identical text + stub (cosine=1.0) + lexical_weight=0.0 → composite=1.0 → score=1.0
        assert!(
            f.score > 0.5,
            "finding score should be high for identical code; got {}",
            f.score
        );
        assert!(
            !f.reasons.is_empty(),
            "finding must have at least one reason"
        );
        assert!(
            f.function_a.file.path != f.function_b.file.path,
            "finding must be cross-file"
        );
    }

    // ── Test B: timing keys present ───────────────────────────────────────────

    #[test]
    fn test_pipeline_timing_keys_present() {
        let dir = TempDir::new().unwrap();
        let dup = "def f(x):\n    return x\n";
        std::fs::write(dir.path().join("a.py"), dup).unwrap();
        std::fs::write(dir.path().join("b.py"), dup).unwrap();

        let config = stub_config(&dir);
        let result = run_pipeline(&[dir.path().to_str().unwrap().into()], &config).unwrap();

        for key in &[
            "collect_files",
            "extract_functions",
            "generate_snippets",
            "embed",
            "similarity",
        ] {
            assert!(
                result.timing.contains_key(*key),
                "missing timing key: {key}"
            );
            assert!(
                *result.timing.get(*key).unwrap() >= 0.0,
                "timing for {key} must be non-negative"
            );
        }
    }

    // ── Test C: non-python files function_count=0, finding_count≥1 ───────────

    #[test]
    fn test_pipeline_non_python_function_count_zero() {
        let dir = TempDir::new().unwrap();
        let js_code = "function add(a, b) { return a + b; }\n".repeat(5);
        std::fs::write(dir.path().join("a.js"), &js_code).unwrap();
        std::fs::write(dir.path().join("b.js"), &js_code).unwrap();

        let mut config = stub_config(&dir);
        config.include_globs = vec!["**/*.js".into()];
        // Use small windows to get WIN snippets
        config.windows.window_lines = 3;
        config.windows.stride_lines = 1;
        config.windows.min_nonempty = 1;

        let result = run_pipeline(&[dir.path().to_str().unwrap().into()], &config).unwrap();
        assert_eq!(
            result.stats.function_count, 0,
            "JS files have no python functions"
        );
        assert_eq!(result.stats.file_count, 2);
        // Identical JS content → WIN snippets with cosine=1.0 → should find clones
        assert!(
            result.stats.finding_count >= 1,
            "identical JS files should produce findings"
        );
    }

    // ── Test D: empty scan (no matching files) ────────────────────────────────

    #[test]
    fn test_pipeline_empty_scan() {
        let dir = TempDir::new().unwrap();
        // Write .py files but scan only for .rs files
        std::fs::write(dir.path().join("a.py"), "def f(): pass\n").unwrap();

        let mut config = stub_config(&dir);
        config.include_globs = vec!["**/*.rs".into()];

        let result = run_pipeline(&[dir.path().to_str().unwrap().into()], &config).unwrap();
        assert_eq!(result.stats.file_count, 0);
        assert_eq!(result.stats.function_count, 0);
        assert_eq!(result.stats.snippet_count, 0);
        assert_eq!(result.stats.finding_count, 0);
        // All timing keys still present even on empty scan
        for key in &[
            "collect_files",
            "extract_functions",
            "generate_snippets",
            "embed",
            "similarity",
        ] {
            assert!(
                result.timing.contains_key(*key),
                "missing timing key: {key}"
            );
        }
    }

    // ── Test I: config snapshot is a JSON object ──────────────────────────────

    #[test]
    fn test_pipeline_config_snapshot_is_object() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.py"), "def f(): pass\n").unwrap();

        let config = stub_config(&dir);
        let result = run_pipeline(&[dir.path().to_str().unwrap().into()], &config).unwrap();
        assert!(
            result.config_snapshot.is_object(),
            "config_snapshot must be a JSON object"
        );
        assert!(
            result.config_snapshot.get("engine").is_some(),
            "config_snapshot must contain 'engine' key"
        );
    }

    // ── Stats parity: candidate_count captured before rollup ─────────────────

    #[test]
    fn test_pipeline_stats_fields_populated() {
        let dir = TempDir::new().unwrap();
        let dup = "def compute(x, y):\n    result = x * 2 + y\n    return result\n";
        std::fs::write(dir.path().join("a.py"), dup).unwrap();
        std::fs::write(dir.path().join("b.py"), dup).unwrap();

        let config = stub_config(&dir);
        let result = run_pipeline(&[dir.path().to_str().unwrap().into()], &config).unwrap();

        // Stats must be populated and internally consistent
        assert_eq!(result.stats.file_count, 2);
        assert_eq!(result.stats.function_count, 2);
        assert!(result.stats.snippet_count >= 2);
        assert_eq!(
            result.stats.cache_hits + result.stats.cache_misses,
            result.stats.snippet_count,
            "hits + misses must equal snippet_count"
        );
        assert_eq!(result.stats.finding_count, result.findings.len());
    }

    // ── DD15: stub cache isolation ────────────────────────────────────────────

    #[test]
    fn test_pipeline_stub_cache_isolation() {
        // Two identical stubs with the same text should get cache hits on the second run.
        // This verifies DD15 isolation does not break the cache (stubs still cache correctly).
        let dir = TempDir::new().unwrap();
        let code = "def foo(x):\n    return x + 1\n";
        std::fs::write(dir.path().join("a.py"), code).unwrap();
        std::fs::write(dir.path().join("b.py"), code).unwrap();

        let config = stub_config(&dir);
        let path = dir.path().to_str().unwrap().into();

        // First run: all misses
        let r1 = run_pipeline(&[path], &config).unwrap();
        assert_eq!(r1.stats.cache_misses, r1.stats.snippet_count);

        // Second run: all hits
        let path2 = dir.path().to_str().unwrap().into();
        let r2 = run_pipeline(&[path2], &config).unwrap();
        assert_eq!(r2.stats.cache_hits, r2.stats.snippet_count);
        assert_eq!(r2.stats.cache_misses, 0);
    }
}
