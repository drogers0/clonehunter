//! Phase 0 Parity Spike — T1a + T1b + T1c
#![allow(dead_code)]
//!
//! Run: `cd spike && cargo run --release`
//! Reads fixture JSON from `spike/fixtures/` (must run generate_references.py first).
//! Outputs a structured summary and sets exit code:
//!   0 = GO or CONDITIONAL GO
//!   1 = NO-GO
//!   2 = INCONCLUSIVE (tooling/infrastructure failure)

mod candle_embed;
mod normalize;
mod treesitter_parse;

use anyhow::{Context, Result};
use candle_embed::{Embedder, RefEmbedding, run_t1a};
use normalize::run_t1c;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use treesitter_parse::{RefFunction, run_t1b};

#[derive(Debug, Clone)]
enum Verdict {
    Pass,
    Conditional(String),
    Fail(String),
    Inconclusive(String),
}

impl Verdict {
    fn label(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Conditional(_) => "CONDITIONAL GO",
            Verdict::Fail(_) => "NO-GO",
            Verdict::Inconclusive(_) => "INCONCLUSIVE",
        }
    }

    fn is_fail(&self) -> bool {
        matches!(self, Verdict::Fail(_))
    }

    fn is_inconclusive(&self) -> bool {
        matches!(self, Verdict::Inconclusive(_))
    }
}

#[derive(Deserialize)]
struct ModelInfo {
    model: String,
    revision: String,
}

#[derive(Deserialize)]
struct NormalizedEntry {
    original: String,
    normalized: String,
}

fn fixtures_dir() -> PathBuf {
    // Run from repo root or from spike/
    let candidates = [
        PathBuf::from("spike/fixtures"),
        PathBuf::from("fixtures"),
    ];
    for p in &candidates {
        if p.exists() {
            return p.clone();
        }
    }
    PathBuf::from("spike/fixtures")
}

fn parse_targets_dir() -> PathBuf {
    let candidates = [
        PathBuf::from("spike/fixtures/parse_targets"),
        PathBuf::from("fixtures/parse_targets"),
    ];
    for p in &candidates {
        if p.exists() {
            return p.clone();
        }
    }
    PathBuf::from("spike/fixtures/parse_targets")
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parse {}", path.display()))
}

fn main() {
    eprintln!("=== Phase 0 Parity Spike ===\n");

    let fx = fixtures_dir();
    let pt = parse_targets_dir();

    // Load reference data
    let model_info: ModelInfo = match load_json(&fx.join("model_info.json")) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("INCONCLUSIVE: cannot load model_info.json: {e}");
            eprintln!("Did you run `uv run python spike/generate_references.py` first?");
            std::process::exit(2);
        }
    };

    let ref_embeddings: Vec<RefEmbedding> = match load_json(&fx.join("python_embeddings.json")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("INCONCLUSIVE: cannot load python_embeddings.json: {e}");
            std::process::exit(2);
        }
    };

    let py_cosines_raw: Vec<Vec<f64>> = match load_json(&fx.join("python_cosines.json")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("INCONCLUSIVE: cannot load python_cosines.json: {e}");
            std::process::exit(2);
        }
    };
    let py_cosines: Vec<Vec<f32>> = py_cosines_raw
        .iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();

    let ref_functions: Vec<RefFunction> = match load_json(&fx.join("python_functions.json")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("INCONCLUSIVE: cannot load python_functions.json: {e}");
            std::process::exit(2);
        }
    };

    let normalized_entries: Vec<NormalizedEntry> =
        match load_json(&fx.join("python_normalized.json")) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("INCONCLUSIVE: cannot load python_normalized.json: {e}");
                std::process::exit(2);
            }
        };

    let py_lexical_raw: Vec<Vec<f64>> = match load_json(&fx.join("python_lexical_scores.json")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("INCONCLUSIVE: cannot load python_lexical_scores.json: {e}");
            std::process::exit(2);
        }
    };

    eprintln!("Fixture corpus: {} snippets", ref_embeddings.len());
    eprintln!("Model: {} @ {}", model_info.model, model_info.revision);
    eprintln!("Reference functions: {}", ref_functions.len());

    // -----------------------------------------------------------------------
    // Load candle model (once, shared between T1a and T1c)
    // -----------------------------------------------------------------------
    eprintln!("\n[Load] Loading candle CodeBERT model...");
    let embedder = match Embedder::load(&model_info.model, &model_info.revision, &fx) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("INCONCLUSIVE: failed to load candle model: {e}");
            std::process::exit(2);
        }
    };

    // -----------------------------------------------------------------------
    // T1a — Candle RoBERTa/CodeBERT numeric parity
    // -----------------------------------------------------------------------
    eprintln!("\n=== T1a: Candle Embedding Parity ===");
    let t1a = run_t1a(&embedder, &ref_embeddings, &py_cosines);

    let t1a_verdict = if t1a.max_cosine_diff >= 1e-3 {
        Verdict::Fail(format!(
            "max cosine diff {:.2e} >= 1e-3",
            t1a.max_cosine_diff
        ))
    } else if t1a.token_id_match_rate < 1.0 {
        Verdict::Fail(format!(
            "token_id_match_rate {:.4} < 1.0",
            t1a.token_id_match_rate
        ))
    } else if !t1a.determinism_pass {
        Verdict::Fail("non-deterministic embeddings".to_string())
    } else if t1a.batch_invariance_max_diff >= 1e-4 {
        // Threshold: 1e-4. Values up to ~1e-5 are expected from padding changing
        // effective sequence lengths for LayerNorm in different batch sizes.
        Verdict::Fail(format!(
            "batch invariance violation {:.2e}",
            t1a.batch_invariance_max_diff
        ))
    } else if !t1a.near_threshold_flips.is_empty() {
        Verdict::Fail(format!(
            "{} near-threshold pairs flip sides",
            t1a.near_threshold_flips.len()
        ))
    } else if t1a.max_cosine_diff >= 1e-4 {
        Verdict::Conditional(format!(
            "max cosine diff {:.2e} in [1e-4, 1e-3) — orchestrator sign-off required before T2",
            t1a.max_cosine_diff
        ))
    } else {
        Verdict::Pass
    };

    eprintln!("  Token ID match rate:     {:.6}", t1a.token_id_match_rate);
    eprintln!("  Max dim absolute diff:   {:.2e}", t1a.max_dim_diff);
    eprintln!("  Mean dim absolute diff:  {:.2e}", t1a.mean_dim_diff);
    eprintln!("  Max cosine diff:         {:.2e}", t1a.max_cosine_diff);
    eprintln!(
        "  Pairs exceeding 1e-4:    {} / {}",
        t1a.pairs_exceeding_1e4, t1a.total_pairs
    );
    eprintln!("  Near-threshold flips:    {}", t1a.near_threshold_flips.len());
    eprintln!("  Determinism:             {}", if t1a.determinism_pass { "PASS" } else { "FAIL" });
    eprintln!("  Batch invariance max:    {:.2e}", t1a.batch_invariance_max_diff);
    eprintln!("  Verdict: {}", t1a_verdict.label());
    if let Verdict::Conditional(ref msg) | Verdict::Fail(ref msg) = t1a_verdict {
        eprintln!("    → {msg}");
    }
    for flip in &t1a.near_threshold_flips {
        eprintln!("    flip: {flip}");
    }

    // -----------------------------------------------------------------------
    // T1b — tree-sitter vs ast function extraction
    // -----------------------------------------------------------------------
    eprintln!("\n=== T1b: Tree-sitter Extraction Parity ===");
    let t1b = run_t1b(&pt, &ref_functions);

    let qname_rate = if t1b.total_ref == 0 {
        1.0
    } else {
        t1b.qname_matches as f64 / t1b.total_ref as f64
    };
    let code_rate = if t1b.total_ref == 0 {
        1.0
    } else {
        t1b.code_text_matches as f64 / t1b.total_ref as f64
    };

    let t1b_verdict = if qname_rate < 1.0 {
        Verdict::Fail(format!(
            "qname match rate {:.4} < 1.0 ({} / {} matched)",
            qname_rate, t1b.qname_matches, t1b.total_ref
        ))
    } else if code_rate < 1.0 && t1b.adjustments_applied.is_empty() {
        Verdict::Fail(format!(
            "code text match rate {:.4} < 1.0 without adjustment ({} / {})",
            code_rate, t1b.code_text_matches, t1b.total_ref
        ))
    } else if !t1b.span_diffs.is_empty() {
        Verdict::Conditional(format!(
            "spans differ for {} functions — documented above",
            t1b.span_diffs.len()
        ))
    } else if code_rate < 1.0 {
        Verdict::Conditional(format!(
            "code text matches only after adjustments: {:?}",
            t1b.adjustments_applied
        ))
    } else {
        Verdict::Pass
    };

    eprintln!("  Reference functions:     {}", t1b.total_ref);
    eprintln!("  Rust-extracted:          {}", t1b.total_rust);
    eprintln!("  Qualified name matches:  {} / {} ({:.1}%)", t1b.qname_matches, t1b.total_ref, qname_rate * 100.0);
    eprintln!("  Code text matches:       {} / {} ({:.1}%)", t1b.code_text_matches, t1b.total_ref, code_rate * 100.0);
    eprintln!("  Span divergences:        {}", t1b.span_diffs.len());
    eprintln!("  Lambda exclusion:        {}", if t1b.lambda_exclusion_ok { "OK" } else { "FAIL (lambda found)" });
    for d in &t1b.span_diffs {
        eprintln!("    {d}");
    }
    eprintln!("  Verdict: {}", t1b_verdict.label());
    if let Verdict::Conditional(ref msg) | Verdict::Fail(ref msg) = t1b_verdict {
        eprintln!("    → {msg}");
    }

    // -----------------------------------------------------------------------
    // T1c — Normalization impact
    // -----------------------------------------------------------------------
    eprintln!("\n=== T1c: Normalization Impact ===");
    let original_texts: Vec<String> = normalized_entries.iter().map(|e| e.original.clone()).collect();
    let py_normalized: Vec<String> = normalized_entries.iter().map(|e| e.normalized.clone()).collect();

    let embed_fn = |texts: &[&str]| -> anyhow::Result<Vec<Vec<f32>>> {
        let (embs, _) = embedder.embed_batch(texts)?;
        Ok(embs)
    };

    let t1c = run_t1c(&original_texts, &py_normalized, &py_lexical_raw, embed_fn);

    let t1c_verdict = if t1c.embedding_cosines.is_empty() {
        Verdict::Inconclusive("embedding failed during T1c".to_string())
    } else if t1c.min_cosine < 0.90 {
        Verdict::Fail(format!(
            "normalization cosine min {:.4} < 0.90",
            t1c.min_cosine
        ))
    } else if t1c.max_lexical_diff > 0.10 {
        Verdict::Fail(format!(
            "lexical diff max {:.4} > 0.10",
            t1c.max_lexical_diff
        ))
    } else if !t1c.near_threshold_flips.is_empty() {
        Verdict::Fail(format!(
            "{} near-threshold lexical pairs flip",
            t1c.near_threshold_flips.len()
        ))
    } else if t1c.min_cosine < 0.99 || t1c.max_lexical_diff > 0.05 {
        Verdict::Conditional(format!(
            "normalization impact: min_cosine={:.4} lexical_diff_max={:.4} — orchestrator sign-off required",
            t1c.min_cosine, t1c.max_lexical_diff
        ))
    } else {
        Verdict::Pass
    };

    eprintln!("  Normalization strategy:  tree-sitter docstring→pass + source passthrough");
    eprintln!("  Embedding cosine (rust-norm vs py-norm):");
    eprintln!("    min:  {:.4}", t1c.min_cosine);
    eprintln!("    mean: {:.4}", t1c.mean_cosine);
    eprintln!("    max:  {:.4}", t1c.max_cosine);
    eprintln!("  Lexical score diff (rust-norm vs py-norm):");
    eprintln!("    max:  {:.4}", t1c.max_lexical_diff);
    eprintln!("    mean: {:.6}", t1c.mean_lexical_diff);
    eprintln!("  Normalization diff categories:");
    for cat in &t1c.diff_categories {
        eprintln!("    - {cat}");
    }
    eprintln!("  Near-threshold lexical flips: {}", t1c.near_threshold_flips.len());
    for flip in &t1c.near_threshold_flips {
        eprintln!("    {flip}");
    }
    eprintln!("  Verdict: {}", t1c_verdict.label());
    if let Verdict::Conditional(ref msg) | Verdict::Fail(ref msg) = t1c_verdict {
        eprintln!("    → {msg}");
    }

    // -----------------------------------------------------------------------
    // Overall GO/NO-GO
    // -----------------------------------------------------------------------
    eprintln!("\n=== OVERALL VERDICT ===");

    let any_fail = t1a_verdict.is_fail() || t1b_verdict.is_fail() || t1c_verdict.is_fail();
    let any_inconclusive =
        t1a_verdict.is_inconclusive() || t1b_verdict.is_inconclusive() || t1c_verdict.is_inconclusive();
    let any_conditional = matches!(t1a_verdict, Verdict::Conditional(_))
        || matches!(t1b_verdict, Verdict::Conditional(_))
        || matches!(t1c_verdict, Verdict::Conditional(_));

    let overall = if any_fail {
        "NO-GO"
    } else if any_inconclusive {
        "INCONCLUSIVE"
    } else if any_conditional {
        "CONDITIONAL GO"
    } else {
        "GO"
    };

    eprintln!("  T1a: {}", t1a_verdict.label());
    eprintln!("  T1b: {}", t1b_verdict.label());
    eprintln!("  T1c: {}", t1c_verdict.label());
    eprintln!();
    eprintln!("  *** {overall} ***");
    eprintln!();

    if overall == "CONDITIONAL GO" {
        eprintln!("  CONDITIONAL mitigations required before T2:");
        if let Verdict::Conditional(ref m) = t1a_verdict {
            eprintln!("    [T1a] {m}");
        }
        if let Verdict::Conditional(ref m) = t1b_verdict {
            eprintln!("    [T1b] {m}");
        }
        if let Verdict::Conditional(ref m) = t1c_verdict {
            eprintln!("    [T1c] {m}");
        }
    }

    if overall == "NO-GO" || overall == "INCONCLUSIVE" {
        eprintln!("  ESCALATION REQUIRED — report to orchestrator before proceeding.");
    }

    // Print key numbers for the memo
    eprintln!("\n=== KEY NUMBERS FOR MEMO ===");
    eprintln!("  token_id_match_rate:      {:.6}", t1a.token_id_match_rate);
    eprintln!("  max_dim_diff:             {:.2e}", t1a.max_dim_diff);
    eprintln!("  mean_dim_diff:            {:.2e}", t1a.mean_dim_diff);
    eprintln!("  max_cosine_diff:          {:.2e}", t1a.max_cosine_diff);
    eprintln!("  pairs_exceeding_1e4:      {} / {}", t1a.pairs_exceeding_1e4, t1a.total_pairs);
    eprintln!("  determinism:              {}", if t1a.determinism_pass { "PASS" } else { "FAIL" });
    eprintln!("  batch_invariance_max:     {:.2e}", t1a.batch_invariance_max_diff);
    eprintln!("  qname_match_rate:         {:.6}", qname_rate);
    eprintln!("  code_text_match_rate:     {:.6}", code_rate);
    eprintln!("  norm_cosine_min:          {:.4}", t1c.min_cosine);
    eprintln!("  norm_cosine_mean:         {:.4}", t1c.mean_cosine);
    eprintln!("  norm_cosine_max:          {:.4}", t1c.max_cosine);
    eprintln!("  lexical_diff_max:         {:.4}", t1c.max_lexical_diff);
    eprintln!("  lexical_diff_mean:        {:.6}", t1c.mean_lexical_diff);

    let exit_code: i32 = if any_fail {
        1
    } else if any_inconclusive {
        2
    } else {
        0
    };

    std::process::exit(exit_code);
}
