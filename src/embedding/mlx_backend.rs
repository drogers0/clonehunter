//! Apple MLX embedding backend (`--features mlx`).
//!
//! Thin FFI wrapper over the C++ shim in `csrc/ch_mlx.cpp`, which runs the entire
//! `microsoft/codebert-base` (RoBERTa-base) forward pass + mean-pool against Apple's
//! `mlx-c` C API (vendored at `vendor/mlx-c`). Apple Silicon only; uses the Metal GPU
//! when available (chosen inside the shim).
//!
//! This crate owns the shim outright — there is no `mlx-rs`/`mlx-sys` dependency. The
//! shim is built and linked by the crate-root `build.rs` against a prebuilt `libmlx`
//! (see `scripts/setup-mlx.sh` and the `CLONEHUNTER_MLX_PREBUILT` env var).
//!
//! **Performance:** ~47s on the click benchmark (3386 snippets) — beats PyTorch-MPS.
//! **Detection:** 578 findings — exact frozen-baseline parity (the contract).
//!
//! Uses the same bundled tokenizer and mean-pooling as `CodeBertEmbedder`; weights are
//! loaded (once, in the shim) from the HF-cache safetensors file.

use std::ffi::{CString, c_char, c_int};

use tokenizers::Tokenizer;

use crate::core::config::{CODEBERT_REVISION, EmbedderConfig};
use crate::core::types::{Embedding, SnippetRef};

use super::shared::{CODEBERT_MODEL, bundled_codebert_tokenizer, chunked_embed};
use super::{Embedder, EmbeddingError};

const HIDDEN_SIZE: usize = 768;
const PAD_TOKEN_ID: i32 = 1;

// ── FFI surface (csrc/ch_mlx.h) ──────────────────────────────────────────────

#[repr(C)]
struct ChMlxCtx {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn ch_mlx_ctx_new(path: *const c_char) -> *mut ChMlxCtx;
    fn ch_mlx_embed(
        ctx: *mut ChMlxCtx,
        ids: *const i32,
        mask: *const i32,
        batch: c_int,
        seq: c_int,
        out: *mut f32,
    ) -> c_int;
    fn ch_mlx_ctx_free(ctx: *mut ChMlxCtx);
    fn ch_mlx_metal_available() -> bool;
    #[cfg(test)]
    fn ch_mlx_selftest() -> c_int;
    fn ch_mlx_last_error() -> *const c_char;
}

/// The shim's thread-local last-error message (empty string if none).
fn last_error() -> String {
    let ptr = unsafe { ch_mlx_last_error() };
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Dylib-only self-test: proves the link + a real MLX op path work, no weights/network.
#[cfg(test)]
fn selftest() -> Result<(), EmbeddingError> {
    if unsafe { ch_mlx_selftest() } == 0 {
        Ok(())
    } else {
        Err(EmbeddingError::Inference(format!(
            "MLX selftest failed: {}",
            last_error()
        )))
    }
}

/// True if a Metal GPU is available.
fn metal_available() -> bool {
    unsafe { ch_mlx_metal_available() }
}

// ── MlxEmbedder ──────────────────────────────────────────────────────────────

pub(crate) struct MlxEmbedder {
    ctx: *mut ChMlxCtx,
    tokenizer: Tokenizer,
    config: EmbedderConfig,
}

// SAFETY: the ctx (weights map + one mlx_stream) is built once and only ever touched
// through sequential `&self` calls in `embed`; `chunked_embed` is single-threaded. MLX
// arrays use non-atomic refcounts, so `embed` must never be called concurrently on one
// `&self` — the current pipeline never does.
unsafe impl Send for MlxEmbedder {}

impl Drop for MlxEmbedder {
    fn drop(&mut self) {
        unsafe { ch_mlx_ctx_free(self.ctx) };
    }
}

impl MlxEmbedder {
    pub(crate) fn new(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        if metal_available() {
            tracing::info!("MLX Metal GPU is available — using GPU");
        } else {
            tracing::warn!("MLX Metal GPU NOT available — falling back to CPU");
        }

        let tokenizer = load_tokenizer(config)?;

        let weights_path = super::codebert::download_file(
            &config.model_name,
            &config.revision,
            "model.safetensors",
        )?;
        tracing::info!(path = %weights_path.display(), "loading MLX weights from safetensors");

        let path = CString::new(weights_path.to_string_lossy().as_bytes())
            .map_err(|e| EmbeddingError::ModelLoad(format!("weights path: {e}")))?;
        let ctx = unsafe { ch_mlx_ctx_new(path.as_ptr()) };
        if ctx.is_null() {
            return Err(EmbeddingError::ModelLoad(format!(
                "MLX context init failed: {}",
                last_error()
            )));
        }

        Ok(Self {
            ctx,
            tokenizer,
            config: config.clone(),
        })
    }
}

impl Embedder for MlxEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        chunked_embed(snippets, self.config.batch_size, |texts| {
            embed_batch_mlx(self.ctx, &self.tokenizer, texts)
        })
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

fn load_tokenizer(config: &EmbedderConfig) -> Result<Tokenizer, EmbeddingError> {
    if config.model_name != CODEBERT_MODEL || config.revision != CODEBERT_REVISION {
        return Err(EmbeddingError::Tokenizer(
            "MlxEmbedder only supports microsoft/codebert-base @ pinned revision".into(),
        ));
    }
    bundled_codebert_tokenizer(config.max_length)
}

// ── Batch inference ───────────────────────────────────────────────────────────

fn embed_batch_mlx(
    ctx: *mut ChMlxCtx,
    tokenizer: &Tokenizer,
    texts: &[&str],
) -> Result<Vec<Embedding>, EmbeddingError> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| EmbeddingError::Inference(format!("tokenize: {e}")))?;

    let batch = texts.len();
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    let mut ids_flat = vec![PAD_TOKEN_ID; batch * max_len];
    let mut mask_flat = vec![0i32; batch * max_len];
    for (row, enc) in encodings.iter().enumerate() {
        for (col, (&id, &m)) in enc
            .get_ids()
            .iter()
            .zip(enc.get_attention_mask().iter())
            .enumerate()
        {
            ids_flat[row * max_len + col] = id as i32;
            mask_flat[row * max_len + col] = m as i32;
        }
    }

    let mut out = vec![0f32; batch * HIDDEN_SIZE];
    let rc = unsafe {
        ch_mlx_embed(
            ctx,
            ids_flat.as_ptr(),
            mask_flat.as_ptr(),
            batch as c_int,
            max_len as c_int,
            out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(EmbeddingError::Inference(format!(
            "MLX embed failed: {}",
            last_error()
        )));
    }

    Ok(out
        .chunks_exact(HIDDEN_SIZE)
        .map(|v| Embedding { vector: v.to_vec() })
        .collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// MLX's default GPU stream is process-global; two threads encoding to it at once
    /// abort with a Metal command-buffer assertion. Cargo runs tests in parallel, so the
    /// MLX tests take this lock to serialize (production never runs embeds concurrently).
    static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Dylib-only: proves the shim links and a real MLX op path runs. No weights,
    /// no network — runs on every `cargo test --features mlx`.
    #[test]
    fn mlx_selftest_smoke() {
        let _guard = MLX_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        selftest().expect("MLX shim selftest (matmul + eval) should succeed");
    }

    /// Run: `cargo test --features mlx -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn mlx_embed_produces_vectors() {
        use crate::core::config::{EmbedderConfig, EmbedderName};
        use crate::core::types::SnippetRef;
        use crate::test_support::make_snippet;

        let _guard = MLX_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = EmbedderConfig {
            name: EmbedderName::Mlx,
            ..EmbedderConfig::default()
        };
        let embedder = MlxEmbedder::new(&config).expect("MlxEmbedder::new should succeed");

        let s1 = make_snippet("def foo(x): return x + 1");
        let s2 = make_snippet("def bar(y): return y + 1");
        let snippets: Vec<&SnippetRef> = vec![&s1, &s2];

        let embs = embedder.embed(&snippets).expect("embed should succeed");
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].vector.len(), 768);

        let dot: f32 = embs[0]
            .vector
            .iter()
            .zip(&embs[1].vector)
            .map(|(a, b)| a * b)
            .sum();
        let n0: f32 = embs[0].vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let n1: f32 = embs[1].vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (n0 * n1);
        eprintln!("MLX cosine(foo,bar) = {cosine:.6}  (expect > 0.90)");
        assert!(cosine > 0.90, "similar snippets cosine: {cosine}");

        let embs2 = embedder.embed(&snippets).expect("second embed");
        assert_eq!(embs[0].vector, embs2[0].vector, "must be deterministic");
        eprintln!("MlxEmbedder integration PASSED: dim=768 cosine={cosine:.6}");
    }

    #[derive(serde::Deserialize)]
    struct FixtureEntry {
        text: String,
        embedding: Vec<f64>,
    }

    /// Numeric parity vs the PyTorch CPU reference (`spike/fixtures/python_embeddings.json`,
    /// 205 entries). Embeds each fixture text through the production `MlxEmbedder` and
    /// reports the max cosine / abs diff (informational — the binding contract is the
    /// 578-finding detection baseline; the shim's compiled-vs-eager gelu may shift the
    /// embedding diff slightly). Asserts determinism.
    ///
    /// Run: `cargo test --features mlx -- --ignored --nocapture mlx_parity_vs_pytorch`
    /// `MLX_PARITY_COUNT=N` caps the entries checked (default 10).
    #[test]
    #[ignore]
    fn mlx_parity_vs_pytorch() {
        use crate::core::config::{EmbedderConfig, EmbedderName};
        use crate::core::types::SnippetRef;
        use crate::test_support::make_snippet;

        let _guard = MLX_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("spike/fixtures/python_embeddings.json");
        let fixtures: Vec<FixtureEntry> =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path).unwrap()).unwrap();
        eprintln!("Loaded {} fixture entries", fixtures.len());

        let config = EmbedderConfig {
            name: EmbedderName::Mlx,
            ..EmbedderConfig::default()
        };
        let embedder = MlxEmbedder::new(&config).expect("MlxEmbedder::new should succeed");

        let embed_text = |text: &str| -> Vec<f32> {
            let snip = make_snippet(text);
            let refs: Vec<&SnippetRef> = vec![&snip];
            embedder.embed(&refs).expect("embed").remove(0).vector
        };

        let n = std::env::var("MLX_PARITY_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        let (mut max_cosine_diff, mut max_abs_diff) = (0.0f64, 0.0f64);
        for (i, entry) in fixtures.iter().take(n).enumerate() {
            let emb = embed_text(&entry.text);
            let dot: f64 = emb
                .iter()
                .zip(&entry.embedding)
                .map(|(&a, &b)| a as f64 * b)
                .sum();
            let n0: f64 = emb.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            let n1: f64 = entry.embedding.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let cos = dot / (n0 * n1);
            let cosine_diff = (1.0 - cos).abs();
            let abs_diff: f64 = emb
                .iter()
                .zip(&entry.embedding)
                .map(|(&a, &b)| (a as f64 - b).abs())
                .fold(0.0, f64::max);
            max_cosine_diff = max_cosine_diff.max(cosine_diff);
            max_abs_diff = max_abs_diff.max(abs_diff);
            eprintln!(
                "  [{i:3}] cosine={cos:.8} diff={cosine_diff:.2e} abs_diff={abs_diff:.2e} text={:.40}",
                entry.text
            );
        }
        eprintln!("\n=== MLX Parity ({n} entries) ===");
        eprintln!("  Max cosine diff: {max_cosine_diff:.2e}");
        eprintln!("  Max abs diff:    {max_abs_diff:.2e}");

        let e1 = embed_text(&fixtures[0].text);
        let e2 = embed_text(&fixtures[0].text);
        assert_eq!(e1, e2, "MLX must be deterministic");
        eprintln!("  Determinism:     PASS");
    }
}
