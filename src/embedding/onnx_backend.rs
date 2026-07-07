//! ONNX Runtime embedding backend (`--features onnx`).
//!
//! Implements the same `Embedder` trait as `CodeBertEmbedder` using ONNX Runtime
//! (`ort` 2.0.0-rc.9) instead of candle. Uses an identical bundled tokenizer and
//! the same mean-pooling formula. Accepts the same `codebert-base` weights via a
//! pre-exported `model.onnx` + `model.onnx.data` file pair.
//!
//! ## Numeric parity vs PyTorch reference
//! - Validated: max cosine diff = 2.38e-7 (vs candle CPU 2.68e-6 — ~11× better)
//! - Deterministic: two passes produce bit-identical results
//!
//! ## Model path resolution (priority order)
//! 1. `CLONEHUNTER_ONNX_MODEL` env var (full path to `model.onnx`)
//! 2. `~/.cache/clonehunter/onnx/codebert-base/model.onnx` (default export location)
//!
//! ## Execution provider
//! CPU only — statically linked ORT (no runtime dylib required).
//!
//! ## ONNX export
//! `microsoft/codebert-base @ 3b0952fed…` via `torch.onnx.export` (dynamo),
//! opset 18, external-data format: `model.onnx` (1.3 MB graph) +
//! `model.onnx.data` (476 MB weights) at `~/.cache/clonehunter/onnx/codebert-base/`.

use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::core::config::{CODEBERT_REVISION, EmbedderConfig};
use crate::core::types::{Embedding, SnippetRef};

use super::shared::{CODEBERT_MODEL, bundled_codebert_tokenizer, chunked_embed, tokenize_padded};
use super::{Embedder, EmbeddingError};

// ── OnnxEmbedder ──────────────────────────────────────────────────────────────

/// ONNX Runtime-backed embedder for `microsoft/codebert-base`.
///
/// Uses the same bundled tokenizer and mean-pooling formula as `CodeBertEmbedder`.
/// Session is immutable after construction; `ort` rc.9's `session.run()` takes
/// `&self` so no locking is needed.
pub(crate) struct OnnxEmbedder {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    config: EmbedderConfig,
}

impl OnnxEmbedder {
    /// Create an OnnxEmbedder with the CPU execution provider.
    pub(crate) fn new(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        tracing::debug!(device = ?config.device, "ONNX backend is CPU-only; --device is ignored");
        let model_path = resolve_model_path()?;
        tracing::info!(path = %model_path.display(), "loading ONNX model");

        let session = ort::session::Session::builder()
            .map_err(|e| EmbeddingError::ModelLoad(format!("ort: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbeddingError::ModelLoad(format!("ort load model: {e}")))?;

        let tokenizer = load_tokenizer(config)?;

        Ok(Self {
            session,
            tokenizer,
            config: config.clone(),
        })
    }
}

// SAFETY: the `Embedder` trait only requires `Send` (the boxed embedder is moved into the
// single-threaded pipeline; it is never shared across threads, so `Sync` is not needed).
// `Session` is immutable after construction and `session.run()` takes `&self`, so moving the
// embedder between threads is sound.
unsafe impl Send for OnnxEmbedder {}

impl Embedder for OnnxEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        chunked_embed(snippets, self.config.batch_size, |texts| {
            embed_batch_ort(&self.session, &self.tokenizer, texts)
        })
    }
}

// ── Model path resolution ─────────────────────────────────────────────────────

fn resolve_model_path() -> Result<PathBuf, EmbeddingError> {
    if let Ok(p) = std::env::var("CLONEHUNTER_ONNX_MODEL") {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Ok(path);
        }
        return Err(EmbeddingError::ModelLoad(format!(
            "CLONEHUNTER_ONNX_MODEL={p} does not exist"
        )));
    }

    let default = default_onnx_model_path();
    if default.exists() {
        return Ok(default);
    }

    Err(EmbeddingError::ModelLoad(format!(
        "ONNX model not found at {p}. \
         Export it with torch.onnx.export or set CLONEHUNTER_ONNX_MODEL. \
         See src/embedding/onnx_backend.rs for instructions.",
        p = default.display()
    )))
}

pub(crate) fn default_onnx_model_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/clonehunter/onnx/codebert-base/model.onnx")
}

// ── Tokenizer loading ─────────────────────────────────────────────────────────

fn load_tokenizer(config: &EmbedderConfig) -> Result<Tokenizer, EmbeddingError> {
    if config.model_name != CODEBERT_MODEL || config.revision != CODEBERT_REVISION {
        return Err(EmbeddingError::Tokenizer(
            "OnnxEmbedder only supports microsoft/codebert-base @ pinned revision; \
             other models require a separate ONNX export"
                .into(),
        ));
    }
    bundled_codebert_tokenizer(config.max_length)
}

// ── Batch inference + mean pooling ────────────────────────────────────────────

/// Run one batch through the ORT session and return mean-pooled F32 embeddings.
///
/// Tokenizes with the bundled tokenizer, creates i64 tensors (no ndarray dependency —
/// uses raw `([batch, seq], &[i64])` tuples), runs the session, then mean-pools
/// `last_hidden_state` with the attention mask.
///
/// Mean-pool formula (same as `CodeBertEmbedder`):
/// `pooled[d] = sum_over_non_pad_tokens(hidden[t][d]) / max(non_pad_count, 1)`
fn embed_batch_ort(
    session: &ort::session::Session,
    tokenizer: &Tokenizer,
    texts: &[&str],
) -> Result<Vec<Embedding>, EmbeddingError> {
    let (batch, max_len, ids_u32, mask_u32) = tokenize_padded(tokenizer, texts)?;
    // ORT needs i64 tensors. token_type_ids = all zeros (RoBERTa).
    let ids_flat: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
    let mask_flat: Vec<i64> = mask_u32.iter().map(|&x| x as i64).collect();
    let type_ids_flat = vec![0i64; batch * max_len];

    // Build tensors from (shape, &[T]) — no ndarray version conflict
    let shape = [batch, max_len];
    let ids_tensor = ort::value::Tensor::<i64>::from_array((shape, ids_flat.as_slice()))
        .map_err(|e| EmbeddingError::Inference(format!("input_ids tensor: {e}")))?;
    let mask_tensor = ort::value::Tensor::<i64>::from_array((shape, mask_flat.as_slice()))
        .map_err(|e| EmbeddingError::Inference(format!("attention_mask tensor: {e}")))?;
    let type_ids_tensor = ort::value::Tensor::<i64>::from_array((shape, type_ids_flat.as_slice()))
        .map_err(|e| EmbeddingError::Inference(format!("token_type_ids tensor: {e}")))?;

    // Run session (rc.9 takes &self — no mutex needed)
    let outputs = session
        .run(
            ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_ids_tensor,
            ]
            .map_err(|e| EmbeddingError::Inference(format!("inputs! build: {e}")))?,
        )
        .map_err(|e| EmbeddingError::Inference(format!("ort run: {e}")))?;

    // Extract last_hidden_state — fail gracefully if a re-exported model names it differently.
    let lhs = outputs.get("last_hidden_state").ok_or_else(|| {
        EmbeddingError::Inference(
            "ONNX model has no 'last_hidden_state' output (re-export with that output name)".into(),
        )
    })?;
    let (lhs_shape, lhs_data) = lhs
        .try_extract_raw_tensor::<f32>()
        .map_err(|e| EmbeddingError::Inference(format!("extract last_hidden_state: {e}")))?;

    // Shape: [batch, seq, hidden]
    if lhs_shape.len() != 3 {
        return Err(EmbeddingError::Inference(format!(
            "expected 3-D last_hidden_state, got rank {}",
            lhs_shape.len()
        )));
    }
    let (b, s, h) = (
        lhs_shape[0] as usize,
        lhs_shape[1] as usize,
        lhs_shape[2] as usize,
    );
    if b != batch {
        return Err(EmbeddingError::Inference(format!(
            "batch mismatch: expected {batch}, got {b}"
        )));
    }
    if s != max_len {
        return Err(EmbeddingError::Inference(format!(
            "sequence length mismatch: expected {max_len}, got {s}"
        )));
    }

    // Mean-pool: sum(hidden * mask) / clamp(mask.sum, 1)
    let mut embeddings = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut pooled = vec![0.0f32; h];
        let mut count = 0.0f32;
        for col in 0..s {
            let m = mask_flat[row * max_len + col] as f32;
            if m > 0.0 {
                let offset = (row * s + col) * h;
                for d in 0..h {
                    pooled[d] += lhs_data[offset + d] * m;
                }
                count += m;
            }
        }
        let scale = count.max(1.0).recip();
        for v in &mut pooled {
            *v *= scale;
        }
        embeddings.push(Embedding { vector: pooled });
    }

    Ok(embeddings)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_path_ends_with_model_onnx() {
        assert_eq!(default_onnx_model_path().file_name().unwrap(), "model.onnx");
    }

    /// End-to-end integration: load ONNX session → embed snippets → validate cosine + det.
    ///
    /// Requires ONNX model at default path or CLONEHUNTER_ONNX_MODEL env var.
    /// Run: `cargo test --features onnx -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn onnx_embed_produces_vectors() {
        use crate::core::config::{EmbedderConfig, EmbedderName};
        use crate::core::types::SnippetRef;
        use crate::test_support::make_snippet;

        let config = EmbedderConfig {
            name: EmbedderName::Onnx,
            ..EmbedderConfig::default()
        };
        let embedder = OnnxEmbedder::new(&config).expect("OnnxEmbedder::new should succeed");

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
        eprintln!("cosine(foo,bar) = {cosine:.6}  (expect > 0.90)");
        assert!(
            cosine > 0.90,
            "similar snippets should have high cosine: {cosine}"
        );

        let embs2 = embedder.embed(&snippets).expect("second embed");
        assert_eq!(embs[0].vector, embs2[0].vector, "must be deterministic");
        eprintln!("OnnxEmbedder integration PASSED: dim=768 cosine={cosine:.6}");
    }
}
