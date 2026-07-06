//! ONNX Runtime embedding backend (experimental, `--features onnx`).
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
//! ## Execution provider selection
//! - **CPU** (always): `OnnxEmbedder::new(config)`. Uses the dynamic-shape export.
//! - **CoreML** (macOS, `--features onnx-coreml`): `OnnxEmbedder::new_coreml(config)`.
//!   Appends the CoreML EP with the ML Program backend (see `build_coreml_session`)
//!   and loads the fixed-sequence-length export; falls back to CPU if registration
//!   fails. Confirmed working with exact detection parity (578 findings on the
//!   `click` benchmark), but slower than the CPU EP for CodeBERT-size batches —
//!   per-inference CoreML dispatch overhead outweighs the accelerator benefit.
//!
//! ## ONNX export note
//! - **CPU (default):** `microsoft/codebert-base @ 3b0952fed…` via `torch.onnx.export`
//!   (dynamo), opset 18, external-data format: `model.onnx` (1.3 MB graph) +
//!   `model.onnx.data` (476 MB weights) at `~/.cache/clonehunter/onnx/codebert-base/`.
//! - **CoreML:** a *fixed-sequence-length* (seq=256, dynamic batch) re-export at
//!   `~/.cache/clonehunter/onnx/codebert-base-static/model.onnx` (single 473 MB file,
//!   TorchScript exporter). The dynamic-shape graph fails to compile to an ML Program
//!   (MIL error -14); fixing the sequence length resolves it. Masked mean-pooling
//!   makes the fixed-length embeddings identical to the dynamic ones.

use std::path::PathBuf;

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::core::config::{CODEBERT_REVISION, EmbedderConfig};
use crate::core::types::{Embedding, SnippetRef};

use super::{Embedder, EmbeddingError};

/// Bundled tokenizer.json shared with `CodeBertEmbedder`.
const CODEBERT_TOKENIZER_BYTES: &[u8] = include_bytes!("tokenizer.json");

/// Codebert model name string.
const CODEBERT_MODEL: &str = "microsoft/codebert-base";

// ── OnnxEmbedder ──────────────────────────────────────────────────────────────

/// Which execution provider the session was created with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnnxEp {
    Cpu,
    CoreMl,
    Cuda,
}

/// ONNX Runtime-backed embedder for `microsoft/codebert-base`.
///
/// Uses the same bundled tokenizer and mean-pooling formula as `CodeBertEmbedder`.
/// Session is immutable after construction; `ort` rc.9's `session.run()` takes
/// `&self` so no locking is needed.
pub(crate) struct OnnxEmbedder {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    config: EmbedderConfig,
    #[allow(dead_code)] // read in #[ignore] integration tests
    pub(crate) ep: OnnxEp,
}

impl OnnxEmbedder {
    /// Create an OnnxEmbedder with the CPU execution provider.
    pub(crate) fn new(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        Self::new_with_ep(config, EpChoice::Cpu)
    }

    /// Attempt CoreML EP first; fall back to CPU on any failure.
    ///
    /// Requires `--features onnx-coreml`. Without it compiles but always uses CPU.
    pub(crate) fn new_coreml(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        Self::new_with_ep(config, EpChoice::CoreMl)
    }

    /// Attempt CUDA EP first; fall back to CPU on any failure.
    ///
    /// Requires `--features onnx-cuda`. Without it compiles but always uses CPU.
    pub(crate) fn new_cuda(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        Self::new_with_ep(config, EpChoice::Cuda)
    }

    fn new_with_ep(config: &EmbedderConfig, choice: EpChoice) -> Result<Self, EmbeddingError> {
        let prefer_static = matches!(choice, EpChoice::CoreMl);
        let model_path = resolve_model_path(prefer_static)?;
        tracing::info!(path = %model_path.display(), ?choice, "loading ONNX model");

        let (session, ep) = build_session(&model_path, choice)?;
        // The CoreML path loads the fixed-sequence-length export, which requires
        // every input padded to exactly `max_length`. Masked mean-pooling makes
        // this numerically identical to batch-longest padding (pad tokens have
        // attention mask 0 and are excluded from both attention and the pool).
        let fixed_len = if prefer_static {
            Some(config.max_length)
        } else {
            None
        };
        let tokenizer = load_tokenizer(config, fixed_len)?;

        Ok(Self {
            session,
            tokenizer,
            config: config.clone(),
            ep,
        })
    }
}

// SAFETY: ort rc.9 Session is Send + Sync (SharedSessionInner is Send + Sync).
unsafe impl Send for OnnxEmbedder {}
unsafe impl Sync for OnnxEmbedder {}

impl Embedder for OnnxEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        if snippets.is_empty() {
            return Ok(vec![]);
        }
        let mut result = Vec::with_capacity(snippets.len());
        for chunk in snippets.chunks(self.config.batch_size) {
            let texts: Vec<&str> = chunk.iter().map(|s| s.text.as_str()).collect();
            let batch = embed_batch_ort(&self.session, &self.tokenizer, &texts)?;
            result.extend(batch);
        }
        Ok(result)
    }

    fn dim(&self) -> usize {
        768
    }
}

// ── Model path resolution ─────────────────────────────────────────────────────

fn resolve_model_path(prefer_static: bool) -> Result<PathBuf, EmbeddingError> {
    if let Ok(p) = std::env::var("CLONEHUNTER_ONNX_MODEL") {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Ok(path);
        }
        return Err(EmbeddingError::ModelLoad(format!(
            "CLONEHUNTER_ONNX_MODEL={p} does not exist"
        )));
    }

    // CoreML needs the fixed-sequence-length export (dynamic-shape graphs fail to
    // compile to an ML Program). Prefer it when present; fall back to the dynamic
    // model otherwise.
    if prefer_static {
        let static_path = default_onnx_static_model_path();
        if static_path.exists() {
            return Ok(static_path);
        }
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

/// Fixed-sequence-length export used by the CoreML EP (see `export_static.py`).
pub(crate) fn default_onnx_static_model_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/clonehunter/onnx/codebert-base-static/model.onnx")
}

// ── EP selection ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum EpChoice {
    Cpu,
    CoreMl,
    Cuda,
}

// ── Session construction ──────────────────────────────────────────────────────

fn build_session(
    model_path: &std::path::Path,
    choice: EpChoice,
) -> Result<(ort::session::Session, OnnxEp), EmbeddingError> {
    let map_load = |e: ort::Error| EmbeddingError::ModelLoad(format!("ort: {e}"));

    match choice {
        EpChoice::CoreMl => {
            match build_coreml_session(model_path) {
                Ok(s) => {
                    tracing::info!("ONNX Runtime: CoreML EP active");
                    return Ok((s, OnnxEp::CoreMl));
                }
                Err(e) => {
                    tracing::warn!("CoreML EP unavailable ({e}), falling back to CPU");
                }
            }
        }
        EpChoice::Cuda => {
            match build_cuda_session(model_path) {
                Ok(s) => {
                    tracing::info!("ONNX Runtime: CUDA EP active");
                    return Ok((s, OnnxEp::Cuda));
                }
                Err(e) => {
                    tracing::warn!("CUDA EP unavailable ({e}), falling back to CPU");
                }
            }
        }
        EpChoice::Cpu => {}
    }

    let session = ort::session::Session::builder()
        .map_err(map_load)?
        .commit_from_file(model_path)
        .map_err(|e| EmbeddingError::ModelLoad(format!("ort load model: {e}")))?;

    Ok((session, OnnxEp::Cpu))
}

#[cfg(feature = "onnx-coreml")]
fn build_coreml_session(model_path: &std::path::Path) -> ort::Result<ort::session::Session> {
    use ort::AsPointer;

    // ORT 1.20 CoreML flag (coreml_provider_factory.h). rc.9's safe
    // `CoreMLExecutionProvider` wrapper only exposes the legacy NeuralNetwork
    // backend (flags 0x001/0x002/0x004), which cannot run our dynamic-shape
    // transformer graph — CoreML fails at inference with error code -1. We append
    // the EP by hand with COREML_FLAG_CREATE_MLPROGRAM so ORT emits an ML Program
    // model, which supports the ops and dynamic sequence lengths CodeBERT needs.
    // Unsupported nodes still fall back to CPU automatically (ORT EP partitioning).
    const COREML_FLAG_CREATE_MLPROGRAM: u32 = 0x010;

    // Exported by the statically-linked onnxruntime (ort declares the same symbol
    // under `all(not(load-dynamic), coreml)`). Appends the CoreML EP to the given
    // session options; returns a null OrtStatusPtr on success.
    unsafe extern "C" {
        fn OrtSessionOptionsAppendExecutionProvider_CoreML(
            options: *mut ort::sys::OrtSessionOptions,
            flags: u32,
        ) -> ort::sys::OrtStatusPtr;
    }

    let mut builder = ort::session::Session::builder()?;
    // SAFETY: `builder.ptr_mut()` is a valid, non-null OrtSessionOptions pointer for
    // the lifetime of `builder`; the extern only reads it and appends an EP entry.
    let status = unsafe {
        OrtSessionOptionsAppendExecutionProvider_CoreML(
            builder.ptr_mut(),
            COREML_FLAG_CREATE_MLPROGRAM,
        )
    };
    if !status.is_null() {
        return Err(ort::Error::new(
            "CoreML MLProgram EP registration returned an error status",
        ));
    }
    builder.commit_from_file(model_path)
}

#[cfg(not(feature = "onnx-coreml"))]
fn build_coreml_session(
    _model_path: &std::path::Path,
) -> Result<ort::session::Session, ort::Error> {
    Err(ort::Error::new("onnx-coreml feature not compiled"))
}

#[cfg(feature = "onnx-cuda")]
fn build_cuda_session(model_path: &std::path::Path) -> ort::Result<ort::session::Session> {
    let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default().build();
    ort::session::Session::builder()?
        .with_execution_providers([cuda_ep])?
        .commit_from_file(model_path)
}

#[cfg(not(feature = "onnx-cuda"))]
fn build_cuda_session(
    _model_path: &std::path::Path,
) -> Result<ort::session::Session, ort::Error> {
    Err(ort::Error::new("onnx-cuda feature not compiled"))
}

// ── Tokenizer loading ─────────────────────────────────────────────────────────

fn load_tokenizer(
    config: &EmbedderConfig,
    fixed_len: Option<usize>,
) -> Result<Tokenizer, EmbeddingError> {
    if config.model_name != CODEBERT_MODEL || config.revision != CODEBERT_REVISION {
        return Err(EmbeddingError::Tokenizer(
            "OnnxEmbedder only supports microsoft/codebert-base @ pinned revision; \
             other models require a separate ONNX export"
                .into(),
        ));
    }

    let mut tokenizer = Tokenizer::from_bytes(CODEBERT_TOKENIZER_BYTES)
        .map_err(|e| EmbeddingError::Tokenizer(format!("bundled tokenizer: {e}")))?;

    // Fixed padding for the CoreML static-shape model; batch-longest otherwise.
    let strategy = match fixed_len {
        Some(n) => PaddingStrategy::Fixed(n),
        None => PaddingStrategy::BatchLongest,
    };
    tokenizer.with_padding(Some(PaddingParams {
        strategy,
        pad_id: 1,
        pad_token: "<pad>".into(),
        ..Default::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: config.max_length,
            ..Default::default()
        }))
        .map_err(|e| EmbeddingError::Tokenizer(format!("truncation config: {e}")))?;

    Ok(tokenizer)
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
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| EmbeddingError::Inference(format!("tokenize: {e}")))?;

    let batch = texts.len();
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    // Flatten to row-major i64 arrays. token_type_ids = all zeros (RoBERTa).
    let mut ids_flat = vec![0i64; batch * max_len];
    let mut mask_flat = vec![0i64; batch * max_len];
    for (row, enc) in encodings.iter().enumerate() {
        let ids = enc.get_ids();
        let mask = enc.get_attention_mask();
        for (col, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
            ids_flat[row * max_len + col] = id as i64;
            mask_flat[row * max_len + col] = m as i64;
        }
    }
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

    // Extract last_hidden_state — output named "last_hidden_state" or index 0
    let (lhs_shape, lhs_data) = outputs["last_hidden_state"]
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
        embeddings.push(Embedding {
            dim: h,
            vector: pooled,
        });
    }

    Ok(embeddings)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_tokenizer_parses() {
        let tok =
            Tokenizer::from_bytes(CODEBERT_TOKENIZER_BYTES).expect("bundled tokenizer must parse");
        assert!(tok.get_vocab_size(false) > 40_000);
    }

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
        use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind, SnippetRef};

        let config = EmbedderConfig {
            name: EmbedderName::Onnx,
            ..EmbedderConfig::default()
        };
        let embedder = OnnxEmbedder::new(&config).expect("OnnxEmbedder::new should succeed");
        eprintln!("EP: {:?}", embedder.ep);

        let make_snip = |text: &str| -> SnippetRef {
            let file = FileRef {
                path: "a.py".into(),
                content_hash: "h".into(),
                language: Language::Python,
            };
            let func = FunctionRef {
                file,
                qualified_name: "f".into(),
                start_line: 1,
                end_line: 3,
                code: text.into(),
                code_hash: "h".into(),
            };
            SnippetRef {
                kind: SnippetKind::Func,
                function: func,
                start_line: 1,
                end_line: 3,
                text: text.into(),
                display_text: text.into(),
                snippet_hash: crate::io::fingerprints::hash_text(text),
            }
        };

        let s1 = make_snip("def foo(x): return x + 1");
        let s2 = make_snip("def bar(y): return y + 1");
        let snippets: Vec<&SnippetRef> = vec![&s1, &s2];

        let embs = embedder.embed(&snippets).expect("embed should succeed");
        assert_eq!(embs.len(), 2);
        assert_eq!(embs[0].dim, 768);

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
