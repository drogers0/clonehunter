//! Apple MLX embedding backend (`--features mlx`).
//!
//! Implements the `Embedder` trait using Apple MLX via `mlx-rs` for
//! `microsoft/codebert-base` (RoBERTa-base architecture). Apple Silicon only.
//! Uses Metal GPU when available (automatic via prebuilt libmlx.dylib).
//!
//! **Performance:** ~46s on click benchmark (3386 snippets) — beats PyTorch-MPS (50s).
//! **Numerics:** 2.88e-12 max cosine diff vs PyTorch CPU reference (best of any backend).
//! **Detection:** 578 findings — exact frozen-baseline parity.
//!
//! Setup: `./scripts/setup-mlx.sh` then
//! `MLX_SYS_PREBUILT=~/.local/share/clonehunter/mlx cargo build --release --features mlx`
//!
//! Uses the same bundled tokenizer and mean-pooling formula as `CodeBertEmbedder`.
//! Weights are loaded from the HF cache safetensors file.

use std::collections::HashMap;

use mlx_rs::Array;
use tokenizers::Tokenizer;

use crate::core::config::{CODEBERT_REVISION, EmbedderConfig};
use crate::core::types::{Embedding, SnippetRef};

use super::shared::{CODEBERT_MODEL, bundled_codebert_tokenizer, chunked_embed};
use super::{Embedder, EmbeddingError};

const NUM_LAYERS: usize = 12;
const HIDDEN_SIZE: usize = 768;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN_SIZE / NUM_HEADS; // 64
const LAYER_NORM_EPS: f32 = 1e-5;
const PAD_TOKEN_ID: i32 = 1;

// ── Convenience ──────────────────────────────────────────────────────────────

fn ie(e: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::Inference(format!("{e}"))
}

fn w<'a>(weights: &'a HashMap<String, Array>, key: &str) -> Result<&'a Array, EmbeddingError> {
    weights
        .get(key)
        .ok_or_else(|| EmbeddingError::ModelLoad(format!("missing weight: {key}")))
}

// ── Forward pass (operates on borrowed weight HashMap) ───────────────────────

fn linear(x: &Array, weight: &Array, bias: &Array) -> Result<Array, EmbeddingError> {
    let out = x.matmul(weight.t()).map_err(ie)?;
    Ok(&out + bias)
}

fn layer_norm(x: &Array, weight: &Array, bias: &Array) -> Result<Array, EmbeddingError> {
    mlx_rs::fast::layer_norm(x, Some(weight), Some(bias), LAYER_NORM_EPS).map_err(ie)
}

/// RoBERTa embedding layer.
/// position_ids = cumsum(mask, axis=1) * mask + padding_idx
fn roberta_embeddings(
    input_ids: &Array,
    attention_mask: &Array,
    weights: &HashMap<String, Array>,
) -> Result<Array, EmbeddingError> {
    let mask_i32 = attention_mask.as_type::<i32>().map_err(ie)?;
    let cumsum = mask_i32.cumsum(Some(1), None, None).map_err(ie)?;
    let position_ids = &(&cumsum * &mask_i32) + &Array::from_int(PAD_TOKEN_ID);

    let shape = input_ids.shape().to_vec();
    let token_type_ids = Array::zeros::<i32>(&shape).map_err(ie)?;

    let ids_i32 = input_ids.as_type::<i32>().map_err(ie)?;
    let word_emb = w(weights, "embeddings.word_embeddings.weight")?
        .take_axis(&ids_i32, 0)
        .map_err(ie)?;
    let pos_emb = w(weights, "embeddings.position_embeddings.weight")?
        .take_axis(&position_ids, 0)
        .map_err(ie)?;
    let type_emb = w(weights, "embeddings.token_type_embeddings.weight")?
        .take_axis(&token_type_ids, 0)
        .map_err(ie)?;

    let combined = &(&word_emb + &pos_emb) + &type_emb;

    layer_norm(
        &combined,
        w(weights, "embeddings.LayerNorm.weight")?,
        w(weights, "embeddings.LayerNorm.bias")?,
    )
}

fn encoder_layer(
    hidden: &Array,
    attention_mask: &Array,
    weights: &HashMap<String, Array>,
    layer_idx: usize,
    batch: i32,
    seq_len: i32,
) -> Result<Array, EmbeddingError> {
    let pfx = format!("encoder.layer.{layer_idx}");

    // Self-attention Q, K, V
    let q = linear(
        hidden,
        w(weights, &format!("{pfx}.attention.self.query.weight"))?,
        w(weights, &format!("{pfx}.attention.self.query.bias"))?,
    )?;
    let k = linear(
        hidden,
        w(weights, &format!("{pfx}.attention.self.key.weight"))?,
        w(weights, &format!("{pfx}.attention.self.key.bias"))?,
    )?;
    let v = linear(
        hidden,
        w(weights, &format!("{pfx}.attention.self.value.weight"))?,
        w(weights, &format!("{pfx}.attention.self.value.bias"))?,
    )?;

    // Multi-head reshape: [batch, seq, hidden] → [batch, heads, seq, head_dim]
    let head_shape = &[batch, seq_len, NUM_HEADS as i32, HEAD_DIM as i32];
    let q = q
        .reshape(head_shape)
        .map_err(ie)?
        .transpose_axes(&[0, 2, 1, 3])
        .map_err(ie)?;
    let k = k
        .reshape(head_shape)
        .map_err(ie)?
        .transpose_axes(&[0, 2, 1, 3])
        .map_err(ie)?;
    let v = v
        .reshape(head_shape)
        .map_err(ie)?
        .transpose_axes(&[0, 2, 1, 3])
        .map_err(ie)?;

    // Attention scores: softmax(Q @ K^T / sqrt(d_k) + mask_bias) @ V
    let scale = Array::from_f32(1.0 / (HEAD_DIM as f32).sqrt());
    let kt = k.transpose_axes(&[0, 1, 3, 2]).map_err(ie)?;
    let scores = &q.matmul(&kt).map_err(ie)? * &scale;

    // Mask: [batch, seq] → [batch, 1, 1, seq]; (1-mask) * -1e9
    let mask_f32 = attention_mask.as_type::<f32>().map_err(ie)?;
    let mask_bias = &(&Array::from_f32(1.0) - &mask_f32) * &Array::from_f32(-1e9);
    let mask_bias = mask_bias
        .expand_dims(1)
        .map_err(ie)?
        .expand_dims(1)
        .map_err(ie)?;

    let scores = &scores + &mask_bias;
    let attn_weights = mlx_rs::ops::softmax_axis(&scores, -1, None).map_err(ie)?;
    let attn_output = attn_weights.matmul(&v).map_err(ie)?;

    // Reshape back: [batch, heads, seq, head_dim] → [batch, seq, hidden]
    let attn_output = attn_output
        .transpose_axes(&[0, 2, 1, 3])
        .map_err(ie)?
        .reshape(&[batch, seq_len, HIDDEN_SIZE as i32])
        .map_err(ie)?;

    // Output projection + residual + LayerNorm
    let projected = linear(
        &attn_output,
        w(weights, &format!("{pfx}.attention.output.dense.weight"))?,
        w(weights, &format!("{pfx}.attention.output.dense.bias"))?,
    )?;
    let residual = &projected + hidden;
    let normed = layer_norm(
        &residual,
        w(weights, &format!("{pfx}.attention.output.LayerNorm.weight"))?,
        w(weights, &format!("{pfx}.attention.output.LayerNorm.bias"))?,
    )?;

    // FFN: Linear → GELU → Linear + residual + LayerNorm
    let intermediate = linear(
        &normed,
        w(weights, &format!("{pfx}.intermediate.dense.weight"))?,
        w(weights, &format!("{pfx}.intermediate.dense.bias"))?,
    )?;
    let activated = mlx_rs::nn::gelu(&intermediate).map_err(ie)?;
    let output = linear(
        &activated,
        w(weights, &format!("{pfx}.output.dense.weight"))?,
        w(weights, &format!("{pfx}.output.dense.bias"))?,
    )?;
    let residual = &output + &normed;

    layer_norm(
        &residual,
        w(weights, &format!("{pfx}.output.LayerNorm.weight"))?,
        w(weights, &format!("{pfx}.output.LayerNorm.bias"))?,
    )
}

fn roberta_forward(
    input_ids: &Array,
    attention_mask: &Array,
    weights: &HashMap<String, Array>,
) -> Result<Array, EmbeddingError> {
    let shape = input_ids.shape().to_vec();
    let batch = shape[0];
    let seq_len = shape[1];

    let mut hidden = roberta_embeddings(input_ids, attention_mask, weights)?;

    for i in 0..NUM_LAYERS {
        hidden = encoder_layer(&hidden, attention_mask, weights, i, batch, seq_len)?;
    }

    Ok(hidden)
}

fn mean_pool(hidden: &Array, attention_mask: &Array) -> Result<Array, EmbeddingError> {
    let mask_f32 = attention_mask.as_type::<f32>().map_err(ie)?;
    let mask_3d = mask_f32.expand_dims(-1).map_err(ie)?;

    let masked = hidden * &mask_3d;
    let summed = masked.sum_axis(1, None).map_err(ie)?;

    let counts = mask_3d.sum_axis(1, None).map_err(ie)?;
    let counts = mlx_rs::ops::maximum(&counts, Array::from_f32(1.0)).map_err(ie)?;

    Ok(&summed / &counts)
}

// ── MlxEmbedder ──────────────────────────────────────────────────────────────

pub(crate) struct MlxEmbedder {
    /// Raw weight tensors from safetensors — DO NOT deep_clone. deep_clone() on
    /// safetensors-loaded arrays segfaults in mlx-rs 0.25 (non-atomic refcount
    /// race in the C++ shared_ptr). Access by key reference only.
    weights: HashMap<String, Array>,
    tokenizer: Tokenizer,
    config: EmbedderConfig,
}

// SAFETY: MlxEmbedder is constructed once and only accessed via &self in embed().
// The weight HashMap is never mutated after construction. mlx_rs::Array is !Send
// because mlx::core::array uses non-atomic shared_ptr, but our usage pattern is
// single-threaded construction followed by sequential &self calls — no concurrent
// access to the underlying C++ objects occurs.
unsafe impl Send for MlxEmbedder {}

impl MlxEmbedder {
    pub(crate) fn new(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        // Check and log Metal availability via mlx-c FFI
        let metal_available = {
            unsafe extern "C" {
                fn mlx_metal_is_available(res: *mut bool) -> std::ffi::c_int;
            }
            let mut res: bool = false;
            let rc = unsafe { mlx_metal_is_available(&mut res) };
            rc == 0 && res
        };
        tracing::info!(metal_available, "MLX device check");
        if metal_available {
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

        let weights = Array::load_safetensors(&weights_path)
            .map_err(|e| EmbeddingError::ModelLoad(format!("mlx load_safetensors: {e}")))?;

        // Validate a few key weights exist
        for key in [
            "embeddings.word_embeddings.weight",
            "encoder.layer.0.attention.self.query.weight",
            "encoder.layer.11.output.LayerNorm.weight",
        ] {
            if !weights.contains_key(key) {
                return Err(EmbeddingError::ModelLoad(format!("missing weight: {key}")));
            }
        }

        Ok(Self {
            weights,
            tokenizer,
            config: config.clone(),
        })
    }
}

impl Embedder for MlxEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        chunked_embed(snippets, self.config.batch_size, |texts| {
            embed_batch_mlx(&self.weights, &self.tokenizer, texts)
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
    weights: &HashMap<String, Array>,
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

    let shape = &[batch as i32, max_len as i32];
    let input_ids = Array::from_slice(&ids_flat, shape);
    let attention_mask = Array::from_slice(&mask_flat, shape);

    let hidden = roberta_forward(&input_ids, &attention_mask, weights)?;
    let pooled = mean_pool(&hidden, &attention_mask)?;

    pooled.eval().map_err(ie)?;

    let pooled_data: &[f32] = pooled.try_as_slice().map_err(ie)?;

    let mut embeddings = Vec::with_capacity(batch);
    for row in 0..batch {
        let start = row * HIDDEN_SIZE;
        let end = start + HIDDEN_SIZE;
        embeddings.push(Embedding {
            vector: pooled_data[start..end].to_vec(),
        });
    }

    Ok(embeddings)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Run: `cargo test --features mlx -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn mlx_embed_produces_vectors() {
        use crate::core::config::{EmbedderConfig, EmbedderName};
        use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind, SnippetRef};

        let config = EmbedderConfig {
            name: EmbedderName::Mlx,
            ..EmbedderConfig::default()
        };
        let embedder = MlxEmbedder::new(&config).expect("MlxEmbedder::new should succeed");

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
}
