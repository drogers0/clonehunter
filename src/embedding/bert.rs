/// Candle BERT embedder for the `faster` preset (MiniLM-L6 and BERT-family models).
///
/// Uses `candle_transformers::models::bert::BertModel` — distinct from the
/// `XLMRobertaModel` used by the `codebert` preset.  BERT and RoBERTa differ in
/// position-ID handling and tokenizer padding conventions; this module uses
/// BERT-correct settings (pad_id=0, "[PAD]", standard position offsets).
///
/// Prototype-grade: no GPU fallback, no degradation tracking, no bundled tokenizer.
/// The tokenizer.json is downloaded from HF Hub on first use (cached locally).
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::core::config::EmbedderConfig;
use crate::core::types::{Embedding, SnippetRef};

use super::{Embedder, EmbeddingError};
use crate::embedding::codebert::{download_file, resolve_device};

// ── BertEmbedder ─────────────────────────────────────────────────────────────

pub(crate) struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    config: EmbedderConfig,
    #[allow(dead_code)] // accessed via dim() — same pattern as CodeBertEmbedder
    hidden_size: usize,
}

impl BertEmbedder {
    /// Load model and tokenizer eagerly. Falls back to CPU on device error (same policy as
    /// `CodeBertEmbedder`, but without full degradation tracking — prototype quality).
    pub(crate) fn new(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        let (device, _deg) = resolve_device(config.device);
        let tokenizer = load_tokenizer(config)?;
        let (model, hidden_size, device) = load_model(config, device)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            config: config.clone(),
            hidden_size,
        })
    }
}

impl Embedder for BertEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        if snippets.is_empty() {
            return Ok(vec![]);
        }
        let mut result = Vec::with_capacity(snippets.len());
        for chunk in snippets.chunks(self.config.batch_size) {
            let texts: Vec<&str> = chunk.iter().map(|s| s.text.as_str()).collect();
            let batch_embs = embed_batch(&self.model, &self.tokenizer, &self.device, &texts)?;
            result.extend(batch_embs);
        }
        Ok(result)
    }

    fn dim(&self) -> usize {
        self.hidden_size
    }
}

// ── Internal loading helpers ──────────────────────────────────────────────────

/// Download and configure tokenizer for a BERT-family model.
///
/// Unlike CodeBERT (which uses a bundled tokenizer.json), BERT models published on
/// HuggingFace include `tokenizer.json` in the repo — we download it via hf-hub.
/// BERT tokenizers use pad_id=0 / "[PAD]" (not RoBERTa's pad_id=1 / "<pad>").
fn load_tokenizer(config: &EmbedderConfig) -> Result<Tokenizer, EmbeddingError> {
    let path = download_file(&config.model_name, &config.revision, "tokenizer.json")?;
    let mut tokenizer = Tokenizer::from_file(&path)
        .map_err(|e| EmbeddingError::Tokenizer(format!("tokenizer from file: {e}")))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id: 0,
        pad_token: "[PAD]".into(),
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

/// Download config.json + weights and construct BertModel.
///
/// Tries model.safetensors first (faster mmap), falls back to pytorch_model.bin.
/// Falls back to CPU if model load fails on non-CPU device.
fn load_model(
    config: &EmbedderConfig,
    device: Device,
) -> Result<(BertModel, usize, Device), EmbeddingError> {
    let config_path = download_file(&config.model_name, &config.revision, "config.json")?;
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| EmbeddingError::ModelLoad(format!("read config.json: {e}")))?;
    let bert_config: BertConfig = serde_json::from_str(&config_str)
        .map_err(|e| EmbeddingError::ModelLoad(format!("parse config.json: {e}")))?;
    let hidden_size = bert_config.hidden_size;

    match try_load_weights(config, &device, &bert_config) {
        Ok(model) => Ok((model, hidden_size, device)),
        Err(e) if !matches!(device, Device::Cpu) => {
            // CPU fallback — matches CodeBertEmbedder's DD4 policy
            let model = try_load_weights(config, &Device::Cpu, &bert_config)?;
            let _ = e; // ignore original error after successful CPU retry
            Ok((model, hidden_size, Device::Cpu))
        }
        Err(e) => Err(e),
    }
}

fn try_load_weights(
    config: &EmbedderConfig,
    device: &Device,
    bert_config: &BertConfig,
) -> Result<BertModel, EmbeddingError> {
    let vb = if let Ok(path) =
        download_file(&config.model_name, &config.revision, "model.safetensors")
    {
        // Safety: mmap of a trusted local HF cache file (same semantics as CodeBertEmbedder)
        unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
            .map_err(|e| EmbeddingError::ModelLoad(format!("safetensors load: {e}")))?
    } else {
        let path = download_file(&config.model_name, &config.revision, "pytorch_model.bin")
            .map_err(|_| {
                EmbeddingError::ModelLoad(format!(
                    "neither model.safetensors nor pytorch_model.bin found for {}/{}",
                    config.model_name, config.revision
                ))
            })?;
        VarBuilder::from_pth(&path, DType::F32, device)
            .map_err(|e| EmbeddingError::ModelLoad(format!("pth load: {e}")))?
    };

    // BertModel::load tries direct key paths first ("embeddings.*", "encoder.*"),
    // then falls back to "{model_type}.*" prefix (e.g. "bert.embeddings.*").
    // sentence-transformers models use the "bert.*" prefix, which matches the fallback. ✓
    BertModel::load(vb, bert_config)
        .map_err(|e| EmbeddingError::ModelLoad(format!("BertModel::load: {e}")))
}

// ── Batch inference ───────────────────────────────────────────────────────────

/// Embed a batch of texts using BertModel with mean pooling.
///
/// Mean pooling formula (same as CodeBertEmbedder, validated against HF sentence-transformers):
/// `pooled = sum(hidden * mask[:,:,None]) / clamp(sum(mask), 1, MAX)`
fn embed_batch(
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
    texts: &[&str],
) -> Result<Vec<Embedding>, EmbeddingError> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), /* add_special_tokens */ true)
        .map_err(|e| EmbeddingError::Inference(format!("tokenize: {e}")))?;

    let batch = texts.len();
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    let ids_flat: Vec<u32> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().copied())
        .collect();
    let mask_flat: Vec<u32> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().copied())
        .collect();
    // Single-sequence encoding: all token_type_ids = 0
    let type_ids_flat: Vec<u32> = vec![0u32; batch * max_len];

    let input_ids = Tensor::from_vec(ids_flat, (batch, max_len), device)
        .map_err(|e| EmbeddingError::Inference(format!("input_ids tensor: {e}")))?;
    let attention_mask = Tensor::from_vec(mask_flat, (batch, max_len), device)
        .map_err(|e| EmbeddingError::Inference(format!("attention_mask tensor: {e}")))?;
    let token_type_ids = Tensor::from_vec(type_ids_flat, (batch, max_len), device)
        .map_err(|e| EmbeddingError::Inference(format!("token_type_ids tensor: {e}")))?;

    // Forward pass → [batch, seq, hidden]
    // BertModel::forward converts attention_mask to additive bias internally (HF convention).
    // The original `attention_mask` (U32 0/1) is untouched and used for mean-pooling below.
    let sequence_output = model
        .forward(&input_ids, &token_type_ids, Some(&attention_mask))
        .map_err(|e| EmbeddingError::Inference(format!("forward pass: {e}")))?;

    // Mean-pool with attention mask (spike-validated formula from CodeBertEmbedder)
    let mask_f32 = attention_mask
        .to_dtype(DType::F32)
        .map_err(|e| EmbeddingError::Inference(format!("mask to f32: {e}")))?
        .unsqueeze(D::Minus1) // [batch, seq, 1]
        .map_err(|e| EmbeddingError::Inference(format!("unsqueeze: {e}")))?;
    let masked = sequence_output
        .broadcast_mul(&mask_f32)
        .map_err(|e| EmbeddingError::Inference(format!("broadcast_mul: {e}")))?;
    let summed = masked
        .sum(1) // [batch, hidden]
        .map_err(|e| EmbeddingError::Inference(format!("sum: {e}")))?;
    let counts = mask_f32
        .sum(1) // [batch, 1]
        .map_err(|e| EmbeddingError::Inference(format!("mask sum: {e}")))?
        .clamp(1f32, f32::MAX)
        .map_err(|e| EmbeddingError::Inference(format!("clamp: {e}")))?;
    let pooled = summed
        .broadcast_div(&counts)
        .map_err(|e| EmbeddingError::Inference(format!("broadcast_div: {e}")))?;

    let pooled_vecs: Vec<Vec<f32>> = pooled
        .to_vec2::<f32>()
        .map_err(|e| EmbeddingError::Inference(format!("to_vec2: {e}")))?;

    Ok(pooled_vecs
        .into_iter()
        .map(|v| {
            let dim = v.len();
            Embedding { vector: v, dim }
        })
        .collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{DeviceName, EmbedderName};

    /// Smoke test: BertEmbedder loads and embeds two structurally similar snippets
    /// with cosine > 0.8. Requires HF Hub access or local cache for
    /// `sentence-transformers/all-MiniLM-L6-v2`.
    ///
    /// Run with: `cargo test -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn minilm_embed_produces_vectors() {
        use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind, SnippetRef};

        let config = EmbedderConfig {
            name: EmbedderName::Faster,
            model_name: "sentence-transformers/all-MiniLM-L6-v2".into(),
            revision: "main".into(),
            max_length: 512,
            batch_size: 32,
            device: DeviceName::Cpu,
        };

        let embedder = BertEmbedder::new(&config).expect("BertEmbedder::new should succeed");
        assert_eq!(embedder.dim(), 384, "all-MiniLM-L6-v2 hidden_size is 384");

        let make_snippet = |text: &str| -> SnippetRef {
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

        let s1 = make_snippet("def foo(x): return x + 1");
        let s2 = make_snippet("def bar(y): return y + 1");
        let snippets: Vec<&SnippetRef> = vec![&s1, &s2];

        let embeddings = embedder.embed(&snippets).expect("embed should succeed");
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].dim, 384);

        let norm0: f32 = embeddings[0]
            .vector
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        let norm1: f32 = embeddings[1]
            .vector
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(norm0 > 0.0, "embedding 0 should be non-zero");
        assert!(norm1 > 0.0, "embedding 1 should be non-zero");

        let dot: f32 = embeddings[0]
            .vector
            .iter()
            .zip(&embeddings[1].vector)
            .map(|(a, b)| a * b)
            .sum();
        let cosine = dot / (norm0 * norm1);
        eprintln!("minilm cosine(foo(x)=x+1, bar(y)=y+1) = {cosine:.6}  (expect > 0.8)");
        assert!(
            cosine > 0.8,
            "similar functions should have high cosine: {cosine}"
        );

        // Determinism
        let emb2 = embedder.embed(&snippets).expect("second embed");
        assert_eq!(embeddings[0].vector, emb2[0].vector);
        assert_eq!(embeddings[1].vector, emb2[1].vector);

        eprintln!("minilm test PASSED: hidden_size=384 cosine={cosine:.6}");
    }
}
