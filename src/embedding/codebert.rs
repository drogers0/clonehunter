use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config as XLMConfig, XLMRobertaModel};
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::core::config::{CODEBERT_REVISION, DeviceName, EmbedderConfig};
use crate::core::types::{Degradation, DegradationKind, Embedding, SnippetRef};

use super::{Embedder, EmbeddingError};

/// Pre-generated tokenizer for `microsoft/codebert-base` @ CODEBERT_REVISION.
///
/// `microsoft/codebert-base` does not publish `tokenizer.json` on HuggingFace — only
/// `vocab.json` + `merges.txt`. This tokenizer was generated from Python's
/// `AutoTokenizer.from_pretrained()` and spike-validated (100% token ID match vs Python).
///
/// Used ONLY when `config.model_name == "microsoft/codebert-base"` AND
/// `config.revision == CODEBERT_REVISION`. For all other models or revisions,
/// `tokenizer.json` is downloaded via hf-hub so the tokenizer always matches the model.
const CODEBERT_TOKENIZER_BYTES: &[u8] = include_bytes!("tokenizer.json");

/// `microsoft/codebert-base` model name — identifies when to use the bundled tokenizer.
const CODEBERT_MODEL: &str = "microsoft/codebert-base";

// ── Intermediate config deserialization ──────────────────────────────────────────

/// Intermediate deserialization struct that tolerates optional fields in codebert's `config.json`.
/// Converts into `XLMConfig` for model loading. Pattern from Phase-0 spike (candle_embed.rs).
#[derive(Deserialize)]
struct CodeBertJson {
    hidden_size: usize,
    layer_norm_eps: f64,
    attention_probs_dropout_prob: f32,
    hidden_dropout_prob: f32,
    num_attention_heads: usize,
    #[serde(default = "default_pos_emb_type")]
    position_embedding_type: String,
    intermediate_size: usize,
    hidden_act: candle_nn::Activation,
    num_hidden_layers: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    pad_token_id: u32,
}

fn default_pos_emb_type() -> String {
    "absolute".to_string()
}

impl From<CodeBertJson> for XLMConfig {
    fn from(c: CodeBertJson) -> Self {
        XLMConfig {
            hidden_size: c.hidden_size,
            layer_norm_eps: c.layer_norm_eps,
            attention_probs_dropout_prob: c.attention_probs_dropout_prob,
            hidden_dropout_prob: c.hidden_dropout_prob,
            num_attention_heads: c.num_attention_heads,
            position_embedding_type: c.position_embedding_type,
            intermediate_size: c.intermediate_size,
            hidden_act: c.hidden_act,
            num_hidden_layers: c.num_hidden_layers,
            vocab_size: c.vocab_size,
            max_position_embeddings: c.max_position_embeddings,
            type_vocab_size: c.type_vocab_size,
            pad_token_id: c.pad_token_id,
        }
    }
}

// ── Device resolution ─────────────────────────────────────────────────────────

/// Map `DeviceName` → `candle_core::Device` with graceful fallback to CPU (DD4).
///
/// Metal/CUDA availability is determined by fallible `Device::new_metal/new_cuda` constructors,
/// NOT by `candle_core::utils::metal_is_available()` / `cuda_is_available()` which are
/// compile-time feature checks, not runtime GPU probes.
///
/// `Auto` → Metal → CUDA → CPU. Auto→CPU is not recorded as a degradation.
pub(crate) fn resolve_device(requested: DeviceName) -> (Device, Option<Degradation>) {
    match requested {
        DeviceName::Cpu => (Device::Cpu, None),

        DeviceName::Mps => (
            Device::Cpu,
            Some(Degradation {
                kind: DegradationKind::DeviceFallback,
                message:
                    "Metal support not compiled (use --features mlx for GPU), falling back to CPU"
                        .into(),
            }),
        ),

        DeviceName::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(d) => (d, None),
                    Err(e) => (
                        Device::Cpu,
                        Some(Degradation {
                            kind: DegradationKind::DeviceFallback,
                            message: format!("CUDA device unavailable ({e}), falling back to CPU"),
                        }),
                    ),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                (
                    Device::Cpu,
                    Some(Degradation {
                        kind: DegradationKind::DeviceFallback,
                        message: "CUDA support not compiled, falling back to CPU".into(),
                    }),
                )
            }
        }

        DeviceName::Auto => {
            #[cfg(feature = "cuda")]
            if let Ok(d) = Device::new_cuda(0) {
                return (d, None);
            }
            // Auto → CPU is not a degradation (expected on most developer machines)
            (Device::Cpu, None)
        }
    }
}

// ── CodeBertEmbedder ──────────────────────────────────────────────────────────

/// Production embedder using `candle` + `XLMRobertaModel`.
///
/// Loads model weights eagerly in `new()` (DD8: fail-fast, better UX than failing mid-scan).
///
/// Uses `XLMRobertaModel` (not `BertModel`) — RoBERTa requires a position-ID offset of 2
/// (`padding_idx + 1`). Using `BertModel` produces ~5% cosine error vs Python transformers
/// (validated in Phase-0 spike T1a).
///
/// No explicit `.eval()` call needed (DD10): candle has no train/eval mode toggle.
/// The spike achieved 2.68e-6 max cosine error vs Python (with Python's `.eval()` active),
/// confirming candle's forward pass produces eval-equivalent outputs.
pub(crate) struct CodeBertEmbedder {
    model: XLMRobertaModel,
    tokenizer: Tokenizer,
    device: Device,
    config: EmbedderConfig,
    #[allow(dead_code)] // reserved for T14 test port (dim() method)
    hidden_size: usize,
    #[allow(dead_code)] // reserved for T14 test port (take_degradations)
    degradations: Vec<Degradation>,
}

impl CodeBertEmbedder {
    /// Load model and tokenizer. Eagerly fails on load errors (DD8).
    /// Falls back to CPU on device construction or model load errors (DD4).
    pub(crate) fn new(config: &EmbedderConfig) -> Result<Self, EmbeddingError> {
        let (device, device_deg) = resolve_device(config.device);
        let mut degradations: Vec<Degradation> = device_deg.into_iter().collect();

        // Load tokenizer
        let tokenizer = load_tokenizer(config)?;

        // Load model config + weights (with CPU fallback on load error)
        let (model, hidden_size, device) = load_model(config, device, &mut degradations)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config: config.clone(),
            hidden_size,
            degradations,
        })
    }

    /// Drain accumulated degradation events (device fallback, etc.).
    #[allow(dead_code)] // reserved for T14 test port
    pub(crate) fn take_degradations(&mut self) -> Vec<Degradation> {
        std::mem::take(&mut self.degradations)
    }
}

impl Embedder for CodeBertEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        if snippets.is_empty() {
            return Ok(vec![]);
        }

        let mut result = Vec::with_capacity(snippets.len());
        for chunk in snippets.chunks(self.config.batch_size) {
            let texts: Vec<&str> = chunk.iter().map(|s| s.text.as_str()).collect();
            let batch_embeddings = embed_batch(&self.model, &self.tokenizer, &self.device, &texts)?;
            result.extend(batch_embeddings);
        }
        Ok(result)
    }

    fn dim(&self) -> usize {
        self.hidden_size
    }
}

// ── Internal loading helpers ──────────────────────────────────────────────────

/// Load tokenizer: bundled bytes for the pinned codebert model, hf-hub otherwise.
fn load_tokenizer(config: &EmbedderConfig) -> Result<Tokenizer, EmbeddingError> {
    let mut tokenizer =
        if config.model_name == CODEBERT_MODEL && config.revision == CODEBERT_REVISION {
            // Use bundled tokenizer (3.5 MB, spike-validated at 100% token ID match)
            Tokenizer::from_bytes(CODEBERT_TOKENIZER_BYTES)
                .map_err(|e| EmbeddingError::Tokenizer(format!("bundled tokenizer: {e}")))?
        } else {
            // Download tokenizer.json via hf-hub for other models/revisions
            let path = download_file(&config.model_name, &config.revision, "tokenizer.json")?;
            Tokenizer::from_file(&path)
                .map_err(|e| EmbeddingError::Tokenizer(format!("tokenizer from file: {e}")))?
        };

    // RoBERTa uses pad_id=1 and "<pad>" token (not 0/"[PAD]" as in BERT)
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
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

/// Load model config + weights. Falls back to CPU if load fails on non-CPU device.
fn load_model(
    config: &EmbedderConfig,
    device: Device,
    degradations: &mut Vec<Degradation>,
) -> Result<(XLMRobertaModel, usize, Device), EmbeddingError> {
    // Download config.json (same revision — never mix snapshots, DD2)
    let config_path = download_file(&config.model_name, &config.revision, "config.json")?;
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| EmbeddingError::ModelLoad(format!("read config.json: {e}")))?;
    let codebert_json: CodeBertJson = serde_json::from_str(&config_str)
        .map_err(|e| EmbeddingError::ModelLoad(format!("parse config.json: {e}")))?;
    let hidden_size = codebert_json.hidden_size;
    let xlm_config: XLMConfig = codebert_json.into();

    // Try to load on the requested device; fall back to CPU on failure
    match try_load_weights(config, &device, &xlm_config) {
        Ok(model) => Ok((model, hidden_size, device)),
        Err(e) if !matches!(device, Device::Cpu) => {
            degradations.push(Degradation {
                kind: DegradationKind::DeviceFallback,
                message: format!("model load failed on {device:?} ({e}), retrying on CPU"),
            });
            let model = try_load_weights(config, &Device::Cpu, &xlm_config)?;
            Ok((model, hidden_size, Device::Cpu))
        }
        Err(e) => Err(e),
    }
}

/// Attempt to load weights on the given device. Tries safetensors first, then .bin.
fn try_load_weights(
    config: &EmbedderConfig,
    device: &Device,
    xlm_config: &XLMConfig,
) -> Result<XLMRobertaModel, EmbeddingError> {
    // Try model.safetensors first (preferred — faster mmap load, no conversion needed)
    let vb = if let Ok(path) =
        download_file(&config.model_name, &config.revision, "model.safetensors")
    {
        // Safety: mmap of a trusted local HF cache file (DD2 — same semantics as spike)
        unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
            .map_err(|e| EmbeddingError::ModelLoad(format!("safetensors load: {e}")))?
    } else {
        // Fall back to pytorch_model.bin
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

    // codebert weight keys start with `embeddings.*` / `encoder.*` — no prefix needed (DD2 spike)
    XLMRobertaModel::new(xlm_config, vb)
        .map_err(|e| EmbeddingError::ModelLoad(format!("XLMRobertaModel::new: {e}")))
}

/// Resolve the root of the HuggingFace local hub cache.
///
/// Priority: `HF_HOME` > `XDG_CACHE_HOME/huggingface` > `~/.cache/huggingface`.
/// Matches the logic in Python's `huggingface_hub` and the Phase-0 spike.
fn hf_cache_root() -> Option<std::path::PathBuf> {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Some(std::path::PathBuf::from(hf_home));
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return Some(std::path::PathBuf::from(xdg).join("huggingface"));
    }
    dirs::home_dir().map(|h| h.join(".cache/huggingface"))
}

/// Check the local HF cache for a file at the specified (model, revision).
///
/// Tries two path forms:
/// 1. `snapshots/{revision}/{filename}` — works when `revision` is a pinned commit SHA.
/// 2. If that fails, reads `refs/{revision}` (a text file containing the resolved commit SHA)
///    and retries with the SHA — works when `revision` is a branch name like "main", because
///    `huggingface_hub` (Python) writes `refs/main → <commit-sha>` at download time.
fn find_in_local_hf_cache(
    model_name: &str,
    revision: &str,
    filename: &str,
) -> Option<std::path::PathBuf> {
    let cache_base = hf_cache_root()?.join("hub");
    let dir_name = format!("models--{}", model_name.replace('/', "--"));
    let model_dir = cache_base.join(&dir_name);

    // 1. Direct lookup (works for pinned SHAs)
    let direct = model_dir.join("snapshots").join(revision).join(filename);
    if direct.exists() {
        return Some(direct);
    }

    // 2. Branch-name resolution via refs/ (works after Python `huggingface_hub` download)
    let refs_path = model_dir.join("refs").join(revision);
    if let Ok(sha) = std::fs::read_to_string(&refs_path) {
        let sha = sha.trim();
        if !sha.is_empty() {
            let resolved = model_dir.join("snapshots").join(sha).join(filename);
            if resolved.exists() {
                return Some(resolved);
            }
        }
    }

    None
}

/// Download a file from HuggingFace Hub (same `(model, revision)` pair — DD2).
///
/// Checks the local HF cache first (same logic as Phase-0 spike) to avoid
/// network round-trips when the model is already downloaded. Falls back to
/// hf-hub API download if not found locally.
///
/// `pub(crate)` so sibling embedding backends (e.g. `bert.rs`) can reuse the
/// same HF cache lookup + download logic without duplicating it.
pub(crate) fn download_file(
    model_name: &str,
    revision: &str,
    filename: &str,
) -> Result<std::path::PathBuf, EmbeddingError> {
    // Check local cache first (avoids network + URL construction on offline machines)
    if let Some(path) = find_in_local_hf_cache(model_name, revision, filename) {
        return Ok(path);
    }
    // Fall back to hf-hub download
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let api = Api::new().map_err(|e| EmbeddingError::ModelLoad(format!("hf-hub init: {e}")))?;
    let repo = api.repo(Repo::with_revision(
        model_name.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    repo.get(filename)
        .map_err(|e| EmbeddingError::ModelLoad(format!("download {filename}: {e}")))
}

// ── Batch inference ───────────────────────────────────────────────────────────

/// Embed a batch of texts using XLMRobertaModel with mean pooling.
///
/// Mean pooling formula (spike-validated, max cosine error 2.68e-6 vs Python):
/// `sum(hidden * mask.unsqueeze(-1)) / clamp(mask.sum(-1), 1, MAX)`
fn embed_batch(
    model: &XLMRobertaModel,
    tokenizer: &Tokenizer,
    device: &Device,
    texts: &[&str],
) -> Result<Vec<Embedding>, EmbeddingError> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), /* add_special_tokens */ true)
        .map_err(|e| EmbeddingError::Inference(format!("tokenize: {e}")))?;

    let token_ids_vecs: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
    let attention_masks: Vec<Vec<u32>> = encodings
        .iter()
        .map(|e| e.get_attention_mask().to_vec())
        .collect();

    let batch = texts.len();
    let max_len = token_ids_vecs.iter().map(|v| v.len()).max().unwrap_or(0);

    let ids_flat: Vec<u32> = token_ids_vecs
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    let mask_flat: Vec<u32> = attention_masks
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    // RoBERTa uses token_type_ids = all zeros
    let type_ids_flat: Vec<u32> = vec![0u32; batch * max_len];

    let input_ids = Tensor::from_vec(ids_flat, (batch, max_len), device)
        .map_err(|e| EmbeddingError::Inference(format!("input_ids tensor: {e}")))?
        .to_dtype(DType::U32)
        .map_err(|e| EmbeddingError::Inference(format!("dtype cast: {e}")))?;
    let attention_mask = Tensor::from_vec(mask_flat, (batch, max_len), device)
        .map_err(|e| EmbeddingError::Inference(format!("attention_mask tensor: {e}")))?
        .to_dtype(DType::U32)
        .map_err(|e| EmbeddingError::Inference(format!("dtype cast: {e}")))?;
    let token_type_ids = Tensor::from_vec(type_ids_flat, (batch, max_len), device)
        .map_err(|e| EmbeddingError::Inference(format!("token_type_ids tensor: {e}")))?
        .to_dtype(DType::U32)
        .map_err(|e| EmbeddingError::Inference(format!("dtype cast: {e}")))?;

    // Forward pass → [batch, seq, hidden]
    // XLMRobertaModel signature: forward(input_ids, attention_mask, token_type_ids, past_kv, enc_hs, enc_mask)
    let sequence_output = model
        .forward(
            &input_ids,
            &attention_mask,
            &token_type_ids,
            None,
            None,
            None,
        )
        .map_err(|e| EmbeddingError::Inference(format!("forward pass: {e}")))?;

    // Mean-pool with attention mask (spike-validated formula)
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
        // Clamp to 1.0 (not Python's 1e-9) — an all-zero mask is impossible since
        // the tokenizer always produces at least [CLS] and [SEP] tokens. Both values
        // give the same (all-zero) pooled vector for fully-masked inputs; the spike
        // validated 2.68e-6 max cosine error vs Python using clamp=1.0.
        .clamp(1f32, f32::MAX)
        .map_err(|e| EmbeddingError::Inference(format!("clamp: {e}")))?;
    let pooled = summed
        .broadcast_div(&counts) // [batch, hidden]
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
    use crate::core::config::DeviceName;

    #[test]
    fn resolve_device_explicit_cpu() {
        let (device, deg) = resolve_device(DeviceName::Cpu);
        assert!(matches!(device, Device::Cpu));
        assert!(deg.is_none());
    }

    #[test]
    fn resolve_device_explicit_mps_without_feature() {
        let (device, deg) = resolve_device(DeviceName::Mps);
        assert!(matches!(device, Device::Cpu));
        let d = deg.expect("should have a degradation");
        assert_eq!(d.kind, DegradationKind::DeviceFallback);
        assert!(
            d.message.contains("Metal"),
            "message should mention Metal: {}",
            d.message
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn resolve_device_explicit_cuda_without_feature() {
        let (device, deg) = resolve_device(DeviceName::Cuda);
        assert!(matches!(device, Device::Cpu));
        let d = deg.expect("should have a degradation");
        assert_eq!(d.kind, DegradationKind::DeviceFallback);
        assert!(
            d.message.contains("CUDA") || d.message.contains("Cuda"),
            "message should mention CUDA: {}",
            d.message
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn resolve_device_auto_cpu() {
        // Without cuda feature, Auto resolves to CPU with no degradation
        let (device, deg) = resolve_device(DeviceName::Auto);
        assert!(matches!(device, Device::Cpu));
        assert!(deg.is_none(), "Auto→CPU should not record a degradation");
    }

    #[test]
    fn bundled_tokenizer_bytes_parse() {
        // Validates that the embedded 3.5 MB tokenizer.json is valid and parseable.
        // Runs without any external dependencies or network access.
        let tokenizer = tokenizers::Tokenizer::from_bytes(CODEBERT_TOKENIZER_BYTES)
            .expect("bundled CODEBERT_TOKENIZER_BYTES must parse as a valid Tokenizer");
        // Sanity check: codebert uses a 50265-token RoBERTa vocabulary
        assert!(
            tokenizer.get_vocab_size(false) > 40_000,
            "expected large RoBERTa vocab, got {}",
            tokenizer.get_vocab_size(false)
        );
    }

    /// End-to-end integration test: bundled tokenizer → local HF cache weights → forward pass → pooling.
    ///
    /// Uses `microsoft/codebert-base` @ `CODEBERT_REVISION` (already in the HF cache from the
    /// Phase-0 spike — no network download required). Verifies:
    /// - dim = 768 (codebert-base hidden_size)
    /// - Embeddings are non-zero
    /// - Determinism: two passes produce identical vectors
    /// - Parity: similar functions have cosine > 0.90 (consistent with spike results)
    ///
    /// Note: `tiny-random-roberta` was tried but its weights use a `roberta.` key prefix that
    /// XLMRobertaModel does not expect (codebert-base weights have no prefix). Using the real
    /// model from cache is both simpler and more meaningful.
    #[test]
    #[ignore] // Requires HF cache from Phase-0 spike. Run: cargo test -- --ignored --nocapture
    fn codebert_embed_produces_vectors() {
        use crate::core::config::CODEBERT_REVISION;
        use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind, SnippetRef};

        let config = EmbedderConfig::default(); // codebert-base @ CODEBERT_REVISION, CPU
        assert_eq!(config.revision, CODEBERT_REVISION);

        let embedder = CodeBertEmbedder::new(&config).expect("real model load should succeed");
        assert_eq!(embedder.dim(), 768, "codebert-base hidden_size is 768");

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
        assert_eq!(embeddings[0].vector.len(), 768);
        assert_eq!(embeddings[0].dim, 768);

        // Vectors should be non-zero
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

        // These two snippets are structurally similar — cosine should be high
        let dot: f32 = embeddings[0]
            .vector
            .iter()
            .zip(&embeddings[1].vector)
            .map(|(a, b)| a * b)
            .sum();
        let cosine = dot / (norm0 * norm1);
        eprintln!("codebert cosine(foo(x)=x+1, bar(y)=y+1) = {cosine:.6}  (expect > 0.9)");
        assert!(
            cosine > 0.9,
            "similar functions should have high cosine: {cosine}"
        );

        // Determinism: two passes should produce identical vectors
        let embeddings2 = embedder
            .embed(&snippets)
            .expect("second embed should succeed");
        assert_eq!(
            embeddings[0].vector, embeddings2[0].vector,
            "embeddings should be deterministic"
        );
        assert_eq!(embeddings[1].vector, embeddings2[1].vector);

        eprintln!(
            "Integration test PASSED: hidden_size=768 norm0={norm0:.6} norm1={norm1:.6} cosine={cosine:.6}"
        );
    }
}
