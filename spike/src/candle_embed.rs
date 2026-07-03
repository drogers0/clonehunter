//! T1a: Candle RoBERTa/CodeBERT embedding parity check.
#![allow(dead_code)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config as XLMConfig, XLMRobertaModel};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

/// Reference datum from python_embeddings.json.
#[derive(Deserialize)]
pub struct RefEmbedding {
    pub text: String,
    pub embedding: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// One embedding produced by the Rust candle path.
#[derive(Serialize)]
pub struct RustEmbedding {
    pub embedding: Vec<f32>,
    pub token_ids: Vec<u32>,
}

/// Intermediate deserialization config that tolerates optional fields in codebert's config.json.
/// Converts into XLMConfig for model loading.
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

pub struct Embedder {
    model: XLMRobertaModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Resolve the local HuggingFace cache path for a model snapshot.
///
/// Python's huggingface_hub always uses `~/.cache/huggingface/hub` (XDG_CACHE_HOME
/// override respected, but defaults to `~/.cache` not `~/Library/Caches` on macOS).
fn resolve_local_cache(model_name: &str, revision: &str) -> Option<PathBuf> {
    // Respect XDG_CACHE_HOME if set; otherwise use ~/.cache
    let cache_base = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("XDG_CACHE_HOME")
                .map(|p| PathBuf::from(p).join("huggingface"))
                .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".cache/huggingface"))
        })
        .join("hub");
    let dir_name = format!("models--{}", model_name.replace('/', "--"));
    let candidate = cache_base.join(dir_name).join("snapshots").join(revision);
    eprintln!("  Looking for local cache: {}", candidate.display());
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Search all local HF cache snapshots for a file (not just the pinned revision).
/// Needed because safetensors and config.json may live in different snapshots.
fn find_in_any_snapshot(model_name: &str, filename: &str) -> Option<PathBuf> {
    let cache_base = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("XDG_CACHE_HOME")
                .map(|p| PathBuf::from(p).join("huggingface"))
                .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".cache/huggingface"))
        })
        .join("hub");
    let dir_name = format!("models--{}", model_name.replace('/', "--"));
    let snapshots_dir = cache_base.join(dir_name).join("snapshots");
    std::fs::read_dir(&snapshots_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(filename))
        .find(|p| p.exists())
}

impl Embedder {
    pub fn load(model_name: &str, revision: &str, fixtures_dir: &Path) -> Result<Self> {
        // Try local HuggingFace cache first (avoids network round-trip)
        let snapshot_dir = resolve_local_cache(model_name, revision);

        let config_path = match &snapshot_dir {
            Some(dir) if dir.join("config.json").exists() => dir.join("config.json"),
            _ => {
                // Fall back to hf-hub download
                eprintln!("  Local cache not found, downloading via hf-hub...");
                download_via_hub(model_name, revision, "config.json")?
            }
        };

        // Tokenizer: loaded from committed fixtures/tokenizer.json (pre-generated)
        // This avoids the missing tokenizer.json issue on microsoft/codebert-base.
        let tokenizer_path = fixtures_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "tokenizer.json not found at {}. Run generate_references.py first.",
                tokenizer_path.display()
            );
        }

        // Find model weights: prefer safetensors (may live in a different snapshot than config).
        let weights_path = match &snapshot_dir {
            Some(dir) if dir.join("model.safetensors").exists() => dir.join("model.safetensors"),
            _ => {
                // safetensors may be in a different snapshot — scan all of them
                if let Some(p) = find_in_any_snapshot(model_name, "model.safetensors") {
                    p
                } else {
                    match &snapshot_dir {
                        Some(dir) if dir.join("pytorch_model.bin").exists() => {
                            dir.join("pytorch_model.bin")
                        }
                        _ => match download_via_hub(model_name, revision, "model.safetensors") {
                            Ok(p) => p,
                            Err(_) => download_via_hub(model_name, revision, "pytorch_model.bin")
                                .context("weights not found locally or via hub")?,
                        },
                    }
                }
            }
        };

        eprintln!("  config:    {}", config_path.display());
        eprintln!("  tokenizer: {}", tokenizer_path.display());
        eprintln!("  weights:   {}", weights_path.display());

        // Load config — deserialize via CodeBertJson to handle missing optional fields,
        // then convert to XLMConfig which has proper RoBERTa position-ID offset logic.
        let config_str = std::fs::read_to_string(&config_path).context("read config.json")?;
        let codebert_json: CodeBertJson =
            serde_json::from_str(&config_str).context("parse config.json")?;
        let xlm_config: XLMConfig = codebert_json.into();

        // Load tokenizer
        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 256,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Load model weights
        let device = Device::Cpu;
        let ext = weights_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let vb = if ext == "bin" {
            eprintln!("  Loading from pytorch_model.bin...");
            VarBuilder::from_pth(&weights_path, DType::F32, &device).context("from_pth")?
        } else {
            eprintln!("  Loading from model.safetensors...");
            // Safety: mmaped read of trusted local cache file
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                    .context("from_mmaped_safetensors")?
            }
        };

        // XLMRobertaModel uses proper RoBERTa position ID offset (padding_idx+1).
        // BertModel would use 0-based IDs causing ~5% cosine error vs Python transformers.
        // codebert weights have no model-type prefix — keys start with embeddings.* / encoder.*
        let model = XLMRobertaModel::new(&xlm_config, vb)
            .context("XLMRobertaModel::new")?;
        eprintln!("  Model loaded (hidden_size={})", xlm_config.hidden_size);

        Ok(Self { model, tokenizer, device })
    }

    /// Embed a batch of texts, returning (embeddings, token_id_vecs).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<(Vec<Vec<f32>>, Vec<Vec<u32>>)> {
        if texts.is_empty() {
            return Ok((vec![], vec![]));
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), /* add_special_tokens */ true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

        let token_ids_vecs: Vec<Vec<u32>> =
            encodings.iter().map(|e| e.get_ids().to_vec()).collect();
        let attention_masks: Vec<Vec<u32>> =
            encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect();

        let max_len = token_ids_vecs.iter().map(|v| v.len()).max().unwrap_or(0);
        let batch = texts.len();

        // Flatten into tensors [batch, seq]
        let ids_flat: Vec<u32> =
            token_ids_vecs.iter().flat_map(|v| v.iter().copied()).collect();
        let mask_flat: Vec<u32> =
            attention_masks.iter().flat_map(|v| v.iter().copied()).collect();
        let type_ids_flat: Vec<u32> = vec![0u32; batch * max_len]; // RoBERTa: all zeros

        let input_ids = Tensor::from_vec(ids_flat, (batch, max_len), &self.device)?
            .to_dtype(DType::U32)?;
        let attention_mask = Tensor::from_vec(mask_flat, (batch, max_len), &self.device)?
            .to_dtype(DType::U32)?;
        let token_type_ids =
            Tensor::from_vec(type_ids_flat, (batch, max_len), &self.device)?.to_dtype(DType::U32)?;

        // Forward pass → [batch, seq, hidden]
        // XLMRobertaModel signature: (input_ids, attention_mask, token_type_ids, past_kv, enc_hs, enc_mask)
        let sequence_output = self
            .model
            .forward(&input_ids, &attention_mask, &token_type_ids, None, None, None)
            .context("XLMRobertaModel::forward")?;

        // Mean pool with attention mask
        let mask_f32 = attention_mask
            .to_dtype(DType::F32)?
            .unsqueeze(candle_core::D::Minus1)?; // [batch, seq, 1]
        let masked = sequence_output.broadcast_mul(&mask_f32)?;
        let summed = masked.sum(1)?; // [batch, hidden]
        let counts = mask_f32.sum(1)?.clamp(1f32, f32::MAX)?; // [batch, 1]
        let pooled = summed.broadcast_div(&counts)?; // [batch, hidden]

        let pooled_vec: Vec<Vec<f32>> =
            pooled.to_vec2::<f32>().context("pooled to_vec2")?;

        Ok((pooled_vec, token_ids_vecs))
    }

    /// Embed texts one at a time (for batch-invariance check).
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let (embeddings, _) = self.embed_batch(&[text])?;
        Ok(embeddings.into_iter().next().unwrap())
    }
}

fn download_via_hub(model_name: &str, revision: &str, filename: &str) -> Result<PathBuf> {
    use hf_hub::{Repo, RepoType, api::sync::Api};
    let api = Api::new().context("hf-hub Api init")?;
    let repo = api.repo(Repo::with_revision(
        model_name.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    repo.get(filename)
        .with_context(|| format!("download {filename} from {model_name}@{revision}"))
}

/// Cosine similarity between two equal-length vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// T1a result details.
pub struct T1aResult {
    pub token_id_match_rate: f64,
    pub max_dim_diff: f32,
    pub mean_dim_diff: f32,
    pub max_cosine_diff: f32,
    pub pairs_exceeding_1e4: usize,
    pub total_pairs: usize,
    pub near_threshold_flips: Vec<String>,
    pub determinism_pass: bool,
    pub batch_invariance_max_diff: f32,
    pub rust_cosines: Vec<Vec<f32>>,
    pub rust_embeddings: Vec<Vec<f32>>,
    pub rust_token_ids: Vec<Vec<u32>>,
}

pub fn run_t1a(
    embedder: &Embedder,
    ref_embeddings: &[RefEmbedding],
    py_cosines: &[Vec<f32>],
) -> T1aResult {
    let texts: Vec<&str> = ref_embeddings.iter().map(|r| r.text.as_str()).collect();
    let n = texts.len();

    eprintln!("\n[T1a] Embedding {n} texts in one batch...");
    let (rust_embeddings, rust_token_ids) = match embedder.embed_batch(&texts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[T1a] FATAL: embed_batch failed: {e}");
            return T1aResult {
                token_id_match_rate: 0.0,
                max_dim_diff: f32::MAX,
                mean_dim_diff: f32::MAX,
                max_cosine_diff: f32::MAX,
                pairs_exceeding_1e4: usize::MAX,
                total_pairs: 0,
                near_threshold_flips: vec![format!("embed_batch error: {e}")],
                determinism_pass: false,
                batch_invariance_max_diff: f32::MAX,
                rust_cosines: vec![],
                rust_embeddings: vec![],
                rust_token_ids: vec![],
            };
        }
    };

    // Token ID comparison (single-item, no padding)
    let mut token_match_count = 0usize;
    let mut token_total = 0usize;
    // Re-tokenize without padding for fair comparison to Python's per-item token_ids
    let single_token_ids: Vec<Vec<u32>> = texts
        .iter()
        .filter_map(|&text| {
            // Use tokenizer without padding for single-item comparison
            let mut tok = embedder.tokenizer.clone();
            tok.with_padding(None);
            tok.encode(text, true).ok().map(|e| e.get_ids().to_vec())
        })
        .collect();

    for (i, (rust_ids, ref_item)) in
        single_token_ids.iter().zip(ref_embeddings.iter()).enumerate()
    {
        let ref_ids = &ref_item.token_ids;
        let cmp_len = rust_ids.len().min(ref_ids.len());
        for j in 0..cmp_len {
            token_total += 1;
            if rust_ids[j] == ref_ids[j] {
                token_match_count += 1;
            }
        }
        token_total += rust_ids.len().abs_diff(ref_ids.len());
        if rust_ids != ref_ids {
            eprintln!(
                "  [T1a] token mismatch at snippet {i} ({}/{}): rust_len={}, ref_len={}",
                ref_item
                    .text
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(40)
                    .collect::<String>(),
                ref_embeddings
                    .get(i)
                    .map(|r| r.token_ids.len())
                    .unwrap_or(0),
                rust_ids.len(),
                ref_ids.len()
            );
        }
    }
    let token_id_match_rate =
        if token_total == 0 { 1.0 } else { token_match_count as f64 / token_total as f64 };

    // Per-dimension absolute difference
    let mut max_dim_diff = 0f32;
    let mut total_dim_diff = 0f64;
    let mut dim_count = 0usize;
    for (rust_emb, ref_item) in rust_embeddings.iter().zip(ref_embeddings.iter()) {
        for (r, p) in rust_emb.iter().zip(ref_item.embedding.iter()) {
            let d = (r - p).abs();
            max_dim_diff = max_dim_diff.max(d);
            total_dim_diff += d as f64;
            dim_count += 1;
        }
    }
    let mean_dim_diff =
        if dim_count > 0 { (total_dim_diff / dim_count as f64) as f32 } else { 0.0 };

    // Pairwise cosine similarity matrix (Rust)
    let rust_cosines: Vec<Vec<f32>> = (0..n)
        .map(|i| (0..n).map(|j| cosine(&rust_embeddings[i], &rust_embeddings[j])).collect())
        .collect();

    // Compare against Python cosines
    let thresholds = [0.90f32, 0.92f32];
    let mut max_cosine_diff = 0f32;
    let mut pairs_exceeding_1e4 = 0usize;
    let mut near_threshold_flips = Vec::new();
    let total_pairs = n * (n - 1) / 2;

    for i in 0..n {
        for j in (i + 1)..n {
            let rc = rust_cosines[i][j];
            let pc = py_cosines[i][j];
            let diff = (rc - pc).abs();
            max_cosine_diff = max_cosine_diff.max(diff);
            if diff > 1e-4 {
                pairs_exceeding_1e4 += 1;
            }
            for &thresh in &thresholds {
                let rc_side = rc >= thresh;
                let pc_side = pc >= thresh;
                let near = (rc - thresh).abs() < 0.02 || (pc - thresh).abs() < 0.02;
                if near && rc_side != pc_side {
                    near_threshold_flips.push(format!(
                        "pair ({i},{j}) flips at thresh={thresh:.2}: py={pc:.4} rust={rc:.4}"
                    ));
                }
            }
        }
    }

    // Determinism check: embed all again
    eprintln!("[T1a] Determinism check (second batch pass)...");
    let (rust_embeddings2, _) = match embedder.embed_batch(&texts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[T1a] Determinism check failed: {e}");
            return T1aResult {
                token_id_match_rate,
                max_dim_diff,
                mean_dim_diff,
                max_cosine_diff,
                pairs_exceeding_1e4,
                total_pairs,
                near_threshold_flips,
                determinism_pass: false,
                batch_invariance_max_diff: f32::MAX,
                rust_cosines,
                rust_embeddings,
                rust_token_ids,
            };
        }
    };

    let determinism_pass =
        rust_embeddings.iter().zip(rust_embeddings2.iter()).all(|(a, b)| a == b);
    if !determinism_pass {
        eprintln!("[T1a] FAIL: non-deterministic embeddings detected");
    }

    // Batch invariance: embed each snippet individually, compare to batch result
    eprintln!("[T1a] Batch invariance check ({n} individual embeddings)...");
    let mut batch_invariance_max_diff = 0f32;
    for (i, text) in texts.iter().enumerate() {
        let single = match embedder.embed_single(text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[T1a] embed_single({i}) failed: {e}");
                batch_invariance_max_diff = f32::MAX;
                break;
            }
        };
        for (s, b) in single.iter().zip(rust_embeddings[i].iter()) {
            batch_invariance_max_diff = batch_invariance_max_diff.max((s - b).abs());
        }
    }

    T1aResult {
        token_id_match_rate,
        max_dim_diff,
        mean_dim_diff,
        max_cosine_diff,
        pairs_exceeding_1e4,
        total_pairs,
        near_threshold_flips,
        determinism_pass,
        batch_invariance_max_diff,
        rust_cosines,
        rust_embeddings,
        rust_token_ids,
    }
}
