use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Which detection engine to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EngineName {
    Semantic,
    Sonarqube,
}

/// Which embedder to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EmbedderName {
    Codebert,
    Faster,
    Stub,
    /// ONNX Runtime backend (experimental). Requires `--features onnx` build and a
    /// pre-exported model.onnx at CLONEHUNTER_ONNX_MODEL or the default cache path.
    Onnx,
}

/// Which vector index to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum IndexName {
    Brute,
    Faiss,
}

/// Which compute device to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DeviceName {
    Auto,
    Cpu,
    Mps,
    Cuda,
}

/// Sliding-window snippet configuration.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct WindowConfig {
    pub window_lines: usize,
    pub stride_lines: usize,
    pub min_nonempty: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_lines: 40,
            stride_lines: 6,
            min_nonempty: 4,
        }
    }
}

/// Call-expansion snippet configuration.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ExpansionConfig {
    pub enabled: bool,
    pub depth: usize,
    pub max_chars: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            depth: 1,
            max_chars: 4000,
        }
    }
}

/// Similarity thresholds and gates.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Thresholds {
    pub func: f64,
    pub win: f64,
    pub exp: f64,
    pub min_window_hits: usize,
    pub lexical_min_ratio: f64,
    pub lexical_weight: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            func: 0.92,
            win: 0.90,
            exp: 0.90,
            min_window_hits: 1,
            lexical_min_ratio: 0.5,
            lexical_weight: 0.3,
        }
    }
}

/// Vector index configuration.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct IndexConfig {
    pub name: IndexName,
    pub top_k: usize,
    pub faiss_nlist: usize,
    pub faiss_nprobe: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            name: IndexName::Brute,
            top_k: 25,
            faiss_nlist: 128,
            faiss_nprobe: 8,
        }
    }
}

/// Embedding cache configuration.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CacheConfig {
    pub path: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            path: "~/.cache/clonehunter".into(),
        }
    }
}

/// Pinned CodeBERT model revision SHA validated in Phase 0 spike (T1a).
/// Max cosine diff vs Python: 2.68e-6. Using "main" would be a moving target.
pub(crate) const CODEBERT_REVISION: &str = "3b0952feddeffad0063f274080e3c23d75e7eb39";

/// Embedder configuration.
///
/// Note: `trust_remote_code` (Python field) is dropped (DD9) — candle compiles model
/// architectures as Rust code; there is no dynamic model loading equivalent.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct EmbedderConfig {
    pub name: EmbedderName,
    pub model_name: String,
    pub revision: String,
    pub max_length: usize,
    pub batch_size: usize,
    pub device: DeviceName,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            name: EmbedderName::Codebert,
            model_name: "microsoft/codebert-base".into(),
            revision: CODEBERT_REVISION.into(),
            max_length: 256,
            batch_size: 16,
            device: DeviceName::Auto,
        }
    }
}

/// A known-good embedder preset (model name + revision + sizing).
#[derive(Debug, Clone)]
pub(crate) struct EmbedderPreset {
    pub model_name: &'static str,
    pub revision: &'static str,
    pub max_length: usize,
    pub batch_size: usize,
}

/// Return the preset for a given embedder name, or `None` for `Stub`.
///
/// Presets give sensible defaults for well-known models. When the user switches
/// the embedder name, `apply_overrides` rebases from this preset and then applies
/// any explicit field overrides on top.
pub(crate) fn embedder_preset(name: EmbedderName) -> Option<EmbedderPreset> {
    match name {
        EmbedderName::Codebert => Some(EmbedderPreset {
            model_name: "microsoft/codebert-base",
            revision: CODEBERT_REVISION,
            max_length: 256,
            batch_size: 16,
        }),
        EmbedderName::Faster => Some(EmbedderPreset {
            // revision = "main" — this model was not spike-validated; "main" is the Python behavior
            model_name: "isuruwijesiri/all-MiniLM-L6-v2-code-search-512",
            revision: "main",
            max_length: 512,
            batch_size: 32,
        }),
        EmbedderName::Stub => None, // No preset; stub uses whatever defaults are in place
        EmbedderName::Onnx => Some(EmbedderPreset {
            // Same model as codebert; ONNX backend does its own weight loading
            model_name: "microsoft/codebert-base",
            revision: CODEBERT_REVISION,
            max_length: 256,
            batch_size: 16,
        }),
    }
}

/// Top-level CloneHunter configuration.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CloneHunterConfig {
    pub engine: EngineName,
    pub include_globs: Vec<String>,
    pub exclude_globs: Vec<String>,
    pub windows: WindowConfig,
    pub expansion: ExpansionConfig,
    pub thresholds: Thresholds,
    pub index: IndexConfig,
    pub cache: CacheConfig,
    pub embedder: EmbedderConfig,
    pub cluster_findings: bool,
    pub cluster_min_size: usize,
}

impl Default for CloneHunterConfig {
    fn default() -> Self {
        Self {
            engine: EngineName::Semantic,
            include_globs: vec!["**/*.py".into()],
            exclude_globs: vec![
                "**/.venv/**".into(),
                "**/venv/**".into(),
                "**/__pycache__/**".into(),
                "**/site-packages/**".into(),
            ],
            windows: WindowConfig::default(),
            expansion: ExpansionConfig::default(),
            thresholds: Thresholds::default(),
            index: IndexConfig::default(),
            cache: CacheConfig::default(),
            embedder: EmbedderConfig::default(),
            cluster_findings: false,
            cluster_min_size: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_claude_md() {
        let c = CloneHunterConfig::default();
        assert_eq!(c.engine, EngineName::Semantic);
        assert_eq!(c.thresholds.func, 0.92);
        assert_eq!(c.thresholds.win, 0.90);
        assert_eq!(c.thresholds.exp, 0.90);
        assert_eq!(c.thresholds.min_window_hits, 1);
        assert!((c.thresholds.lexical_min_ratio - 0.5).abs() < f64::EPSILON);
        assert!((c.thresholds.lexical_weight - 0.3).abs() < f64::EPSILON);
        assert_eq!(c.windows.window_lines, 40);
        assert_eq!(c.windows.stride_lines, 6);
        assert_eq!(c.windows.min_nonempty, 4);
        assert!(!c.expansion.enabled);
        assert_eq!(c.expansion.depth, 1);
        assert_eq!(c.expansion.max_chars, 4000);
        assert_eq!(c.index.name, IndexName::Brute);
        assert_eq!(c.index.top_k, 25);
        assert_eq!(c.index.faiss_nlist, 128);
        assert_eq!(c.index.faiss_nprobe, 8);
        assert_eq!(c.embedder.name, EmbedderName::Codebert);
        assert_eq!(c.embedder.model_name, "microsoft/codebert-base");
        assert_eq!(c.embedder.revision, CODEBERT_REVISION);
        assert_eq!(c.embedder.max_length, 256);
        assert_eq!(c.embedder.batch_size, 16);
        assert_eq!(c.embedder.device, DeviceName::Auto);
        assert_eq!(c.cache.path, "~/.cache/clonehunter");
        assert_eq!(c.include_globs, vec!["**/*.py"]);
        assert!(!c.cluster_findings);
        assert_eq!(c.cluster_min_size, 2);
    }

    #[test]
    fn codebert_preset_matches_defaults() {
        let p = embedder_preset(EmbedderName::Codebert).unwrap();
        assert_eq!(p.model_name, "microsoft/codebert-base");
        assert_eq!(p.revision, CODEBERT_REVISION);
        assert_eq!(p.max_length, 256);
        assert_eq!(p.batch_size, 16);
    }

    #[test]
    fn faster_preset_values() {
        let p = embedder_preset(EmbedderName::Faster).unwrap();
        assert_eq!(
            p.model_name,
            "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
        );
        assert_eq!(p.revision, "main");
        assert_eq!(p.max_length, 512);
        assert_eq!(p.batch_size, 32);
    }

    #[test]
    fn stub_has_no_preset() {
        assert!(embedder_preset(EmbedderName::Stub).is_none());
    }
}
