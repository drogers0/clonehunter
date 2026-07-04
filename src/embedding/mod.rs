mod bert;
mod cache;
mod codebert;
#[cfg(feature = "mlx")]
mod mlx_backend;
#[cfg(feature = "onnx")]
mod onnx_backend;
mod stub;

use std::collections::HashMap;

use thiserror::Error;

use crate::core::config::{EmbedderConfig, EmbedderName};
use crate::core::types::{Embedding, SnippetRef};
use crate::io::fingerprints::embed_cache_key;

pub(crate) use bert::BertEmbedder;
pub(crate) use cache::EmbeddingCache;
pub(crate) use codebert::CodeBertEmbedder;
#[cfg(feature = "mlx")]
pub(crate) use mlx_backend::MlxEmbedder;
#[cfg(feature = "onnx")]
pub(crate) use onnx_backend::OnnxEmbedder;
pub(crate) use stub::StubEmbedder;

// ── Error types (DD5) ─────────────────────────────────────────────────────────

/// Errors from the embedding subsystem (DD5).
///
/// `Cache` is a separate variant from `Inference` so T10 can distinguish recoverable
/// cache failures (log + continue) from fatal inference errors (abort scan).
#[derive(Debug, Error)]
pub(crate) enum EmbeddingError {
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("device error: {0}")]
    #[allow(dead_code)] // reserved for T14 test port
    Device(String),
    #[error("cache error: {0}")]
    Cache(String),
}

// ── Embedder trait (DD3) ─────────────────────────────────────────────────────

/// Abstraction over embedding backends (DD3).
///
/// `Box<dyn Embedder>` is the T10 pipeline entry point. Two implementations:
/// - `CodeBertEmbedder`: candle `XLMRobertaModel` (production)
/// - `StubEmbedder`: deterministic SHA-256 (tests, CI, smoke runs without torch)
pub(crate) trait Embedder: Send {
    /// Embed a batch of snippet references. Returns one `Embedding` per snippet in input order.
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError>;

    /// Embedding dimension (e.g. 768 for codebert-base, 16 for stub).
    #[allow(dead_code)] // reserved for T14 test port
    fn dim(&self) -> usize;
}

// ── Factory (Step 7.6) ────────────────────────────────────────────────────────

/// Create the appropriate embedder for the given config (DD3, DD8).
///
/// The `CLONEHUNTER_EMBEDDER=stub` env var override is handled by T12 (CLI) which sets
/// `config.embedder.name = Stub` before reaching the pipeline. This factory only
/// honors the config value.
pub(crate) fn create_embedder(
    config: &EmbedderConfig,
) -> Result<Box<dyn Embedder>, EmbeddingError> {
    match config.name {
        EmbedderName::Stub => Ok(Box::new(StubEmbedder::new(16))),
        EmbedderName::Codebert => Ok(Box::new(CodeBertEmbedder::new(config)?)),
        // `faster` routes through the candle BERT path (BertModel), not XLMRobertaModel.
        // Using XLMRobertaModel for BERT-family weights (e.g. MiniLM) was broken — the
        // weight key layout and position-ID handling differ. BertEmbedder fixes this.
        EmbedderName::Faster => Ok(Box::new(BertEmbedder::new(config)?)),
        EmbedderName::Mlx => {
            #[cfg(feature = "mlx")]
            {
                Ok(Box::new(MlxEmbedder::new(config)?))
            }
            #[cfg(not(feature = "mlx"))]
            {
                Err(EmbeddingError::ModelLoad(
                    "EmbedderName::Mlx requires the `mlx` cargo feature: \
                     cargo build --features mlx"
                        .into(),
                ))
            }
        }
        EmbedderName::Onnx => {
            #[cfg(feature = "onnx")]
            {
                // Check CLONEHUNTER_ONNX_COREML=1 to enable CoreML EP attempt
                let use_coreml = std::env::var("CLONEHUNTER_ONNX_COREML")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if use_coreml {
                    Ok(Box::new(OnnxEmbedder::new_coreml(config)?))
                } else {
                    Ok(Box::new(OnnxEmbedder::new(config)?))
                }
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(EmbeddingError::ModelLoad(
                    "EmbedderName::Onnx requires the `onnx` cargo feature: \
                     cargo build --features onnx"
                        .into(),
                ))
            }
        }
    }
}

// ── embed_with_cache (DD7) ────────────────────────────────────────────────────

/// Full pipeline integration: cache lookup → embed misses → cache write → assemble (DD7).
///
/// Returns `(embeddings, cache_hits, cache_misses)` where hits + misses == snippets.len().
/// Hits and misses are counted per snippet position (not per unique key), so duplicate snippets
/// each count independently.
///
/// Cardinality is validated before writing to the cache — a length mismatch returns
/// `EmbeddingError::Inference` rather than panicking or writing partial results.
pub(crate) fn embed_with_cache(
    snippets: &[&SnippetRef],
    embedder: &dyn Embedder,
    cache: &EmbeddingCache,
    config: &EmbedderConfig,
) -> Result<(Vec<Embedding>, usize, usize), EmbeddingError> {
    if snippets.is_empty() {
        return Ok((vec![], 0, 0));
    }

    // 1. Compute cache key for each snippet position
    let keys: Vec<String> = snippets
        .iter()
        .map(|s| {
            embed_cache_key(
                &config.model_name,
                &config.revision,
                config.max_length,
                &s.snippet_hash,
            )
        })
        .collect();

    // 2. Batch cache lookup
    let key_refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
    let cached = cache
        .get_many(&key_refs)
        .map_err(|e| EmbeddingError::Cache(format!("cache read: {e}")))?;

    // 3. Identify misses — per position (not per unique key), matching Python behavior
    let miss_indices: Vec<usize> = (0..snippets.len())
        .filter(|i| !cached.contains_key(&keys[*i]))
        .collect();
    let cache_hits = snippets.len() - miss_indices.len();
    let cache_misses = miss_indices.len();

    // 4. Embed misses
    let miss_snippets: Vec<&SnippetRef> = miss_indices.iter().map(|&i| snippets[i]).collect();
    let miss_embeddings = if miss_snippets.is_empty() {
        vec![]
    } else {
        embedder.embed(&miss_snippets)?
    };

    // 4b. Defensive cardinality check — prevents partial cache writes and assembly panics
    if miss_embeddings.len() != miss_indices.len() {
        return Err(EmbeddingError::Inference(format!(
            "embedder returned {} embeddings for {} snippets",
            miss_embeddings.len(),
            miss_indices.len()
        )));
    }

    // 5. Write new embeddings to cache
    // Note: if two miss positions share the same cache key (duplicate snippets), the
    // HashMap coalesces them to a single entry (last write wins). This is safe because
    // any deterministic embedder produces identical vectors for identical text, so the
    // overwritten value is semantically equivalent. Assembly (step 6) reads from `cached`
    // (snapshotted before set_many, so still reports misses for duplicates) and consumes
    // one embedding per position from miss_iter — no value is lost or mis-assigned.
    let new_entries: HashMap<String, Embedding> = miss_indices
        .iter()
        .zip(miss_embeddings.iter())
        .map(|(&i, emb)| (keys[i].clone(), emb.clone()))
        .collect();
    if !new_entries.is_empty() {
        cache
            .set_many(&new_entries)
            .map_err(|e| EmbeddingError::Cache(format!("cache write: {e}")))?;
    }

    // 6. Assemble results in original order
    let mut result = Vec::with_capacity(snippets.len());
    let mut miss_iter = miss_embeddings.into_iter();
    for key in &keys {
        if let Some(emb) = cached.get(key) {
            result.push(emb.clone());
        } else {
            // SAFETY: cardinality validated above
            result.push(miss_iter.next().expect("cardinality validated above"));
        }
    }

    Ok((result, cache_hits, cache_misses))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::EmbedderConfig;
    use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind};
    use tempfile::TempDir;

    fn make_snippet(text: &str) -> SnippetRef {
        let file = FileRef {
            path: "test.py".into(),
            content_hash: "abc".into(),
            language: Language::Python,
        };
        let func = FunctionRef {
            file,
            qualified_name: "f".into(),
            start_line: 1,
            end_line: 3,
            code: text.into(),
            code_hash: "c".into(),
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
    }

    fn stub_config(_cache_dir: &TempDir) -> EmbedderConfig {
        EmbedderConfig {
            name: EmbedderName::Stub,
            model_name: "stub".into(),
            revision: "none".into(),
            max_length: 16,
            batch_size: 8,
            device: crate::core::config::DeviceName::Cpu,
        }
    }

    fn open_cache(dir: &TempDir) -> EmbeddingCache {
        EmbeddingCache::new(dir.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn embed_with_cache_empty() {
        let dir = TempDir::new().unwrap();
        let embedder = StubEmbedder::new(16);
        let cache = open_cache(&dir);
        let config = stub_config(&dir);

        let (result, hits, misses) = embed_with_cache(&[], &embedder, &cache, &config).unwrap();
        assert!(result.is_empty());
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    #[test]
    fn embed_with_cache_all_misses() {
        let dir = TempDir::new().unwrap();
        let embedder = StubEmbedder::new(16);
        let cache = open_cache(&dir);
        let config = stub_config(&dir);

        let s1 = make_snippet("def foo(): pass");
        let s2 = make_snippet("def bar(): return 1");
        let snippets: Vec<&SnippetRef> = vec![&s1, &s2];

        let (result, hits, misses) =
            embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(hits, 0);
        assert_eq!(misses, 2);

        // Verify the embeddings were written to cache (second call should be all hits)
        let (result2, hits2, misses2) =
            embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();
        assert_eq!(hits2, 2);
        assert_eq!(misses2, 0);
        assert_eq!(result[0].vector, result2[0].vector);
        assert_eq!(result[1].vector, result2[1].vector);
    }

    #[test]
    fn embed_with_cache_all_hits() {
        let dir = TempDir::new().unwrap();
        let embedder = StubEmbedder::new(16);
        let cache = open_cache(&dir);
        let config = stub_config(&dir);

        let s1 = make_snippet("def alpha(): pass");
        let snippets: Vec<&SnippetRef> = vec![&s1];

        // Prime the cache
        embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();

        // Second call: all hits
        let (_, hits, misses) = embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();
        assert_eq!(hits, 1);
        assert_eq!(misses, 0);
    }

    #[test]
    fn embed_with_cache_mixed() {
        let dir = TempDir::new().unwrap();
        let embedder = StubEmbedder::new(16);
        let cache = open_cache(&dir);
        let config = stub_config(&dir);

        let s1 = make_snippet("def a(): pass");
        let s2 = make_snippet("def b(): pass");
        let s3 = make_snippet("def c(): pass");

        // Prime with s1 and s3
        embed_with_cache(&[&s1, &s3], &embedder, &cache, &config).unwrap();

        // Now embed s1, s2, s3 — s1 and s3 are hits, s2 is a miss
        let snippets: Vec<&SnippetRef> = vec![&s1, &s2, &s3];
        let (result, hits, misses) =
            embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
        assert_eq!(result.len(), 3);

        // Results should match direct stub embedding
        let direct = embedder.embed(&snippets).unwrap();
        for (cached_emb, direct_emb) in result.iter().zip(direct.iter()) {
            assert_eq!(
                cached_emb.vector, direct_emb.vector,
                "cached and direct should match"
            );
        }
    }

    #[test]
    fn embed_with_cache_duplicate_key_snippets() {
        // Two snippets with identical text share the same cache key.
        // Both should be counted per-position (DD7):
        //   - First call: both are misses (key not yet in cache).
        //   - Second call: both are hits (key written by first call).
        // Assembly must return the correct embedding at each position.
        let dir = TempDir::new().unwrap();
        let embedder = StubEmbedder::new(16);
        let cache = open_cache(&dir);
        let config = stub_config(&dir);

        let s1 = make_snippet("def dup(): pass");
        let s2 = make_snippet("def dup(): pass"); // same text → same cache key
        let snippets: Vec<&SnippetRef> = vec![&s1, &s2];

        // First call: both are misses
        let (result, hits, misses) =
            embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();
        assert_eq!(hits, 0);
        assert_eq!(misses, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].vector, result[1].vector); // same text → same embedding

        // Second call: both are hits
        let (result2, hits2, misses2) =
            embed_with_cache(&snippets, &embedder, &cache, &config).unwrap();
        assert_eq!(hits2, 2);
        assert_eq!(misses2, 0);
        assert_eq!(result[0].vector, result2[0].vector);
        assert_eq!(result[1].vector, result2[1].vector);
    }

    #[test]
    fn embed_with_cache_cardinality_mismatch() {
        let dir = TempDir::new().unwrap();
        let cache = open_cache(&dir);
        let config = stub_config(&dir);

        // Broken embedder that returns wrong number of results
        struct BrokenEmbedder;
        impl Embedder for BrokenEmbedder {
            fn embed(&self, _: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
                Ok(vec![]) // Always returns empty — mismatch with any non-empty input
            }
            fn dim(&self) -> usize {
                16
            }
        }

        let s1 = make_snippet("def x(): pass");
        let snippets: Vec<&SnippetRef> = vec![&s1];
        let result = embed_with_cache(&snippets, &BrokenEmbedder, &cache, &config);
        assert!(matches!(result, Err(EmbeddingError::Inference(_))));
    }
}
