//! Shared helpers for the codebert-family embedding backends (candle / onnx / mlx).
//!
//! All three backends embed `microsoft/codebert-base` with the same bundled tokenizer,
//! the same RoBERTa padding/truncation config, and the same batch-chunking loop. This
//! module is the single canonicalization point for that shared surface (CLAUDE.md:
//! "one canonicalization point per concept") — a padding change applied here reaches
//! every backend, instead of drifting across three feature-gated copies.

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::core::types::{Embedding, SnippetRef};

use super::EmbeddingError;

/// Bundled `tokenizer.json` for `microsoft/codebert-base` @ CODEBERT_REVISION.
///
/// `microsoft/codebert-base` does not publish `tokenizer.json` on HuggingFace — only
/// `vocab.json` + `merges.txt`. This tokenizer was generated from Python's
/// `AutoTokenizer.from_pretrained()` and spike-validated (100% token ID match vs Python).
/// Single definition shared by all backends (the compiler dedups the embedded bytes).
pub(super) const CODEBERT_TOKENIZER_BYTES: &[u8] = include_bytes!("tokenizer.json");

/// Canonical `microsoft/codebert-base` model name — identifies when to use the bundled tokenizer.
pub(super) const CODEBERT_MODEL: &str = "microsoft/codebert-base";

/// Apply the codebert (RoBERTa) padding + truncation config to an existing tokenizer.
///
/// RoBERTa uses `pad_id=1` and `"<pad>"` (not BERT's 0/`"[PAD]"`). This exact config is
/// what the frozen embedding parity was validated against — do not change it without
/// regenerating the detection baseline.
pub(super) fn configure_codebert_tokenizer(
    mut tokenizer: Tokenizer,
    max_length: usize,
) -> Result<Tokenizer, EmbeddingError> {
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id: 1,
        pad_token: "<pad>".into(),
        ..Default::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| EmbeddingError::Tokenizer(format!("truncation config: {e}")))?;
    Ok(tokenizer)
}

/// Build the bundled codebert tokenizer (used by all backends for the pinned model).
pub(super) fn bundled_codebert_tokenizer(max_length: usize) -> Result<Tokenizer, EmbeddingError> {
    let tokenizer = Tokenizer::from_bytes(CODEBERT_TOKENIZER_BYTES)
        .map_err(|e| EmbeddingError::Tokenizer(format!("bundled tokenizer: {e}")))?;
    configure_codebert_tokenizer(tokenizer, max_length)
}

/// Batch a snippet slice by `batch_size` and concatenate per-batch embeddings, in order.
///
/// Each backend supplies only its inner batch inference via the `embed_batch` closure;
/// the empty-input short-circuit and the `chunks(batch_size)` loop live here once.
pub(super) fn chunked_embed<F>(
    snippets: &[&SnippetRef],
    batch_size: usize,
    mut embed_batch: F,
) -> Result<Vec<Embedding>, EmbeddingError>
where
    F: FnMut(&[&str]) -> Result<Vec<Embedding>, EmbeddingError>,
{
    if snippets.is_empty() {
        return Ok(vec![]);
    }
    let mut result = Vec::with_capacity(snippets.len());
    for chunk in snippets.chunks(batch_size) {
        let texts: Vec<&str> = chunk.iter().map(|s| s.text.as_str()).collect();
        result.extend(embed_batch(&texts)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_tokenizer_parses_with_roberta_vocab() {
        let tok = bundled_codebert_tokenizer(256).expect("bundled tokenizer must build");
        assert!(
            tok.get_vocab_size(false) > 40_000,
            "expected large RoBERTa vocab, got {}",
            tok.get_vocab_size(false)
        );
    }

    #[test]
    fn chunked_embed_empty_is_empty() {
        let out = chunked_embed(&[], 8, |_| unreachable!("should not be called")).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn chunked_embed_batches_in_order() {
        use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind};
        let mk = |t: &str| SnippetRef {
            kind: SnippetKind::Func,
            function: FunctionRef {
                file: FileRef {
                    path: "a".into(),
                    content_hash: "h".into(),
                    language: Language::Python,
                    content: "".into(),
                },
                qualified_name: "f".into(),
                start_line: 1,
                end_line: 1,
                code: t.into(),
                code_hash: "c".into(),
            },
            start_line: 1,
            end_line: 1,
            text: t.into(),
            display_text: t.into(),
            snippet_hash: "s".into(),
        };
        let a = mk("a");
        let b = mk("b");
        let c = mk("c");
        let snips: Vec<&SnippetRef> = vec![&a, &b, &c];
        // batch_size=2 → chunks ["a","b"], ["c"]; embedding = one f32 = text length.
        let out = chunked_embed(&snips, 2, |texts| {
            Ok(texts
                .iter()
                .map(|t| Embedding {
                    vector: vec![t.len() as f32],
                })
                .collect())
        })
        .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].vector, vec![1.0]);
        assert_eq!(out[2].vector, vec![1.0]);
    }
}
