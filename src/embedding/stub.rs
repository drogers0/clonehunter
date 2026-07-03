use sha2::{Digest, Sha256};

use crate::core::types::{Embedding, SnippetRef};

use super::{Embedder, EmbeddingError};

/// Deterministic embedder for tests and local smoke runs.
///
/// Produces 16-dim (default) embeddings via SHA-256 → normalize.
/// No candle/torch dependency — safe to use in all test environments.
///
/// Algorithm matches Python's `StubEmbedder` exactly (DD11):
/// - Digest first `dim` bytes of SHA-256(text.as_bytes())
/// - Divide each byte by 255.0 **in f64** (Python `float` is f64)
/// - L2 normalize in f64
/// - Cast to f32 for storage
pub(crate) struct StubEmbedder {
    dim: usize,
}

impl StubEmbedder {
    pub(crate) fn new(dim: usize) -> Self {
        debug_assert!(
            dim <= 32,
            "StubEmbedder dim must be ≤ 32 (SHA-256 digest length)"
        );
        Self { dim }
    }

    fn embed_one(&self, text: &str) -> Embedding {
        let digest = Sha256::digest(text.as_bytes());
        // Python: [b / 255.0 for b in digest[:dim]] — Python float is f64
        let values: Vec<f64> = digest[..self.dim]
            .iter()
            .map(|&b| b as f64 / 255.0)
            .collect();
        let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm = if norm == 0.0 { 1.0 } else { norm };
        let vector: Vec<f32> = values.iter().map(|v| (*v / norm) as f32).collect();
        Embedding {
            vector,
            dim: self.dim,
        }
    }
}

impl Embedder for StubEmbedder {
    fn embed(&self, snippets: &[&SnippetRef]) -> Result<Vec<Embedding>, EmbeddingError> {
        Ok(snippets.iter().map(|s| self.embed_one(&s.text)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{FileRef, FunctionRef, Language, SnippetKind};

    fn make_snippet(text: &str) -> SnippetRef {
        let file = FileRef {
            path: "test.py".into(),
            content_hash: "abc".into(),
            language: Language::Python,
        };
        let func = FunctionRef {
            file,
            qualified_name: "test_func".into(),
            start_line: 1,
            end_line: 5,
            code: text.into(),
            code_hash: "abc".into(),
        };
        SnippetRef {
            kind: SnippetKind::Func,
            function: func,
            start_line: 1,
            end_line: 5,
            text: text.into(),
            display_text: text.into(),
            snippet_hash: crate::io::fingerprints::hash_text(text),
        }
    }

    #[test]
    fn stub_deterministic() {
        let stub = StubEmbedder::new(16);
        let s = make_snippet("def f(): pass");
        let r1 = stub.embed(&[&s]).unwrap();
        let r2 = stub.embed(&[&s]).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn stub_different_texts() {
        let stub = StubEmbedder::new(16);
        let s1 = make_snippet("def foo(): pass");
        let s2 = make_snippet("def bar(): return 1");
        let r1 = stub.embed(&[&s1]).unwrap();
        let r2 = stub.embed(&[&s2]).unwrap();
        assert_ne!(r1[0].vector, r2[0].vector);
    }

    #[test]
    fn stub_dim_16() {
        let stub = StubEmbedder::new(16);
        let s = make_snippet("x = 1");
        let result = stub.embed(&[&s]).unwrap();
        assert_eq!(result[0].dim, 16);
        assert_eq!(result[0].vector.len(), 16);
    }

    #[test]
    fn stub_values_match_python() {
        // Python reference:
        //   import hashlib, math
        //   text = "def f(): pass"
        //   digest = hashlib.sha256(text.encode("utf-8")).digest()
        //   values = [b / 255.0 for b in digest[:16]]
        //   norm = math.sqrt(sum(v*v for v in values))
        //   normalized = [v/norm for v in values]
        let stub = StubEmbedder::new(16);
        let text = "def f(): pass";
        let s = make_snippet(text);
        let result = stub.embed(&[&s]).unwrap();

        // Compute expected the same way as Python
        let digest = Sha256::digest(text.as_bytes());
        let values: Vec<f64> = digest[..16].iter().map(|&b| b as f64 / 255.0).collect();
        let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
        let expected: Vec<f32> = values.iter().map(|v| (*v / norm) as f32).collect();

        for (got, exp) in result[0].vector.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-7, "got={got} exp={exp}");
        }
    }

    #[test]
    fn stub_normalized() {
        let stub = StubEmbedder::new(16);
        let s = make_snippet("def something(): return 42");
        let result = stub.embed(&[&s]).unwrap();
        let norm: f32 = result[0].vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "L2 norm = {norm}");
    }

    #[test]
    fn stub_empty_input() {
        let stub = StubEmbedder::new(16);
        let result = stub.embed(&[]).unwrap();
        assert!(result.is_empty());
    }
}
