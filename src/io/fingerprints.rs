use sha2::{Digest, Sha256};

/// SHA-256 hex digest of UTF-8 encoded text.
/// Matches Python: `hashlib.sha256(text.encode("utf-8")).hexdigest()`
pub(crate) fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

/// Cache key for embeddings.
/// Format: `sha256("{backend}:{model_name}:{model_revision}:{max_tokens}:{snippet_hash}")`.
///
/// The `backend` discriminator (codebert/onnx/mlx/stub) IS keyed: the backends run the same
/// model but are not bit-identical, so sharing entries across them would silently return
/// another backend's vectors on a warm cache. Device and batch_size are NOT keyed — same
/// embeddings regardless of CPU/CUDA or batch size (matches Python).
pub(crate) fn embed_cache_key(
    backend: &str,
    model_name: &str,
    model_revision: &str,
    max_tokens: usize,
    snippet_hash: &str,
) -> String {
    let payload = format!("{backend}:{model_name}:{model_revision}:{max_tokens}:{snippet_hash}");
    hash_text(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_text_sha256() {
        // Verify against known Python output:
        // hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert_eq!(
            hash_text("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn hash_text_empty() {
        // hashlib.sha256(b"").hexdigest()
        assert_eq!(
            hash_text(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn embed_cache_key_format() {
        let key = embed_cache_key("codebert", "microsoft/codebert-base", "main", 256, "abc123");
        let expected = hash_text("codebert:microsoft/codebert-base:main:256:abc123");
        assert_eq!(key, expected);
    }

    #[test]
    fn embed_cache_key_excludes_device_and_batch() {
        // Exclusion is structural: device and batch_size are not parameters to embed_cache_key.
        let k1 = embed_cache_key("codebert", "m", "r", 256, "s");
        let k2 = embed_cache_key("codebert", "m", "r", 256, "s");
        assert_eq!(k1, k2);
    }

    #[test]
    fn embed_cache_key_changes_with_revision() {
        let k1 = embed_cache_key("codebert", "m", "rev1", 256, "s");
        let k2 = embed_cache_key("codebert", "m", "rev2", 256, "s");
        assert_ne!(k1, k2);
    }

    #[test]
    fn embed_cache_key_changes_with_backend() {
        // Different backends must NOT share cache entries (they are not bit-identical).
        let k1 = embed_cache_key("codebert", "m", "r", 256, "s");
        let k2 = embed_cache_key("mlx", "m", "r", 256, "s");
        assert_ne!(k1, k2);
    }
}
