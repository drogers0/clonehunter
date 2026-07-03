// allow dead_code until T12 wires cli::run to the pipeline
#![allow(dead_code)]

mod brute;

use crate::core::config::IndexName;
use crate::core::types::{Degradation, DegradationKind, Embedding};

pub(crate) use brute::BruteIndex;

/// Vector index trait — the one interface actually subclassed (per CLAUDE.md).
/// `Send + Sync` because rayon workers share a read-only index (DD4).
pub(crate) trait VectorIndex: Send + Sync {
    /// Build the index from embedding vectors and their corresponding IDs.
    fn build(&mut self, vectors: &[Embedding], ids: &[String]);

    /// Query the index for the top-k nearest neighbors of `vector`.
    /// Returns `(id, cosine_similarity)` pairs in descending score order.
    fn query(&self, vector: &Embedding, k: usize) -> Vec<(String, f64)>;
}

/// Create a vector index for the given config name.
/// `IndexName::Faiss` returns a `BruteIndex` + a `Degradation` (DD2):
/// no ANN backend is available in this Rust port, so faiss silently degrades to brute.
pub(crate) fn create_index(name: IndexName) -> (Box<dyn VectorIndex>, Vec<Degradation>) {
    match name {
        IndexName::Brute => (Box::new(BruteIndex::new()), vec![]),
        IndexName::Faiss => {
            let degradation = Degradation {
                kind: DegradationKind::IndexFallback,
                message: "--index faiss requested but no ANN backend available; \
                          falling back to brute"
                    .into(),
            };
            (Box::new(BruteIndex::new()), vec![degradation])
        }
    }
}
