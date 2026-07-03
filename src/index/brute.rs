// allow dead_code until T12 wires cli::run to the pipeline
#![allow(dead_code)]

use ndarray::{Array1, Array2};

use crate::core::types::Embedding;

use super::VectorIndex;

/// Brute-force cosine index backed by an (N, D) f32 matrix.
///
/// All arithmetic is done in f32 to match Python's `np.float32` precision (DD1).
/// Scores are widened to f64 only at the `query()` return boundary.
pub(crate) struct BruteIndex {
    ids: Vec<String>,
    /// Row-major matrix of embedding vectors, f32 precision to match Python.
    matrix: Option<Array2<f32>>,
    /// Precomputed L2 norms per row (zero norms replaced with 1.0 to avoid div-by-zero).
    norms: Option<Array1<f32>>,
}

impl BruteIndex {
    pub(crate) fn new() -> Self {
        Self {
            ids: Vec::new(),
            matrix: None,
            norms: None,
        }
    }
}

impl VectorIndex for BruteIndex {
    fn build(&mut self, vectors: &[Embedding], ids: &[String]) {
        assert_eq!(
            vectors.len(),
            ids.len(),
            "vectors length ({}) != ids length ({})",
            vectors.len(),
            ids.len()
        );
        self.ids = ids.to_vec();
        if vectors.is_empty() {
            self.matrix = None;
            self.norms = None;
            return;
        }
        let dim = vectors[0].dim;
        let n = vectors.len();
        // Build (N, D) matrix — f32 to match Python's np.float32
        let mut data = Vec::with_capacity(n * dim);
        for v in vectors {
            data.extend_from_slice(&v.vector);
        }
        let matrix = Array2::from_shape_vec((n, dim), data)
            .expect("embedding dimensions must be consistent");
        // Compute L2 norms per row: norm = sqrt(row · row)
        let norms_raw: Array1<f32> = matrix
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt())
            .collect();
        // Replace zero norms with 1.0 to avoid division by zero
        let norms = norms_raw.mapv(|v| if v == 0.0 { 1.0 } else { v });
        self.matrix = Some(matrix);
        self.norms = Some(norms);
    }

    /// Cosine similarity query.
    ///
    /// Computes `dots = matrix @ q`, `scores = dots / (norms * norm_q)` — all in f32
    /// (matching Python), then stable-descending sort (matching `np.argsort(-scores,
    /// kind="stable")`). Scores are widened to f64 at the return boundary (DD1).
    fn query(&self, vector: &Embedding, k: usize) -> Vec<(String, f64)> {
        if k == 0 {
            return vec![];
        }
        let (matrix, norms) = match (&self.matrix, &self.norms) {
            (Some(m), Some(n)) if !self.ids.is_empty() => (m, n),
            _ => return vec![],
        };

        // Build query vector as f32 Array1
        let q = Array1::from_vec(vector.vector.clone());
        let norm_q: f32 = q.dot(&q).sqrt();
        let norm_q = if norm_q == 0.0 { 1.0_f32 } else { norm_q };

        // matrix @ q — (N, D) × (D,) → (N,) dot products, all in f32
        let dots: Array1<f32> = matrix.dot(&q);

        // scores[i] = dots[i] / (norms[i] * norm_q) — element-wise, f32
        let scores: Array1<f32> = dots
            .iter()
            .zip(norms.iter())
            .map(|(&d, &n)| d / (n * norm_q))
            .collect();

        // Stable descending sort: matches Python's np.argsort(-scores, kind="stable").
        // Rust's sort_by is guaranteed stable; equal scores preserve original index order.
        // partial_cmp (not total_cmp) so -0.0 and +0.0 compare as Equal, matching numpy.
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top-k; cast each score to f64 at output boundary (DD1)
        let top_k = k.min(indices.len());
        indices[..top_k]
            .iter()
            .map(|&i| (self.ids[i].clone(), scores[i] as f64))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Embedding;

    fn emb(v: Vec<f32>) -> Embedding {
        let dim = v.len();
        Embedding { vector: v, dim }
    }

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| i.to_string()).collect()
    }

    #[test]
    fn test_brute_empty_build_query_returns_empty() {
        let mut idx = BruteIndex::new();
        idx.build(&[], &[]);
        assert!(idx.query(&emb(vec![1.0, 0.0]), 5).is_empty());
    }

    #[test]
    fn test_brute_k_zero_returns_empty() {
        let mut idx = BruteIndex::new();
        idx.build(&[emb(vec![1.0, 0.0])], &ids(1));
        assert!(idx.query(&emb(vec![1.0, 0.0]), 0).is_empty());
    }

    #[test]
    fn test_brute_cosine_exact_orthogonal() {
        // [1,0] vs [0,1]: cosine similarity = 0; [1,0] vs [1,0]: = 1
        let mut idx = BruteIndex::new();
        idx.build(
            &[emb(vec![1.0, 0.0]), emb(vec![0.0, 1.0])],
            &["a".into(), "b".into()],
        );
        let results = idx.query(&emb(vec![1.0, 0.0]), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(results[1].0, "b");
        assert!((results[1].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_brute_stable_sort_tie_preserves_insertion_order() {
        // Three identical vectors → all cosine similarities = 1.0.
        // Stable sort must preserve insertion order: 0, 1, 2.
        let mut idx = BruteIndex::new();
        idx.build(
            &[
                emb(vec![1.0, 0.0]),
                emb(vec![1.0, 0.0]),
                emb(vec![1.0, 0.0]),
            ],
            &["0".into(), "1".into(), "2".into()],
        );
        let results = idx.query(&emb(vec![1.0, 0.0]), 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "0");
        assert_eq!(results[1].0, "1");
        assert_eq!(results[2].0, "2");
    }

    #[test]
    fn test_brute_zero_norm_vector_no_panic() {
        // A zero embedding vector should produce norm=1.0 (clamp), no division by zero.
        let mut idx = BruteIndex::new();
        idx.build(
            &[emb(vec![0.0, 0.0]), emb(vec![1.0, 0.0])],
            &["zero".into(), "one".into()],
        );
        let results = idx.query(&emb(vec![1.0, 0.0]), 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_brute_k_larger_than_corpus_returns_all() {
        let mut idx = BruteIndex::new();
        idx.build(&[emb(vec![1.0, 0.0]), emb(vec![0.0, 1.0])], &ids(2));
        let results = idx.query(&emb(vec![1.0, 0.0]), 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_brute_basic_ranking() {
        // Three vectors at different angles; query [1,1] should rank nearest first.
        let mut idx = BruteIndex::new();
        let rt2 = (2.0f32).sqrt() / 2.0;
        idx.build(
            &[
                emb(vec![1.0, 0.0]),  // cosine([1,1], [1,0]) = 1/√2
                emb(vec![0.0, 1.0]),  // cosine([1,1], [0,1]) = 1/√2
                emb(vec![-1.0, 0.0]), // cosine([1,1], [-1,0]) = -1/√2
            ],
            &["a".into(), "b".into(), "c".into()],
        );
        let results = idx.query(&emb(vec![1.0, 1.0]), 3);
        // Both 'a' and 'b' have score 1/√2; 'c' has -1/√2. Stable sort puts a before b.
        assert_eq!(results.len(), 3);
        assert!((results[0].1 - rt2 as f64).abs() < 1e-5);
        assert!((results[1].1 - rt2 as f64).abs() < 1e-5);
        assert!(results[2].1 < 0.0);
        // Stable sort: 'a' (index 0) before 'b' (index 1) since equal scores
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "b");
        assert_eq!(results[2].0, "c");
    }
}
