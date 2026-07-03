use std::collections::HashMap;

use rayon::prelude::*;

use crate::core::config::Thresholds;
use crate::core::types::{CandidateMatch, Embedding, SnippetKind, SnippetRef};
use crate::index::VectorIndex;

use super::lexical::lexical_similarity;

/// Per-kind threshold lookup. Matches Python's `_threshold_for_kind` exactly.
/// Uses the NEIGHBOR snippet's kind (snippet_b / `other`), not the query snippet's kind.
fn threshold_for_kind(kind: SnippetKind, thresholds: &Thresholds) -> f64 {
    match kind {
        SnippetKind::Func => thresholds.func,
        SnippetKind::Win => thresholds.win,
        SnippetKind::Exp => thresholds.exp,
    }
}

/// Format the evidence string (DD13).
/// Matches Python: `f"{snip.kind}->{other.kind}|emb={score:.3f}|lex={lexical:.3f}|comp={composite:.3f}"`.
fn format_evidence(
    kind_a: SnippetKind,
    kind_b: SnippetKind,
    emb_score: f64,
    lex_score: f64,
    composite: f64,
) -> String {
    fn kind_str(k: SnippetKind) -> &'static str {
        match k {
            SnippetKind::Func => "FUNC",
            SnippetKind::Win => "WIN",
            SnippetKind::Exp => "EXP",
        }
    }
    format!(
        "{}->{}|emb={:.3}|lex={:.3}|comp={:.3}",
        kind_str(kind_a),
        kind_str(kind_b),
        emb_score,
        lex_score,
        composite,
    )
}

/// Retrieve clone candidates via vector similarity + composite scoring.
///
/// The index is built ONCE and shared read-only across rayon workers (DD4), replacing
/// Python's per-worker index rebuild via multiprocessing. The SET of `(a, b, score)`
/// results is identical run-to-run (determinism); ORDER may vary under rayon scheduling.
///
/// Applies two gates before emitting a `CandidateMatch`:
/// 1. Lexical gate: `lexical >= lexical_min_ratio` (strict less-than exclusion)
/// 2. Per-kind threshold: `composite >= threshold_for_kind(other.kind, ...)`
///
/// Self-hash skip: a snippet is never matched against itself (by snippet_hash equality).
pub(crate) fn retrieve_candidates(
    snippets: &[SnippetRef],
    embeddings: &[Embedding],
    index: &dyn VectorIndex,
    thresholds: &Thresholds,
    top_k: usize,
) -> Vec<CandidateMatch> {
    assert_eq!(
        snippets.len(),
        embeddings.len(),
        "snippets length ({}) != embeddings length ({})",
        snippets.len(),
        embeddings.len()
    );
    if snippets.is_empty() {
        return vec![];
    }

    // Build snippet_hash → index lookup for neighbor resolution.
    let id_to_idx: HashMap<&str, usize> = snippets
        .iter()
        .enumerate()
        .map(|(i, s)| (s.snippet_hash.as_str(), i))
        .collect();

    // Parallel retrieval: rayon par_iter over (snippet, embedding) pairs.
    // The shared `index` is read-only (&dyn VectorIndex: Sync), so all workers can query it.
    let pairs: Vec<(&SnippetRef, &Embedding)> = snippets.iter().zip(embeddings.iter()).collect();

    pairs
        .par_iter()
        .flat_map(|&(snip, emb)| {
            let neighbors = index.query(emb, top_k);
            let mut local_matches = Vec::new();
            for (neighbor_id, emb_score) in neighbors {
                // Self-hash skip: a snippet must not match itself.
                if neighbor_id == snip.snippet_hash {
                    continue;
                }
                let other_idx = match id_to_idx.get(neighbor_id.as_str()) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let other = &snippets[other_idx];

                let lexical = lexical_similarity(&snip.text, &other.text);
                let composite = (1.0 - thresholds.lexical_weight) * emb_score
                    + thresholds.lexical_weight * lexical;

                // Lexical gate (first gate — also applied in rollup).
                if thresholds.lexical_min_ratio > 0.0 && lexical < thresholds.lexical_min_ratio {
                    continue;
                }

                // Per-kind threshold applied to the NEIGHBOR's kind (DD: candidates.py:151).
                let threshold = threshold_for_kind(other.kind, thresholds);
                if composite >= threshold {
                    local_matches.push(CandidateMatch {
                        snippet_a: snip.clone(),
                        snippet_b: other.clone(),
                        similarity: composite,
                        evidence: format_evidence(
                            snip.kind, other.kind, emb_score, lexical, composite,
                        ),
                    });
                }
            }
            local_matches
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Thresholds;
    use crate::core::types::{Embedding, FileRef, FunctionRef, Language, SnippetKind, SnippetRef};
    use crate::index::BruteIndex;

    fn make_file() -> FileRef {
        FileRef {
            path: "x.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
        }
    }

    fn make_func(file: &FileRef, qname: &str) -> FunctionRef {
        FunctionRef {
            file: file.clone(),
            qualified_name: qname.into(),
            start_line: 1,
            end_line: 5,
            code: "pass".into(),
            code_hash: qname.into(),
        }
    }

    fn make_snip(kind: SnippetKind, func: &FunctionRef, text: &str, hash: &str) -> SnippetRef {
        SnippetRef {
            kind,
            function: func.clone(),
            start_line: 1,
            end_line: 5,
            text: text.into(),
            display_text: text.into(),
            snippet_hash: hash.into(),
        }
    }

    fn emb(v: Vec<f32>) -> Embedding {
        let dim = v.len();
        Embedding { vector: v, dim }
    }

    fn default_thresholds() -> Thresholds {
        Thresholds {
            func: 0.5,
            win: 0.5,
            exp: 0.5,
            min_window_hits: 1,
            lexical_min_ratio: 0.0,
            lexical_weight: 0.0,
        }
    }

    fn build_index(snippets: &[SnippetRef], embeddings: &[Embedding]) -> BruteIndex {
        let mut idx = BruteIndex::new();
        let ids: Vec<String> = snippets.iter().map(|s| s.snippet_hash.clone()).collect();
        idx.build(embeddings, &ids);
        idx
    }

    #[test]
    fn test_retrieve_candidates_empty() {
        let idx = BruteIndex::new();
        let results = retrieve_candidates(&[], &[], &idx, &default_thresholds(), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_candidates_self_hash_skip() {
        // A single snippet should never match itself.
        let file = make_file();
        let func = make_func(&file, "f");
        let snip = make_snip(SnippetKind::Func, &func, "def foo(): pass", "hash_a");
        let embeddings = vec![emb(vec![1.0, 0.0])];
        let idx = build_index(&[snip.clone()], &embeddings);
        let results = retrieve_candidates(&[snip], &embeddings, &idx, &default_thresholds(), 5);
        assert!(results.is_empty(), "snippet must not match itself");
    }

    #[test]
    fn test_retrieve_candidates_basic_match() {
        // Two snippets with identical embeddings → cosine=1.0, should produce a match.
        let file = make_file();
        let func = make_func(&file, "f");
        let s1 = make_snip(SnippetKind::Func, &func, "def foo(): pass", "hash_a");
        let s2 = make_snip(SnippetKind::Func, &func, "def foo(): pass", "hash_b");
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![1.0, 0.0])];
        let idx = build_index(&[s1.clone(), s2.clone()], &embeddings);
        let results = retrieve_candidates(&[s1, s2], &embeddings, &idx, &default_thresholds(), 5);
        assert!(
            !results.is_empty(),
            "identical embeddings must produce candidates"
        );
    }

    #[test]
    fn test_retrieve_candidates_lexical_gate() {
        // Two snippets with high embedding similarity but completely disjoint tokens.
        // With lexical_min_ratio=0.9, they should be filtered out.
        let file = make_file();
        let func = make_func(&file, "f");
        let s1 = make_snip(SnippetKind::Func, &func, "alpha beta gamma", "hash_a");
        let s2 = make_snip(SnippetKind::Func, &func, "delta epsilon zeta", "hash_b");
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![1.0, 0.0])];
        let idx = build_index(&[s1.clone(), s2.clone()], &embeddings);
        let strict = Thresholds {
            func: 0.5,
            win: 0.5,
            exp: 0.5,
            min_window_hits: 1,
            lexical_min_ratio: 0.9,
            lexical_weight: 0.0,
        };
        let results = retrieve_candidates(&[s1, s2], &embeddings, &idx, &strict, 5);
        assert!(
            results.is_empty(),
            "disjoint tokens must be filtered by lexical gate"
        );
    }

    #[test]
    fn test_evidence_format() {
        // Verify DD13 evidence string format.
        let ev = format_evidence(SnippetKind::Func, SnippetKind::Win, 0.950, 0.800, 0.905);
        assert_eq!(ev, "FUNC->WIN|emb=0.950|lex=0.800|comp=0.905");
    }

    #[test]
    fn test_threshold_applied_to_neighbor_kind_not_query_kind() {
        // Threshold must come from OTHER.kind (snippet_b/neighbor), not the query snippet's kind.
        // Set func_threshold impossible (1.5) and win_threshold passable (0.5).
        // When a FUNC snippet queries a WIN neighbor: threshold = win = 0.5 → passes.
        // If threshold were from query kind (FUNC, 1.5), no match would be found.
        let file = make_file();
        let func = make_func(&file, "f");
        let s1 = make_snip(SnippetKind::Func, &func, "def foo(): pass", "hash_a");
        let s2 = make_snip(SnippetKind::Win, &func, "def foo(): pass", "hash_b");
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![1.0, 0.0])];
        let idx = build_index(&[s1.clone(), s2.clone()], &embeddings);
        let thresholds = Thresholds {
            func: 1.5, // impossible — query kind threshold, must NOT be used
            win: 0.5,  // neighbor kind threshold, must be used
            exp: 1.5,
            min_window_hits: 1,
            lexical_min_ratio: 0.0,
            lexical_weight: 0.0,
        };
        let results = retrieve_candidates(&[s1, s2], &embeddings, &idx, &thresholds, 5);
        // s1 (FUNC) → s2 (WIN): uses win threshold (0.5) → composite 1.0 ≥ 0.5 → match found
        let func_to_win = results
            .iter()
            .any(|m| m.snippet_a.kind == SnippetKind::Func && m.snippet_b.kind == SnippetKind::Win);
        assert!(
            func_to_win,
            "threshold must use neighbor (WIN) kind, not query (FUNC) kind"
        );
    }

    #[test]
    fn test_lexical_weight_affects_composite_score() {
        // lexical_weight=0.5 with disjoint tokens should halve composite vs weight=0.0.
        // With embedding cosine=1.0 and lexical=0.0:
        //   weight=0.0 → composite=1.0 ≥ threshold(0.9) → match found
        //   weight=0.5 → composite=0.5 < threshold(0.9) → no match
        let file = make_file();
        let func = make_func(&file, "f");
        let s1 = make_snip(SnippetKind::Func, &func, "alpha beta gamma", "hash_a");
        let s2 = make_snip(SnippetKind::Func, &func, "delta epsilon zeta", "hash_b");
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![1.0, 0.0])];
        let idx = build_index(&[s1.clone(), s2.clone()], &embeddings);
        let no_lex_weight = Thresholds {
            func: 0.9,
            win: 0.9,
            exp: 0.9,
            min_window_hits: 1,
            lexical_min_ratio: 0.0,
            lexical_weight: 0.0, // composite = 1.0*emb → 1.0 ≥ 0.9 → match
        };
        let with_lex_weight = Thresholds {
            func: 0.9,
            win: 0.9,
            exp: 0.9,
            min_window_hits: 1,
            lexical_min_ratio: 0.0,
            lexical_weight: 0.5, // composite = 0.5*1.0 + 0.5*0.0 = 0.5 < 0.9 → no match
        };
        let without = retrieve_candidates(
            &[s1.clone(), s2.clone()],
            &embeddings,
            &idx,
            &no_lex_weight,
            5,
        );
        let with_w = retrieve_candidates(&[s1, s2], &embeddings, &idx, &with_lex_weight, 5);
        assert!(
            !without.is_empty(),
            "with lexical_weight=0 and embedding cosine=1.0, should find matches"
        );
        assert!(
            with_w.is_empty(),
            "with lexical_weight=0.5 and disjoint tokens, composite=0.5 < threshold=0.9"
        );
    }
}
