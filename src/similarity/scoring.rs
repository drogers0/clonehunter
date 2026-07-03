use crate::core::types::CandidateMatch;

/// Maximum similarity across a group of candidate matches.
/// Returns 0.0 for an empty slice, matching Python's `best_score`.
pub(crate) fn best_score(matches: &[CandidateMatch]) -> f64 {
    matches.iter().map(|m| m.similarity).fold(0.0_f64, f64::max)
}

/// Variant accepting a slice of references (avoids cloning in `compute_reasons`).
pub(super) fn best_score_refs(matches: &[&CandidateMatch]) -> f64 {
    matches.iter().map(|m| m.similarity).fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{
        CandidateMatch, FileRef, FunctionRef, Language, SnippetKind, SnippetRef,
    };

    fn make_match(sim: f64) -> CandidateMatch {
        let file = FileRef {
            path: "x.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
        };
        let func = FunctionRef {
            file,
            qualified_name: "f".into(),
            start_line: 1,
            end_line: 5,
            code: "pass".into(),
            code_hash: "c".into(),
        };
        let snip = SnippetRef {
            kind: SnippetKind::Func,
            function: func,
            start_line: 1,
            end_line: 5,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: "h".into(),
        };
        CandidateMatch {
            snippet_a: snip.clone(),
            snippet_b: snip,
            similarity: sim,
            evidence: "".into(),
        }
    }

    #[test]
    fn test_best_score_empty() {
        assert_eq!(best_score(&[]), 0.0);
    }

    #[test]
    fn test_best_score_max() {
        let matches = vec![make_match(0.7), make_match(0.95), make_match(0.8)];
        assert!((best_score(&matches) - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_best_score_refs_empty() {
        assert_eq!(best_score_refs(&[]), 0.0);
    }

    #[test]
    fn test_best_score_refs_max() {
        let m1 = make_match(0.7);
        let m2 = make_match(0.95);
        let refs = vec![&m1, &m2];
        assert!((best_score_refs(&refs) - 0.95).abs() < 1e-9);
    }
}
