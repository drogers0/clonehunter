use crate::core::types::CandidateMatch;

/// Maximum similarity across a group of candidate matches.
/// Returns 0.0 for an empty iterator, matching Python's `best_score`.
///
/// Generic over any iterator of `&CandidateMatch`, so both `&[CandidateMatch]` (owned slices)
/// and reference collections (`Vec<&CandidateMatch>` via `.iter().copied()`) share one impl.
pub(crate) fn best_score<'a, I>(matches: I) -> f64
where
    I: IntoIterator<Item = &'a CandidateMatch>,
{
    matches
        .into_iter()
        .map(|m| m.similarity)
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{make_match as build_match, make_snippet};

    fn make_match(sim: f64) -> CandidateMatch {
        let snip = make_snippet("t");
        build_match(snip.clone(), snip, sim)
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
    fn test_best_score_over_refs() {
        // Same generic accepts a Vec<&CandidateMatch> via .iter().copied().
        let m1 = make_match(0.7);
        let m2 = make_match(0.95);
        let refs = [&m1, &m2];
        assert!((best_score(refs.iter().copied()) - 0.95).abs() < 1e-9);
    }
}
