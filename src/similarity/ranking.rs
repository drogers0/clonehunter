#![allow(dead_code)] // T10 (pipeline) will wire these up

use crate::core::types::{CandidateMatch, SnippetKind, SnippetRef};

fn span_len(snippet: &SnippetRef) -> usize {
    snippet.end_line.saturating_sub(snippet.start_line) + 1
}

/// Rank a candidate match by snippet-kind pair (DD9).
/// FUNC+FUNC=3, FUNC+anything=2, WIN+WIN=1, EXP+EXP or mixed-EXP=0.
pub(crate) fn kind_rank(m: &CandidateMatch) -> i32 {
    let a = m.snippet_a.kind;
    let b = m.snippet_b.kind;
    if a == SnippetKind::Func && b == SnippetKind::Func {
        return 3;
    }
    if a == SnippetKind::Func || b == SnippetKind::Func {
        return 2;
    }
    if a == SnippetKind::Win && b == SnippetKind::Win {
        return 1;
    }
    0
}

/// Select the representative evidence match from a group (DD9).
///
/// Returns the match with the maximum rank key — a 7-tuple matching Python's exact tuple:
/// `(kind_rank, min(len_a, len_b), similarity, -start_a, -end_a, -start_b, -end_b)`.
///
/// Uses `reduce` (not `max_by_key`) so that on ties the FIRST maximum is kept, matching
/// Python's `max(matches, key=_rank)` which also returns the first maximal element.
/// f64 similarity is converted to u64 via `to_bits()` for total ordering — safe because
/// all candidates have already passed threshold gates so similarity is in [0.0, 1.0].
pub(crate) fn best_match(matches: &[CandidateMatch]) -> Option<&CandidateMatch> {
    matches.iter().reduce(|best, m| {
        if rank_key(m) > rank_key(best) {
            m
        } else {
            best
        }
    })
}

/// Comparable rank key. Matches Python's `_rank` tuple exactly (DD9).
fn rank_key(m: &CandidateMatch) -> (i32, usize, u64, i64, i64, i64, i64) {
    let len_a = span_len(&m.snippet_a);
    let len_b = span_len(&m.snippet_b);
    (
        kind_rank(m),
        len_a.min(len_b),
        // to_bits() gives a monotone u64 for non-negative f64 (all thresholded scores ≥ 0)
        m.similarity.to_bits(),
        -(m.snippet_a.start_line as i64),
        -(m.snippet_a.end_line as i64),
        -(m.snippet_b.start_line as i64),
        -(m.snippet_b.end_line as i64),
    )
}

// Helper for tests: returns a pair of snippet hashes for comparison.
impl CandidateMatch {
    #[cfg(test)]
    fn snippet_hash_pair(&self) -> (String, String) {
        (
            self.snippet_a.snippet_hash.clone(),
            self.snippet_b.snippet_hash.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{
        CandidateMatch, FileRef, FunctionRef, Language, SnippetKind, SnippetRef,
    };

    fn make_file(path: &str) -> FileRef {
        FileRef {
            path: path.into(),
            content_hash: "h".into(),
            language: Language::Python,
        }
    }

    fn make_func(file: &FileRef, qname: &str, start: usize, end: usize) -> FunctionRef {
        FunctionRef {
            file: file.clone(),
            qualified_name: qname.into(),
            start_line: start,
            end_line: end,
            code: "pass".into(),
            code_hash: qname.into(),
        }
    }

    fn make_snip(
        kind: SnippetKind,
        func: &FunctionRef,
        start: usize,
        end: usize,
        hash: &str,
    ) -> SnippetRef {
        SnippetRef {
            kind,
            function: func.clone(),
            start_line: start,
            end_line: end,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: hash.into(),
        }
    }

    fn make_match(snip_a: SnippetRef, snip_b: SnippetRef, sim: f64) -> CandidateMatch {
        CandidateMatch {
            snippet_a: snip_a,
            snippet_b: snip_b,
            similarity: sim,
            evidence: "".into(),
        }
    }

    #[test]
    fn test_kind_rank_values() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 10);
        let f = make_snip(SnippetKind::Func, &func, 1, 10, "f");
        let w = make_snip(SnippetKind::Win, &func, 1, 5, "w");
        let e = make_snip(SnippetKind::Exp, &func, 1, 8, "e");

        assert_eq!(kind_rank(&make_match(f.clone(), f.clone(), 0.9)), 3);
        assert_eq!(kind_rank(&make_match(f.clone(), w.clone(), 0.9)), 2);
        assert_eq!(kind_rank(&make_match(w.clone(), f.clone(), 0.9)), 2);
        assert_eq!(kind_rank(&make_match(w.clone(), w.clone(), 0.9)), 1);
        assert_eq!(kind_rank(&make_match(e.clone(), e.clone(), 0.9)), 0);
        assert_eq!(kind_rank(&make_match(w.clone(), e.clone(), 0.9)), 0);
    }

    #[test]
    fn test_best_match_empty_returns_none() {
        assert!(best_match(&[]).is_none());
    }

    #[test]
    fn test_best_match_prefers_func_func_over_win_win() {
        let file = make_file("x.py");
        let fa = make_func(&file, "a", 1, 10);
        let fb = make_func(&file, "b", 20, 30);
        let f1 = make_snip(SnippetKind::Func, &fa, 1, 10, "f1");
        let f2 = make_snip(SnippetKind::Func, &fb, 20, 30, "f2");
        let w1 = make_snip(SnippetKind::Win, &fa, 1, 5, "w1");
        let w2 = make_snip(SnippetKind::Win, &fb, 20, 25, "w2");

        let func_match = make_match(f1, f2, 0.91);
        let win_match = make_match(w1, w2, 0.99); // higher sim but lower kind_rank
        let matches = vec![win_match.clone(), func_match.clone()];
        let best = best_match(&matches).unwrap();
        assert_eq!(best.snippet_a.kind, SnippetKind::Func);
    }

    #[test]
    fn test_best_match_order_independent() {
        // Same matches in two different orders must yield the same result.
        let file = make_file("x.py");
        let fa = make_func(&file, "a", 1, 10);
        let fb = make_func(&file, "b", 20, 30);
        let f1 = make_snip(SnippetKind::Func, &fa, 1, 10, "f1");
        let f2 = make_snip(SnippetKind::Func, &fb, 20, 30, "f2");
        let w1 = make_snip(SnippetKind::Win, &fa, 1, 5, "w1");
        let w2 = make_snip(SnippetKind::Win, &fb, 20, 25, "w2");

        let m1 = make_match(f1, f2, 0.95);
        let m2 = make_match(w1, w2, 0.92);

        let result_a = best_match(&[m1.clone(), m2.clone()])
            .unwrap()
            .snippet_hash_pair();
        let result_b = best_match(&[m2, m1]).unwrap().snippet_hash_pair();
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn test_best_match_tie_break_prefers_lower_start_line() {
        // Same kind_rank, same min_len, same similarity → prefer lower start_line.
        let file = make_file("x.py");
        let fa = make_func(&file, "a", 1, 10);
        let fb = make_func(&file, "b", 20, 30);
        let s1 = make_snip(SnippetKind::Win, &fa, 1, 5, "s1"); // start=1
        let s2 = make_snip(SnippetKind::Win, &fa, 3, 7, "s2"); // start=3 (higher → prefer s1)
        let t1 = make_snip(SnippetKind::Win, &fb, 20, 24, "t1");

        let m1 = make_match(s1.clone(), t1.clone(), 0.95);
        let m2 = make_match(s2, t1, 0.95);
        // m1 and m2 have same kind_rank=1, same min_len=5, same sim=0.95.
        // Tie-break: -start_a is larger (less negative) for lower start → m1 wins.
        let arr = [m2, m1];
        let best = best_match(&arr).unwrap();
        assert_eq!(best.snippet_a.start_line, 1);
    }
}
