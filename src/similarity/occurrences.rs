use std::collections::HashMap;

use crate::core::types::CandidateMatch;

type Interval = (usize, usize);

/// Returns `true` if all matches in the group share the same function identity (self-clone).
pub(crate) fn is_self_clone(matches: &[CandidateMatch]) -> bool {
    matches
        .first()
        .is_some_and(|m| m.snippet_a.function.identity() == m.snippet_b.function.identity())
}

/// Merge strictly overlapping spans into contiguous intervals.
///
/// Uses STRICT overlap (`start <= prev_end`), NOT adjacency — two touching-but-distinct
/// occurrences are never collapsed. This differs from `covered_lines` in rollup which
/// uses adjacency (`start <= prev_end + 1`). See DD11.
fn merge_overlapping(spans: &[(usize, usize)]) -> Vec<Interval> {
    let mut sorted: Vec<(usize, usize)> = spans.to_vec();
    sorted.sort_unstable();
    let mut merged: Vec<Interval> = Vec::new();
    for (start, end) in sorted {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                // Strict overlap: extend if needed
                if end > last.1 {
                    last.1 = end;
                }
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

/// Connected occurrence-components for a self-clone group.
///
/// Pools every snippet span from both sides of a self-clone group's matches, merges
/// strictly-overlapping spans into physical "occurrences", then unions the occurrences
/// connected by a match into components (union-find with path halving). This resolves
/// N-way chained self-clones (R1↔R2↔R3) into the correct global partition.
pub(crate) struct SelfCloneOccurrences {
    occ: Vec<Interval>,
    parent: Vec<usize>,
}

impl SelfCloneOccurrences {
    pub(crate) fn new(matches: &[CandidateMatch]) -> Self {
        // Pool all spans from both sides via a HashSet to deduplicate.
        let mut span_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for m in matches {
            span_set.insert((m.snippet_a.start_line, m.snippet_a.end_line));
            span_set.insert((m.snippet_b.start_line, m.snippet_b.end_line));
        }
        let spans_vec: Vec<(usize, usize)> = span_set.into_iter().collect();
        let occ = merge_overlapping(&spans_vec);
        let parent: Vec<usize> = (0..occ.len()).collect();
        let mut inst = Self { occ, parent };
        // Union occurrences connected by each match.
        for m in matches {
            let ia = inst.index_of(m.snippet_a.start_line, m.snippet_a.end_line);
            let ib = inst.index_of(m.snippet_b.start_line, m.snippet_b.end_line);
            inst.union(ia, ib);
        }
        inst
    }

    /// Linear scan to find the merged occurrence containing a given span.
    /// Returns `None` for spans that don't map to any occurrence (unreachable for well-formed data).
    fn index_of(&self, start: usize, end: usize) -> Option<usize> {
        self.occ
            .iter()
            .position(|&(o_start, o_end)| o_start <= start && end <= o_end)
    }

    /// Union-find with path halving (exact match of Python's `_find`).
    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]]; // path halving
            i = self.parent[i];
        }
        i
    }

    /// Union by minimum index (smaller index becomes the root). No-op if either span is unmapped.
    fn union(&mut self, a: Option<usize>, b: Option<usize>) {
        let (Some(a), Some(b)) = (a, b) else {
            return;
        };
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra.max(rb)] = ra.min(rb);
        }
    }

    /// The merged occurrence that contains the given span.
    pub(crate) fn occurrence_for(&self, start: usize, end: usize) -> Interval {
        match self.index_of(start, end) {
            Some(i) => self.occ[i],
            None => (start, end),
        }
    }

    /// `sum(lengths) - max(length)` per connected component — the duplicated line count.
    /// Requires `&mut self` because `find` does path halving.
    pub(crate) fn duplicated_lines(&mut self) -> usize {
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.occ.len() {
            let root = self.find(i);
            let (start, end) = self.occ[i];
            let len = end - start + 1;
            components.entry(root).or_default().push(len);
        }
        components
            .values()
            .map(|lens| {
                let sum: usize = lens.iter().sum();
                let max: usize = *lens.iter().max().unwrap_or(&0);
                sum - max
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{
        CandidateMatch, FileRef, FunctionRef, Language, SnippetKind, SnippetRef,
    };

    fn make_self_clone_match(
        func: &FunctionRef,
        a_start: usize,
        a_end: usize,
        a_hash: &str,
        b_start: usize,
        b_end: usize,
        b_hash: &str,
    ) -> CandidateMatch {
        let snip_a = SnippetRef {
            kind: SnippetKind::Win,
            function: func.clone(),
            start_line: a_start,
            end_line: a_end,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: a_hash.into(),
        };
        let snip_b = SnippetRef {
            kind: SnippetKind::Win,
            function: func.clone(),
            start_line: b_start,
            end_line: b_end,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: b_hash.into(),
        };
        CandidateMatch {
            snippet_a: snip_a,
            snippet_b: snip_b,
            similarity: 0.95,
            evidence: "".into(),
        }
    }

    fn self_clone_func() -> FunctionRef {
        let file = FileRef {
            path: "x.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        FunctionRef {
            file,
            qualified_name: "f".into(),
            start_line: 1,
            end_line: 3000,
            code: "pass".into(),
            code_hash: "c".into(),
        }
    }

    #[test]
    fn test_is_self_clone_true() {
        let func = self_clone_func();
        let m = make_self_clone_match(&func, 1, 10, "a", 20, 30, "b");
        assert!(is_self_clone(&[m]));
    }

    #[test]
    fn test_is_self_clone_false_different_functions() {
        let file = FileRef {
            path: "x.py".into(),
            content_hash: "h".into(),
            language: Language::Python,
            content: "".into(),
        };
        let fa = FunctionRef {
            file: file.clone(),
            qualified_name: "a".into(),
            start_line: 1,
            end_line: 10,
            code: "p".into(),
            code_hash: "a".into(),
        };
        let fb = FunctionRef {
            file,
            qualified_name: "b".into(),
            start_line: 20,
            end_line: 30,
            code: "p".into(),
            code_hash: "b".into(),
        };
        let sa = SnippetRef {
            kind: SnippetKind::Func,
            function: fa,
            start_line: 1,
            end_line: 10,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: "sa".into(),
        };
        let sb = SnippetRef {
            kind: SnippetKind::Func,
            function: fb,
            start_line: 20,
            end_line: 30,
            text: "t".into(),
            display_text: "t".into(),
            snippet_hash: "sb".into(),
        };
        let m = CandidateMatch {
            snippet_a: sa,
            snippet_b: sb,
            similarity: 0.95,
            evidence: "".into(),
        };
        assert!(!is_self_clone(&[m]));
    }

    #[test]
    fn test_is_self_clone_empty_is_false() {
        assert!(!is_self_clone(&[]));
    }

    #[test]
    fn test_two_occurrence_self_clone_duplicated_lines() {
        // Two disjoint spans in the same function: sum-max = (26 + 26) - 26 = 26.
        let func = self_clone_func();
        let m = make_self_clone_match(&func, 1800, 1825, "a", 2100, 2125, "b");
        let mut occ = SelfCloneOccurrences::new(&[m]);
        assert_eq!(occ.duplicated_lines(), 26);
    }

    #[test]
    fn test_chained_nway_self_clone_duplicated_lines() {
        // Three occurrences chained by two pairwise matches sharing the long middle region.
        // R1=(1000,1005) 6 lines, R2=(2000,2100) 101 lines merged with R2b=(2005,2105) → (2000,2105) 106 lines,
        // R3=(2900,2905) 6 lines.
        // sum = 6 + 106 + 6 = 118; max = 106; sum - max = 12.
        let func = self_clone_func();
        let r1 = SnippetRef {
            kind: SnippetKind::Exp,
            function: func.clone(),
            start_line: 1000,
            end_line: 1005,
            text: "s".into(),
            display_text: "s".into(),
            snippet_hash: "r1".into(),
        };
        let r2 = SnippetRef {
            kind: SnippetKind::Exp,
            function: func.clone(),
            start_line: 2000,
            end_line: 2100,
            text: "s".into(),
            display_text: "s".into(),
            snippet_hash: "r2".into(),
        };
        let r3 = SnippetRef {
            kind: SnippetKind::Exp,
            function: func.clone(),
            start_line: 2900,
            end_line: 2905,
            text: "s".into(),
            display_text: "s".into(),
            snippet_hash: "r3".into(),
        };
        let r2b = SnippetRef {
            kind: SnippetKind::Exp,
            function: func.clone(),
            start_line: 2005,
            end_line: 2105,
            text: "s".into(),
            display_text: "s".into(),
            snippet_hash: "r2b".into(),
        };
        let m1 = CandidateMatch {
            snippet_a: r1,
            snippet_b: r2,
            similarity: 0.95,
            evidence: "".into(),
        };
        let m2 = CandidateMatch {
            snippet_a: r3,
            snippet_b: r2b,
            similarity: 0.95,
            evidence: "".into(),
        };
        let mut occ = SelfCloneOccurrences::new(&[m1, m2]);
        assert_eq!(occ.duplicated_lines(), 12);
    }

    #[test]
    fn test_merge_strict_overlap_not_adjacent() {
        // Spans (1, 5) and (6, 10) are adjacent but NOT strictly overlapping.
        // They should NOT be merged by merge_overlapping.
        let merged = merge_overlapping(&[(1, 5), (6, 10)]);
        assert_eq!(
            merged.len(),
            2,
            "adjacent spans must NOT be merged (strict overlap only)"
        );
    }

    #[test]
    fn test_merge_strict_overlap_merges_overlapping() {
        // Spans (1, 5) and (5, 10): start(5) <= end(5) → strictly overlapping → merged.
        let merged = merge_overlapping(&[(1, 5), (5, 10)]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], (1, 10));
    }
}
