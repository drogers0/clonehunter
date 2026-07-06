use std::collections::HashMap;

use crate::core::config::Thresholds;
use crate::core::types::{CandidateMatch, Finding, SnippetKind, SnippetRef};

use super::lexical::lexical_similarity;
use super::occurrences::{SelfCloneOccurrences, is_self_clone};
use super::ranking::kind_rank;
use super::scoring::best_score;

/// Roll up candidate matches into findings, one per function pair.
///
/// Pipeline (exact Python order):
/// 1. Filter overlapping matches (self-clone range overlap; cross-fn containment).
/// 2. Filter by lexical gate (second gate — first is in retrieve_candidates).
/// 3. Dedupe symmetric duplicates within span granularity.
/// 4. Normalize a/b orientation (THE single canonicalization point, DD8).
/// 5. Group by function-identity pair.
/// 6. Emit a Finding only if ≥1 reason is earned.
pub(crate) fn rollup_findings(
    matches: Vec<CandidateMatch>,
    thresholds: &Thresholds,
) -> Vec<Finding> {
    let filtered = filter_overlapping_matches(&matches);
    let filtered = filter_lexical_matches(&filtered, thresholds.lexical_min_ratio);
    let filtered = dedupe_matches(&filtered);

    // Normalize orientation + group by function pair, preserving insertion order.
    let mut grouped: HashMap<(String, String), Vec<CandidateMatch>> = HashMap::new();
    let mut group_order: Vec<(String, String)> = Vec::new();
    for m in filtered {
        let m = normalize_orientation(m);
        let key = fn_pair_key(&m);
        if !grouped.contains_key(&key) {
            group_order.push(key.clone());
        }
        grouped.entry(key).or_default().push(m);
    }

    let mut findings = Vec::new();
    for key in &group_order {
        let group = &grouped[key];
        let func_a = group[0].snippet_a.function.clone();
        let func_b = group[0].snippet_b.function.clone();
        let score = best_score(group);
        let reasons = compute_reasons(group, thresholds);
        if !reasons.is_empty() {
            findings.push(Finding {
                function_a: func_a,
                function_b: func_b,
                score,
                duplicated_lines: duplicated_lines(group),
                evidence: group.clone(),
                reasons,
                metadata: std::collections::BTreeMap::new(),
            });
        }
    }
    findings
}

/// Normalize a/b orientation. THE SINGLE CANONICALIZATION POINT (DD8).
///
/// Compares function identity strings first; breaks ties by start_line.
/// This is the one place every per-side aggregator branches from.
fn normalize_orientation(m: CandidateMatch) -> CandidateMatch {
    if is_canonical_order(&m.snippet_a, &m.snippet_b) {
        m
    } else {
        CandidateMatch {
            snippet_a: m.snippet_b,
            snippet_b: m.snippet_a,
            similarity: m.similarity,
            evidence: m.evidence,
        }
    }
}

/// Return `true` if `a` should be on side A (canonical order). (DD8)
/// Matches Python's `_is_canonical_order` exactly.
fn is_canonical_order(a: &SnippetRef, b: &SnippetRef) -> bool {
    let id_a = a.function.identity();
    let id_b = b.function.identity();
    if id_a != id_b {
        return id_a < id_b;
    }
    a.start_line <= b.start_line
}

/// Grouping key: ordered function-identity pair, always (smaller, larger).
fn fn_pair_key(m: &CandidateMatch) -> (String, String) {
    let a = m.snippet_a.function.identity();
    let b = m.snippet_b.function.identity();
    if a <= b { (a, b) } else { (b, a) }
}

/// Compute the reasons a finding should be emitted (DD7).
///
/// - `"func_threshold"`: there is ≥1 FUNC-touching match with best score ≥ thresholds.func
/// - `"exp_threshold"`: there is ≥1 EXP-touching match with best score ≥ thresholds.exp
/// - `"min_window_hits"`: COUNT of WIN-touching matches ≥ thresholds.min_window_hits
///
/// Note: `min_window_hits` is purely a COUNT gate — no score threshold applies to WIN.
fn compute_reasons(matches: &[CandidateMatch], thresholds: &Thresholds) -> Vec<String> {
    let func_hits: Vec<&CandidateMatch> = matches
        .iter()
        .filter(|m| m.snippet_a.kind == SnippetKind::Func || m.snippet_b.kind == SnippetKind::Func)
        .collect();
    let win_hits: Vec<&CandidateMatch> = matches
        .iter()
        .filter(|m| m.snippet_a.kind == SnippetKind::Win || m.snippet_b.kind == SnippetKind::Win)
        .collect();
    let exp_hits: Vec<&CandidateMatch> = matches
        .iter()
        .filter(|m| m.snippet_a.kind == SnippetKind::Exp || m.snippet_b.kind == SnippetKind::Exp)
        .collect();

    let mut reasons = Vec::new();
    if !func_hits.is_empty() && best_score(func_hits.iter().copied()) >= thresholds.func {
        reasons.push("func_threshold".into());
    }
    if !exp_hits.is_empty() && best_score(exp_hits.iter().copied()) >= thresholds.exp {
        reasons.push("exp_threshold".into());
    }
    if win_hits.len() >= thresholds.min_window_hits {
        reasons.push("min_window_hits".into());
    }
    reasons
}

/// Filter: self-clones only when ranges are disjoint; cross-function same-file
/// drops when functions overlap (structural containment).
fn filter_overlapping_matches(matches: &[CandidateMatch]) -> Vec<CandidateMatch> {
    let mut filtered = Vec::new();
    for m in matches {
        let func_a_id = m.snippet_a.function.identity();
        let func_b_id = m.snippet_b.function.identity();

        // Self-clone: allow only if snippet ranges are disjoint.
        if func_a_id == func_b_id {
            if overlap_len(
                m.snippet_a.start_line,
                m.snippet_a.end_line,
                m.snippet_b.start_line,
                m.snippet_b.end_line,
            ) > 0
            {
                continue;
            }
            filtered.push(m.clone());
            continue;
        }

        // Cross-function same-file: structural containment → drop.
        if m.snippet_a.function.file.path == m.snippet_b.function.file.path
            && overlap_len(
                m.snippet_a.function.start_line,
                m.snippet_a.function.end_line,
                m.snippet_b.function.start_line,
                m.snippet_b.function.end_line,
            ) > 0
        {
            continue;
        }
        filtered.push(m.clone());
    }
    filtered
}

fn overlap_len(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> usize {
    let start = a_start.max(b_start);
    let end = a_end.min(b_end);
    if start > end { 0 } else { end - start + 1 }
}

/// Second lexical gate in rollup (first is in retrieve_candidates — see "Known limitations").
fn filter_lexical_matches(matches: &[CandidateMatch], min_ratio: f64) -> Vec<CandidateMatch> {
    if min_ratio <= 0.0 {
        return matches.to_vec();
    }
    matches
        .iter()
        .filter(|m| lexical_similarity(&m.snippet_a.text, &m.snippet_b.text) >= min_ratio)
        .cloned()
        .collect()
}

/// Dedupe symmetric and span-identical matches, preserving insertion order (DD10).
/// Keeps the match with highest similarity; on tie, prefers higher kind_rank.
fn dedupe_matches(matches: &[CandidateMatch]) -> Vec<CandidateMatch> {
    type SpanKey = (String, usize, usize);
    type DedupeKey = (SpanKey, SpanKey);

    let mut best: HashMap<DedupeKey, CandidateMatch> = HashMap::new();
    let mut order: Vec<DedupeKey> = Vec::new();

    for m in matches {
        let a_key: SpanKey = (
            m.snippet_a.function.identity(),
            m.snippet_a.start_line,
            m.snippet_a.end_line,
        );
        let b_key: SpanKey = (
            m.snippet_b.function.identity(),
            m.snippet_b.start_line,
            m.snippet_b.end_line,
        );
        let key: DedupeKey = if a_key <= b_key {
            (a_key, b_key)
        } else {
            (b_key, a_key)
        };

        match best.get(&key) {
            Some(existing) => {
                if m.similarity > existing.similarity
                    || (m.similarity == existing.similarity && kind_rank(m) > kind_rank(existing))
                {
                    best.insert(key, m.clone());
                }
            }
            None => {
                order.push(key.clone());
                best.insert(key, m.clone());
            }
        }
    }
    order.iter().map(|k| best[k].clone()).collect()
}

/// Compute duplicated line count for a group.
///
/// Self-clone groups use `SelfCloneOccurrences` (connected-component sum-max) to
/// correctly handle N-way chained occurrences. Cross-function groups use
/// `covered_lines` with adjacency merging and take the min of both sides.
fn duplicated_lines(matches: &[CandidateMatch]) -> usize {
    if matches.is_empty() {
        return 0;
    }
    if is_self_clone(matches) {
        let mut occ = SelfCloneOccurrences::new(matches);
        return occ.duplicated_lines();
    }
    let spans_a: Vec<(usize, usize)> = matches
        .iter()
        .map(|m| (m.snippet_a.start_line, m.snippet_a.end_line))
        .collect();
    let spans_b: Vec<(usize, usize)> = matches
        .iter()
        .map(|m| (m.snippet_b.start_line, m.snippet_b.end_line))
        .collect();
    covered_lines(&spans_a).min(covered_lines(&spans_b))
}

/// Count covered lines with ADJACENCY merging (`start <= prev_end + 1` extends the span).
/// This differs from `merge_overlapping` in occurrences.rs which uses STRICT overlap.
fn covered_lines(spans: &[(usize, usize)]) -> usize {
    if spans.is_empty() {
        return 0;
    }
    let mut sorted: Vec<(usize, usize)> = spans.to_vec();
    sorted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in sorted {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 + 1 {
                if end > last.1 {
                    last.1 = end;
                }
                continue;
            }
        }
        merged.push((start, end));
    }
    merged.iter().map(|(s, e)| e - s + 1).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Thresholds;
    use crate::core::types::{
        CandidateMatch, FileRef, FunctionRef, Language, SnippetKind, SnippetRef,
    };

    // ── Test helpers ──────────────────────────────────────────────────────────

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
        text: &str,
        hash: &str,
    ) -> SnippetRef {
        SnippetRef {
            kind,
            function: func.clone(),
            start_line: start,
            end_line: end,
            text: text.into(),
            display_text: text.into(),
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

    fn thresholds(func: f64, win: f64, exp: f64, mwh: usize, lmr: f64) -> Thresholds {
        Thresholds {
            func,
            win,
            exp,
            min_window_hits: mwh,
            lexical_min_ratio: lmr,
            lexical_weight: 0.3,
        }
    }

    // ── Overlap / self-clone filtering ────────────────────────────────────────

    #[test]
    fn test_rollup_min_window_hits() {
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 5);
        let fn_b = make_func(&file, "b", 10, 14);
        let a1 = make_snip(SnippetKind::Win, &fn_a, 1, 3, "a1", "a1");
        let b1 = make_snip(SnippetKind::Win, &fn_b, 10, 12, "b1", "b1");
        let a2 = make_snip(SnippetKind::Win, &fn_a, 2, 4, "a2", "a2");
        let b2 = make_snip(SnippetKind::Win, &fn_b, 11, 13, "b2", "b2");
        let matches = vec![make_match(a1, b1, 0.5), make_match(a2, b2, 0.5)];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 2, 0.0));
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_rollup_filters_overlapping_windows_same_function() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 10);
        let a = make_snip(SnippetKind::Win, &func, 1, 5, "a", "a");
        let b = make_snip(SnippetKind::Win, &func, 4, 8, "b", "b");
        let findings = rollup_findings(
            vec![make_match(a, b, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_keeps_non_overlapping_windows_same_function() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 40);
        let a = make_snip(SnippetKind::Win, &func, 1, 10, "same", "a");
        let b = make_snip(SnippetKind::Win, &func, 21, 30, "same", "b");
        let findings = rollup_findings(
            vec![make_match(a, b, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(!findings.is_empty());
    }

    #[test]
    fn test_rollup_drops_identical_windows_same_function() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 30);
        let a = make_snip(SnippetKind::Win, &func, 5, 24, "a", "a");
        let b = make_snip(SnippetKind::Win, &func, 5, 24, "b", "b");
        let findings = rollup_findings(
            vec![make_match(a, b, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_drops_identical_func_self_match() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 10);
        let a = make_snip(SnippetKind::Func, &func, 1, 10, "a", "a");
        let b = make_snip(SnippetKind::Func, &func, 1, 10, "b", "b");
        let findings = rollup_findings(
            vec![make_match(a, b, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_drops_overlapping_func_win_same_function() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 30);
        let f = make_snip(SnippetKind::Func, &func, 1, 30, "f", "f");
        let w = make_snip(SnippetKind::Win, &func, 5, 24, "w", "w");
        let findings = rollup_findings(
            vec![make_match(f, w, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_drops_overlapping_expansions_same_function() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 60);
        let a = make_snip(SnippetKind::Exp, &func, 10, 35, "a", "a");
        let b = make_snip(SnippetKind::Exp, &func, 20, 45, "b", "b");
        let findings = rollup_findings(
            vec![make_match(a, b, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_drops_overlapping_functions_same_file() {
        let file = make_file("x.py");
        let outer = make_func(&file, "outer", 1, 40);
        let inner = make_func(&file, "inner", 10, 20);
        let o = make_snip(SnippetKind::Func, &outer, 1, 40, "o", "o");
        let i = make_snip(SnippetKind::Func, &inner, 10, 20, "i", "i");
        let findings = rollup_findings(
            vec![make_match(o, i, 1.0)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.0),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_applies_lexical_filter() {
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 5);
        let fn_b = make_func(&file, "b", 10, 14);
        let a = make_snip(
            SnippetKind::Win,
            &fn_a,
            1,
            3,
            "def alpha():\n    return 1",
            "a1",
        );
        let b = make_snip(
            SnippetKind::Win,
            &fn_b,
            10,
            12,
            "def beta():\n    return 2",
            "b1",
        );
        // Jaccard(alpha,return,1 vs beta,return,2) = |{def,return}| / |{def,alpha,return,1,beta,2}| = 2/6 < 0.6
        let findings = rollup_findings(
            vec![make_match(a, b, 0.99)],
            &thresholds(0.9, 0.9, 0.9, 1, 0.6),
        );
        assert!(findings.is_empty());
    }

    #[test]
    fn test_rollup_distinct_functions_with_same_code_hash() {
        let fa = make_func(&make_file("a.py"), "a", 1, 2);
        let fb = make_func(&make_file("b.py"), "b", 1, 2);
        let fc = make_func(&make_file("c.py"), "c", 1, 2);
        let a = make_snip(SnippetKind::Func, &fa, 1, 2, "pass", "a");
        let b = make_snip(SnippetKind::Func, &fb, 1, 2, "pass", "b");
        let c = make_snip(SnippetKind::Func, &fc, 1, 2, "pass", "c");
        let matches = vec![make_match(a.clone(), b, 0.99), make_match(a, c, 0.99)];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 1, 0.0));
        assert_eq!(findings.len(), 2);
    }

    // ── Orientation normalization ──────────────────────────────────────────────

    #[test]
    fn test_rollup_normalizes_self_clone_orientation() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 3000);
        let early_1 = make_snip(SnippetKind::Win, &func, 1800, 1820, "s", "w1");
        let late_1 = make_snip(SnippetKind::Win, &func, 2100, 2120, "s", "w2");
        let late_2 = make_snip(SnippetKind::Win, &func, 2105, 2125, "s", "w3");
        let early_2 = make_snip(SnippetKind::Win, &func, 1805, 1825, "s", "w4");
        let matches = vec![
            make_match(early_1, late_1, 0.9),
            // Orientation flipped — candidate generation is bidirectional
            make_match(late_2, early_2, 0.9),
        ];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 2, 0.0));
        assert_eq!(findings.len(), 1);
        let ev = &findings[0].evidence;
        let max_end_a = ev.iter().map(|m| m.snippet_a.end_line).max().unwrap();
        let min_start_b = ev.iter().map(|m| m.snippet_b.start_line).min().unwrap();
        assert!(
            max_end_a < min_start_b,
            "side A and side B evidence bounds must be disjoint"
        );
    }

    // ── Duplicated lines ──────────────────────────────────────────────────────

    #[test]
    fn test_rollup_nway_chained_self_clone_duplicated_lines() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 3000);
        let r1 = make_snip(SnippetKind::Exp, &func, 1000, 1005, "s", "r1");
        let r2 = make_snip(SnippetKind::Exp, &func, 2000, 2100, "s", "r2");
        let r3 = make_snip(SnippetKind::Exp, &func, 2900, 2905, "s", "r3");
        let r2b = make_snip(SnippetKind::Exp, &func, 2005, 2105, "s", "r2b");
        let matches = vec![
            make_match(r1, r2, 0.95),
            make_match(r3, r2b, 0.95), // orientation flipped on second edge
        ];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 1, 0.0));
        assert_eq!(findings.len(), 1);
        // merged middle: (2000,2105) = 106 lines; R1=6, R3=6
        // sum-max = (6 + 106 + 6) - 106 = 12
        assert_eq!(findings[0].duplicated_lines, 12);
    }

    #[test]
    fn test_rollup_two_occurrence_self_clone_duplicated_lines_unchanged() {
        let file = make_file("x.py");
        let func = make_func(&file, "f", 1, 3000);
        let early_1 = make_snip(SnippetKind::Win, &func, 1800, 1820, "s", "w1");
        let late_1 = make_snip(SnippetKind::Win, &func, 2100, 2120, "s", "w2");
        let late_2 = make_snip(SnippetKind::Win, &func, 2105, 2125, "s", "w3");
        let early_2 = make_snip(SnippetKind::Win, &func, 1805, 1825, "s", "w4");
        // Python fixture: early_1=(1800,1820), late_1=(2100,2120), late_2=(2105,2125), early_2=(1805,1825)
        // Spans: {(1800,1820),(2100,2120),(2105,2125),(1805,1825)}
        // After merge_overlapping:
        //   sort: (1800,1820),(1805,1825),(2100,2120),(2105,2125)
        //   (1800,1820) → merged[0]
        //   (1805,1825): start(1805) <= end(1820) → strict overlap → extend to (1800,1825)
        //   (2100,2120) → start(2100) > 1825 → new merged[1]
        //   (2105,2125): start(2105) <= end(2120) → extend to (2100,2125)
        // occ = [(1800,1825),(2100,2125)], each 26 lines
        // matches: m1 a=(1800,1820) b=(2100,2120); m2 a=(2105,2125) b=(1805,1825)
        // index_of: (1800,1820) in (1800,1825) → occ[0]; (2100,2120) in (2100,2125) → occ[1]
        //           (2105,2125) in (2100,2125) → occ[1]; (1805,1825) in (1800,1825) → occ[0]
        // unions: union(0,1), union(1,0) → one component {0,1}
        // sum=26+26=52, max=26, sum-max=26 ✓
        let matches = vec![
            make_match(early_1, late_1, 0.9),
            make_match(late_2, early_2, 0.9),
        ];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 2, 0.0));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].duplicated_lines, 26);
    }

    #[test]
    fn test_rollup_cross_function_multiwindow_duplicated_lines_unchanged() {
        // covered_a = (1-10)=10 + (20-29)=10 = 20
        // covered_b = (101-110)=10 + (120-139)=20 = 30
        // min(20, 30) = 20
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 50);
        let fn_b = make_func(&file, "b", 100, 150);
        let a1 = make_snip(SnippetKind::Win, &fn_a, 1, 10, "a1", "a1");
        let b1 = make_snip(SnippetKind::Win, &fn_b, 101, 110, "b1", "b1");
        let a2 = make_snip(SnippetKind::Win, &fn_a, 20, 29, "a2", "a2");
        let b2 = make_snip(SnippetKind::Win, &fn_b, 120, 139, "b2", "b2");
        let matches = vec![make_match(a1, b1, 0.95), make_match(a2, b2, 0.95)];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 2, 0.0));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].duplicated_lines, 20);
    }

    #[test]
    fn test_rollup_canonicalizes_cross_function_orientation() {
        let file = make_file("z.py");
        let fn_a = make_func(&file, "a_fn", 1, 20);
        let fn_b = make_func(&file, "b_fn", 100, 120);
        let sa = make_snip(SnippetKind::Func, &fn_a, 1, 20, "s", "sa");
        let sb = make_snip(SnippetKind::Func, &fn_b, 100, 120, "s", "sb");
        // Supply B-before-A: snippet_a=sb (b_fn), snippet_b=sa (a_fn)
        let matches = vec![make_match(sb, sa, 0.95)];
        let findings = rollup_findings(matches, &thresholds(0.9, 0.9, 0.9, 2, 0.0));
        assert_eq!(findings.len(), 1);
        assert!(findings[0].function_a.identity() < findings[0].function_b.identity());
    }

    // ── Threshold edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_func_threshold_edge() {
        // FUNC at exactly threshold → emitted; 1 ULP below → not emitted.
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 2);
        let fn_b = make_func(&file, "b", 10, 12);
        let sa = make_snip(SnippetKind::Func, &fn_a, 1, 2, "a", "a1");
        let sb = make_snip(SnippetKind::Func, &fn_b, 10, 12, "b", "b10");
        let t = thresholds(0.95, 0.9, 0.9, 2, 0.0);
        let at = rollup_findings(vec![make_match(sa.clone(), sb.clone(), 0.95)], &t);
        let below = rollup_findings(vec![make_match(sa, sb, 0.9499)], &t);
        assert!(!at.is_empty());
        assert!(below.is_empty());
    }

    #[test]
    fn test_win_threshold_edge() {
        // Two WIN matches meeting min_window_hits=2 → finding emitted even when score < win threshold.
        // min_window_hits is a COUNT gate, not a score gate (DD7).
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 20);
        let fn_b = make_func(&file, "b", 30, 50);
        let a1 = make_snip(SnippetKind::Win, &fn_a, 1, 3, "a1", "a1");
        let b1 = make_snip(SnippetKind::Win, &fn_b, 30, 32, "b1", "b1");
        let a2 = make_snip(SnippetKind::Win, &fn_a, 4, 6, "a2", "a2");
        let b2 = make_snip(SnippetKind::Win, &fn_b, 33, 35, "b2", "b2");
        let t = thresholds(0.95, 0.9, 0.9, 2, 0.0);
        let at = rollup_findings(
            vec![
                make_match(a1.clone(), b1.clone(), 0.9),
                make_match(a2.clone(), b2.clone(), 0.9),
            ],
            &t,
        );
        let below = rollup_findings(
            vec![make_match(a1, b1, 0.8999), make_match(a2, b2, 0.8999)],
            &t,
        );
        assert!(
            !at.is_empty(),
            "at-threshold WIN count gate must produce a finding"
        );
        assert!(
            !below.is_empty(),
            "below-threshold WIN count gate still produces finding (count-only gate)"
        );
    }

    #[test]
    fn test_exp_threshold_edge() {
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 2);
        let fn_b = make_func(&file, "b", 10, 12);
        let sa = make_snip(SnippetKind::Exp, &fn_a, 1, 2, "a", "a1");
        let sb = make_snip(SnippetKind::Exp, &fn_b, 10, 12, "b", "b10");
        let t = thresholds(0.95, 0.9, 0.9, 2, 0.0);
        let at = rollup_findings(vec![make_match(sa.clone(), sb.clone(), 0.9)], &t);
        let below = rollup_findings(vec![make_match(sa, sb, 0.8999)], &t);
        assert!(!at.is_empty());
        assert!(below.is_empty());
    }

    #[test]
    fn test_min_window_hits_below_count() {
        // 2 WIN matches with min_window_hits=3 → count(2) < 3 → no finding.
        let file = make_file("x.py");
        let fn_a = make_func(&file, "a", 1, 20);
        let fn_b = make_func(&file, "b", 30, 50);
        let a1 = make_snip(SnippetKind::Win, &fn_a, 1, 3, "a1", "a1");
        let b1 = make_snip(SnippetKind::Win, &fn_b, 30, 32, "b1", "b1");
        let a2 = make_snip(SnippetKind::Win, &fn_a, 4, 6, "a2", "a2");
        let b2 = make_snip(SnippetKind::Win, &fn_b, 33, 35, "b2", "b2");
        let t = thresholds(0.95, 0.9, 0.9, 3, 0.0);
        let findings =
            rollup_findings(vec![make_match(a1, b1, 0.95), make_match(a2, b2, 0.95)], &t);
        assert!(
            findings.is_empty(),
            "count 2 < min_window_hits 3 must not emit a finding"
        );
    }
}
