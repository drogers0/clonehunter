use std::collections::{BTreeMap, HashMap};

use crate::core::types::{Finding, FunctionRef};

/// Assign `cluster_id` metadata to each finding via union-find over function identities.
///
/// Two findings that share a function identity end up in the same cluster.
/// Cluster IDs are assigned in finding-iteration order (first finding's cluster gets id=1).
/// Path halving is used in the `find` inner function (matches Python's implementation).
pub(crate) fn cluster_findings(findings: &[Finding]) -> Vec<Finding> {
    if findings.is_empty() {
        return vec![];
    }

    let mut parent: HashMap<String, String> = HashMap::new();

    // Path-halving union-find over function identity strings (DD12).
    fn find(parent: &mut HashMap<String, String>, x: &str) -> String {
        let mut x = x.to_string();
        loop {
            let px = parent.get(&x).cloned().unwrap_or_else(|| x.clone());
            if px == x {
                return x;
            }
            let ppx = parent.get(&px).cloned().unwrap_or_else(|| px.clone());
            parent.insert(x.clone(), ppx.clone());
            x = ppx;
        }
    }

    fn union(parent: &mut HashMap<String, String>, a: &str, b: &str) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(rb, ra);
        }
    }

    // Initialise all function identities, then union each finding's pair.
    for finding in findings {
        let a = finding.function_a.identity();
        let b = finding.function_b.identity();
        parent.entry(a.clone()).or_insert(a.clone());
        parent.entry(b.clone()).or_insert(b.clone());
        union(
            &mut parent,
            &finding.function_a.identity(),
            &finding.function_b.identity(),
        );
    }

    // Assign cluster IDs in finding-iteration order. NOTE: the integer `cluster_id` values are
    // therefore finding-order-dependent — cluster *membership* is deterministic, but the ID
    // integers are not stable across rayon-unordered runs, so do not snapshot/assert on them.
    let mut clusters: HashMap<String, usize> = HashMap::new();
    let mut next_id = 1usize;
    let mut result = Vec::with_capacity(findings.len());

    for finding in findings {
        let root = find(&mut parent, &finding.function_a.identity());
        let cluster_id = *clusters.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        let mut meta: BTreeMap<String, String> = finding.metadata.clone();
        meta.insert("cluster_id".into(), cluster_id.to_string());
        result.push(Finding {
            function_a: finding.function_a.clone(),
            function_b: finding.function_b.clone(),
            score: finding.score,
            duplicated_lines: finding.duplicated_lines,
            evidence: finding.evidence.clone(),
            reasons: finding.reasons.clone(),
            metadata: meta,
        });
    }
    result
}

/// Keep only findings belonging to clusters with at least `min_size` findings.
/// Returns a clone of `findings` unchanged when `min_size <= 1`.
pub(crate) fn filter_clusters(findings: &[Finding], min_size: usize) -> Vec<Finding> {
    if min_size <= 1 {
        return findings.to_vec();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in findings {
        if let Some(cid) = f.metadata.get("cluster_id") {
            *counts.entry(cid.clone()).or_insert(0) += 1;
        }
    }
    findings
        .iter()
        .filter(|f| {
            f.metadata
                .get("cluster_id")
                .and_then(|cid| counts.get(cid))
                .is_some_and(|&c| c >= min_size)
        })
        .cloned()
        .collect()
}

/// A clone group derived from findings (DD2/DD3/DD4). Borrows from the input slice.
pub(crate) struct CloneGroup<'a> {
    /// Stable, re-numbered group id (NOT the finding-order-dependent `cluster_id` metadata int).
    pub id: usize,
    /// Unique member functions, sorted by `identity()`; always non-empty.
    pub locations: Vec<&'a FunctionRef>,
    /// Indices into the input slice for this group's pairwise findings.
    pub finding_indices: Vec<usize>,
    pub max_score: f64,
    pub max_duplicated_lines: usize,
}

/// Derive stable clone groups from findings — the single grouping source of truth (DD2).
///
/// Partitions findings into groups (by `cluster_id` when a clustered run, else one singleton per
/// finding — DD4), collects each group's unique member functions sorted by `identity()`, then
/// sorts the groups by their full sorted location-identity vector (a total order) and assigns
/// sequential ids `1..`. Membership, ordering, and ids are reproducible run-to-run (DD3).
pub(crate) fn build_groups(findings: &[Finding]) -> Vec<CloneGroup<'_>> {
    use std::collections::HashSet;
    if findings.is_empty() {
        return Vec::new();
    }

    // Partition finding indices into clone groups:
    //  - clustered run → group by the `cluster_id` metadata value (stable partition);
    //  - unclustered   → each finding is its own singleton group (a pairwise clone = a
    //                    2-location group). Keeps `groups` uniform so it fully replaces the
    //                    flat `findings` array (DD4).
    let clustered = findings
        .iter()
        .any(|f| f.metadata.contains_key("cluster_id"));
    let partitions: Vec<Vec<usize>> = if clustered {
        // A finding without a `cluster_id` (only possible with a mixed/hand-built input — the
        // pipeline tags every finding or none) is its own singleton, never merged with other
        // untagged findings; that would silently combine unrelated clones.
        let mut by_cid: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut untagged: Vec<Vec<usize>> = Vec::new();
        for (i, f) in findings.iter().enumerate() {
            match f.metadata.get("cluster_id") {
                Some(cid) => by_cid.entry(cid.clone()).or_default().push(i),
                None => untagged.push(vec![i]),
            }
        }
        by_cid.into_values().chain(untagged).collect()
    } else {
        (0..findings.len()).map(|i| vec![i]).collect()
    };

    let mut groups: Vec<CloneGroup> = partitions
        .into_iter()
        .map(|idxs| {
            let mut locs: Vec<&FunctionRef> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            for &i in &idxs {
                for func in [&findings[i].function_a, &findings[i].function_b] {
                    if seen.insert(func.identity()) {
                        locs.push(func);
                    }
                }
            }
            locs.sort_by_key(|a| a.identity());
            let max_score = idxs.iter().map(|&i| findings[i].score).fold(0.0, f64::max);
            let max_duplicated_lines = idxs
                .iter()
                .map(|&i| findings[i].duplicated_lines)
                .max()
                .unwrap_or(0);
            CloneGroup {
                id: 0,
                locations: locs,
                finding_indices: idxs,
                max_score,
                max_duplicated_lines,
            }
        })
        .collect();

    // Total-order sort by the full sorted location-identity vector, then sequential ids.
    // `sort_by_cached_key` computes each group's identity vector once rather than re-`identity()`ing
    // (a heap alloc per location) on every comparison.
    groups.sort_by_cached_key(|g| g.locations.iter().map(|f| f.identity()).collect::<Vec<_>>());
    for (n, g) in groups.iter_mut().enumerate() {
        g.id = n + 1;
    }
    groups
}

/// Derive the two `ScanStats` group metrics from findings (DD5): `(group_count,
/// grouped_function_count)`. The function count is **de-duplicated** across groups — unclustered
/// singleton groups can repeat a function (X in both X↔Y and X↔Z), so a plain sum would
/// double-count. The single canonical computation, shared by every producer of `ScanStats`.
pub(crate) fn group_stats(findings: &[Finding]) -> (usize, usize) {
    use std::collections::HashSet;
    let groups = build_groups(findings);
    let grouped_function_count = groups
        .iter()
        .flat_map(|g| g.locations.iter().map(|f| f.identity()))
        .collect::<HashSet<_>>()
        .len();
    (groups.len(), grouped_function_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        make_finding as build_finding, make_function, make_match, make_snippet_for,
    };

    fn make_finding(a_name: &str, b_name: &str) -> Finding {
        let fn_a = make_function(&format!("{a_name}.py"), a_name, 1, 2, "pass");
        let fn_b = make_function(&format!("{b_name}.py"), b_name, 1, 2, "pass");
        let snip = make_snippet_for(&fn_a, "pass");
        let m = make_match(snip.clone(), snip, 1.0);
        build_finding(fn_a, fn_b, 1.0, 2, vec![m], &["func_threshold"])
    }

    #[test]
    fn test_cluster_empty() {
        assert!(cluster_findings(&[]).is_empty());
    }

    #[test]
    fn test_cluster_assigns_ids() {
        // Two findings sharing function "a" → same cluster.
        let findings = vec![make_finding("a", "b"), make_finding("a", "c")];
        let clustered = cluster_findings(&findings);
        assert_eq!(clustered.len(), 2);
        let cid0 = clustered[0].metadata.get("cluster_id").unwrap();
        let cid1 = clustered[1].metadata.get("cluster_id").unwrap();
        assert_eq!(
            cid0, cid1,
            "findings sharing function 'a' must be in the same cluster"
        );
    }

    #[test]
    fn test_cluster_min_size_filter_keeps_cluster_of_two() {
        let findings = vec![make_finding("a", "b"), make_finding("a", "b")];
        let clustered = cluster_findings(&findings);
        let filtered = filter_clusters(&clustered, 2);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_cluster_min_size_filter_drops_singleton() {
        // Two different clusters: (a,b) with 2 findings, (x,y) with 1 finding.
        let findings = vec![
            make_finding("a", "b"),
            make_finding("a", "b"),
            make_finding("x", "y"),
        ];
        let clustered = cluster_findings(&findings);
        let filtered = filter_clusters(&clustered, 2);
        assert_eq!(filtered.len(), 2, "singleton cluster must be dropped");
        assert!(
            filtered
                .iter()
                .all(|f| f.function_a.qualified_name == "a" || f.function_b.qualified_name == "a")
        );
    }

    #[test]
    fn test_filter_min_size_one_returns_all() {
        let findings = vec![make_finding("a", "b")];
        let clustered = cluster_findings(&findings);
        let filtered = filter_clusters(&clustered, 1);
        assert_eq!(filtered.len(), 1);
    }

    // ── build_groups ─────────────────────────────────────────────────────────

    /// Tag a finding with an explicit `cluster_id` (simulates a clustered run).
    fn with_cid(mut f: Finding, cid: &str) -> Finding {
        f.metadata.insert("cluster_id".into(), cid.into());
        f
    }

    #[test]
    fn build_groups_empty() {
        assert!(build_groups(&[]).is_empty());
    }

    #[test]
    fn build_groups_unclustered_one_group_per_finding() {
        // No cluster_id → each finding is its own 2-location singleton group (NOT dropped).
        let findings = vec![make_finding("a", "b"), make_finding("c", "d")];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 2);
        for g in &groups {
            assert_eq!(g.locations.len(), 2);
            assert_eq!(g.finding_indices.len(), 1);
        }
    }

    #[test]
    fn build_groups_clustered_merges_into_one() {
        // Three functions A,B,C all in one cluster (A–B, A–C, B–C) → one group of {A,B,C}.
        let findings = vec![
            with_cid(make_finding("a", "b"), "7"),
            with_cid(make_finding("a", "c"), "7"),
            with_cid(make_finding("b", "c"), "7"),
        ];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.finding_indices.len(), 3);
        assert_eq!(g.locations.len(), 3);
        // locations sorted by identity() = "{path}:{qname}:1:2"
        let ids: Vec<String> = g
            .locations
            .iter()
            .map(|f| f.qualified_name.clone())
            .collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn build_groups_two_clusters_get_sequential_ids() {
        // Two independent clusters → two groups, ids 1/2 by full-vector order.
        let findings = vec![
            with_cid(make_finding("x", "y"), "2"),
            with_cid(make_finding("a", "b"), "1"),
        ];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 2);
        // Group with {a,b} sorts before {x,y} regardless of cluster_id label order.
        assert_eq!(groups[0].id, 1);
        assert_eq!(groups[0].locations[0].qualified_name, "a");
        assert_eq!(groups[1].id, 2);
        assert_eq!(groups[1].locations[0].qualified_name, "x");
    }

    #[test]
    fn build_groups_untagged_finding_in_clustered_mode_stays_singleton() {
        // Mixed input: one finding carries a cluster_id, one does not. The untagged finding must
        // NOT be merged with other untagged findings into a bogus group — it stays a singleton.
        let findings = vec![
            with_cid(make_finding("a", "b"), "1"),
            make_finding("x", "y"), // no cluster_id
        ];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 2, "untagged finding is its own group");
        // Each group has exactly its own pair — nothing cross-merged.
        for g in &groups {
            assert_eq!(g.finding_indices.len(), 1);
            assert_eq!(g.locations.len(), 2);
        }
    }

    #[test]
    fn build_groups_self_clone_single_location() {
        // function_a.identity() == function_b.identity() → one location, retained (DD9).
        let fn_self = make_function("s.py", "f", 1, 2, "pass");
        let snip = make_snippet_for(&fn_self, "pass");
        let m = make_match(snip.clone(), snip, 1.0);
        let finding = build_finding(fn_self.clone(), fn_self, 1.0, 2, vec![m], &["self_clone"]);
        let findings = [finding];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].locations.len(), 1);
    }

    #[test]
    fn build_groups_deterministic_under_shuffle() {
        // Same partition, different order and different integer cluster_id labels → identical
        // membership, ids, ordering, and per-group maxima.
        let base = |cid_ab: &str, cid_xy: &str| {
            vec![
                {
                    let mut f = with_cid(make_finding("a", "b"), cid_ab);
                    f.score = 0.8;
                    f.duplicated_lines = 5;
                    f
                },
                {
                    let mut f = with_cid(make_finding("a", "c"), cid_ab);
                    f.score = 0.95;
                    f.duplicated_lines = 9;
                    f
                },
                with_cid(make_finding("x", "y"), cid_xy),
            ]
        };
        let ordered = base("1", "2");
        let g1 = build_groups(&ordered);
        let mut shuffled = base("5", "3");
        shuffled.reverse();
        let g2 = build_groups(&shuffled);

        let summarize = |gs: &[CloneGroup]| -> Vec<(usize, Vec<String>, f64, usize)> {
            gs.iter()
                .map(|g| {
                    (
                        g.id,
                        g.locations.iter().map(|f| f.identity()).collect(),
                        g.max_score,
                        g.max_duplicated_lines,
                    )
                })
                .collect()
        };
        assert_eq!(summarize(&g1), summarize(&g2));
        // The {a,b,c} group carries the per-group maxima.
        let abc = g1.iter().find(|g| g.locations.len() == 3).unwrap();
        assert_eq!(abc.max_score, 0.95);
        assert_eq!(abc.max_duplicated_lines, 9);
    }
}
