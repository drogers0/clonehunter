use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::types::{Finding, FunctionRef};

/// A clone group derived from findings. Groups are the single presentation/source-of-truth view:
/// findings sharing any function identity merge into one family via union-find, and the
/// representative is the most-connected location within that family.
pub(crate) struct CloneGroup<'a> {
    /// Stable, re-numbered group id.
    pub id: usize,
    /// Unique member functions, sorted by `identity()`; always non-empty.
    pub locations: Vec<&'a FunctionRef>,
    /// Index into `locations` for the representative function.
    pub representative: usize,
    /// Indices into the input slice for this group's pairwise findings.
    pub finding_indices: Vec<usize>,
    pub max_score: f64,
    pub max_duplicated_lines: usize,
}

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

/// Derive stable clone groups from findings via unconditional union-find over function identity.
///
/// Membership, ordering, ids, representative selection, and per-group finding ordering are all
/// deterministic run-to-run. Groups sort by their full sorted location-identity vector and are
/// re-numbered sequentially `1..`.
pub(crate) fn build_groups(findings: &[Finding]) -> Vec<CloneGroup<'_>> {
    if findings.is_empty() {
        return Vec::new();
    }

    let mut parent: HashMap<String, String> = HashMap::new();
    for finding in findings {
        let a = finding.function_a.identity();
        let b = finding.function_b.identity();
        parent.entry(a.clone()).or_insert_with(|| a.clone());
        parent.entry(b.clone()).or_insert_with(|| b.clone());
        union(&mut parent, &a, &b);
    }

    let mut partitions: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, finding) in findings.iter().enumerate() {
        let root = find(&mut parent, &finding.function_a.identity());
        partitions.entry(root).or_default().push(idx);
    }

    let mut groups: Vec<CloneGroup<'_>> = partitions
        .into_values()
        .map(|mut finding_indices| {
            finding_indices.sort_by_key(|&idx| {
                (
                    findings[idx].function_a.identity(),
                    findings[idx].function_b.identity(),
                )
            });

            let mut seen = HashSet::new();
            let mut locations = Vec::new();
            for &idx in &finding_indices {
                for func in [&findings[idx].function_a, &findings[idx].function_b] {
                    let identity = func.identity();
                    if seen.insert(identity) {
                        locations.push(func);
                    }
                }
            }
            locations.sort_by_key(|func| func.identity());

            let mut incident_counts: HashMap<String, usize> = locations
                .iter()
                .map(|func| (func.identity(), 0usize))
                .collect();
            for &idx in &finding_indices {
                let finding = &findings[idx];
                let id_a = finding.function_a.identity();
                let id_b = finding.function_b.identity();
                if id_a == id_b {
                    continue;
                }
                *incident_counts.entry(id_a).or_insert(0) += 1;
                *incident_counts.entry(id_b).or_insert(0) += 1;
            }

            let representative = locations
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    let left_id = left.identity();
                    let right_id = right.identity();
                    incident_counts[&left_id]
                        .cmp(&incident_counts[&right_id])
                        .then_with(|| right_id.cmp(&left_id))
                })
                .map(|(idx, _)| idx)
                .expect("group has at least one location");

            CloneGroup {
                id: 0,
                representative,
                max_score: finding_indices
                    .iter()
                    .map(|&idx| findings[idx].score)
                    .fold(0.0, f64::max),
                max_duplicated_lines: finding_indices
                    .iter()
                    .map(|&idx| findings[idx].duplicated_lines)
                    .max()
                    .unwrap_or(0),
                locations,
                finding_indices,
            }
        })
        .collect();

    groups.sort_by_cached_key(|group| {
        group
            .locations
            .iter()
            .map(|func| func.identity())
            .collect::<Vec<_>>()
    });
    for (idx, group) in groups.iter_mut().enumerate() {
        group.id = idx + 1;
    }
    groups
}

/// Derive the two `ScanStats` group metrics from findings: `(group_count, grouped_function_count)`.
/// The function count is de-duplicated across groups.
pub(crate) fn group_stats(findings: &[Finding]) -> (usize, usize) {
    let groups = build_groups(findings);
    let grouped_function_count = groups
        .iter()
        .flat_map(|group| group.locations.iter().map(|func| func.identity()))
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
        make_finding_with_score(a_name, b_name, 1.0, 2)
    }

    fn make_finding_with_score(
        a_name: &str,
        b_name: &str,
        score: f64,
        duplicated_lines: usize,
    ) -> Finding {
        let fn_a = make_function(&format!("{a_name}.py"), a_name, 1, 2, "pass");
        let fn_b = make_function(&format!("{b_name}.py"), b_name, 1, 2, "pass");
        let snip_a = make_snippet_for(&fn_a, "pass");
        let snip_b = make_snippet_for(&fn_b, "pass");
        let m = make_match(snip_a, snip_b, score);
        build_finding(
            fn_a,
            fn_b,
            score,
            duplicated_lines,
            vec![m],
            &["func_threshold"],
        )
    }

    #[test]
    fn build_groups_empty() {
        assert!(build_groups(&[]).is_empty());
    }

    #[test]
    fn build_groups_star_family_uses_hub_representative() {
        let findings = vec![
            make_finding_with_score("hub", "leaf_b", 0.8, 5),
            make_finding_with_score("hub", "leaf_a", 0.95, 9),
            make_finding_with_score("hub", "leaf_c", 0.9, 7),
        ];

        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.locations.len(), 4);
        assert_eq!(group.locations[group.representative].qualified_name, "hub");
        assert_eq!(group.max_score, 0.95);
        assert_eq!(group.max_duplicated_lines, 9);
        assert_eq!(group.finding_indices, vec![1, 0, 2]);
    }

    #[test]
    fn build_groups_two_disjoint_findings_make_two_families() {
        let findings = [make_finding("a", "b"), make_finding("x", "y")];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].id, 1);
        assert_eq!(
            groups[0]
                .locations
                .iter()
                .map(|func| func.qualified_name.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(groups[1].id, 2);
        assert_eq!(
            groups[1]
                .locations
                .iter()
                .map(|func| func.qualified_name.as_str())
                .collect::<Vec<_>>(),
            vec!["x", "y"]
        );
    }

    #[test]
    fn build_groups_deterministic_under_shuffle() {
        let ordered = vec![
            make_finding_with_score("hub", "leaf_b", 0.8, 5),
            make_finding_with_score("hub", "leaf_a", 0.95, 9),
            make_finding_with_score("x", "y", 0.7, 4),
        ];
        let shuffled = vec![
            make_finding_with_score("x", "y", 0.7, 4),
            make_finding_with_score("hub", "leaf_a", 0.95, 9),
            make_finding_with_score("hub", "leaf_b", 0.8, 5),
        ];

        let g1 = build_groups(&ordered);
        let g2 = build_groups(&shuffled);

        let normalize = |groups: &[CloneGroup<'_>], findings: &[Finding]| {
            groups
                .iter()
                .map(|group| {
                    (
                        group.id,
                        group
                            .locations
                            .iter()
                            .map(|func| func.identity())
                            .collect::<Vec<_>>(),
                        group.representative,
                        group
                            .finding_indices
                            .iter()
                            .map(|&idx| {
                                (
                                    findings[idx].function_a.identity(),
                                    findings[idx].function_b.identity(),
                                )
                            })
                            .collect::<Vec<_>>(),
                        group.max_score,
                        group.max_duplicated_lines,
                    )
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(normalize(&g1, &ordered), normalize(&g2, &shuffled));
    }

    #[test]
    fn build_groups_representative_tiebreaks_by_identity() {
        let findings = vec![
            make_finding("a", "b"),
            make_finding("a", "c"),
            make_finding("b", "c"),
        ];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 1);
        let group = &groups[0];
        assert_eq!(group.locations[group.representative].qualified_name, "a");
    }

    #[test]
    fn build_groups_self_clone_single_location_family() {
        let func = make_function("s.py", "self_clone", 10, 20, "pass");
        let snip_a = make_snippet_for(&func, "pass");
        let snip_b = make_snippet_for(&func, "pass");
        let finding = build_finding(
            func.clone(),
            func,
            1.0,
            2,
            vec![make_match(snip_a, snip_b, 1.0)],
            &["self_clone"],
        );

        let findings = [finding];
        let groups = build_groups(&findings);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].locations.len(), 1);
        assert_eq!(groups[0].representative, 0);
    }
}
