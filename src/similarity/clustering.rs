use std::collections::{BTreeMap, HashMap};

use crate::core::types::Finding;

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

    // Assign cluster IDs in finding-iteration order.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{
        CandidateMatch, FileRef, Finding, FunctionRef, Language, SnippetKind, SnippetRef,
    };
    use std::collections::BTreeMap;

    fn make_finding(a_name: &str, b_name: &str) -> Finding {
        let file_a = FileRef {
            path: format!("{a_name}.py"),
            content_hash: "h".into(),
            language: Language::Python,
        };
        let file_b = FileRef {
            path: format!("{b_name}.py"),
            content_hash: "h".into(),
            language: Language::Python,
        };
        let fn_a = FunctionRef {
            file: file_a,
            qualified_name: a_name.into(),
            start_line: 1,
            end_line: 2,
            code: "pass".into(),
            code_hash: a_name.into(),
        };
        let fn_b = FunctionRef {
            file: file_b,
            qualified_name: b_name.into(),
            start_line: 1,
            end_line: 2,
            code: "pass".into(),
            code_hash: b_name.into(),
        };
        let snip = SnippetRef {
            kind: SnippetKind::Func,
            function: fn_a.clone(),
            start_line: 1,
            end_line: 2,
            text: "pass".into(),
            display_text: "pass".into(),
            snippet_hash: a_name.into(),
        };
        let m = CandidateMatch {
            snippet_a: snip.clone(),
            snippet_b: snip,
            similarity: 1.0,
            evidence: "".into(),
        };
        Finding {
            function_a: fn_a,
            function_b: fn_b,
            score: 1.0,
            duplicated_lines: 2,
            evidence: vec![m],
            reasons: vec!["func_threshold".into()],
            metadata: BTreeMap::new(),
        }
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
}
