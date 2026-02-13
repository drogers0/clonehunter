from __future__ import annotations

from dataclasses import replace

from clonehunter.core.types import Finding


def cluster_findings(findings: list[Finding]) -> list[Finding]:
    if not findings:
        return []
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return parent.get(x, x)

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for finding in findings:
        a = finding.function_a.identity
        b = finding.function_b.identity
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    clusters: dict[str, int] = {}
    clustered: list[Finding] = []
    next_id = 1
    for finding in findings:
        root = find(finding.function_a.identity)
        if root not in clusters:
            clusters[root] = next_id
            next_id += 1
        cluster_id = clusters[root]
        meta = dict(finding.metadata)
        meta["cluster_id"] = str(cluster_id)
        clustered.append(replace(finding, metadata=meta))
    return clustered


def filter_clusters(findings: list[Finding], min_size: int) -> list[Finding]:
    if min_size <= 1:
        return findings
    counts: dict[str, int] = {}
    for finding in findings:
        cluster_id = finding.metadata.get("cluster_id")
        if cluster_id is None:
            continue
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    return [f for f in findings if counts.get(f.metadata.get("cluster_id", ""), 0) >= min_size]
