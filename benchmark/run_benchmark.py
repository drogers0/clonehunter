#!/usr/bin/env python3
"""Benchmark suite for CloneHunter.

Clones mid-sized open-source repositories at pinned commit SHAs and runs
``clonehunter scan`` on each **twice** (cold cache, then warm cache),
capturing detection, caching, and timing metrics.

Usage:
    # Run benchmark and print results
    python benchmark/run_benchmark.py

    # Save current results as the baseline
    python benchmark/run_benchmark.py --save-baseline

    # Compare against a saved baseline
    python benchmark/run_benchmark.py --compare-baseline

    # Skip cloning (reuse already-cloned repos)
    python benchmark/run_benchmark.py --skip-clone

    # Only run a subset
    python benchmark/run_benchmark.py --repos click requests
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
REPOS_DIR = BENCHMARK_DIR / "repos"
BASELINE_PATH = BENCHMARK_DIR / "baseline.json"

# Pinned repos: name → (url, tag, expected_commit_sha).
# Tags for shallow cloning, SHAs for integrity verification.
REPOS: dict[str, tuple[str, str, str]] = {
    "click": (
        "https://github.com/pallets/click.git",
        "8.1.8",
        "934813e4d421071a1b3db3973c02fe2721359a6e",
    ),
    "requests": (
        "https://github.com/psf/requests.git",
        "v2.32.3",
        "0e322af87745eff34caffe4df68456ebc20d9068",
    ),
    "attrs": (
        "https://github.com/python-attrs/attrs.git",
        "24.3.0",
        "598494a618410490cfbe0c896b7a544f6d23e0d9",
    ),
    "rich": (
        "https://github.com/Textualize/rich.git",
        "v13.9.4",
        "43d3b04725ab9731727fb1126e35980c62f32377",
    ),
}

# Consistent scan settings so results are deterministic (modulo model
# non-determinism on different hardware).
SCAN_FLAGS: list[str] = [
    "--format",
    "json",
    "--repotype",
    "python",
    "--engine",
    "semantic",
    "--embedder",
    "codebert",
    "--index",
    "brute",
    "--threshold-func",
    "0.92",
    "--threshold-win",
    "0.90",
    "--threshold-exp",
    "0.90",
    "--min-window-hits",
    "2",
    "--lexical-min-ratio",
    "0.5",
    "--lexical-weight",
    "0.3",
    "--window-lines",
    "12",
    "--stride-lines",
    "6",
    "--min-nonempty",
    "4",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RepoMetrics:
    """Metrics collected from a single clonehunter scan."""

    # Detection counts (should stay stable across refactors)
    file_count: int = 0
    function_count: int = 0
    snippet_count: int = 0
    candidate_count: int = 0
    finding_count: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    # Timing - cold run (empty cache)
    cold_time_collect_files: float = 0.0
    cold_time_extract_functions: float = 0.0
    cold_time_generate_snippets: float = 0.0
    cold_time_embed: float = 0.0
    cold_time_similarity: float = 0.0
    cold_time_total: float = 0.0
    cold_cache_hits: int = 0
    cold_cache_misses: int = 0

    # Timing - warm run (fully cached embeddings)
    warm_time_collect_files: float = 0.0
    warm_time_extract_functions: float = 0.0
    warm_time_generate_snippets: float = 0.0
    warm_time_embed: float = 0.0
    warm_time_similarity: float = 0.0
    warm_time_total: float = 0.0
    warm_cache_hits: int = 0
    warm_cache_misses: int = 0

    # Per-finding scores (sorted, for stable comparison)
    finding_scores: list[float] = field(default_factory=lambda: list[float]())

    # Per-finding file pairs (sorted, for checking we find the same clones)
    finding_pairs: list[str] = field(default_factory=lambda: list[str]())


@dataclass
class BenchmarkResult:
    """Full benchmark run result."""

    timestamp: str = ""
    environment: dict[str, str] = field(
        default_factory=lambda: dict[str, str](),
    )
    results: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict[str, dict[str, Any]](),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list[str],
    *,
    check: bool = False,
    capture_output: bool = False,
    text: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def _get_pkg_version(pkg: str) -> str:
    """Get installed package version, or 'not installed'."""
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "not installed"


def _get_clonehunter_git_sha() -> str:
    """Get the current git commit SHA of the clonehunter repo."""
    result = subprocess.run(
        ["git", "-C", str(BENCHMARK_DIR.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _scan_flag(name: str) -> str:
    """Extract a --flag value from SCAN_FLAGS."""
    for i, flag in enumerate(SCAN_FLAGS):
        if flag == name and i + 1 < len(SCAN_FLAGS):
            return SCAN_FLAGS[i + 1]
    return "unknown"


def _get_torch_device() -> str:
    """Return the device that ``resolve_device("auto")`` would pick."""
    try:
        from clonehunter.embedding.codebert_embedder import resolve_device

        return resolve_device("auto")
    except Exception:
        return "unknown"


def collect_environment() -> dict[str, str]:
    """Collect environment metadata for reproducibility."""
    return {
        # Code version
        "clonehunter_version": _get_pkg_version("clonehunter"),
        "clonehunter_git_sha": _get_clonehunter_git_sha(),
        # Dependencies
        "python_version": platform.python_version(),
        "numpy_version": _get_pkg_version("numpy"),
        "torch_version": _get_pkg_version("torch"),
        "transformers_version": _get_pkg_version("transformers"),
        "faiss_version": _get_pkg_version("faiss-cpu"),
        # Scan configuration (affects results/timing)
        "engine": _scan_flag("--engine"),
        "embedder": _scan_flag("--embedder"),
        "index": _scan_flag("--index"),
        "repotype": _scan_flag("--repotype"),
        "threshold_func": _scan_flag("--threshold-func"),
        "threshold_win": _scan_flag("--threshold-win"),
        "threshold_exp": _scan_flag("--threshold-exp"),
        "min_window_hits": _scan_flag("--min-window-hits"),
        "lexical_min_ratio": _scan_flag("--lexical-min-ratio"),
        "lexical_weight": _scan_flag("--lexical-weight"),
        "window_lines": _scan_flag("--window-lines"),
        "stride_lines": _scan_flag("--stride-lines"),
        "min_nonempty": _scan_flag("--min-nonempty"),
        # Hardware
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": str(os.cpu_count() or "unknown"),
        # Torch device (resolved from "auto")
        "torch_device": _get_torch_device(),
    }


def clone_repo(name: str, url: str, tag: str, expected_sha: str) -> Path:
    """Clone a repo at a pinned tag and verify the commit SHA."""
    dest = REPOS_DIR / name
    if dest.exists():
        # Verify existing clone matches expected SHA
        result = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        actual_sha = result.stdout.strip()
        if actual_sha != expected_sha:
            print(f"  [{name}] SHA mismatch: expected {expected_sha[:12]}, got {actual_sha[:12]}")
            print(f"  [{name}] Removing stale clone and re-cloning...")
            shutil.rmtree(dest)
        else:
            print(f"  [{name}] Already cloned, SHA verified: {actual_sha[:12]}")
            return dest

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", tag, url, str(dest)],
        check=True,
    )

    # Verify commit SHA
    result = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    actual_sha = result.stdout.strip()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"[{name}] SHA verification failed!\n"
            f"  Expected: {expected_sha}\n"
            f"  Actual:   {actual_sha}\n"
            f"  Tag '{tag}' may have been moved upstream."
        )
    print(f"  [{name}] Cloned {url} @ {tag} (SHA: {actual_sha[:12]})")
    return dest


def run_scan(repo_path: Path, out_json: Path, cache_path: Path) -> float:
    """Run ``clonehunter scan`` and return wall-clock time."""
    cmd = [
        sys.executable,
        "-m",
        "clonehunter",
        "scan",
        str(repo_path),
        "--out",
        str(out_json),
        "--cache-path",
        str(cache_path),
        *SCAN_FLAGS,
    ]
    t0 = time.perf_counter()
    result = _run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"clonehunter scan failed (exit {result.returncode})")
    return elapsed


def _parse_json(json_path: Path) -> dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
    return result


def _extract_timing(
    data: dict[str, Any],
    wall_time: float,
    prefix: str,
) -> dict[str, float]:
    """Extract timing fields from a scan result, prefixed for cold/warm."""
    timing = data["timing"]
    return {
        f"{prefix}_time_collect_files": timing.get("collect_files", 0.0),
        f"{prefix}_time_extract_functions": timing.get("extract_functions", 0.0),
        f"{prefix}_time_generate_snippets": timing.get("generate_snippets", 0.0),
        f"{prefix}_time_embed": timing.get("embed", 0.0),
        f"{prefix}_time_similarity": timing.get("similarity", 0.0),
        f"{prefix}_time_total": wall_time,
    }


def _extract_cache(data: dict[str, Any], prefix: str) -> dict[str, int]:
    """Extract cache hit/miss from a scan result, prefixed for cold/warm."""
    stats = data["stats"]
    return {
        f"{prefix}_cache_hits": stats.get("cache_hits", 0),
        f"{prefix}_cache_misses": stats.get("cache_misses", 0),
    }


def parse_metrics(
    cold_json: Path,
    cold_wall: float,
    warm_json: Path,
    warm_wall: float,
    repo_path: Path | None = None,
) -> RepoMetrics:
    """Parse both cold and warm JSON reports into a single RepoMetrics."""
    cold_data = _parse_json(cold_json)
    warm_data = _parse_json(warm_json)

    # Use warm run for detection metrics (should be identical to cold)
    stats = warm_data["stats"]

    # Strip repo path prefix for readable finding pairs
    prefix = str(repo_path) + "/" if repo_path else ""

    def _rel(p: str) -> str:
        return p.removeprefix(prefix) if prefix else p

    # Build sorted list of finding scores and file pairs for stable comparison
    finding_scores: list[float] = []
    finding_pairs: list[str] = []
    for finding in warm_data.get("findings", []):
        finding_scores.append(round(finding["score"], 6))
        fa = finding["function_a"]
        fb = finding["function_b"]
        pair = "::".join(
            sorted(
                [
                    _rel(fa["file"]["path"]) + ":" + fa["qualified_name"],
                    _rel(fb["file"]["path"]) + ":" + fb["qualified_name"],
                ]
            )
        )
        finding_pairs.append(pair)

    finding_scores.sort()
    finding_pairs.sort()

    return RepoMetrics(
        file_count=stats["file_count"],
        function_count=stats["function_count"],
        snippet_count=stats["snippet_count"],
        candidate_count=stats["candidate_count"],
        finding_count=stats["finding_count"],
        cache_hits=stats.get("cache_hits", 0),
        cache_misses=stats.get("cache_misses", 0),
        **_extract_timing(cold_data, cold_wall, "cold"),
        **_extract_cache(cold_data, "cold"),
        **_extract_timing(warm_data, warm_wall, "warm"),
        **_extract_cache(warm_data, "warm"),
        finding_scores=finding_scores,
        finding_pairs=finding_pairs,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

DETECTION_FIELDS = [
    "file_count",
    "function_count",
    "snippet_count",
    "candidate_count",
    "finding_count",
]
DETECTION_LABELS = {
    "file_count": "Files",
    "function_count": "Functions",
    "snippet_count": "Snippets",
    "candidate_count": "Candidates",
    "finding_count": "Findings",
}

CACHE_FIELDS = [
    "cold_cache_hits",
    "cold_cache_misses",
    "warm_cache_hits",
    "warm_cache_misses",
]
CACHE_LABELS = {
    "cold_cache_hits": "Cold hits",
    "cold_cache_misses": "Cold misses",
    "warm_cache_hits": "Warm hits",
    "warm_cache_misses": "Warm misses",
}

COLD_TIMING_FIELDS = [
    "cold_time_collect_files",
    "cold_time_extract_functions",
    "cold_time_generate_snippets",
    "cold_time_embed",
    "cold_time_similarity",
    "cold_time_total",
]
WARM_TIMING_FIELDS = [
    "warm_time_collect_files",
    "warm_time_extract_functions",
    "warm_time_generate_snippets",
    "warm_time_embed",
    "warm_time_similarity",
    "warm_time_total",
]
TIMING_LABELS = {
    "cold_time_collect_files": "Collect files",
    "cold_time_extract_functions": "Extract funcs",
    "cold_time_generate_snippets": "Gen snippets",
    "cold_time_embed": "Embed",
    "cold_time_similarity": "Similarity",
    "cold_time_total": "TOTAL (wall)",
    "warm_time_collect_files": "Collect files",
    "warm_time_extract_functions": "Extract funcs",
    "warm_time_generate_snippets": "Gen snippets",
    "warm_time_embed": "Embed",
    "warm_time_similarity": "Similarity",
    "warm_time_total": "TOTAL (wall)",
}


def print_summary(all_metrics: dict[str, RepoMetrics]) -> None:
    """Print a summary table of all results."""
    repos = list(all_metrics.keys())
    col_w = max(14, *(len(r) for r in repos))

    header = f"{'Metric':<20s}" + "".join(f"{r:>{col_w}s}" for r in repos)
    sep = "-" * len(header)

    # Detection metrics
    print("\n" + sep)
    print("DETECTION METRICS")
    print(sep)
    print(header)
    print(sep)
    for fld in DETECTION_FIELDS:
        label = DETECTION_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            val = getattr(all_metrics[repo], fld)
            row += f"{val:>{col_w},d}"
        print(row)

    # Cache metrics
    print("\n" + sep)
    print("CACHE METRICS")
    print(sep)
    print(header)
    print(sep)
    for fld in CACHE_FIELDS:
        label = CACHE_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            val = getattr(all_metrics[repo], fld)
            row += f"{val:>{col_w},d}"
        print(row)
    # Cache effectiveness
    row = f"{'Warm hit rate':<20s}"
    for repo in repos:
        m = all_metrics[repo]
        total = m.warm_cache_hits + m.warm_cache_misses
        rate = m.warm_cache_hits / total * 100 if total else 0.0
        row += f"{rate:>{col_w}.1f}%"
    print(row)

    # Cold timing
    print("\n" + sep)
    print("TIMING — COLD RUN (empty cache, seconds)")
    print(sep)
    print(header)
    print(sep)
    for fld in COLD_TIMING_FIELDS:
        label = TIMING_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            val = getattr(all_metrics[repo], fld)
            row += f"{val:>{col_w}.3f}"
        print(row)

    # Warm timing
    print("\n" + sep)
    print("TIMING — WARM RUN (cached embeddings, seconds)")
    print(sep)
    print(header)
    print(sep)
    for fld in WARM_TIMING_FIELDS:
        label = TIMING_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            val = getattr(all_metrics[repo], fld)
            row += f"{val:>{col_w}.3f}"
        print(row)

    # Speedup from caching
    row = f"{'Speedup (cold/warm)':<20s}"
    for repo in repos:
        m = all_metrics[repo]
        speedup = m.cold_time_total / m.warm_time_total if m.warm_time_total > 0 else 0
        row += f"{speedup:>{col_w}.2f}x"
    print(row)
    print(sep)

    # Finding details
    print("\nFINDING DETAILS")
    print(sep)
    for repo in repos:
        m = all_metrics[repo]
        print(f"\n  {repo} ({m.finding_count} findings):")
        if m.finding_pairs:
            for pair in m.finding_pairs:
                print(f"    - {pair}")
        else:
            print("    (none)")
    print()


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def save_baseline(all_metrics: dict[str, RepoMetrics]) -> None:
    """Save current results as the baseline."""
    data = BenchmarkResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        environment=collect_environment(),
        results={name: asdict(m) for name, m in all_metrics.items()},
    )
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(data), f, indent=2)
    print(f"\nBaseline saved to {BASELINE_PATH}")


def load_baseline() -> dict[str, RepoMetrics] | None:
    """Load a previously saved baseline."""
    if not BASELINE_PATH.exists():
        return None
    with open(BASELINE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, RepoMetrics] = {}
    for name, d in data["results"].items():
        result[name] = RepoMetrics(**d)
    return result


def compare_baseline(
    current: dict[str, RepoMetrics],
    baseline: dict[str, RepoMetrics],
    timing_tolerance: float = 0.30,
) -> bool:
    """Compare current results against baseline.

    Detection and cache metrics must match exactly. Timing is compared with a
    tolerance (default 30%) and reported but does not cause failure.

    Returns True if detection metrics match.
    """
    repos = sorted(set(current.keys()) & set(baseline.keys()))
    if not repos:
        print("No overlapping repos between current run and baseline.")
        return False

    all_ok = True

    col_w = max(14, *(len(r) for r in repos))

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    # Detection metrics (must match exactly)
    print("\nDetection metrics (exact match required):")
    print("-" * 80)
    for fld in DETECTION_FIELDS:
        label = DETECTION_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            bval = getattr(baseline[repo], fld)
            cval = getattr(current[repo], fld)
            delta = cval - bval
            marker = "" if delta == 0 else " ***"
            if delta != 0:
                all_ok = False
            row += f"{bval:>{col_w},d}{cval:>{col_w},d}{delta:>+{col_w},d}{marker}"
        print(row)

    # Cache metrics (must match exactly)
    print("\nCache metrics (exact match required):")
    print("-" * 80)
    for fld in CACHE_FIELDS:
        label = CACHE_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            bval = getattr(baseline[repo], fld)
            cval = getattr(current[repo], fld)
            delta = cval - bval
            marker = "" if delta == 0 else " ***"
            if delta != 0:
                all_ok = False
            row += f"{bval:>{col_w},d}{cval:>{col_w},d}{delta:>+{col_w},d}{marker}"
        print(row)

    # Finding pairs comparison
    print("\nFinding pairs (must match exactly):")
    print("-" * 80)
    for repo in repos:
        b_pairs = set(baseline[repo].finding_pairs)
        c_pairs = set(current[repo].finding_pairs)
        added = c_pairs - b_pairs
        removed = b_pairs - c_pairs
        if not added and not removed:
            print(f"  {repo}: OK (all {len(c_pairs)} pairs match)")
        else:
            all_ok = False
            if added:
                print(f"  {repo}: NEW findings:")
                for p in sorted(added):
                    print(f"    + {p}")
            if removed:
                print(f"  {repo}: MISSING findings:")
                for p in sorted(removed):
                    print(f"    - {p}")

    # Finding scores comparison
    print("\nFinding scores (must match within 1e-4):")
    print("-" * 80)
    for repo in repos:
        b_scores = baseline[repo].finding_scores
        c_scores = current[repo].finding_scores
        if len(b_scores) != len(c_scores):
            all_ok = False
            print(
                f"  {repo}: MISMATCH - baseline has "
                f"{len(b_scores)} scores, "
                f"current has {len(c_scores)}"
            )
            continue
        diffs = [abs(b - c) for b, c in zip(b_scores, c_scores, strict=True)]
        max_diff = max(diffs) if diffs else 0.0
        if max_diff > 1e-4:
            all_ok = False
            print(f"  {repo}: DRIFT - max score difference = {max_diff:.6f}")
        else:
            print(f"  {repo}: OK (max diff = {max_diff:.6f})")

    # Timing (informational, with tolerance)
    print(f"\nTiming comparison — cold run (informational, tolerance = {timing_tolerance:.0%}):")
    print("-" * 80)
    for fld in COLD_TIMING_FIELDS:
        label = TIMING_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            bval = getattr(baseline[repo], fld)
            cval = getattr(current[repo], fld)
            if bval > 0:
                pct = (cval - bval) / bval
                if pct > timing_tolerance:
                    marker = " SLOW"
                elif pct < -timing_tolerance:
                    marker = " fast"
                else:
                    marker = ""
                row += f"{bval:>{col_w}.3f}{cval:>{col_w}.3f}{pct:>+{col_w}.1%}{marker}"
            else:
                row += f"{bval:>{col_w}.3f}{cval:>{col_w}.3f}{'n/a':>{col_w}s}"
        print(row)

    print(f"\nTiming comparison — warm run (informational, tolerance = {timing_tolerance:.0%}):")
    print("-" * 80)
    for fld in WARM_TIMING_FIELDS:
        label = TIMING_LABELS[fld]
        row = f"{label:<20s}"
        for repo in repos:
            bval = getattr(baseline[repo], fld)
            cval = getattr(current[repo], fld)
            if bval > 0:
                pct = (cval - bval) / bval
                if pct > timing_tolerance:
                    marker = " SLOW"
                elif pct < -timing_tolerance:
                    marker = " fast"
                else:
                    marker = ""
                row += f"{bval:>{col_w}.3f}{cval:>{col_w}.3f}{pct:>+{col_w}.1%}{marker}"
            else:
                row += f"{bval:>{col_w}.3f}{cval:>{col_w}.3f}{'n/a':>{col_w}s}"
        print(row)

    print("\n" + "=" * 80)
    if all_ok:
        print("RESULT: PASS - All detection and cache metrics match the baseline.")
    else:
        print("RESULT: FAIL - Metrics differ from the baseline!")
    print("=" * 80)
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CloneHunter benchmark suite")
    parser.add_argument(
        "--repos",
        nargs="*",
        choices=list(REPOS.keys()),
        default=None,
        help="Only run these repos (default: all)",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning (reuse existing repos)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as the baseline for future comparison",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results against the saved baseline",
    )
    parser.add_argument(
        "--timing-tolerance",
        type=float,
        default=0.30,
        help="Timing regression tolerance as fraction (default: 0.30 = 30%%)",
    )
    args = parser.parse_args()

    selected = args.repos or list(REPOS.keys())

    print("=" * 60)
    print("CloneHunter Benchmark")
    print("=" * 60)

    # 1. Clone repos
    if not args.skip_clone:
        print("\n--- Cloning repositories ---")
        for name in selected:
            url, tag, sha = REPOS[name]
            clone_repo(name, url, tag, sha)
    else:
        print("\n--- Skipping clone (--skip-clone) ---")
        # Still verify SHAs of existing clones
        for name in selected:
            dest = REPOS_DIR / name
            if not dest.exists():
                continue
            _, _, expected_sha = REPOS[name]
            result = subprocess.run(
                ["git", "-C", str(dest), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            actual_sha = result.stdout.strip()
            if actual_sha != expected_sha:
                print(
                    f"  WARNING: [{name}] SHA mismatch"
                    f" - expected {expected_sha[:12]},"
                    f" got {actual_sha[:12]}"
                )
                print("           Re-run without --skip-clone to fix.")
            else:
                print(f"  [{name}] SHA verified: {actual_sha[:12]}")

    # 2. Run scans (cold + warm for each repo)
    all_metrics: dict[str, RepoMetrics] = {}
    output_dir = BENCHMARK_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in selected:
        repo_path = REPOS_DIR / name
        if not repo_path.exists():
            print(f"\n  [{name}] SKIP - repo not found at {repo_path}")
            continue

        # Use an isolated cache directory per repo so cold/warm is controlled
        cache_dir = BENCHMARK_DIR / "cache" / name
        cold_json = output_dir / f"{name}_cold.json"
        warm_json = output_dir / f"{name}_warm.json"

        # --- Cold run: wipe cache to force all embeddings from scratch ---
        print(f"\n--- Scanning {name} (cold — empty cache) ---")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            cold_wall = run_scan(repo_path, cold_json, cache_dir)
            cold_data = _parse_json(cold_json)
            cold_stats = cold_data["stats"]
            print(
                f"  [{name}] Cold done in {cold_wall:.1f}s — "
                f"{cold_stats['finding_count']} findings, "
                f"cache: {cold_stats.get('cache_hits', 0)} hits / "
                f"{cold_stats.get('cache_misses', 0)} misses"
            )
        except Exception as exc:
            print(f"  [{name}] Cold run FAILED: {exc}", file=sys.stderr)
            continue

        # --- Warm run: reuse the cache populated by the cold run ---
        print(f"\n--- Scanning {name} (warm — cached embeddings) ---")
        try:
            warm_wall = run_scan(repo_path, warm_json, cache_dir)
            warm_data = _parse_json(warm_json)
            warm_stats = warm_data["stats"]
            print(
                f"  [{name}] Warm done in {warm_wall:.1f}s — "
                f"{warm_stats['finding_count']} findings, "
                f"cache: {warm_stats.get('cache_hits', 0)} hits / "
                f"{warm_stats.get('cache_misses', 0)} misses"
            )
        except Exception as exc:
            print(f"  [{name}] Warm run FAILED: {exc}", file=sys.stderr)
            continue

        metrics = parse_metrics(cold_json, cold_wall, warm_json, warm_wall, repo_path=repo_path)
        all_metrics[name] = metrics

        speedup = cold_wall / warm_wall if warm_wall > 0 else 0
        print(f"  [{name}] Cache speedup: {speedup:.2f}x")

    if not all_metrics:
        print("\nNo results collected!", file=sys.stderr)
        sys.exit(1)

    # 3. Print summary
    print_summary(all_metrics)

    # 4. Save / compare baseline
    if args.save_baseline:
        save_baseline(all_metrics)

    if args.compare_baseline:
        baseline = load_baseline()
        if baseline is None:
            print(f"\nNo baseline found at {BASELINE_PATH}.")
            print("Run with --save-baseline first.")
            sys.exit(1)
        ok = compare_baseline(all_metrics, baseline, args.timing_tolerance)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
