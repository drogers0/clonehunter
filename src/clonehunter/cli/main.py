from __future__ import annotations

import argparse

from clonehunter.cli.commands.diff import run_diff
from clonehunter.cli.commands.scan import REPO_TYPE_PRESETS, ScanOptions, run_scan


def _add_common_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=["json", "html", "sarif"],
        default="html",
        help="Output report format.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Auto-derived from --format when omitted.",
    )


def _add_common_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=["semantic", "sonarqube"],
        default=None,
        help="Scan engine override; defaults to config value.",
    )
    parser.add_argument(
        "--embedder",
        choices=["codebert", "faster", "stub"],
        default=None,
        help="Embedding backend override; defaults to config value.",
    )
    parser.add_argument(
        "--index",
        choices=["brute", "faiss"],
        default=None,
        help="Vector index override; defaults to config value.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default=None,
        help="Embedder device override.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clonehunter",
        description="Find semantic code clones in repositories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan",
        help="Scan a repository for semantic clones",
        description="Scan one or more paths for duplicate code patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scan.add_argument("path", nargs="*", default=["."], help="Files/directories to scan.")
    _add_common_report_args(scan)
    _add_common_override_args(scan)
    scan.add_argument(
        "--threshold-func", type=float, default=None, help="Function threshold override [0,1]."
    )
    scan.add_argument(
        "--threshold-win", type=float, default=None, help="Window threshold override [0,1]."
    )
    scan.add_argument(
        "--threshold-exp", type=float, default=None, help="Expansion threshold override [0,1]."
    )
    scan.add_argument(
        "--min-window-hits", type=int, default=None, help="Minimum window hits required."
    )
    scan.add_argument(
        "--lexical-min-ratio", type=float, default=None, help="Lexical floor in [0,1]."
    )
    scan.add_argument(
        "--lexical-weight", type=float, default=None, help="Lexical blend weight in [0,1]."
    )
    scan.add_argument("--window-lines", type=int, default=None, help="Window size in lines.")
    scan.add_argument("--stride-lines", type=int, default=None, help="Window stride in lines.")
    scan.add_argument(
        "--min-nonempty", type=int, default=None, help="Minimum non-empty lines per window."
    )
    scan.add_argument("--expand-calls", action="store_true", help="Enable call-expansion snippets.")
    scan.add_argument(
        "--expand-depth", type=int, default=None, help="Call-expansion depth override."
    )
    scan.add_argument(
        "--expand-max-chars", type=int, default=None, help="Call-expansion size cap override."
    )
    scan.add_argument("--cache-path", default=None, help="Embedding cache directory override.")
    scan.add_argument("--cluster", action="store_true", help="Enable cluster post-processing.")
    scan.add_argument("--cluster-min-size", type=int, default=None, help="Minimum cluster size.")
    scan.add_argument(
        "--repotype",
        action="append",
        choices=sorted(REPO_TYPE_PRESETS),
        default=None,
        help="Repeatable language preset globs. Use --repotype none to disable preset globs.",
    )
    scan.add_argument(
        "--include-globs",
        action="append",
        default=None,
        help="Repeatable include glob appended after config/preset globs.",
    )
    scan.add_argument(
        "--exclude-globs",
        action="append",
        default=None,
        help="Repeatable exclude glob appended after config/preset globs.",
    )
    diff = sub.add_parser(
        "diff",
        help="Scan only changed files",
        description="Scan files changed from a git base reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    diff.add_argument(
        "path", nargs="*", default=["."], help="Files/directories to scope changed-file discovery."
    )
    diff.add_argument(
        "--base", default="HEAD", help="Git base revision for changed-file detection."
    )
    _add_common_report_args(diff)
    _add_common_override_args(diff)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.out is None:
        ext = {"json": "json", "html": "html", "sarif": "sarif"}[args.format]
        args.out = f"clonehunter_report.{ext}"
    if args.command == "scan":
        run_scan(
            ScanOptions(
                paths=args.path,
                fmt=args.format,
                out_path=args.out,
                embedder=args.embedder,
                index=args.index,
                engine_name=args.engine,
                threshold_func=args.threshold_func,
                threshold_win=args.threshold_win,
                threshold_exp=args.threshold_exp,
                min_window_hits=args.min_window_hits,
                lexical_min_ratio=args.lexical_min_ratio,
                lexical_weight=args.lexical_weight,
                window_lines=args.window_lines,
                stride_lines=args.stride_lines,
                min_nonempty=args.min_nonempty,
                expand_calls=args.expand_calls,
                expand_depth=args.expand_depth,
                expand_max_chars=args.expand_max_chars,
                cache_path=args.cache_path,
                cluster=args.cluster,
                cluster_min_size=args.cluster_min_size,
                repotypes=args.repotype,
                include_globs=args.include_globs,
                exclude_globs=args.exclude_globs,
                device=args.device,
            )
        )
    elif args.command == "diff":
        run_diff(
            args.base,
            args.format,
            args.out,
            args.path,
            args.engine,
            args.embedder,
            args.index,
            args.device,
        )
