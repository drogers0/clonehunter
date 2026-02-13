from __future__ import annotations

import argparse

from clonehunter.cli.commands.diff import run_diff
from clonehunter.cli.commands.scan import REPO_TYPE_PRESETS, ScanOptions, run_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clonehunter")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan a repository for semantic clones")
    scan.add_argument("path", nargs="*", default=["."])
    scan.add_argument("--format", choices=["json", "html", "sarif"], default="html")
    scan.add_argument("--out", default=None)
    scan.add_argument("--engine", choices=["semantic", "sonarqube"], default=None)
    scan.add_argument("--embedder", choices=["codebert", "stub"], default=None)
    scan.add_argument("--index", choices=["brute", "faiss"], default=None)
    scan.add_argument("--threshold-func", type=float, default=None)
    scan.add_argument("--threshold-win", type=float, default=None)
    scan.add_argument("--threshold-exp", type=float, default=None)
    scan.add_argument("--min-window-hits", type=int, default=None)
    scan.add_argument("--lexical-min-ratio", type=float, default=None)
    scan.add_argument("--lexical-weight", type=float, default=None)
    scan.add_argument("--window-lines", type=int, default=None)
    scan.add_argument("--stride-lines", type=int, default=None)
    scan.add_argument("--min-nonempty", type=int, default=None)
    scan.add_argument("--expand-calls", action="store_true")
    scan.add_argument("--expand-depth", type=int, default=None)
    scan.add_argument("--expand-max-chars", type=int, default=None)
    scan.add_argument("--cache-path", default=None)
    scan.add_argument("--cluster", action="store_true")
    scan.add_argument("--cluster-min-size", type=int, default=None)
    scan.add_argument(
        "--repotype", action="append", choices=sorted(REPO_TYPE_PRESETS), default=None
    )
    scan.add_argument("--include-globs", action="append", default=None)
    scan.add_argument("--exclude-globs", action="append", default=None)

    diff = sub.add_parser("diff", help="Scan only changed files")
    diff.add_argument("--base", default="HEAD")
    diff.add_argument("--format", choices=["json", "html", "sarif"], default="html")
    diff.add_argument("--out", default=None)
    diff.add_argument("--engine", choices=["semantic", "sonarqube"], default=None)
    diff.add_argument("--embedder", choices=["codebert", "stub"], default=None)
    diff.add_argument("--index", choices=["brute", "faiss"], default=None)

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
            )
        )
    if args.command == "diff":
        run_diff(args.base, args.format, args.out, args.embedder, args.index)
