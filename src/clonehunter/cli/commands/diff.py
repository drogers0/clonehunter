from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from clonehunter.cli.commands.overrides import build_base_overrides, clean_overrides
from clonehunter.cli.commands.scan import ScanOptions, run_scan
from clonehunter.core.config_loader import load_config
from clonehunter.core.types import ScanRequest, ScanResult
from clonehunter.io.git import GitError, changed_files
from clonehunter.model.registry import get_engine
from clonehunter.reporting.html_reporter import HtmlReporter
from clonehunter.reporting.json_reporter import JsonReporter
from clonehunter.reporting.sarif_reporter import SarifReporter


def run_diff(
    base: str,
    fmt: str,
    out_path: str,
    paths: list[str],
    engine_name: str | None,
    embedder: str | None,
    index: str | None,
    device: str | None = None,
) -> None:
    requested_paths = paths or ["."]
    try:
        files = changed_files(base, requested_paths)
    except GitError as exc:
        raise SystemExit(f"Failed to determine changed files: {exc}") from exc

    if not files:
        run_scan(
            ScanOptions(
                paths=[],
                fmt=fmt,
                out_path=out_path,
                engine_name=engine_name,
                embedder=embedder,
                index=index,
                device=device,
            )
        )
        return

    __import__("clonehunter.engines")

    overrides = build_base_overrides(
        engine_name=engine_name,
        embedder=embedder,
        index=index,
        device=device,
        env_embedder=os.environ.get("CLONEHUNTER_EMBEDDER"),
    )
    config = load_config(Path.cwd(), clean_overrides(overrides))
    engine = get_engine(config.engine)
    result = engine.scan(ScanRequest(paths=requested_paths, config=config))

    changed = {_normalize_repo_path(file_path) for file_path in files}
    filtered = [
        f
        for f in result.findings
        if _normalize_repo_path(f.function_a.file.path) in changed
        or _normalize_repo_path(f.function_b.file.path) in changed
    ]
    stats = replace(result.stats, finding_count=len(filtered))
    filtered_result = ScanResult(
        findings=filtered,
        stats=stats,
        config_snapshot=result.config_snapshot,
        timing=result.timing,
    )

    if fmt == "json":
        JsonReporter().write(filtered_result, out_path)
    elif fmt == "html":
        HtmlReporter().write(filtered_result, out_path)
    else:
        SarifReporter().write(filtered_result, out_path)


def _normalize_repo_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        with suppress(ValueError):
            path = path.relative_to(Path.cwd())
    normalized = path.as_posix()
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized
