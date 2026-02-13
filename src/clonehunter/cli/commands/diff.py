from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from clonehunter.cli.commands.scan import ScanOptions, run_scan
from clonehunter.core.config_loader import load_config
from clonehunter.core.types import ScanRequest, ScanResult
from clonehunter.io.git import changed_files
from clonehunter.model.registry import get_engine
from clonehunter.reporting.html_reporter import HtmlReporter
from clonehunter.reporting.json_reporter import JsonReporter
from clonehunter.reporting.sarif_reporter import SarifReporter


def run_diff(
    base: str,
    fmt: str,
    out_path: str,
    embedder: str | None,
    index: str | None,
) -> None:
    files = changed_files(base)
    if not files:
        run_scan(
            ScanOptions(
                paths=[],
                fmt=fmt,
                out_path=out_path,
                embedder=embedder,
                index=index,
            )
        )
        return

    __import__("clonehunter.engines")

    env_embedder = os.environ.get("CLONEHUNTER_EMBEDDER", "").strip().lower()
    overrides: dict[str, object] = {}
    if embedder:
        overrides["embedder"] = {"name": embedder}
    if index:
        overrides["index"] = {"name": index}
    if env_embedder == "stub" and "embedder" not in overrides:
        overrides["embedder"] = {"name": "stub"}
    config = load_config(Path.cwd(), overrides)
    engine = get_engine(config.engine)
    result = engine.scan(ScanRequest(paths=["."], config=config))

    changed = set(files)
    filtered = [
        f
        for f in result.findings
        if f.function_a.file.path in changed or f.function_b.file.path in changed
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
