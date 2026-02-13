from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from clonehunter.core.config_loader import load_config
from clonehunter.core.types import ScanRequest
from clonehunter.model.registry import get_engine
from clonehunter.reporting.html_reporter import HtmlReporter
from clonehunter.reporting.json_reporter import JsonReporter
from clonehunter.reporting.sarif_reporter import SarifReporter

REPO_TYPE_PRESETS: dict[str, tuple[list[str], list[str]]] = {
    "dotnet": (
        ["**/*.cs", "**/*.vb", "**/*.fs"],
        ["**/bin/**", "**/obj/**", "**/packages/**", "**/.vs/**"],
    ),
    "go": (
        ["**/*.go"],
        ["**/vendor/**", "**/bin/**", "**/dist/**", "**/.git/**"],
    ),
    "java": (
        ["**/*.java"],
        ["**/target/**", "**/build/**", "**/.gradle/**", "**/out/**"],
    ),
    "kotlin": (
        ["**/*.kt", "**/*.kts"],
        ["**/build/**", "**/.gradle/**", "**/out/**"],
    ),
    # Special alias resolved in `resolve_repotype_globs`.
    "monorepo": ([], []),
    "none": ([], []),
    "node": (
        ["**/*.js", "**/*.mjs", "**/*.cjs", "**/*.ts"],
        [
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "**/.next/**",
            "**/.turbo/**",
            "**/coverage/**",
        ],
    ),
    "php": (
        ["**/*.php"],
        ["**/vendor/**", "**/node_modules/**", "**/storage/**", "**/bootstrap/cache/**"],
    ),
    "python": (
        ["**/*.py"],
        ["**/.venv/**", "**/venv/**", "**/__pycache__/**", "**/site-packages/**"],
    ),
    "react": (
        ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx"],
        ["**/node_modules/**", "**/.next/**", "**/dist/**", "**/build/**", "**/coverage/**"],
    ),
    "ruby": (
        ["**/*.rb", "**/*.rake"],
        ["**/vendor/**", "**/tmp/**", "**/log/**", "**/coverage/**"],
    ),
    "rust": (
        ["**/*.rs"],
        ["**/target/**"],
    ),
    "swift": (
        ["**/*.swift"],
        ["**/.build/**", "**/DerivedData/**", "**/build/**"],
    ),
    "cpp": (
        ["**/*.c", "**/*.cc", "**/*.cpp", "**/*.cxx", "**/*.h", "**/*.hh", "**/*.hpp", "**/*.hxx"],
        ["**/build/**", "**/out/**", "**/bin/**", "**/obj/**", "**/cmake-build-*/**"],
    ),
}


@dataclass(frozen=True, slots=True)
class ScanOptions:
    paths: list[str]
    fmt: str
    out_path: str
    embedder: str | None = None
    index: str | None = None
    engine_name: str | None = None
    threshold_func: float | None = None
    threshold_win: float | None = None
    threshold_exp: float | None = None
    min_window_hits: int | None = None
    lexical_min_ratio: float | None = None
    lexical_weight: float | None = None
    window_lines: int | None = None
    stride_lines: int | None = None
    min_nonempty: int | None = None
    expand_calls: bool = False
    expand_depth: int | None = None
    expand_max_chars: int | None = None
    cache_path: str | None = None
    cluster: bool = False
    cluster_min_size: int | None = None
    repotypes: list[str] | None = None
    include_globs: list[str] | None = None
    exclude_globs: list[str] | None = None


def run_scan(options: ScanOptions) -> None:
    env_embedder = os.environ.get("CLONEHUNTER_EMBEDDER", "").strip().lower()
    overrides: dict[str, object] = {}
    if options.engine_name:
        overrides["engine"] = options.engine_name
    if options.embedder:
        overrides["embedder"] = {"name": options.embedder}
    if options.index:
        overrides["index"] = {"name": options.index}
    if env_embedder == "stub" and "embedder" not in overrides:
        overrides["embedder"] = {"name": "stub"}
    if (
        options.threshold_func is not None
        or options.threshold_win is not None
        or options.threshold_exp is not None
        or options.min_window_hits is not None
        or options.lexical_min_ratio is not None
        or options.lexical_weight is not None
    ):
        overrides["thresholds"] = {
            "func": options.threshold_func,
            "win": options.threshold_win,
            "exp": options.threshold_exp,
            "min_window_hits": options.min_window_hits,
            "lexical_min_ratio": options.lexical_min_ratio,
            "lexical_weight": options.lexical_weight,
        }
    if (
        options.window_lines is not None
        or options.stride_lines is not None
        or options.min_nonempty is not None
    ):
        overrides["windows"] = {
            "window_lines": options.window_lines,
            "stride_lines": options.stride_lines,
            "min_nonempty": options.min_nonempty,
        }
    if (
        options.expand_calls
        or options.expand_depth is not None
        or options.expand_max_chars is not None
    ):
        overrides["expansion"] = {
            "enabled": True
            if (
                options.expand_calls
                or options.expand_depth is not None
                or options.expand_max_chars is not None
            )
            else None,
            "depth": options.expand_depth,
            "max_chars": options.expand_max_chars,
        }
    if options.cache_path:
        overrides["cache"] = {"path": options.cache_path}
    if options.cluster:
        overrides["cluster_findings"] = True
    if options.cluster_min_size is not None:
        overrides["cluster_min_size"] = options.cluster_min_size

    __import__("clonehunter.engines")

    config = load_config(Path.cwd(), _clean_overrides(overrides))
    repotype_include, repotype_exclude = resolve_repotype_globs(
        effective_repotypes(options.repotypes)
    )
    include_globs, exclude_globs = merge_globs(
        config.include_globs,
        config.exclude_globs,
        repotype_include,
        repotype_exclude,
    )
    include_globs, exclude_globs = merge_globs(
        include_globs,
        exclude_globs,
        options.include_globs or [],
        options.exclude_globs or [],
    )
    config = replace(config, include_globs=include_globs, exclude_globs=exclude_globs)
    engine = get_engine(config.engine)
    result = engine.scan(ScanRequest(paths=options.paths, config=config))

    if options.fmt == "json":
        JsonReporter().write(result, options.out_path)
    elif options.fmt == "html":
        HtmlReporter().write(result, options.out_path)
    else:
        SarifReporter().write(result, options.out_path)


def _clean_overrides(overrides: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            value_dict = cast(dict[str, object], value)
            filtered: dict[str, object] = {k: v for k, v in value_dict.items() if v is not None}
            if filtered:
                cleaned[key] = filtered
        elif value is not None:
            cleaned[key] = value
    return cleaned


def merge_globs(
    base_include: list[str],
    base_exclude: list[str],
    cli_include: list[str],
    cli_exclude: list[str],
) -> tuple[list[str], list[str]]:
    include = _dedupe(base_include + cli_include)
    exclude = _dedupe(base_exclude + cli_exclude)

    # CLI entries override conflicting pyproject entries.
    for pattern in cli_include:
        exclude = [value for value in exclude if value != pattern]
    for pattern in cli_exclude:
        include = [value for value in include if value != pattern]
    return include, exclude


def resolve_repotype_globs(repotypes: list[str]) -> tuple[list[str], list[str]]:
    include: list[str] = []
    exclude: list[str] = []
    for repotype in repotypes:
        if repotype == "monorepo":
            for key, (repotype_include, repotype_exclude) in REPO_TYPE_PRESETS.items():
                if key == "monorepo":
                    continue
                include.extend(repotype_include)
                exclude.extend(repotype_exclude)
            continue
        repotype_include, repotype_exclude = REPO_TYPE_PRESETS[repotype]
        include.extend(repotype_include)
        exclude.extend(repotype_exclude)
    return _dedupe(include), _dedupe(exclude)


def effective_repotypes(repotypes: list[str] | None) -> list[str]:
    if repotypes:
        filtered = [repotype for repotype in repotypes if repotype != "none"]
        return filtered
    return ["monorepo"]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
