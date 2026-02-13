from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from clonehunter._compat.toml import loads as toml_loads
from clonehunter.core.config import CloneHunterConfig


def load_config(root: Path, overrides: dict[str, Any] | None = None) -> CloneHunterConfig:
    overrides = overrides or {}
    config = CloneHunterConfig()
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        data = toml_loads(pyproject.read_text(encoding="utf-8"))
        tool_cfg: dict[str, Any] = data.get("tool", {}).get("clonehunter", {})
        config = _apply_config(config, tool_cfg)
    config = _apply_config(config, overrides)
    return config


def _apply_config(config: CloneHunterConfig, cfg: dict[str, Any]) -> CloneHunterConfig:
    if not cfg:
        return config
    if "engine" in cfg:
        config = replace(config, engine=cfg["engine"])
    if "include_globs" in cfg:
        config = replace(config, include_globs=list(cfg["include_globs"]))
    if "exclude_globs" in cfg:
        config = replace(config, exclude_globs=list(cfg["exclude_globs"]))
    if "cluster_findings" in cfg:
        config = replace(config, cluster_findings=bool(cfg["cluster_findings"]))
    if "cluster_min_size" in cfg:
        config = replace(config, cluster_min_size=int(cfg["cluster_min_size"]))
    if "windows" in cfg:
        w = cfg["windows"]
        config = replace(
            config,
            windows=replace(
                config.windows,
                window_lines=w.get("window_lines", config.windows.window_lines),
                stride_lines=w.get("stride_lines", config.windows.stride_lines),
                min_nonempty=w.get("min_nonempty", config.windows.min_nonempty),
            ),
        )
    if "expansion" in cfg:
        e = cfg["expansion"]
        config = replace(
            config,
            expansion=replace(
                config.expansion,
                enabled=e.get("enabled", config.expansion.enabled),
                depth=e.get("depth", config.expansion.depth),
                max_chars=e.get("max_chars", config.expansion.max_chars),
            ),
        )
    if "thresholds" in cfg:
        t = cfg["thresholds"]
        config = replace(
            config,
            thresholds=replace(
                config.thresholds,
                func=t.get("func", config.thresholds.func),
                win=t.get("win", config.thresholds.win),
                exp=t.get("exp", config.thresholds.exp),
                min_window_hits=t.get("min_window_hits", config.thresholds.min_window_hits),
                lexical_min_ratio=t.get("lexical_min_ratio", config.thresholds.lexical_min_ratio),
                lexical_weight=t.get("lexical_weight", config.thresholds.lexical_weight),
            ),
        )
    if "index" in cfg:
        i = cfg["index"]
        config = replace(
            config,
            index=replace(
                config.index,
                name=i.get("name", config.index.name),
                top_k=i.get("top_k", config.index.top_k),
                faiss_nlist=i.get("faiss_nlist", config.index.faiss_nlist),
                faiss_nprobe=i.get("faiss_nprobe", config.index.faiss_nprobe),
            ),
        )
    if "cache" in cfg:
        c = cfg["cache"]
        config = replace(config, cache=replace(config.cache, path=c.get("path", config.cache.path)))
    if "embedder" in cfg:
        e = cfg["embedder"]
        config = replace(
            config,
            embedder=replace(
                config.embedder,
                name=e.get("name", config.embedder.name),
                model_name=e.get("model_name", config.embedder.model_name),
                revision=e.get("revision", config.embedder.revision),
                max_length=e.get("max_length", config.embedder.max_length),
                batch_size=e.get("batch_size", config.embedder.batch_size),
                device=e.get("device", config.embedder.device),
            ),
        )
    return config
