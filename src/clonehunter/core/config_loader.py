from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from clonehunter._compat.toml import loads as toml_loads
from clonehunter.core.config import EMBEDDER_PRESETS, CloneHunterConfig
from clonehunter.core.errors import ConfigError

_VALID_ENGINES = frozenset(("semantic", "sonarqube"))
_VALID_EMBEDDERS = frozenset(("codebert", "faster", "stub"))
_VALID_INDEXES = frozenset(("brute", "faiss"))
_TRUE_BOOL_TOKENS = frozenset(("1", "true", "yes", "on"))
_FALSE_BOOL_TOKENS = frozenset(("0", "false", "no", "off"))


def load_config(root: Path, overrides: dict[str, Any] | None = None) -> CloneHunterConfig:
    overrides = overrides or {}
    config = CloneHunterConfig()
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        data = toml_loads(pyproject.read_text(encoding="utf-8"))
        tool_cfg = data.get("tool", {}).get("clonehunter", {})
        if not isinstance(tool_cfg, Mapping):
            raise ConfigError("[tool.clonehunter] must be a TOML table")
        config = _apply_config(config, cast(Mapping[str, Any], tool_cfg))
    config = _apply_config(config, overrides)
    validate_config(config)
    return config


def _apply_config(config: CloneHunterConfig, cfg: Mapping[str, Any]) -> CloneHunterConfig:
    if not cfg:
        return config
    if "engine" in cfg:
        config = replace(config, engine=_coerce_choice("engine", cfg["engine"], _VALID_ENGINES))
    if "include_globs" in cfg:
        config = replace(config, include_globs=_coerce_globs("include_globs", cfg["include_globs"]))
    if "exclude_globs" in cfg:
        config = replace(config, exclude_globs=_coerce_globs("exclude_globs", cfg["exclude_globs"]))
    if "cluster_findings" in cfg:
        config = replace(
            config,
            cluster_findings=_coerce_bool("cluster_findings", cfg["cluster_findings"]),
        )
    if "cluster_min_size" in cfg:
        config = replace(
            config, cluster_min_size=_coerce_int("cluster_min_size", cfg["cluster_min_size"])
        )
    if "windows" in cfg:
        w = _coerce_mapping("windows", cfg["windows"])
        config = replace(
            config,
            windows=replace(
                config.windows,
                window_lines=_coerce_int(
                    "windows.window_lines", w.get("window_lines", config.windows.window_lines)
                ),
                stride_lines=_coerce_int(
                    "windows.stride_lines", w.get("stride_lines", config.windows.stride_lines)
                ),
                min_nonempty=_coerce_int(
                    "windows.min_nonempty", w.get("min_nonempty", config.windows.min_nonempty)
                ),
            ),
        )
    if "expansion" in cfg:
        e = _coerce_mapping("expansion", cfg["expansion"])
        config = replace(
            config,
            expansion=replace(
                config.expansion,
                enabled=_coerce_bool(
                    "expansion.enabled", e.get("enabled", config.expansion.enabled)
                ),
                depth=_coerce_int("expansion.depth", e.get("depth", config.expansion.depth)),
                max_chars=_coerce_int(
                    "expansion.max_chars", e.get("max_chars", config.expansion.max_chars)
                ),
            ),
        )
    if "thresholds" in cfg:
        t = _coerce_mapping("thresholds", cfg["thresholds"])
        config = replace(
            config,
            thresholds=replace(
                config.thresholds,
                func=_coerce_float("thresholds.func", t.get("func", config.thresholds.func)),
                win=_coerce_float("thresholds.win", t.get("win", config.thresholds.win)),
                exp=_coerce_float("thresholds.exp", t.get("exp", config.thresholds.exp)),
                min_window_hits=_coerce_int(
                    "thresholds.min_window_hits",
                    t.get("min_window_hits", config.thresholds.min_window_hits),
                ),
                lexical_min_ratio=_coerce_float(
                    "thresholds.lexical_min_ratio",
                    t.get("lexical_min_ratio", config.thresholds.lexical_min_ratio),
                ),
                lexical_weight=_coerce_float(
                    "thresholds.lexical_weight",
                    t.get("lexical_weight", config.thresholds.lexical_weight),
                ),
            ),
        )
    if "index" in cfg:
        i = _coerce_mapping("index", cfg["index"])
        config = replace(
            config,
            index=replace(
                config.index,
                name=_coerce_choice("index.name", i.get("name", config.index.name), _VALID_INDEXES),
                top_k=_coerce_int("index.top_k", i.get("top_k", config.index.top_k)),
                faiss_nlist=_coerce_int(
                    "index.faiss_nlist", i.get("faiss_nlist", config.index.faiss_nlist)
                ),
                faiss_nprobe=_coerce_int(
                    "index.faiss_nprobe", i.get("faiss_nprobe", config.index.faiss_nprobe)
                ),
            ),
        )
    if "cache" in cfg:
        c = _coerce_mapping("cache", cfg["cache"])
        config = replace(
            config,
            cache=replace(
                config.cache, path=_coerce_str("cache.path", c.get("path", config.cache.path))
            ),
        )
    if "embedder" in cfg:
        e = _coerce_mapping("embedder", cfg["embedder"])
        name = _coerce_choice(
            "embedder.name", e.get("name", config.embedder.name), _VALID_EMBEDDERS
        )
        p = EMBEDDER_PRESETS.get(name, {})
        cur = config.embedder
        config = replace(
            config,
            embedder=replace(
                cur,
                name=name,
                model_name=_coerce_str(
                    "embedder.model_name",
                    e.get("model_name", p.get("model_name", cur.model_name)),
                ),
                revision=_coerce_str(
                    "embedder.revision", e.get("revision", p.get("revision", cur.revision))
                ),
                max_length=_coerce_int(
                    "embedder.max_length",
                    e.get("max_length", p.get("max_length", cur.max_length)),
                ),
                batch_size=_coerce_int(
                    "embedder.batch_size",
                    e.get("batch_size", p.get("batch_size", cur.batch_size)),
                ),
                device=_coerce_str("embedder.device", e.get("device", p.get("device", cur.device))),
                trust_remote_code=_coerce_bool(
                    "embedder.trust_remote_code",
                    e.get("trust_remote_code", p.get("trust_remote_code", cur.trust_remote_code)),
                ),
            ),
        )
    return config


def validate_config(config: CloneHunterConfig) -> None:
    if config.engine not in _VALID_ENGINES:
        raise ConfigError(_invalid_choice_message("engine", config.engine, _VALID_ENGINES))
    if config.embedder.name not in _VALID_EMBEDDERS:
        raise ConfigError(
            _invalid_choice_message("embedder.name", config.embedder.name, _VALID_EMBEDDERS)
        )
    if config.index.name not in _VALID_INDEXES:
        raise ConfigError(_invalid_choice_message("index.name", config.index.name, _VALID_INDEXES))

    if config.embedder.batch_size <= 0:
        raise ConfigError("embedder.batch_size must be > 0")
    if config.embedder.max_length <= 0:
        raise ConfigError("embedder.max_length must be > 0")

    if config.index.top_k <= 0:
        raise ConfigError("index.top_k must be > 0")
    if config.index.faiss_nlist <= 0:
        raise ConfigError("index.faiss_nlist must be > 0")
    if config.index.faiss_nprobe <= 0:
        raise ConfigError("index.faiss_nprobe must be > 0")

    if config.windows.window_lines <= 0:
        raise ConfigError("windows.window_lines must be > 0")
    if config.windows.stride_lines <= 0:
        raise ConfigError("windows.stride_lines must be > 0")
    if config.windows.min_nonempty < 0:
        raise ConfigError("windows.min_nonempty must be >= 0")

    _validate_unit_interval("thresholds.func", config.thresholds.func)
    _validate_unit_interval("thresholds.win", config.thresholds.win)
    _validate_unit_interval("thresholds.exp", config.thresholds.exp)
    _validate_unit_interval("thresholds.lexical_min_ratio", config.thresholds.lexical_min_ratio)
    _validate_unit_interval("thresholds.lexical_weight", config.thresholds.lexical_weight)

    if config.thresholds.min_window_hits < 1:
        raise ConfigError("thresholds.min_window_hits must be >= 1")

    if config.cluster_min_size < 1:
        raise ConfigError("cluster_min_size must be >= 1")

    if config.expansion.depth < 0:
        raise ConfigError("expansion.depth must be >= 0")
    if config.expansion.max_chars <= 0:
        raise ConfigError("expansion.max_chars must be > 0")


def _coerce_mapping(field_name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be a table/object")
    return cast(Mapping[str, Any], value)


def _coerce_globs(field_name: str, value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast(Sequence[Any], value)
        globs: list[str] = []
        for idx, item in enumerate(sequence):
            if not isinstance(item, str):
                raise ConfigError(f"{field_name}[{idx}] must be a string")
            globs.append(item)
        return globs
    raise ConfigError(f"{field_name} must be a string or list of strings")


def _coerce_bool(field_name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOL_TOKENS:
            return True
        if normalized in _FALSE_BOOL_TOKENS:
            return False
    raise ConfigError(
        f"{field_name} must be a boolean or one of: "
        f"{', '.join(sorted(_TRUE_BOOL_TOKENS | _FALSE_BOOL_TOKENS))}"
    )


def _coerce_int(field_name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ConfigError(f"{field_name} must be an integer")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ConfigError(f"{field_name} must be an integer") from exc
    raise ConfigError(f"{field_name} must be an integer")


def _coerce_float(field_name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ConfigError(f"{field_name} must be a number") from exc
    raise ConfigError(f"{field_name} must be a number")


def _coerce_str(field_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string")
    return value


def _coerce_choice(field_name: str, value: Any, choices: frozenset[str]) -> str:
    choice = _coerce_str(field_name, value)
    if choice not in choices:
        raise ConfigError(_invalid_choice_message(field_name, choice, choices))
    return choice


def _invalid_choice_message(field_name: str, value: str, choices: frozenset[str]) -> str:
    return f"{field_name} must be one of: {', '.join(sorted(choices))} (got {value!r})"


def _validate_unit_interval(field_name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{field_name} must be between 0 and 1")
