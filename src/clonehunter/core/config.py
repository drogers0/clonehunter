from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class WindowConfig:
    window_lines: int = 40
    stride_lines: int = 6
    min_nonempty: int = 4


@dataclass(frozen=True, slots=True)
class ExpansionConfig:
    enabled: bool = False
    depth: int = 1
    max_chars: int = 4000


@dataclass(frozen=True, slots=True)
class Thresholds:
    func: float = 0.92
    win: float = 0.90
    exp: float = 0.90
    min_window_hits: int = 2
    lexical_min_ratio: float = 0.5
    lexical_weight: float = 0.3


@dataclass(frozen=True, slots=True)
class IndexConfig:
    name: Literal["brute", "faiss"] = "brute"
    top_k: int = 25
    faiss_nlist: int = 128
    faiss_nprobe: int = 8


@dataclass(frozen=True, slots=True)
class CacheConfig:
    path: str = "~/.cache/clonehunter"


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    name: Literal["codebert", "stub"] = "codebert"
    model_name: str = "microsoft/codebert-base"
    revision: str = "main"
    max_length: int = 256
    batch_size: int = 16
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class CloneHunterConfig:
    engine: Literal["semantic", "sonarqube"] = "semantic"
    include_globs: list[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_globs: list[str] = field(
        default_factory=lambda: [
            "**/.venv/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/site-packages/**",
        ]
    )
    windows: WindowConfig = WindowConfig()
    expansion: ExpansionConfig = ExpansionConfig()
    thresholds: Thresholds = Thresholds()
    index: IndexConfig = IndexConfig()
    cache: CacheConfig = CacheConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    cluster_findings: bool = False
    cluster_min_size: int = 2
