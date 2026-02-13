from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

Language = Literal["python", "text"]
SnippetKind = Literal["FUNC", "WIN", "EXP"]


@dataclass(frozen=True, slots=True)
class FileRef:
    path: str
    content_hash: str
    language: Language


@dataclass(frozen=True, slots=True)
class FunctionRef:
    file: FileRef
    qualified_name: str
    start_line: int
    end_line: int
    code: str
    code_hash: str

    @property
    def identity(self) -> str:
        return f"{self.file.path}:{self.qualified_name}:{self.start_line}:{self.end_line}"


@dataclass(frozen=True, slots=True)
class SnippetRef:
    kind: SnippetKind
    function: FunctionRef
    start_line: int
    end_line: int
    text: str
    snippet_hash: str


@dataclass(frozen=True, slots=True)
class Embedding:
    vector: Sequence[float]
    dim: int


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    snippet_a: SnippetRef
    snippet_b: SnippetRef
    similarity: float
    evidence: str


@dataclass(frozen=True, slots=True)
class Finding:
    function_a: FunctionRef
    function_b: FunctionRef
    score: float
    duplicated_lines: int
    evidence: list[CandidateMatch]
    reasons: list[str]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class ScanStats:
    file_count: int
    function_count: int
    snippet_count: int
    candidate_count: int
    finding_count: int
    cache_hits: int
    cache_misses: int


@dataclass(frozen=True, slots=True)
class ScanResult:
    findings: list[Finding]
    stats: ScanStats
    config_snapshot: dict[str, str]
    timing: dict[str, float]


if TYPE_CHECKING:
    from clonehunter.core.config import CloneHunterConfig


@dataclass(frozen=True, slots=True)
class ScanRequest:
    paths: list[str]
    config: CloneHunterConfig
