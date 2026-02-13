from __future__ import annotations

from abc import ABC, abstractmethod

from clonehunter.core.types import (
    Embedding,
    FileRef,
    FunctionRef,
    ScanRequest,
    ScanResult,
    SnippetRef,
)


class Engine(ABC):
    @abstractmethod
    def scan(self, request: ScanRequest) -> ScanResult: ...


class Extractor(ABC):
    @abstractmethod
    def supports(self, file: FileRef) -> bool: ...

    @abstractmethod
    def extract_functions(self, file: FileRef) -> list[FunctionRef]: ...


class SnippetGenerator(ABC):
    @abstractmethod
    def generate(self, functions: list[FunctionRef]) -> list[SnippetRef]: ...


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, snippets: list[SnippetRef]) -> list[Embedding]: ...


class VectorIndex(ABC):
    @abstractmethod
    def build(self, vectors: list[Embedding], ids: list[str]) -> None: ...

    @abstractmethod
    def query(self, vector: Embedding, k: int) -> list[tuple[str, float]]: ...


class Reporter(ABC):
    @abstractmethod
    def write(self, result: ScanResult, out_path: str) -> None: ...
