from __future__ import annotations

from clonehunter.core.pipeline import run_pipeline
from clonehunter.core.types import ScanRequest, ScanResult
from clonehunter.model.interfaces import Engine


class SemanticEngine(Engine):
    def scan(self, request: ScanRequest) -> ScanResult:
        return run_pipeline(request.paths, request.config)
