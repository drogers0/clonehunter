import builtins
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from clonehunter.core.config import CloneHunterConfig
from clonehunter.engines.sonarqube_engine import ScanRequest, SonarQubeEngine


def test_sonarqube_engine_requires_env():
    engine = SonarQubeEngine()
    with pytest.raises(RuntimeError):
        engine.scan(ScanRequest(paths=["."], config=CloneHunterConfig()))


def test_faiss_index_installed():
    from clonehunter.index.faiss_index import FaissIndex

    FaissIndex()


def test_faiss_index_missing_dependency(monkeypatch: pytest.MonkeyPatch):
    from clonehunter.index.faiss_index import FaissIndex

    real_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "faiss":
            raise ModuleNotFoundError("faiss")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(RuntimeError):
        FaissIndex()
