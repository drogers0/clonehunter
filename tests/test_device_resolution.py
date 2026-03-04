from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from clonehunter.embedding.codebert_embedder import resolve_device


def _mps_backend() -> Any:
    # Torch stub coverage differs between environments; cast for test monkeypatching.
    return cast(Any, torch).backends.mps


def _cuda_module() -> Any:
    return cast(Any, torch).cuda


def test_explicit_cpu_passthrough() -> None:
    assert resolve_device("cpu", torch) == "cpu"


def test_explicit_mps_passthrough() -> None:
    assert resolve_device("mps", torch) == "mps"


def test_explicit_cuda_passthrough() -> None:
    assert resolve_device("cuda", torch) == "cuda"


def test_auto_picks_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mps_backend(), "is_available", lambda: True)
    assert resolve_device("auto", torch) == "mps"


def test_auto_picks_cuda_when_no_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mps_backend(), "is_available", lambda: False)
    monkeypatch.setattr(_cuda_module(), "is_available", lambda: True)
    assert resolve_device("auto", torch) == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mps_backend(), "is_available", lambda: False)
    monkeypatch.setattr(_cuda_module(), "is_available", lambda: False)
    assert resolve_device("auto", torch) == "cpu"
