from __future__ import annotations

import pytest
import torch

from clonehunter.embedding.codebert_embedder import resolve_device


def test_explicit_cpu_passthrough() -> None:
    assert resolve_device("cpu") == "cpu"


def test_explicit_mps_passthrough() -> None:
    assert resolve_device("mps") == "mps"


def test_explicit_cuda_passthrough() -> None:
    assert resolve_device("cuda") == "cuda"


def test_auto_picks_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_auto_picks_cuda_when_no_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
