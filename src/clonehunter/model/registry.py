from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from clonehunter.model.interfaces import Engine


@dataclass(frozen=True, slots=True)
class EngineRegistration:
    name: str
    factory: Callable[[], Engine]


_REGISTRY: dict[str, EngineRegistration] = {}


def register_engine(name: str, factory: Callable[[], Engine]) -> None:
    _REGISTRY[name] = EngineRegistration(name=name, factory=factory)


def get_engine(name: str) -> Engine:
    if name not in _REGISTRY:
        raise KeyError(f"Engine not registered: {name}")
    return _REGISTRY[name].factory()
