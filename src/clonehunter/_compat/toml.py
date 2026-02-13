from __future__ import annotations

from typing import Any, Protocol, cast


class _TomlLoader(Protocol):
    def loads(self, s: str, /, *, parse_float: Any = ...) -> dict[str, Any]: ...


def loads(text: str) -> dict[str, Any]:
    try:
        import tomllib

        loader = cast(_TomlLoader, tomllib)
        return loader.loads(text)
    except ModuleNotFoundError:  # pragma: no cover - only used on Python < 3.11
        import tomli

        loader = cast(_TomlLoader, tomli)
        return loader.loads(text)
