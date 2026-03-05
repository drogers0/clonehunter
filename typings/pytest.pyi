from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

class Config: ...

class MonkeyPatch:
    def setenv(self, name: str, value: str) -> None: ...
    def setitem(self, mapping: Any, name: str, value: Any) -> None: ...
    def setattr(self, target: Any, name: str, value: Any) -> None: ...
    def chdir(self, path: str | Path) -> None: ...

class RaisesContext(AbstractContextManager[Any]):
    def __enter__(self) -> Any: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...

_F = TypeVar("_F", bound=Callable[..., Any])

class MarkDecorator:
    def __call__(self, func: _F) -> _F: ...

class MarkNamespace:
    def parametrize(self, argnames: str | tuple[str, ...], argvalues: Any) -> MarkDecorator: ...

mark: MarkNamespace

def raises(
    expected_exception: type[BaseException],
    *,
    match: str | None = None,
) -> RaisesContext: ...
def skip(reason: str) -> None: ...
def importorskip(
    modname: str,
    minversion: str | None = None,
    reason: str | None = None,
) -> Any: ...
