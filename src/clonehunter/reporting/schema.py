from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from clonehunter._compat.toml import loads as toml_loads


def _release_version() -> str:
    try:
        return version("clonehunter")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject.exists():
            data = toml_loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project")
            if isinstance(project, dict):
                project_dict = cast(dict[str, Any], project)
                project_version = project_dict.get("version")
                if isinstance(project_version, str):
                    return project_version
        return "0.0.0"


SCHEMA_VERSION = _release_version()
