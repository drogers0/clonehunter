from __future__ import annotations

from pathlib import Path

import pytest

import clonehunter.cli.commands.scan as scan_cmd
from clonehunter.cli.commands.scan import ScanOptions, run_scan
from clonehunter.core.config import CloneHunterConfig
from clonehunter.core.types import ScanRequest, ScanResult, ScanStats


class _FakeEngine:
    def scan(self, request: ScanRequest) -> ScanResult:
        return ScanResult(
            findings=[],
            stats=ScanStats(
                file_count=0,
                function_count=0,
                snippet_count=0,
                candidate_count=0,
                finding_count=0,
                cache_hits=0,
                cache_misses=0,
            ),
            config_snapshot={},
            timing={},
        )


class _NullJsonReporter:
    def write(self, scan_result: ScanResult, out_path: str) -> None:
        return None


def test_run_scan_loads_config_from_target_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = tmp_path / "caller"
    caller.mkdir()
    target_repo = tmp_path / "target_repo"
    target_repo.mkdir()

    captured: dict[str, Path] = {}

    def _load_config(root: Path, overrides: dict[str, object]) -> CloneHunterConfig:
        captured["root"] = root
        return CloneHunterConfig()

    def _get_engine(_name: str) -> _FakeEngine:
        return _FakeEngine()

    monkeypatch.chdir(caller)
    monkeypatch.setattr(scan_cmd, "load_config", _load_config)
    monkeypatch.setattr(scan_cmd, "get_engine", _get_engine)
    monkeypatch.setattr(scan_cmd, "JsonReporter", _NullJsonReporter)

    run_scan(
        ScanOptions(
            paths=[str(target_repo)],
            fmt="json",
            out_path=str(tmp_path / "out.json"),
        )
    )

    assert captured["root"] == target_repo.resolve()


def test_run_scan_loads_config_from_repo_root_when_path_is_nested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = tmp_path / "caller"
    caller.mkdir()
    target_repo = tmp_path / "target_repo"
    nested = target_repo / "src" / "pkg"
    nested.mkdir(parents=True)
    (target_repo / "pyproject.toml").write_text(
        """
[tool.clonehunter]
engine = "semantic"
""",
        encoding="utf-8",
    )

    captured: dict[str, Path] = {}

    def _load_config(root: Path, overrides: dict[str, object]) -> CloneHunterConfig:
        captured["root"] = root
        return CloneHunterConfig()

    def _get_engine(_name: str) -> _FakeEngine:
        return _FakeEngine()

    monkeypatch.chdir(caller)
    monkeypatch.setattr(scan_cmd, "load_config", _load_config)
    monkeypatch.setattr(scan_cmd, "get_engine", _get_engine)
    monkeypatch.setattr(scan_cmd, "JsonReporter", _NullJsonReporter)

    run_scan(
        ScanOptions(
            paths=[str(nested)],
            fmt="json",
            out_path=str(tmp_path / "out.json"),
        )
    )

    assert captured["root"] == target_repo.resolve()
