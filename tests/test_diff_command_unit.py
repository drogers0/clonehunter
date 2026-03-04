from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

import clonehunter.cli.main as cli_main
from clonehunter.cli.commands import diff as diff_cmd
from clonehunter.cli.commands.overrides import build_base_overrides
from clonehunter.cli.commands.scan import ScanOptions
from clonehunter.cli.main import main
from clonehunter.core.config import CloneHunterConfig
from clonehunter.core.types import FileRef, Finding, FunctionRef, ScanResult, ScanStats
from clonehunter.io.git import GitError


def _finding(path_a: str, path_b: str) -> Finding:
    file_a = FileRef(path=path_a, content_hash="a", language="python")
    file_b = FileRef(path=path_b, content_hash="b", language="python")
    func_a = FunctionRef(
        file=file_a,
        qualified_name="a",
        start_line=1,
        end_line=2,
        code="pass",
        code_hash="ha",
    )
    func_b = FunctionRef(
        file=file_b,
        qualified_name="b",
        start_line=1,
        end_line=2,
        code="pass",
        code_hash="hb",
    )
    return Finding(
        function_a=func_a,
        function_b=func_b,
        score=1.0,
        duplicated_lines=2,
        evidence=[],
        reasons=[],
        metadata={},
    )


def test_run_diff_exits_on_git_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(base: str, paths: list[str]) -> list[str]:
        raise GitError("git unavailable")

    monkeypatch.setattr(diff_cmd, "changed_files", _boom)

    with pytest.raises(SystemExit) as exc_info:
        diff_cmd.run_diff("HEAD", "json", "out.json", ["."], None, None, None, None)
    assert "Failed to determine changed files: git unavailable" in str(exc_info.value)


def test_run_diff_no_files_delegates_to_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, ScanOptions] = {}

    def _changed_files(base: str, paths: list[str]) -> list[str]:
        return []

    def _run_scan(options: ScanOptions) -> None:
        captured["options"] = options

    monkeypatch.setattr(diff_cmd, "changed_files", _changed_files)
    monkeypatch.setattr(diff_cmd, "run_scan", _run_scan)

    diff_cmd.run_diff(
        "HEAD",
        "json",
        "out.json",
        ["src"],
        "sonarqube",
        "stub",
        "brute",
        "cpu",
    )

    options = captured["options"]
    assert options.paths == []
    assert options.engine_name == "sonarqube"
    assert options.embedder == "stub"
    assert options.index == "brute"
    assert options.device == "cpu"


def test_run_diff_filters_findings_to_changed_files(monkeypatch: pytest.MonkeyPatch) -> None:
    changed_rel = "src/a.py"
    changed_abs = str(Path.cwd() / changed_rel)
    result = ScanResult(
        findings=[_finding(changed_abs, "src/unchanged.py"), _finding("src/b.py", "src/c.py")],
        stats=ScanStats(
            file_count=3,
            function_count=3,
            snippet_count=3,
            candidate_count=2,
            finding_count=2,
            cache_hits=0,
            cache_misses=0,
        ),
        config_snapshot={},
        timing={},
    )

    class _FakeEngine:
        def scan(self, request: object) -> ScanResult:
            return result

    class _CaptureJsonReporter:
        written: ClassVar[list[tuple[ScanResult, str]]] = []

        def write(self, scan_result: ScanResult, out_path: str) -> None:
            self.written.append((scan_result, out_path))

    def _changed_files(base: str, paths: list[str]) -> list[str]:
        return [changed_rel]

    def _load_config(root: Path, overrides: dict[str, object]) -> CloneHunterConfig:
        return CloneHunterConfig()

    def _get_engine(engine_name: str) -> _FakeEngine:
        return _FakeEngine()

    monkeypatch.setattr(diff_cmd, "changed_files", _changed_files)
    monkeypatch.setattr(diff_cmd, "load_config", _load_config)
    monkeypatch.setattr(diff_cmd, "get_engine", _get_engine)
    monkeypatch.setattr(diff_cmd, "JsonReporter", _CaptureJsonReporter)

    diff_cmd.run_diff("HEAD", "json", "out.json", ["."], None, None, None, None)

    assert len(_CaptureJsonReporter.written) == 1
    filtered_result, out_path = _CaptureJsonReporter.written[0]
    assert out_path == "out.json"
    assert filtered_result.stats.finding_count == 1
    assert filtered_result.findings[0].function_a.file.path == changed_abs


def test_main_plumbs_diff_engine_and_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    def _run_diff(
        base: str,
        fmt: str,
        out_path: str,
        paths: list[str],
        engine_name: str | None,
        embedder: str | None,
        index: str | None,
        device: str | None,
    ) -> None:
        captured.extend([base, fmt, out_path, paths, engine_name, embedder, index, device])

    monkeypatch.setattr(cli_main, "run_diff", _run_diff)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clonehunter",
            "diff",
            "src",
            "--base",
            "HEAD~1",
            "--format",
            "json",
            "--out",
            "out.json",
            "--engine",
            "sonarqube",
            "--embedder",
            "stub",
            "--index",
            "brute",
            "--device",
            "cpu",
        ],
    )

    main()

    assert captured == [
        "HEAD~1",
        "json",
        "out.json",
        ["src"],
        "sonarqube",
        "stub",
        "brute",
        "cpu",
    ]


def test_build_base_overrides_uses_env_stub_when_not_explicit() -> None:
    assert build_base_overrides(env_embedder="stub") == {"embedder": {"name": "stub"}}


def test_build_base_overrides_keeps_explicit_embedder_with_device() -> None:
    overrides = build_base_overrides(embedder="faster", device="cpu", env_embedder="stub")
    assert overrides == {"embedder": {"name": "faster", "device": "cpu"}}
