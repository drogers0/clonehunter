from pathlib import Path

from clonehunter.core.config import CloneHunterConfig, EmbedderConfig, Thresholds, WindowConfig
from clonehunter.core.pipeline import run_pipeline


def test_pipeline_smoke():
    result = run_pipeline(
        ["fixtures/tiny_repo"], CloneHunterConfig(embedder=EmbedderConfig(name="stub"))
    )
    assert result.stats.file_count >= 1
    assert result.stats.function_count >= 1


def test_pipeline_non_python_implicit_windows_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code = "function add(a, b) {\n  return a + b;\n}\n"
    (repo / "a.js").write_text(code, encoding="utf-8")
    (repo / "b.js").write_text(code, encoding="utf-8")

    result = run_pipeline(
        [str(repo)],
        CloneHunterConfig(
            include_globs=["**/*.js"],
            exclude_globs=[],
            embedder=EmbedderConfig(name="stub"),
            windows=WindowConfig(window_lines=3, stride_lines=1, min_nonempty=1),
            thresholds=Thresholds(
                func=0.99,
                win=0.80,
                exp=0.99,
                min_window_hits=1,
                lexical_min_ratio=0.0,
                lexical_weight=0.3,
            ),
        ),
    )
    assert result.stats.file_count == 2
    assert result.stats.function_count == 0
    assert result.stats.finding_count >= 1
    finding = result.findings[0]
    assert finding.function_a.qualified_name != "<file>"
    assert finding.function_b.qualified_name != "<file>"


def test_pipeline_non_python_allows_cross_file_types(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code = "function add(a, b) {\n  return a + b;\n}\n"
    (repo / "a.js").write_text(code, encoding="utf-8")
    (repo / "b.ts").write_text(code, encoding="utf-8")

    result = run_pipeline(
        [str(repo)],
        CloneHunterConfig(
            include_globs=["**/*.js", "**/*.ts"],
            exclude_globs=[],
            embedder=EmbedderConfig(name="stub"),
            windows=WindowConfig(window_lines=3, stride_lines=1, min_nonempty=1),
            thresholds=Thresholds(
                func=0.99,
                win=0.80,
                exp=0.99,
                min_window_hits=1,
                lexical_min_ratio=0.0,
                lexical_weight=0.3,
            ),
        ),
    )
    assert result.stats.file_count == 2
    assert result.stats.finding_count >= 1
