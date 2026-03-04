from pathlib import Path

import pytest

from clonehunter.core.config_loader import load_config
from clonehunter.core.errors import ConfigError


def _assert_config_error_contains(
    tmp_path: Path,
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_config(tmp_path, overrides)
    assert expected_message in str(exc_info.value)


def test_load_config_from_pyproject(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.clonehunter]
include_globs = ["src/**/*.py"]

[tool.clonehunter.thresholds]
func = 0.8
min_window_hits = 3

[tool.clonehunter.index]
name = "brute"
""",
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.include_globs == ["src/**/*.py"]
    assert config.thresholds.func == 0.8
    assert config.thresholds.min_window_hits == 3
    assert config.index.name == "brute"


def test_default_device_is_auto(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path)
    assert config.embedder.device == "auto"


def test_embedder_preset_faster(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path, {"embedder": {"name": "faster"}})
    assert config.embedder.name == "faster"
    assert config.embedder.model_name == "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
    assert config.embedder.max_length == 512
    assert config.embedder.batch_size == 32


def test_embedder_preset_with_explicit_override(tmp_path: Path) -> None:
    """Explicit field overrides should win over preset defaults."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path, {"embedder": {"name": "faster", "batch_size": 64}})
    assert config.embedder.name == "faster"
    assert config.embedder.model_name == "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
    assert config.embedder.batch_size == 64


def test_embedder_preset_codebert_unchanged(tmp_path: Path) -> None:
    """Selecting codebert should keep the original defaults."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path, {"embedder": {"name": "codebert"}})
    assert config.embedder.model_name == "microsoft/codebert-base"
    assert config.embedder.max_length == 256
    assert config.embedder.batch_size == 16


def test_scalar_glob_values_are_coerced_to_singleton_lists(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path, {"include_globs": "**/*.py", "exclude_globs": "**/dist/**"})
    assert config.include_globs == ["**/*.py"]
    assert config.exclude_globs == ["**/dist/**"]


def test_glob_lists_reject_non_string_values(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    for field_name in ["include_globs", "exclude_globs"]:
        _assert_config_error_contains(
            tmp_path=tmp_path,
            overrides={field_name: ["**/*.py", 7]},
            expected_message=field_name,
        )


def test_cluster_findings_parses_false_string_strictly(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    config = load_config(tmp_path, {"cluster_findings": "false"})
    assert config.cluster_findings is False


def test_cluster_findings_rejects_invalid_bool_tokens(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    _assert_config_error_contains(
        tmp_path=tmp_path,
        overrides={"cluster_findings": "maybe"},
        expected_message="cluster_findings",
    )


def test_invalid_enum_like_names_fail_at_config_load(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    cases: list[tuple[dict[str, object], str]] = [
        ({"engine": "unknown"}, "engine"),
        ({"embedder": {"name": "unknown"}}, "embedder.name"),
        ({"index": {"name": "unknown"}}, "index.name"),
    ]
    for overrides, field_name in cases:
        _assert_config_error_contains(
            tmp_path=tmp_path,
            overrides=overrides,
            expected_message=field_name,
        )


def test_numeric_validation_runs_during_config_loading(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    cases: list[tuple[dict[str, object], str]] = [
        ({"embedder": {"batch_size": 0}}, "embedder.batch_size"),
        ({"index": {"top_k": 0}}, "index.top_k"),
        ({"thresholds": {"func": 1.1}}, "thresholds.func"),
        ({"thresholds": {"win": -0.1}}, "thresholds.win"),
        ({"thresholds": {"exp": 1.1}}, "thresholds.exp"),
        ({"thresholds": {"min_window_hits": 0}}, "thresholds.min_window_hits"),
        ({"thresholds": {"lexical_min_ratio": 1.1}}, "thresholds.lexical_min_ratio"),
        ({"thresholds": {"lexical_weight": -0.1}}, "thresholds.lexical_weight"),
        ({"cluster_min_size": 0}, "cluster_min_size"),
        ({"expansion": {"depth": -1}}, "expansion.depth"),
        ({"expansion": {"max_chars": 0}}, "expansion.max_chars"),
    ]
    for overrides, message in cases:
        _assert_config_error_contains(
            tmp_path=tmp_path,
            overrides=overrides,
            expected_message=message,
        )
