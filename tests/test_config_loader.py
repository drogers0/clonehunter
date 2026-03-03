from pathlib import Path

from clonehunter.core.config_loader import load_config


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
