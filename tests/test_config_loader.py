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
