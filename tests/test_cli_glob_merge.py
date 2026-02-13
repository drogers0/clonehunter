from clonehunter.cli.commands.scan import effective_repotypes, merge_globs, resolve_repotype_globs
from clonehunter.cli.main import build_parser


def test_merge_globs_additive_when_no_conflict() -> None:
    include, exclude = merge_globs(
        base_include=["**/*.py"],
        base_exclude=["**/.venv/**"],
        cli_include=["**/*.tsx"],
        cli_exclude=["**/node_modules/**"],
    )
    assert include == ["**/*.py", "**/*.tsx"]
    assert exclude == ["**/.venv/**", "**/node_modules/**"]


def test_merge_globs_cli_wins_on_conflict() -> None:
    include, exclude = merge_globs(
        base_include=["**/*.py", "**/dist/**"],
        base_exclude=["**/dist/**", "**/*.tsx"],
        cli_include=["**/*.tsx"],
        cli_exclude=["**/dist/**"],
    )
    assert "**/*.tsx" in include
    assert "**/*.tsx" not in exclude
    assert "**/dist/**" in exclude
    assert "**/dist/**" not in include


def test_scan_parser_accepts_repeated_glob_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "scan",
            ".",
            "--include-globs",
            "**/*.js",
            "--include-globs",
            "**/*.tsx",
            "--exclude-globs",
            "**/node_modules/**",
            "--exclude-globs",
            "**/dist/**",
        ]
    )
    assert args.include_globs == ["**/*.js", "**/*.tsx"]
    assert args.exclude_globs == ["**/node_modules/**", "**/dist/**"]


def test_scan_parser_accepts_mixed_repotype_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["scan", ".", "--repotype", "python", "--repotype", "react", "--repotype", "rust"]
    )
    assert args.repotype == ["python", "react", "rust"]


def test_resolve_repotype_globs_combines_and_dedupes() -> None:
    include, exclude = resolve_repotype_globs(["react", "cpp", "react"])
    assert "**/*.tsx" in include
    assert "**/*.cpp" in include
    assert "**/node_modules/**" in exclude
    assert "**/build/**" in exclude


def test_resolve_monorepo_has_broad_language_support() -> None:
    include, exclude = resolve_repotype_globs(["monorepo"])
    assert "**/*.py" in include
    assert "**/*.tsx" in include
    assert "**/*.cpp" in include
    assert "**/*.java" in include
    assert "**/*.swift" in include
    assert "**/node_modules/**" in exclude
    assert "**/target/**" in exclude


def test_effective_repotypes_defaults_to_monorepo() -> None:
    assert effective_repotypes(None) == ["monorepo"]
    assert effective_repotypes([]) == ["monorepo"]
    assert effective_repotypes(["python"]) == ["python"]
    assert effective_repotypes(["none"]) == []
    assert effective_repotypes(["none", "react"]) == ["react"]


def test_explicit_cli_globs_override_repotype_conflicts() -> None:
    repotype_include, repotype_exclude = resolve_repotype_globs(["react"])
    include, exclude = merge_globs(
        ["**/*.py"],
        ["**/*.tsx"],
        repotype_include,
        repotype_exclude,
    )
    include, exclude = merge_globs(include, exclude, ["**/node_modules/**"], [])
    assert "**/node_modules/**" in include
    assert "**/node_modules/**" not in exclude


def test_scan_parser_accepts_repotype_none() -> None:
    parser = build_parser()
    args = parser.parse_args(["scan", ".", "--repotype", "none"])
    assert args.repotype == ["none"]
