from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path


class GitError(RuntimeError):
    """Raised when git changed-file discovery fails."""


def changed_files(base: str = "HEAD", paths: Sequence[str] | None = None) -> list[str]:
    tracked = _git_name_only(_with_paths(["diff", "--name-only", base], paths))
    untracked = _git_name_only(_with_paths(["ls-files", "--others", "--exclude-standard"], paths))
    seen: set[str] = set()
    files: list[str] = []
    for raw_path in tracked + untracked:
        normalized = str(Path(raw_path))
        if normalized in seen:
            continue
        seen.add(normalized)
        files.append(normalized)
    return files


def _with_paths(args: list[str], paths: Sequence[str] | None) -> list[str]:
    if not paths:
        return args
    return [*args, "--", *paths]


def _git_name_only(args: list[str]) -> list[str]:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout).strip()
        if not details:
            details = f"git {' '.join(args)} failed with exit code {exc.returncode}"
        raise GitError(details) from exc
    except OSError as exc:
        raise GitError(str(exc)) from exc
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
