from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from clonehunter.core.types import FileRef, Language
from clonehunter.io.fingerprints import hash_text


def _detect_language(path: Path) -> Language:
    if path.suffix == ".py":
        return "python"
    return "text"


def _matches(globs: list[str], rel_path: Path) -> bool:
    rel = rel_path.as_posix()
    if rel.startswith("./"):
        rel = rel[2:]
    rel_posix = PurePosixPath(rel)
    for glob in globs:
        pattern = glob.lstrip("./")
        if rel_posix.match(pattern):
            return True
        if pattern.startswith("**/") and rel_posix.match(pattern[3:]):
            return True
        if pattern.endswith("/**"):
            base = pattern[:-3]
            if rel_posix.match(base):
                return True
            if rel.startswith(f"{base}/"):
                return True
        if "/**" in pattern:
            base = pattern.split("/**")[0].lstrip("./")
            if base.startswith("**/"):
                base = base[3:]
            if rel == base or rel.startswith(f"{base}/") or f"/{base}/" in rel:
                return True
    return False


def _relative_path(path: Path) -> Path:
    if path.is_absolute():
        try:
            return path.relative_to(Path.cwd())
        except ValueError:
            return Path(path.name)
    return path


def _iter_files(
    paths: Iterable[str], include_globs: list[str], exclude_globs: list[str]
) -> list[Path]:
    gathered: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for root, dirnames, files in os.walk(p):
                root_path = Path(root)
                rel_root = root_path.relative_to(p)
                # prune excluded directories early
                dirnames[:] = [d for d in dirnames if not _matches(exclude_globs, rel_root / d)]
                for name in files:
                    rel_file = rel_root / name
                    if not _matches(include_globs, rel_file):
                        continue
                    if _matches(exclude_globs, rel_file):
                        continue
                    gathered.append(root_path / name)
        elif p.is_file():
            rel = _relative_path(p)
            if _matches(include_globs, rel) and not _matches(exclude_globs, rel):
                gathered.append(p)
    return gathered


def collect_files(
    paths: list[str], include_globs: list[str], exclude_globs: list[str]
) -> list[FileRef]:
    results: list[FileRef] = []
    all_files = _iter_files(paths, include_globs, exclude_globs)
    for path in all_files:
        language = _detect_language(path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        results.append(FileRef(path=str(path), content_hash=hash_text(content), language=language))
    return results
