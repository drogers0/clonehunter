from __future__ import annotations

from pathlib import Path

from clonehunter.core.types import FileRef, FunctionRef
from clonehunter.io.fingerprints import hash_text


def extract_file_unit(file: FileRef) -> list[FunctionRef]:
    try:
        with open(file.path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError:
        return []
    if not source.strip():
        return []
    end_line = max(1, len(source.splitlines()))
    return [
        FunctionRef(
            file=file,
            qualified_name=Path(file.path).name,
            start_line=1,
            end_line=end_line,
            code=source,
            code_hash=hash_text(source),
        )
    ]
