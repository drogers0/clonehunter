from __future__ import annotations

import subprocess
from pathlib import Path


def changed_files(base: str = "HEAD") -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [str(Path(f)) for f in files]
