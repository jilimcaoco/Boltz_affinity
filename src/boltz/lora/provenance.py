"""Code provenance recorded on every training run.

``boltz_version`` alone cannot identify the code that trained an artifact when
the package is an editable install of a working tree, so each run also records
the git commit of that tree and whether tracked files had local changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional


def _git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def repo_state() -> dict[str, Any]:
    """Git commit / dirty flag of the installed ``boltz`` source, plus torch.

    Every field is ``None`` when it cannot be determined (not a git checkout,
    git missing, ...); this never raises.
    """
    import torch

    import boltz

    src = Path(boltz.__file__).resolve().parent
    sha = _git(["rev-parse", "HEAD"], src)
    status = _git(["status", "--porcelain", "--untracked-files=no"], src) if sha else None
    return {
        "boltz_git_sha": sha or None,
        "boltz_git_dirty": None if status is None else bool(status),
        "torch_version": str(torch.__version__),
    }


__all__ = ["repo_state"]
