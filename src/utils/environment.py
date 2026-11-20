"""What produced a result: code version and library versions, stored in every run summary.

Config + seed say *what* was run; this says *with which code*. A summary from a dirty
working tree (uncommitted changes) is flagged, because its commit hash alone can't
rebuild it. Every field degrades to None instead of failing a training run.
"""

import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def environment(device: str = "cpu") -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    env = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),   # tracked files modified?
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cupy": None,
        "gpu": None,
    }
    if device == "gpu":
        try:
            import cupy as cp
            env["cupy"] = cp.__version__
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = props["name"]
            env["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:                                         # no CuPy / no driver: keep None
            pass
    if sys.version_info < (3, 11):
        env["warning"] = "tested on Python 3.11"
    return env
