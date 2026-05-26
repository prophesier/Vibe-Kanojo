"""PID file helpers used by the OLV server and Discord bridge.

Each long-running service writes its PID to ``pids/<name>.pid`` at the
project root on startup and removes the file on clean exit. ``restart.bat``
reads these files to know which processes to kill before pulling new code
and re-launching everything.

The ``pids/`` directory is gitignored — these files are transient runtime
state, not configuration.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Optional


def _pid_dir(root: Optional[Path] = None) -> Path:
    base = (root or Path.cwd()) / "pids"
    base.mkdir(exist_ok=True)
    return base


def write_pid(name: str, root: Optional[Path] = None) -> None:
    """Write the current process PID to pids/<name>.pid and clean it up on exit."""
    path = _pid_dir(root) / f"{name}.pid"
    path.write_text(str(os.getpid()))
    atexit.register(_cleanup, path)


def read_pid(name: str, root: Optional[Path] = None) -> Optional[int]:
    """Return the PID stored in pids/<name>.pid, or None if missing/unreadable."""
    path = _pid_dir(root) / f"{name}.pid"
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
