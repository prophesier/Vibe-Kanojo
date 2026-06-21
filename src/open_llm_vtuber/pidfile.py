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
import json
import os
import time
from pathlib import Path
from typing import Optional


def _pid_dir(root: Optional[Path] = None) -> Path:
    base = (root or Path.cwd()) / "pids"
    base.mkdir(exist_ok=True)
    return base


_BACKFILL_SETTLED = "backfill_settled.json"


def mark_backfill_settled(root: Optional[Path] = None, conf_uid: str = "") -> None:
    """Record that startup memory backfill has settled — i.e. the system prompt
    is now stable and won't be rewritten by fact extraction/pruning anymore.

    The Discord bot waits for this before announcing restart-complete, so the
    user isn't invited to talk while backfill is still changing facts (which
    would shift the system prompt mid-session and break the OpenAI prompt
    cache). Always overwrites, so the freshest settle wins; readers compare
    ``settled_at`` against their own reference time to ignore stale markers
    left by a previous run.
    """
    path = _pid_dir(root) / _BACKFILL_SETTLED
    try:
        path.write_text(
            json.dumps({"settled_at": time.time(), "conf_uid": conf_uid})
        )
    except Exception:
        pass


def read_backfill_settled_at(root: Optional[Path] = None) -> Optional[float]:
    """Return the epoch time startup backfill last settled, or None if unknown."""
    path = _pid_dir(root) / _BACKFILL_SETTLED
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        val = data.get("settled_at")
        return float(val) if val is not None else None
    except (ValueError, OSError, TypeError):
        return None


def write_pid(name: str, root: Optional[Path] = None) -> None:
    """Write the current process PID to pids/<name>.pid and clean it up on exit.

    The atexit hook ensures graceful shutdowns leave no stale file behind.
    Hard kills (taskkill /F) bypass atexit, so restart.bat treats a present
    PID file as best-effort and deletes it after the kill attempt regardless.
    """
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
