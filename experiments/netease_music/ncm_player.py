"""Local audio playback, controllable across processes.

Playback runs as a detached ``ffplay`` child rather than inside the calling
process. That is the whole point: an alarm fired by the server and a "stop the
music" tool call handled by the MCP server live in *different* processes, and
only an out-of-process player can be started by one and stopped by the other.
An in-process mixer (pygame et al.) would bind the audio to whoever started
it. ffplay also plays whatever NetEase hands back — mp3, m4a, flac — without a
codec matrix to maintain, and it is already installed.

State lives in ``playback.json`` next to this file: whoever holds the file can
see what is playing and stop it. A stale entry (process already gone) is
detected on read, so a crash can't leave the state wedged.

This module deliberately has no NetEase dependency — it plays a path. The
alarm path in the main server can import it without pulling in the API client.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

HERE = pathlib.Path(__file__).resolve().parent
STATE_FILE = HERE / "playback.json"

_IS_WINDOWS = sys.platform == "win32"


class PlaybackError(Exception):
    """Playback could not be started. ``str(e)`` is safe to show."""


def ffplay_path() -> str:
    exe = shutil.which("ffplay")
    if not exe:
        raise PlaybackError("ffplay が見つかりません（ffmpeg をインストールして）。")
    return exe


def _pid_alive(pid: int) -> bool:
    """True if the pid is still running. Never signals the process — on
    Windows ``os.kill(pid, 0)`` would *terminate* it rather than probe it."""
    if _IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _kill(pid: int) -> None:
    if _IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _read_state() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(STATE_FILE.read_text("utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def _clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def status() -> Optional[Dict[str, Any]]:
    """What is playing right now, or ``None``. Clears state left behind by a
    player that has already exited."""
    state = _read_state()
    if state is None:
        return None
    pid = int(state.get("pid", 0))
    if not pid or not _pid_alive(pid):
        _clear_state()
        return None
    state["playing_for"] = round(time.time() - state.get("started_at", time.time()))
    return state


def stop() -> bool:
    """Stop playback. Returns True if something was actually playing."""
    state = _read_state()
    _clear_state()
    if not state:
        return False
    pid = int(state.get("pid", 0))
    if pid and _pid_alive(pid):
        _kill(pid)
        return True
    return False


def play(
    path: str | pathlib.Path,
    *,
    label: str = "",
    loop: bool = False,
    volume: int = 70,
    song_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Play a local audio file, replacing whatever was playing.

    ``loop`` repeats until stopped — that is the wake-alarm mode; music
    requested in conversation plays once.
    """
    audio = pathlib.Path(path)
    if not audio.exists():
        raise PlaybackError("音声ファイルが見つかりません。")
    exe = ffplay_path()
    stop()

    args = [
        exe,
        "-nodisp",  # audio only, no video window
        "-autoexit",
        "-hide_banner",
        "-loglevel",
        "error",
        "-volume",
        str(max(0, min(int(volume), 100))),
    ]
    if loop:
        args += ["-loop", "0"]
    args.append(str(audio))

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if _IS_WINDOWS:
        # Detach so playback outlives the caller and never flashes a console.
        kwargs["creationflags"] = (
            subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(args, **kwargs)
    except OSError as e:
        raise PlaybackError(f"再生を開始できません: {e}") from e

    state = {
        "pid": proc.pid,
        "label": label or audio.stem,
        "path": str(audio),
        "song_id": song_id,
        "loop": loop,
        "volume": volume,
        "started_at": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    return state
