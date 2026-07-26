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

from loguru import logger

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
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            # Can't tell — assume alive so stop() still tries to kill it.
            return True
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return True
    return True


def _kill(pid: int) -> None:
    if _IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _read_state() -> Optional[Dict[str, Any]]:
    # Broad by design: this sits on the path of every user message (the server
    # stops a ringing wake alarm there), so a locked or half-written state file
    # must degrade to "nothing playing", never raise.
    try:
        state = json.loads(STATE_FILE.read_text("utf-8"))
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _clear_state() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _state_pid(state: Dict[str, Any]) -> int:
    try:
        return int(state.get("pid") or 0)
    except (TypeError, ValueError):
        return 0


def status() -> Optional[Dict[str, Any]]:
    """What is playing right now, or ``None``. Clears state left behind by a
    player that has already exited. Never raises."""
    state = _read_state()
    if state is None:
        return None
    pid = _state_pid(state)
    if not pid or not _pid_alive(pid):
        _clear_state()
        return None
    try:
        started = float(state.get("started_at") or time.time())
    except (TypeError, ValueError):
        started = time.time()
    state["playing_for"] = round(time.time() - started)
    return state


def stop() -> bool:
    """Stop playback. Returns True if something was actually playing.
    Never raises."""
    state = _read_state()
    _clear_state()
    if not state:
        return False
    pid = _state_pid(state)
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
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    except OSError as e:
        # The sound is already playing; losing the state file only costs us the
        # ability to stop or report it. Don't undo a ringing alarm over this.
        logger.warning(f"[player] could not write {STATE_FILE.name}: {e}")
    return state
