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
detected on read, so a crash can't leave the state wedged. ``playback.lock``
next to it serialises the moments that mutate that state, because those are
the moments where two processes could otherwise strand a player nobody can
reach any more.

This module deliberately has no NetEase dependency — it plays a path. The
alarm path in the main server can import it without pulling in the API client.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, Optional

from loguru import logger

HERE = pathlib.Path(__file__).resolve().parent
STATE_FILE = HERE / "playback.json"
LOCK_FILE = HERE / "playback.lock"

_IS_WINDOWS = sys.platform == "win32"

# Lock tuning. Contention means two processes reached for the player in the
# same millisecond, so the wait is deliberately short — and a caller that
# cannot get the lock proceeds anyway (see _state_lock).
_LOCK_WAIT_S = 2.0
_LOCK_POLL_S = 0.02
_LOCK_STALE_S = 30.0


class PlaybackError(Exception):
    """Playback could not be started. ``str(e)`` is safe to show."""


def ffplay_path() -> str:
    exe = shutil.which("ffplay")
    if not exe:
        raise PlaybackError("ffplay が見つかりません（ffmpeg をインストールして）。")
    return exe


def _lock_is_stale() -> bool:
    """A lock file older than ``_LOCK_STALE_S`` was left behind by a process
    that died holding it; the operations it guards take milliseconds."""
    try:
        return time.time() - LOCK_FILE.stat().st_mtime > _LOCK_STALE_S
    except OSError:
        return False


def _unlink_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


@contextlib.contextmanager
def _state_lock() -> Iterator[None]:
    """Serialise the start/stop sequence across processes.

    Without it, two processes can each stop "nothing", each spawn a player,
    and the second one's state file wins — leaving the first ffplay running at
    wake volume, on repeat, with its pid on no record at all. That is the one
    failure mode here that nobody can recover from without Task Manager.

    Never raises and never blocks for long: if the lock cannot be taken it
    proceeds without it, because a wedged lock must not be able to prevent an
    alarm from being silenced. Only the holder releases it — a caller that
    gave up must not delete a lock somebody else is still using.
    """
    fd: Optional[int] = None
    # One deadline over every path through the loop, so no failure mode here —
    # including an unlink that keeps failing on a stale lock — can spin.
    deadline = time.monotonic() + _LOCK_WAIT_S
    while fd is None and time.monotonic() < deadline:
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_is_stale():
                _unlink_lock()  # its holder died; take it over
            else:
                time.sleep(_LOCK_POLL_S)
        except OSError as e:
            logger.warning(f"[player] could not take the lock: {e}")
            break
    if fd is None:
        logger.warning("[player] lock unavailable; proceeding without it.")
    try:
        yield
    finally:
        if fd is not None:
            # Close before unlinking: Windows refuses to delete a file that
            # still has an open handle on it.
            try:
                os.close(fd)
            except OSError:
                pass
            _unlink_lock()


def _process_alive(state: Dict[str, Any], *, unknown: bool) -> bool:
    """True if the player recorded in ``state`` is still running.

    Matches the image name as well as the pid. Pids get recycled, and this
    module kills whole process *trees* (``taskkill /T``) — acting on a
    recycled pid would take down an unrelated program.

    ``unknown`` is the answer to give when the OS won't say: True before a
    kill, so ``stop()`` still tries; False after one, so a kill that worked is
    not reported as a failure.

    Never signals the process: on Windows ``os.kill(pid, 0)`` *terminates* it
    rather than probing it.
    """
    pid = _state_pid(state)
    if not pid:
        return False
    exe = str(state.get("exe") or "")
    if _IS_WINDOWS:
        args = ["tasklist", "/FI", f"PID eq {pid}"]
        if exe:
            args += ["/FI", f"IMAGENAME eq {exe}"]
        # CSV quotes every field, so the pid can be matched as a whole column
        # instead of as a substring that a memory figure could also contain.
        args += ["/FO", "CSV", "/NH"]
        try:
            out = subprocess.run(
                args,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return unknown
        return f'"{pid}"' in out
    try:
        comm = pathlib.Path(f"/proc/{pid}/comm")
        if comm.exists():
            return not exe or comm.read_text().strip() == pathlib.Path(exe).name
    except OSError:
        return unknown
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return unknown
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
    if not _process_alive(state, unknown=True):
        _clear_state()
        return None
    try:
        started = float(state.get("started_at") or time.time())
    except (TypeError, ValueError):
        started = time.time()
    state["playing_for"] = round(time.time() - started)
    return state


def _stop_locked(state: Dict[str, Any]) -> bool:
    """Kill the recorded player and clear the state. Caller holds the lock.
    Returns True only if a running player was actually killed."""
    pid = _state_pid(state)
    if not pid or not _process_alive(state, unknown=True):
        _clear_state()  # it exited on its own
        return False
    _kill(pid)
    if _process_alive(state, unknown=False):
        # Keep the state: claiming a stop we did not achieve would leave music
        # playing with nothing on record able to address it.
        logger.warning(f"[player] pid {pid} survived the kill; state kept to retry.")
        return False
    _clear_state()
    return True


def stop(*, only_wake: bool = False) -> bool:
    """Stop playback. Returns True only if a player was actually killed.

    ``only_wake`` restricts this to a ringing wake alarm and leaves music the
    character started in conversation alone. The server stops on that setting
    for every incoming message: answering has to silence the alarm, but must
    not kill a song あさひ asked for two lines earlier.

    Never raises.
    """
    state = _read_state()
    if state is None or (only_wake and not state.get("wake")):
        return False  # nothing to do, and nothing worth taking the lock for
    with _state_lock():
        state = _read_state()  # another process may have moved on meanwhile
        if state is None or (only_wake and not state.get("wake")):
            return False
        return _stop_locked(state)


def play(
    path: str | pathlib.Path,
    *,
    label: str = "",
    loop: bool = False,
    volume: int = 70,
    song_id: Optional[int] = None,
    wake: bool = False,
) -> Dict[str, Any]:
    """Play a local audio file, replacing whatever was playing.

    ``loop`` repeats until stopped — that is the wake-alarm mode; music
    requested in conversation plays once. ``wake`` marks the playback as an
    alarm ringing, which is what lets ``stop(only_wake=True)`` tell the two
    apart.
    """
    audio = pathlib.Path(path)
    if not audio.exists():
        raise PlaybackError("音声ファイルが見つかりません。")
    exe = ffplay_path()

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

    # Stopping the old player, spawning the new one and recording its pid have
    # to be one indivisible step — see _state_lock.
    with _state_lock():
        previous = _read_state()
        if previous is not None and not _stop_locked(previous):
            # Couldn't kill it. Two songs at once is bad; refusing to ring an
            # alarm is worse, so start anyway and leave a trail.
            if _process_alive(previous, unknown=False):
                logger.warning(
                    f"[player] pid {_state_pid(previous)} is still playing and "
                    "will be replaced in the state file."
                )
        try:
            proc = subprocess.Popen(args, **kwargs)
        except OSError as e:
            raise PlaybackError(f"再生を開始できません: {e}") from e

        state = {
            "pid": proc.pid,
            "exe": pathlib.Path(exe).name,  # guards against pid reuse
            "label": label or audio.stem,
            "path": str(audio),
            "song_id": song_id,
            "loop": loop,
            "wake": bool(wake),
            "volume": volume,
            "started_at": time.time(),
        }
        try:
            STATE_FILE.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), "utf-8"
            )
        except OSError as e:
            # The sound is already playing; losing the state file only costs us
            # the ability to stop or report it. Don't undo a ringing alarm.
            logger.warning(f"[player] could not write {STATE_FILE.name}: {e}")
    return state
