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

The one unrecoverable failure here is a player whose pid nobody recorded:
music at wake volume, on repeat, with nothing left able to address it. Two
mechanisms guard against it, and they are worth keeping apart:

  * ``playback.lock`` serialises the start/stop sequence, which is what stops
    two processes from each spawning a player and one record overwriting the
    other. It is best-effort — a caller that cannot take it proceeds anyway,
    because a wedged lock must never be able to keep an alarm ringing.
  * the ``orphans`` list in the state file, which is *not* best-effort: a
    player we tried and failed to kill keeps its pid on record, so the next
    stop can try again. Nothing is ever dropped from the file merely because
    something newer took its place.

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
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

HERE = pathlib.Path(__file__).resolve().parent
STATE_FILE = HERE / "playback.json"
LOCK_FILE = HERE / "playback.lock"

_IS_WINDOWS = sys.platform == "win32"

# tasklist/taskkill answer in well under a second on a healthy machine; this is
# the bound on a sick one. It also sets the scale of everything below, because
# the lock is held across a few of these at worst.
_CMD_TIMEOUT_S = 5.0
# Waiting this long means the holder is in serious trouble — a healthy critical
# section is milliseconds. Long enough that giving up is genuinely the
# degraded path, not something normal contention can reach.
_LOCK_WAIT_S = 20.0
_LOCK_POLL_S = 0.02
# Only ever break a lock whose holder is certainly gone. Far beyond any hold a
# living process could manage, so this can't delete a lock still in use.
_LOCK_STALE_S = 300.0
# Players we failed to kill and still owe a retry. A cap purely so a
# pathological machine can't grow the file without bound.
_MAX_ORPHANS = 4

# stop() outcomes. Three states, because "nothing was playing" and "something
# is playing and would not stop" must never be reported as the same thing.
STOPPED = "stopped"
IDLE = "idle"
FAILED = "failed"


class PlaybackError(Exception):
    """Playback could not be started. ``str(e)`` is safe to show."""


def ffplay_path() -> str:
    exe = shutil.which("ffplay")
    if not exe:
        raise PlaybackError("ffplay が見つかりません（ffmpeg をインストールして）。")
    return exe


# --------------------------------------------------------------- state file


def _read_state() -> Optional[Dict[str, Any]]:
    # Broad by design: this sits on the path of every user message (the server
    # stops a ringing wake alarm there), so a locked or half-written state file
    # must degrade to "nothing playing", never raise.
    try:
        state = json.loads(STATE_FILE.read_text("utf-8"))
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _write_state(state: Dict[str, Any]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
    except OSError as e:
        # The sound is already playing; losing the state file only costs us the
        # ability to stop or report it. Don't undo a ringing alarm over this.
        logger.warning(f"[player] could not write {STATE_FILE.name}: {e}")


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


def _same_entry(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Whether two state records describe the same player. pid alone is not
    enough — pids get recycled — so the start time comes along."""
    return _state_pid(a) == _state_pid(b) and a.get("started_at") == b.get("started_at")


def _players(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every player this state still holds a pid for: the current one, plus any
    earlier one we failed to kill."""
    entries = [state]
    for orphan in state.get("orphans") or []:
        if isinstance(orphan, dict):
            entries.append(orphan)
    return [e for e in entries if _state_pid(e)]


# --------------------------------------------------------------------- lock


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
    and the second one's state file wins — leaving the first ffplay running
    with its pid on no record at all.

    Never raises and always terminates: if the lock cannot be taken it
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


# ------------------------------------------------------------ the OS itself


def _process_alive(entry: Dict[str, Any]) -> bool:
    """True if the player recorded in ``entry`` is still running.

    Matches the image name as well as the pid. Pids get recycled, and this
    module kills whole process *trees* (``taskkill /T``) — acting on a
    recycled pid would take down an unrelated program.

    **When the OS won't say, the answer is "still running."** A failing
    ``tasklist`` prints nothing and returns non-zero, which by output alone is
    indistinguishable from "no such process"; reading that as "already dead"
    is exactly how a stop that never happened gets reported as done, and the
    music becomes something nobody can reach. Guessing the other way only
    costs us a stale record, which the next successful probe clears.

    Never signals the process: on Windows ``os.kill(pid, 0)`` *terminates* it
    rather than probing it.
    """
    pid = _state_pid(entry)
    if not pid:
        return False
    exe = str(entry.get("exe") or "")
    if _IS_WINDOWS:
        args = ["tasklist", "/FI", f"PID eq {pid}"]
        if exe:
            args += ["/FI", f"IMAGENAME eq {exe}"]
        # CSV quotes every field, so the pid can be matched as a whole column
        # instead of as a substring that a memory figure could also contain.
        args += ["/FO", "CSV", "/NH"]
        try:
            done = subprocess.run(
                args,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=_CMD_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"[player] could not run tasklist for {pid}: {e}")
            return True
        if done.returncode != 0:
            logger.warning(f"[player] tasklist failed ({done.returncode}) for {pid}.")
            return True
        return f'"{pid}"' in done.stdout
    try:
        comm = pathlib.Path(f"/proc/{pid}/comm")
        if comm.exists():
            return not exe or comm.read_text().strip() == pathlib.Path(exe).name
    except OSError:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
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
                timeout=_CMD_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


# ------------------------------------------------------------------ public


def status() -> Optional[Dict[str, Any]]:
    """What is playing right now, or ``None``. Clears state left behind by a
    player that has already exited. Never raises."""
    state = _read_state()
    if state is None:
        return None
    if not _process_alive(state):
        # Probing took a moment, and another process may have started
        # something in it. Only drop the record if it still describes the
        # player we just found dead — dropping a newer one would orphan music
        # that is genuinely playing.
        with _state_lock():
            current = _read_state()
            if current is not None and _same_entry(current, state):
                _clear_state()
        return None
    try:
        started = float(state.get("started_at") or time.time())
    except (TypeError, ValueError):
        started = time.time()
    state["playing_for"] = round(time.time() - started)
    return state


def _reap(state: Dict[str, Any]) -> tuple:
    """Kill every player in ``state``. Caller holds the lock.

    Returns ``(killed_anything, survivors)``. Survivors are the players that
    would not die; they stay on record so a later stop can try again, and so a
    new player can never quietly take their place in the file.
    """
    killed = False
    survivors: List[Dict[str, Any]] = []
    for entry in _players(state):
        if not _process_alive(entry):
            continue  # exited on its own
        _kill(_state_pid(entry))
        if _process_alive(entry):
            # "unknown" counts as alive here: a stop we cannot confirm must
            # not be reported as done, or the music becomes unreachable.
            logger.warning(f"[player] pid {_state_pid(entry)} would not stop.")
            survivors.append(entry)
        else:
            killed = True
    return killed, survivors


def _orphan_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The minimum needed to find and kill this player later."""
    return {
        "pid": _state_pid(entry),
        "exe": entry.get("exe", ""),
        "started_at": entry.get("started_at"),
        "label": entry.get("label", ""),
    }


def _survivor_state(
    base: Dict[str, Any], survivors: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """A state record that keeps players we could not kill on the books. The
    flags come from ``base``, so a wake alarm that refused to die still reads
    as a wake alarm and the next stop will go after it again."""
    keep = dict(base)
    keep.update(
        {
            "pid": _state_pid(survivors[0]),
            "exe": survivors[0].get("exe", ""),
            "started_at": survivors[0].get("started_at"),
            "label": survivors[0].get("label", ""),
            "orphans": [_orphan_record(e) for e in survivors[1 : _MAX_ORPHANS + 1]],
        }
    )
    return keep


def _stop_locked(state: Dict[str, Any]) -> str:
    """Stop everything ``state`` knows about. Caller holds the lock."""
    killed, survivors = _reap(state)
    if survivors:
        _write_state(_survivor_state(state, survivors))
        return FAILED
    _clear_state()
    return STOPPED if killed else IDLE


def stop(*, only_wake: bool = False) -> str:
    """Stop playback. Returns ``STOPPED``, ``IDLE`` or ``FAILED``.

    ``only_wake`` restricts this to a ringing wake alarm and leaves music the
    character started in conversation alone. The server stops on that setting
    for every incoming message: answering has to silence the alarm, but must
    not kill a song あさひ asked for two lines earlier. Orphans are reaped
    either way — nobody chose to keep those playing.

    Never raises.
    """

    def _skip(s: Optional[Dict[str, Any]]) -> bool:
        return s is None or (only_wake and not s.get("wake") and not s.get("orphans"))

    state = _read_state()
    if _skip(state):
        return IDLE  # nothing to do, and nothing worth taking the lock for
    with _state_lock():
        state = _read_state()  # another process may have moved on meanwhile
        if _skip(state):
            return IDLE
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
        survivors: List[Dict[str, Any]] = []
        previous = _read_state()
        if previous is not None:
            _, survivors = _reap(previous)
            if survivors:
                # Two songs at once is bad; refusing to ring an alarm is worse.
                # Start anyway — but carry the pids we could not kill into the
                # new record, because overwriting them is precisely how music
                # becomes unstoppable.
                logger.warning(
                    f"[player] {len(survivors)} player(s) would not stop; "
                    "carrying their pids forward."
                )
        try:
            proc = subprocess.Popen(args, **kwargs)
        except OSError as e:
            if survivors:
                # No new player to record, but the old ones are still audible
                # and still ours to stop.
                _write_state(_survivor_state(previous or {}, survivors))
            raise PlaybackError(f"再生を開始できません: {e}") from e

        state: Dict[str, Any] = {
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
        if survivors:
            state["orphans"] = [_orphan_record(e) for e in survivors[:_MAX_ORPHANS]]
        _write_state(state)
    return state
