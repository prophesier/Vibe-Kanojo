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

  * an OS-owned lock on ``playback.lock`` serialises every start/state-mutation
    sequence. The OS releases it if its process dies, so a stale file is
    harmless and no age-based lock breaking is needed. Starting new playback
    fails closed if the lock is unavailable; an emergency stop may still try
    the pids already on record, but never rewrites shared state without it.
  * the ``orphans`` list in the state file: a
    player we tried and failed to kill keeps its pid on record, so the next
    stop can try again. Nothing is ever dropped from the file merely because
    something newer took its place.

Two flags on a record answer two different questions, and conflating them
breaks the alarm: ``wake`` means "this is an alarm ringing" (so answering
silences it, and so nothing tries to start a second one on top), while
``residual`` means "nobody chose this, it just would not die" (so answering
reaps it too). A residual is audible but it is not the alarm.

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
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger

HERE = pathlib.Path(__file__).resolve().parent
STATE_FILE = HERE / "playback.json"
LOCK_FILE = HERE / "playback.lock"

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl

# tasklist/taskkill answer in well under a second on a healthy machine; this is
# the bound on a sick one. It also sets the scale of everything below, because
# the lock is held across a few of these at worst.
_CMD_TIMEOUT_S = 5.0
# Waiting this long means the holder is in serious trouble — a healthy critical
# section is milliseconds. Long enough that giving up is genuinely the
# degraded path, not something normal contention can reach.
_LOCK_WAIT_S = 20.0
_LOCK_POLL_S = 0.02

# stop() outcomes. Three states, because "nothing was playing" and "something
# is playing and would not stop" must never be reported as the same thing.
STOPPED = "stopped"
IDLE = "idle"
FAILED = "failed"

# Process-probe outcomes. UNKNOWN deliberately remains distinct from ALIVE:
# an unconfirmed pid must stay on record, but it must never authorize taskkill.
ALIVE = "alive"
DEAD = "dead"
UNKNOWN = "unknown"


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


def _write_state(state: Dict[str, Any]) -> bool:
    """Atomically replace the state file.

    Readers run in another process, so they must see either the old complete
    record or the new complete record — never half of a JSON document. The
    return value matters to ``play``: a player whose pid could not be persisted
    must be terminated through its still-live ``Popen`` handle.
    """
    tmp = STATE_FILE.with_name(
        f".{STATE_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        logger.warning(f"[player] could not write {STATE_FILE.name}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


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


def _players(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every player this state still holds a pid for: the current one, plus any
    earlier one we failed to kill."""
    entries = [state]
    for orphan in state.get("orphans") or []:
        if isinstance(orphan, dict):
            # Anything sitting in ``orphans`` is a residual by construction:
            # nobody asked for it, it just would not die. Marking it here is
            # what keeps that fact attached to the entry if it later gets
            # promoted to primary.
            entries.append(dict(orphan, residual=True))
    unique: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        pid = _state_pid(entry)
        if not pid:
            continue
        key = (pid, entry.get("started_at"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


# --------------------------------------------------------------------- lock


def _try_os_lock(fd: int) -> bool:
    """Try once to acquire the one-byte inter-process lock."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if _IS_WINDOWS:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _release_os_lock(fd: int) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        if _IS_WINDOWS:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def _state_lock() -> Iterator[bool]:
    """Serialise the start/stop sequence across processes.

    Without it, two processes can each stop "nothing", each spawn a player,
    and the second one's state file wins — leaving the first ffplay running
    with its pid on no record at all.

    Yields whether the lock was acquired. The lock is owned by the OS rather
    than by the presence/age of the file: it is released automatically on
    process death, while a harmless ``playback.lock`` file may remain forever.
    Callers may attempt an emergency kill from a snapshot when acquisition
    times out, but they must not spawn or mutate shared state without it.
    """
    fd: Optional[int] = None
    acquired = False
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_RDWR)
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        deadline = time.monotonic() + _LOCK_WAIT_S
        while not acquired:
            acquired = _try_os_lock(fd)
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(_LOCK_POLL_S)
    except OSError as e:
        logger.warning(f"[player] could not open the lock: {e}")
    if not acquired:
        logger.warning("[player] lock unavailable.")
    try:
        yield acquired
    finally:
        if fd is not None:
            if acquired:
                _release_os_lock(fd)
            try:
                os.close(fd)
            except OSError:
                pass


# ------------------------------------------------------------ the OS itself


def _probe_process(entry: Dict[str, Any]) -> str:
    """Return ``ALIVE``, ``DEAD`` or ``UNKNOWN`` for a recorded player.

    Matches the image name as well as the pid. Pids get recycled, and this
    module kills whole process *trees* (``taskkill /T``) — acting on a
    recycled pid would take down an unrelated program.

    An inconclusive probe is neither alive nor dead. It keeps the pid on record
    and makes stop return FAILED, but — crucially — it does not authorize a
    signal. Otherwise a tasklist outage plus pid reuse could make the safety
    fallback kill an unrelated process.

    Never signals the process: on Windows ``os.kill(pid, 0)`` *terminates* it
    rather than probing it.
    """
    pid = _state_pid(entry)
    if not pid:
        return DEAD
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
            return UNKNOWN
        if done.returncode != 0:
            logger.warning(f"[player] tasklist failed ({done.returncode}) for {pid}.")
            return UNKNOWN
        return ALIVE if f'"{pid}"' in done.stdout else DEAD
    try:
        comm = pathlib.Path(f"/proc/{pid}/comm")
        if comm.exists():
            if exe and comm.read_text().strip() != pathlib.Path(exe).name:
                return DEAD
            return ALIVE
    except OSError:
        return UNKNOWN
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return DEAD
    except OSError:
        return UNKNOWN
    # A live pid is insufficient when its recorded executable could not be
    # verified (for example on a non-/proc Unix). Preserve, but do not signal.
    return UNKNOWN if exe else ALIVE


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
    if _probe_process(state) == DEAD:
        # The current player may have finished while an older unkillable one
        # remains in ``orphans``. Re-read under the lock, probe every recorded
        # pid and promote a survivor instead of deleting the whole ledger.
        with _state_lock() as acquired:
            current = _read_state() if acquired else state
            if current is None:
                return None
            survivors = [
                entry for entry in _players(current) if _probe_process(entry) != DEAD
            ]
            if not survivors:
                if acquired:
                    _clear_state()
                return None
            state = _state_from_entries(current, survivors)
            if acquired:
                _write_state(state)
    try:
        started = float(state.get("started_at") or time.time())
    except (TypeError, ValueError):
        started = time.time()
    state["playing_for"] = round(time.time() - started)
    return state


def _reap_entries(entries: List[Dict[str, Any]]) -> tuple:
    """Try to kill confirmed players in ``entries``.

    Returns ``(killed_anything, survivors)``. Survivors are the players that
    would not die *or whose identity/liveness could not be confirmed*. Unknown
    entries stay on record but are never signalled.
    """
    killed = False
    survivors: List[Dict[str, Any]] = []
    for entry in entries:
        before = _probe_process(entry)
        if before == DEAD:
            continue  # exited on its own
        if before == UNKNOWN:
            logger.warning(
                f"[player] pid {_state_pid(entry)} could not be verified; not killing."
            )
            survivors.append(entry)
            continue
        _kill(_state_pid(entry))
        after = _probe_process(entry)
        if after != DEAD:
            logger.warning(f"[player] pid {_state_pid(entry)} would not stop.")
            survivors.append(entry)
        else:
            killed = True
    return killed, survivors


_ENTRY_KEYS = (
    "exe",
    "started_at",
    "label",
    "path",
    "song_id",
    "loop",
    "wake",
    "volume",
    "residual",
)


def _orphan_record(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The fields needed to stop and accurately promote this player later."""
    record = {"pid": _state_pid(entry)}
    record.update({key: entry.get(key) for key in _ENTRY_KEYS if key in entry})
    return record


def _state_from_entries(
    base: Dict[str, Any], entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build one state record without dropping any still-addressable pid."""
    if not entries:
        return {}
    primary = entries[0]
    keep = dict(base)
    keep.pop("orphans", None)
    # Do not let metadata from a dead former primary describe a promoted
    # orphan. New-format orphan records carry these fields themselves.
    for key in ("pid", *_ENTRY_KEYS):
        keep.pop(key, None)
    keep.update(_orphan_record(primary))
    if len(entries) > 1:
        keep["orphans"] = [_orphan_record(entry) for entry in entries[1:]]
    return keep


def _stop_targets(
    state: Dict[str, Any], *, only_wake: bool
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(preserved, targets)`` for this stop request."""
    entries = _players(state)
    if only_wake and not state.get("wake") and not state.get("residual"):
        # The current song was explicitly requested and must survive the
        # automatic "user replied" stop. Residual players were not — including
        # one that has since been promoted into the primary slot.
        return entries[:1], entries[1:]
    return [], entries


def _stop_locked(state: Dict[str, Any], *, only_wake: bool = False) -> str:
    """Stop everything ``state`` knows about. Caller holds the lock."""
    preserved, targets = _stop_targets(state, only_wake=only_wake)
    killed, survivors = _reap_entries(targets)
    remaining = preserved + survivors
    if remaining:
        if not _write_state(_state_from_entries(state, remaining)):
            return FAILED
    else:
        _clear_state()
    if survivors:
        return FAILED
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
        return s is None or (
            only_wake
            and not s.get("wake")
            and not s.get("residual")
            and not s.get("orphans")
        )

    state = _read_state()
    if only_wake and _skip(state):
        return IDLE  # nothing to do, and nothing worth taking the lock for
    with _state_lock() as acquired:
        if not acquired:
            if state is None:
                return IDLE
            # Emergency degraded path: try only confirmed pids from the
            # snapshot, but never rewrite shared state without serialization.
            preserved, targets = _stop_targets(state, only_wake=only_wake)
            del preserved
            killed, survivors = _reap_entries(targets)
            if survivors:
                return FAILED
            return STOPPED if killed else IDLE
        state = _read_state()  # another process may have moved on meanwhile
        if _skip(state):
            return IDLE
        return _stop_locked(state, only_wake=only_wake)


def _terminate_unrecorded(proc: subprocess.Popen) -> None:
    """Best effort for a child whose pid could not be persisted.

    The Popen handle identifies the exact process we just created, so unlike a
    state-file pid it is safe to terminate without a separate identity probe.
    """
    try:
        proc.kill()
        proc.wait(timeout=_CMD_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        logger.critical(
            f"[player] could not terminate unrecorded pid {proc.pid}; "
            "manual intervention may be required."
        )


def play(
    path: str | pathlib.Path,
    *,
    label: str = "",
    loop: bool = False,
    volume: int = 40,  # both callers pass their own; this is just a sane floor
    song_id: Optional[int] = None,
    wake: bool = False,
    keep_proc: bool = False,
    only_if_idle: bool = False,
) -> Dict[str, Any]:
    """Play a local audio file, replacing whatever was playing.

    ``loop`` repeats until stopped — that is the wake-alarm mode; music
    requested in conversation plays once. ``wake`` marks the playback as an
    alarm ringing, which is what lets ``stop(only_wake=True)`` tell the two
    apart.

    ``keep_proc`` returns the child's ``Popen`` handle under the (in-memory
    only, never persisted) ``"proc"`` key, so the continuous-playlist loop can
    wait on the exact process it started and read its exit code — the state
    file cannot answer "did it end naturally or was it killed", the handle can.

    ``only_if_idle`` makes this a polite start: if any recorded player is
    still alive — or cannot be confirmed dead — raise instead of reaping it.
    The playlist loop uses it so that the next chained track can never
    replace audio someone else started (e.g. a wake alarm that fired while
    the next song was downloading). An UNKNOWN probe refuses too: failing to
    start is recoverable, killing a ringing alarm is not.
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
    with _state_lock() as acquired:
        if not acquired:
            raise PlaybackError(
                "プレイヤーが別の処理中です。安全に再生を開始できませんでした。"
            )
        survivors: List[Dict[str, Any]] = []
        previous = _read_state()
        if previous is not None:
            if only_if_idle:
                blockers = [
                    entry
                    for entry in _players(previous)
                    if _probe_process(entry) != DEAD
                ]
                if blockers:
                    raise PlaybackError(
                        "別の再生が進行中のため、次の曲は開始しませんでした。"
                    )
            _, survivors = _reap_entries(_players(previous))
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
                _write_state(_state_from_entries(previous or {}, survivors))
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
            state["orphans"] = [_orphan_record(entry) for entry in survivors]
        if not _write_state(state):
            _terminate_unrecorded(proc)
            raise PlaybackError(
                "再生状態を安全に保存できなかったため、開始を取り消しました。"
            )
    if keep_proc:
        # Added AFTER the state was serialised: the handle lives only in the
        # returned dict, never in playback.json.
        state["proc"] = proc
    return state
