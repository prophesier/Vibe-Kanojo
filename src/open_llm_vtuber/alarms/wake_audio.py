"""Wake-alarm audio — the part of an alarm that can actually wake someone.

A Discord message cannot wake a sleeping person. An alarm marked ``wake``
therefore also plays music through this machine's speakers, on repeat, until
something stops it. Crucially that playback does **not** go through the
character's reply: the model decides what an alarm *says*, but whether a sound
happens must not depend on a tool call the model might not make.

Song choice, in order of preference:
  1. a random track from the playlist named in ``wake_playlist`` (conf.yaml),
     downloaded on the spot if it isn't cached yet — so the pool is his own
     playlist and it stays fresh;
  2. any song already in the cache, if the network or the account is
     unavailable — a wake alarm has to ring with no internet;
  3. silence, logged, with the spoken reminder still delivered.

The audio comes from the NetEase music experiment
(``experiments/netease_music``). Its player runs ffplay as a detached child
process, which is what makes this work across process boundaries: the server
starts the music, and the music MCP server (a *different* process) can stop it
when the character calls ``music_stop``. Nothing here needs that MCP server to
be running — the import is direct and in-process.

**Nothing in this module may raise.** It is called from the alarm scheduler and
from the handler for every incoming user message; あさひ is usually away from
the machine, so a crash here is not something he could come and fix. Every
entry point returns a value and logs instead.
"""

from __future__ import annotations

import asyncio
import pathlib
import random
import sys
from typing import Any, Optional, Tuple

from loguru import logger

# Loud enough to wake him, measured on his speakers rather than guessed:
# あさひ 07-27 found 55 already noisy, so this sits just above that.
DEFAULT_WAKE_VOLUME = 60
# Longest a wake alarm may keep ringing unattended. Answering stops it sooner;
# this is only the backstop for "nobody is home". It lives here rather than in
# the server because it also bounds how long a *record* of ringing is believed
# — see ``ringing()``.
MAX_RING_SECONDS = 900
# Cap on the "fetch a fresh song" path so a slow network delays the alarm by
# seconds, not minutes; on timeout we fall back to the cache.
_FETCH_TIMEOUT_S = 25

try:
    # experiments/netease_music, relative to src/open_llm_vtuber/alarms.
    _MUSIC_DIR: Optional[pathlib.Path] = (
        pathlib.Path(__file__).resolve().parents[3] / "experiments" / "netease_music"
    )
except Exception:  # pragma: no cover - import must never break the server
    _MUSIC_DIR = None

_modules: Optional[Tuple[Any, Any]] = None
_load_failed = False


def _load() -> Optional[Tuple[Any, Any]]:
    """Import the music experiment's player + client, once."""
    global _modules, _load_failed
    if _modules is not None or _load_failed:
        return _modules
    try:
        if _MUSIC_DIR is None or not (_MUSIC_DIR / "ncm_player.py").exists():
            logger.info("[wake] music module not installed; wake alarms stay silent.")
            _load_failed = True
            return None
        # The music modules import each other by bare name, so the directory
        # has to be importable — not just the files loadable.
        if str(_MUSIC_DIR) not in sys.path:
            sys.path.insert(0, str(_MUSIC_DIR))
        import ncm_client  # noqa: PLC0415
        import ncm_player  # noqa: PLC0415

        _modules = (ncm_player, ncm_client)
    except Exception as e:
        logger.warning(f"[wake] could not load the music module: {e}")
        _load_failed = True
    return _modules


def available() -> bool:
    """True if a wake alarm could make a sound right now (cache only — the
    playlist path additionally needs the network)."""
    mods = _load()
    if mods is None:
        return False
    try:
        return bool(mods[1].cached_songs())
    except Exception as e:
        logger.warning(f"[wake] could not read the music cache: {e}")
        return False


async def _from_playlist(playlist: str) -> Optional[Tuple[str, str]]:
    """A random track from the named playlist as ``(path, label)``, fetching
    the audio if it isn't cached. None on any failure."""
    mods = _load()
    if mods is None:
        return None
    _, ncm_client = mods
    client = ncm_client.NeteaseClient()
    try:
        playlists = await client.user_playlists()
        # Ownership decides ties: several collected playlists are also called
        # "<someone>喜欢的音乐", and being woken by a stranger's ♥ collection
        # is not what "my playlist" meant.
        match = ncm_client.find_playlist(playlists, playlist)
        if match is None:
            names = "、".join(p.name for p in playlists if p.owned) or "(なし)"
            logger.warning(
                f"[wake] playlist {playlist!r} not found; available: {names}"
            )
            return None
        tracks = await client.playlist_tracks(match.id)
        if not tracks:
            logger.warning(f"[wake] playlist {match.name!r} is empty.")
            return None
        song = random.choice(tracks)
        path = await client.fetch_audio(song)
        return str(path), song.label()
    except Exception as e:
        logger.warning(f"[wake] could not take a song from {playlist!r}: {e}")
        return None
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


def _from_cache() -> Optional[Tuple[str, str]]:
    mods = _load()
    if mods is None:
        return None
    try:
        songs = mods[1].cached_songs()
        if not songs:
            return None
        song = random.choice(songs)
        label = f"{song.get('name', '')} / {song.get('artists', '')}".strip(" /")
        return song["path"], label
    except Exception as e:
        logger.warning(f"[wake] could not pick a cached song: {e}")
        return None


async def pick(playlist: str = "") -> Optional[Tuple[str, str]]:
    """Choose what to ring with, as ``(path, label)``, downloading it if it
    isn't cached yet. None if nothing is available. Never raises.

    This is the slow half — a download can take seconds — which is why it is
    separate from :func:`play`. The caller gets a moment in between to check
    whether the reason for ringing still holds: if あさひ speaks while the
    song is still coming down the wire, he is already awake, and starting the
    music at that point would be noise he never asked for.
    """
    if _load() is None:
        return None
    choice = None
    if playlist:
        try:
            choice = await asyncio.wait_for(
                _from_playlist(playlist), timeout=_FETCH_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[wake] fetching from {playlist!r} took over {_FETCH_TIMEOUT_S}s; "
                "falling back to the cache."
            )
        except Exception as e:
            logger.warning(f"[wake] playlist path failed: {e}")
    if choice is None:
        choice = _from_cache()
    if choice is None:
        logger.warning("[wake] no song available — the wake alarm will be silent.")
    return choice


def play(path: str, label: str, volume: int = DEFAULT_WAKE_VOLUME) -> Optional[str]:
    """Start looping the chosen song. Returns its label, or None if it could
    not be played. Never raises."""
    mods = _load()
    if mods is None:
        return None
    try:
        # wake=True is what lets stop(only_wake=True) tell an alarm ringing
        # apart from a song the character put on during a conversation.
        mods[0].play(path, label=label, loop=True, volume=volume, wake=True)
    except Exception as e:
        logger.warning(f"[wake] playback failed: {e}")
        return None
    logger.info(f"[wake] playing {label!r} on repeat at volume {volume}.")
    return label


async def start(playlist: str = "", volume: int = DEFAULT_WAKE_VOLUME) -> Optional[str]:
    """:func:`pick` then :func:`play`, for callers with nothing to re-check in
    between. Returns the label, or None. Never raises."""
    choice = await pick(playlist)
    if choice is None:
        return None
    return play(choice[0], choice[1], volume)


def stop() -> bool:
    """Stop a ringing wake alarm. True if one was actually stopped.

    Only the alarm: music the character started in conversation keeps playing.
    This runs on the path of every incoming user message, so without that
    distinction あさひ's next sentence would kill any song he had asked for.
    Never raises.

    Blocking — it shells out to the OS. Callers on an event loop should hand
    it to a thread.
    """
    mods = _load()
    if mods is None:
        return False
    player = mods[0]
    try:
        outcome = player.stop(only_wake=True)
    except Exception as e:
        logger.warning(f"[wake] could not stop playback: {e}")
        return False
    if outcome == player.FAILED:
        # Worth shouting about: the alarm is still audible and he is probably
        # standing in front of it wondering why talking didn't help.
        logger.error("[wake] the alarm would not stop — it is still ringing.")
        return False
    return outcome == player.STOPPED


def ringing() -> Optional[str]:
    """The song a wake alarm is ringing with right now, or None. Music the
    character started in conversation does not count. Never raises."""
    mods = _load()
    if mods is None:
        return None
    try:
        state = mods[0].status()
    except Exception:
        return None
    if not state or not state.get("wake"):
        return None
    if state.get("residual"):
        # Audible, but not the alarm: a leftover player that would not die and
        # has since been promoted into the primary slot. Answering a message
        # still reaps it — but counting it as "already ringing" here would stop
        # the real alarm from ever starting, and would have the character say
        # it was playing a song nobody chose.
        logger.info("[wake] a residual player is audible; that is not the alarm.")
        return None
    try:
        playing_for = float(state.get("playing_for") or 0)
    except (TypeError, ValueError):
        playing_for = 0.0
    if playing_for > MAX_RING_SECONDS:
        # Past the backstop, so this isn't something still ringing — it's a
        # record we failed to clear (the player reports "still running" when
        # the OS won't answer, which is the safe guess for stopping and the
        # unsafe one here). Don't let it suppress the next alarm forever.
        logger.warning(f"[wake] ignoring a {playing_for:.0f}s-old ringing record.")
        return None
    return str(state.get("label") or "") or "音楽"


def is_playing() -> bool:
    """True if a wake alarm is ringing right now. Ordinary music doesn't
    count — this answers "is the alarm still going", not "is audio audible"."""
    return ringing() is not None
