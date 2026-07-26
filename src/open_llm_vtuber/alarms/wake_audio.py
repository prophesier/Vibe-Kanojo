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

DEFAULT_WAKE_VOLUME = 85
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
        wanted = playlist.strip().lower()
        match = next((p for p in playlists if wanted in p.name.lower()), None)
        if match is None:
            names = "、".join(p.name for p in playlists[:8]) or "(なし)"
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


async def start(playlist: str = "", volume: int = DEFAULT_WAKE_VOLUME) -> Optional[str]:
    """Start looping a wake song. Returns its label, or None if nothing could
    be played. Never raises."""
    mods = _load()
    if mods is None:
        return None
    ncm_player, _ = mods

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
        return None

    path, label = choice
    try:
        ncm_player.play(path, label=label, loop=True, volume=volume)
    except Exception as e:
        logger.warning(f"[wake] playback failed: {e}")
        return None
    logger.info(f"[wake] playing {label!r} on repeat at volume {volume}.")
    return label


def stop() -> bool:
    """Stop wake playback. True if something was actually playing. Never
    raises — this runs on the path of every incoming user message."""
    mods = _load()
    if mods is None:
        return False
    try:
        return bool(mods[0].stop())
    except Exception as e:
        logger.warning(f"[wake] could not stop playback: {e}")
        return False


def is_playing() -> bool:
    mods = _load()
    if mods is None:
        return False
    try:
        return mods[0].status() is not None
    except Exception:
        return False
