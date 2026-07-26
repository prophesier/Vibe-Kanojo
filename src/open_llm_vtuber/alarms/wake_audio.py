"""Wake-alarm audio — the part of an alarm that can actually wake someone.

A Discord message cannot wake a sleeping person. An alarm marked ``wake``
therefore also plays music through this machine's speakers, on repeat, until
something stops it. Crucially that playback does **not** go through the
character's reply: the model decides what an alarm *says*, but whether a sound
happens must not depend on a tool call the model might not make.

The audio comes from the NetEase music experiment
(``experiments/netease_music``). Its player runs ffplay as a detached child
process, which is what makes this work across process boundaries — the server
starts the music, and the music MCP server (a different process) can stop it
when the character calls ``music_stop``. Songs are taken from that module's
on-disk cache, so a wake alarm still rings with no network.

Everything here degrades to a logged no-op: if the music module is missing, or
nothing is cached yet, the alarm still delivers its spoken reminder — it just
does it quietly.
"""

from __future__ import annotations

import pathlib
import random
import sys
from typing import Any, Optional, Tuple

from loguru import logger

# experiments/netease_music, relative to this file (src/open_llm_vtuber/alarms).
_MUSIC_DIR = (
    pathlib.Path(__file__).resolve().parents[3] / "experiments" / "netease_music"
)

DEFAULT_WAKE_VOLUME = 85

_modules: Optional[Tuple[Any, Any]] = None
_load_failed = False


def _load() -> Optional[Tuple[Any, Any]]:
    """Import the music experiment's player + cache reader, once."""
    global _modules, _load_failed
    if _modules is not None or _load_failed:
        return _modules
    if not (_MUSIC_DIR / "ncm_player.py").exists():
        logger.info("[wake] music module not installed; wake alarms stay silent.")
        _load_failed = True
        return None
    try:
        # The music modules import each other by bare name, so the directory
        # has to be importable — not just the files loadable.
        sys.path.insert(0, str(_MUSIC_DIR))
        import ncm_client  # noqa: PLC0415
        import ncm_player  # noqa: PLC0415

        _modules = (ncm_player, ncm_client)
    except Exception as e:
        logger.warning(f"[wake] could not load the music module: {e}")
        _load_failed = True
    return _modules


def available() -> bool:
    """True if a wake alarm could make a sound right now."""
    mods = _load()
    if mods is None:
        return False
    _, ncm_client = mods
    return bool(ncm_client.cached_songs())


def start(volume: int = DEFAULT_WAKE_VOLUME) -> Optional[str]:
    """Start looping a random cached song. Returns its label, or None if
    nothing could be played."""
    mods = _load()
    if mods is None:
        return None
    ncm_player, ncm_client = mods
    songs = ncm_client.cached_songs()
    if not songs:
        logger.warning("[wake] no cached songs — wake alarm will be silent.")
        return None
    song = random.choice(songs)
    label = f"{song.get('name', '')} / {song.get('artists', '')}".strip(" /")
    try:
        ncm_player.play(
            song["path"],
            label=label,
            loop=True,
            volume=volume,
            song_id=song.get("id"),
        )
    except Exception as e:
        logger.warning(f"[wake] playback failed: {e}")
        return None
    logger.info(f"[wake] playing {label!r} on repeat at volume {volume}.")
    return label


def stop() -> bool:
    """Stop wake playback. True if something was actually playing."""
    mods = _load()
    if mods is None:
        return False
    ncm_player, _ = mods
    try:
        return bool(ncm_player.stop())
    except Exception as e:
        logger.warning(f"[wake] could not stop playback: {e}")
        return False


def is_playing() -> bool:
    mods = _load()
    if mods is None:
        return False
    ncm_player, _ = mods
    try:
        return ncm_player.status() is not None
    except Exception:
        return False
