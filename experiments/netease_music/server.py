"""NetEase Cloud Music MCP server — search and play music on this machine.

Tools (playback only — nothing here can post, follow, rate, or otherwise
touch the account; the session cookie is read-only as far as these tools are
concerned):
  - music_search(keyword)          -> matching songs
  - music_play(keyword)            -> play the best match through the speakers
  - music_playlists()              -> the user's own playlists
  - music_play_playlist(name)      -> play a random song from one of them
  - music_now_playing()            -> what is playing, if anything
  - music_stop()                   -> stop playback

Song and artist names are metadata and are reported freely — that is what a
music player does. Lyrics are a different thing entirely, and no tool here
fetches or returns them.

Robustness contract (same as the Uber server):
  - every tool ALWAYS returns a short string, never raises;
  - a hard per-tool timeout guarantees a tool call can never hang the chat
    turn — the character can always keep talking;
  - failures are logged to ncm_mcp.log so problems stay visible.

Run (registered in mcp_servers.json):  python experiments/netease_music/server.py
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import pathlib
import random
import sys
import time

# Make sibling modules importable regardless of the launch cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

import ncm_player  # noqa: E402
from ncm_client import (  # noqa: E402
    NeteaseClient,
    NeteaseUnavailable,
    Song,
)

HERE = pathlib.Path(__file__).resolve().parent


def _setup_logging() -> None:
    """Only when run as the server: importing this module (tests do) must not
    reconfigure someone else's logging or drop a log file in the repo."""
    logger.remove()  # don't write to stdout — stdout is the MCP stdio channel!
    logger.add(sys.stderr, level="INFO")
    logger.add(str(HERE / "ncm_mcp.log"), rotation="2 MB", retention=3, level="INFO")


# One budget for the WHOLE call, not per step. The MCP client gives a tool
# call 30 seconds total (mcpp/mcp_client.py), so a search and a download that
# each got their own 25s could let the client give up while this server was
# still working — and the song would then start playing well after the
# character had already said something else about it.
_TOOL_TIMEOUT = 25  # seconds — under the MCP client's 30s read timeout
# Reserved at the end of that budget for starting the player itself. Starting
# it runs in a thread, and a thread already inside a blocking OS call cannot
# be cancelled — so if the outer timeout landed there, the tool would report
# a timeout and the music would start anyway, a few seconds later. Giving the
# step its own slice keeps the outer bound from ever landing mid-play.
_PLAYER_BUDGET_S = 8.0
# Leave enough event-loop time for the inner player timeout to be caught and
# translated before the outer whole-tool timeout fires.
_PLAYER_RETURN_MARGIN_S = 1.0
_DEFAULT_VOLUME = 70

# When the current tool call must be finished, so the steps inside it can
# divide up what's left rather than each assuming the whole budget.
_deadline: contextvars.ContextVar[float] = contextvars.ContextVar("ncm_deadline")


def _time_left() -> float:
    """Raw seconds left in the current whole-tool budget."""
    try:
        return _deadline.get() - time.monotonic()
    except LookupError:  # called outside a tool (tests, direct use)
        return float(_TOOL_TIMEOUT)


def _remaining(reserve: float = 0.0) -> float:
    """Usable seconds left after reserving a later step."""
    return max(0.0, _time_left() - reserve)


_RESULT_NOTE = (
    "\n\n（メモ：この結果はあなたの参照用で会話履歴には残らない。"
    "曲名やアーティスト名など伝えたいことは、ユーザーへの返信の中で"
    "必ず自分の言葉で伝えること。この内容は次のターンには消えている。）"
)

mcp = FastMCP("netease-music")
_client = NeteaseClient()


def _tool(fn):
    """Give a tool its one timeout and its one catch-all, around the whole body.

    Applied under ``@mcp.tool()``, so the body below can be written as plain
    linear code that raises — every failure still comes back to the character
    as a short sentence it can say out loud, and no tool call can outlive the
    client's patience.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs) -> str:
        _deadline.set(time.monotonic() + _TOOL_TIMEOUT)
        try:
            return await asyncio.wait_for(fn(*args, **kwargs), timeout=_TOOL_TIMEOUT)
        except NeteaseUnavailable as e:
            logger.warning(f"{fn.__name__}: unavailable: {e}")
            return str(e)
        except asyncio.TimeoutError:
            logger.warning(f"{fn.__name__}: timed out after {_TOOL_TIMEOUT}s")
            return "音楽サービスの応答が遅すぎます（タイムアウト）。"
        except Exception:
            logger.exception(f"{fn.__name__}: unexpected error")
            return "音楽の処理中に問題が発生しました。"

    return wrapper


def _format(songs: list[Song]) -> str:
    """One line per song. The id is what disambiguates covers, remixes and
    re-recordings that share a title — pass it to music_play(song_id=...)."""
    return "\n".join(
        f"{i}. {s.label()}"
        + (f"（{s.album}）" if s.album and s.album != s.name else "")
        + f" [{s.duration_ms // 60000}:{s.duration_ms // 1000 % 60:02d}]"
        + f" id={s.id}"
        for i, s in enumerate(songs, 1)
    )


async def _player(fn, *args, **kwargs):
    """Run a blocking player call off the event loop, bounded.

    Raises ``asyncio.TimeoutError`` if it doesn't come back in time. A thread
    already inside a blocking OS call cannot be cancelled, so the call may
    still succeed afterwards — every caller therefore has to say "couldn't
    confirm" rather than pick an outcome to report.
    """
    return await asyncio.wait_for(
        asyncio.to_thread(fn, *args, **kwargs), timeout=_PLAYER_BUDGET_S
    )


async def _play_song(song: Song, volume: int) -> str:
    # Download under whatever is left of the budget, keeping the player's slice
    # in hand — the download can be cancelled cleanly, the start cannot.
    reserve = _PLAYER_BUDGET_S + _PLAYER_RETURN_MARGIN_S
    fetch_budget = _remaining(reserve=reserve)
    if fetch_budget <= 0:
        return (
            f"「{song.label()}」は検索に時間がかかり、"
            "安全に再生を開始できる時間が残っていなかったため開始しませんでした。"
        )
    try:
        path = await asyncio.wait_for(_client.fetch_audio(song), timeout=fetch_budget)
    except asyncio.TimeoutError:
        return (
            f"「{song.label()}」の取得が時間内に終わらなかったため、"
            "再生は開始しませんでした。"
        )
    # Earlier search/playlist calls share the outer budget and may have used
    # more time than expected. Never dispatch an uncancellable thread unless
    # its entire uncertainty window plus response margin still fits.
    if _time_left() < reserve:
        return (
            f"「{song.label()}」は取得できましたが、安全に再生開始を確認できる"
            "時間が残っていなかったため開始しませんでした。"
        )
    try:
        await _player(
            ncm_player.play,
            path,
            label=song.label(),
            loop=False,
            volume=volume,
            song_id=song.id,
        )
    except asyncio.TimeoutError:
        logger.warning(f"play {song.id} did not confirm within {_PLAYER_BUDGET_S}s")
        return (
            f"「{song.label()}」の再生を開始できたか確認できませんでした"
            "（少し遅れて鳴り出すかもしれない）。"
        )
    logger.info(f"playing {song.id} {song.label()}")
    return f"再生開始: {song.label()}"


@mcp.tool()
@_tool
async def music_search(keyword: str, limit: int = 8) -> str:
    """Search NetEase Cloud Music for songs. Returns song titles, artists and
    durations. Use this when the user wants to know what is available, or when
    you want to confirm which version of a song before playing it."""
    songs = await _client.search(keyword, limit)
    if not songs:
        return f"「{keyword}」に一致する曲は見つかりませんでした。"
    return f"「{keyword}」の検索結果:\n{_format(songs)}{_RESULT_NOTE}"


@mcp.tool()
@_tool
async def music_play(
    keyword: str = "", song_id: int = 0, volume: int = _DEFAULT_VOLUME
) -> str:
    """Play music on the user's computer speakers, replacing whatever was
    playing. volume is 0-100; the song plays once (music_stop ends it early).

    Give either a keyword ("米津玄師 Lemon" — title, artist, or both) or an
    exact song_id from music_search. Titles are NOT unique: covers, remixes,
    live versions and karaoke tracks share them constantly, so a keyword plays
    the top match and lists the other candidates with their ids. When the user
    means a particular recording, or the top match was wrong, search first and
    play the id."""
    if song_id:
        songs = await _client.songs_by_id([song_id])
        if not songs:
            return f"id={song_id} の曲が見つかりませんでした。"
        return await _play_song(songs[0], volume)

    if not keyword.strip():
        return "曲名かアーティスト名、または song_id を指定してください。"
    songs = await _client.search(keyword, 5)
    if not songs:
        return f"「{keyword}」に一致する曲は見つかりませんでした。"
    message = await _play_song(songs[0], volume)
    if len(songs) < 2:
        return message
    others = "、".join(f"{s.label()} (id={s.id})" for s in songs[1:4])
    return f"{message}\n（同名の他候補: {others} — 違ったら song_id で指定し直して）"


@mcp.tool()
@_tool
async def music_playlists() -> str:
    """List the user's own NetEase playlists (name and track count), so you
    can pick one to play with music_play_playlist."""
    playlists = await _client.user_playlists()
    if not playlists:
        return "プレイリストが見つかりません（ログインが必要かも）。"
    listing = "\n".join(f"- {p.name}（{p.count}曲）" for p in playlists)
    return f"プレイリスト:\n{listing}{_RESULT_NOTE}"


@mcp.tool()
@_tool
async def music_play_playlist(name: str, volume: int = _DEFAULT_VOLUME) -> str:
    """Play a random song from one of the user's playlists, matched by name
    (partial names are fine). Use music_playlists first if you don't know
    what exists."""
    playlists = await _client.user_playlists()
    wanted = name.strip().lower()
    match = next(
        (p for p in playlists if wanted and wanted in p.name.lower()),
        None,
    )
    if match is None:
        available = "、".join(p.name for p in playlists[:10]) or "（なし）"
        return f"「{name}」というプレイリストは見つかりません。候補: {available}"

    tracks = await _client.playlist_tracks(match.id)
    if not tracks:
        return f"「{match.name}」は空でした。"
    message = await _play_song(random.choice(tracks), volume)
    return f"{message}（{match.name} から）"


@mcp.tool()
@_tool
async def music_now_playing() -> str:
    """What is playing on the user's speakers right now, if anything."""
    try:
        state = await _player(ncm_player.status)
    except asyncio.TimeoutError:
        return "再生状況を確認できませんでした。"
    if state is None:
        return "今は何も再生していません。"
    elapsed = state.get("playing_for", 0)
    loop = "（繰り返し中）" if state.get("loop") else ""
    wake = "（アラーム）" if state.get("wake") else ""
    return f"再生中: {state.get('label', '?')}{wake}{loop} — {elapsed}秒経過"


@mcp.tool()
@_tool
async def music_stop() -> str:
    """Stop whatever is playing on the user's speakers. This is also how an
    alarm that woke the user up gets silenced."""
    try:
        outcome = await _player(ncm_player.stop)
    except asyncio.TimeoutError:
        # Not "nothing was playing" and not "stopped": we genuinely don't know.
        return "止められたか確認できませんでした（まだ鳴っているかもしれない）。"
    if outcome == ncm_player.FAILED:
        # Never fold this into "nothing was playing": the music is still
        # audible, and the user needs to hear that rather than a shrug.
        return "再生を止められませんでした（プレイヤーが応答しない）。まだ鳴っている。"
    if outcome == ncm_player.STOPPED:
        return "再生を止めました。"
    return "今は何も再生していません。"


if __name__ == "__main__":
    _setup_logging()
    mcp.run()
