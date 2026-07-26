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
import pathlib
import random
import sys

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
logger.remove()  # don't write to stdout — stdout is the MCP stdio channel!
logger.add(sys.stderr, level="INFO")
logger.add(str(HERE / "ncm_mcp.log"), rotation="2 MB", retention=3, level="INFO")

_TOOL_TIMEOUT = 25  # seconds — under the MCP client's 30s read timeout
_DEFAULT_VOLUME = 70

_RESULT_NOTE = (
    "\n\n（メモ：この結果はあなたの参照用で会話履歴には残らない。"
    "曲名やアーティスト名など伝えたいことは、ユーザーへの返信の中で"
    "必ず自分の言葉で伝えること。この内容は次のターンには消えている。）"
)

mcp = FastMCP("netease-music")
_client = NeteaseClient()


async def _run(coro, what: str):
    """Run a client coroutine with a hard timeout; never raise. Returns
    ``(ok, value_or_message)``."""
    try:
        return True, await asyncio.wait_for(coro, timeout=_TOOL_TIMEOUT)
    except NeteaseUnavailable as e:
        logger.warning(f"{what}: unavailable: {e}")
        return False, str(e)
    except asyncio.TimeoutError:
        logger.warning(f"{what}: timed out after {_TOOL_TIMEOUT}s")
        return False, "音楽サービスの応答が遅すぎます（タイムアウト）。"
    except Exception:
        logger.exception(f"{what}: unexpected error")
        return False, "音楽の処理中に問題が発生しました。"


def _format(songs: list[Song]) -> str:
    return "\n".join(
        f"{i}. {s.label()}"
        + (f"（{s.album}）" if s.album and s.album != s.name else "")
        + f" [{s.duration_ms // 60000}:{s.duration_ms // 1000 % 60:02d}]"
        for i, s in enumerate(songs, 1)
    )


async def _play_song(song: Song, volume: int) -> str:
    path = await _client.fetch_audio(song)
    ncm_player.play(
        path, label=song.label(), loop=False, volume=volume, song_id=song.id
    )
    logger.info(f"playing {song.id} {song.label()}")
    return f"再生開始: {song.label()}"


@mcp.tool()
async def music_search(keyword: str, limit: int = 8) -> str:
    """Search NetEase Cloud Music for songs. Returns song titles, artists and
    durations. Use this when the user wants to know what is available, or when
    you want to confirm which version of a song before playing it."""
    ok, result = await _run(_client.search(keyword, limit), "music_search")
    if not ok:
        return result
    if not result:
        return f"「{keyword}」に一致する曲は見つかりませんでした。"
    return f"「{keyword}」の検索結果:\n{_format(result)}{_RESULT_NOTE}"


@mcp.tool()
async def music_play(keyword: str, volume: int = _DEFAULT_VOLUME) -> str:
    """Play music on the user's computer speakers. Takes a song title, an
    artist name, or both ("米津玄師 Lemon"), plays the best match, and replaces
    whatever was playing. volume is 0-100. The song plays once; use
    music_stop to end it early."""
    ok, songs = await _run(_client.search(keyword, 5), "music_play/search")
    if not ok:
        return songs
    if not songs:
        return f"「{keyword}」に一致する曲は見つかりませんでした。"
    ok, message = await _run(_play_song(songs[0], volume), "music_play/fetch")
    return message


@mcp.tool()
async def music_playlists() -> str:
    """List the user's own NetEase playlists (name and track count), so you
    can pick one to play with music_play_playlist."""
    ok, result = await _run(_client.user_playlists(), "music_playlists")
    if not ok:
        return result
    if not result:
        return "プレイリストが見つかりません（ログインが必要かも）。"
    listing = "\n".join(f"- {p.name}（{p.count}曲）" for p in result)
    return f"プレイリスト:\n{listing}{_RESULT_NOTE}"


@mcp.tool()
async def music_play_playlist(name: str, volume: int = _DEFAULT_VOLUME) -> str:
    """Play a random song from one of the user's playlists, matched by name
    (partial names are fine). Use music_playlists first if you don't know
    what exists."""
    ok, playlists = await _run(_client.user_playlists(), "music_play_playlist/list")
    if not ok:
        return playlists
    wanted = name.strip().lower()
    match = next(
        (p for p in playlists if wanted and wanted in p.name.lower()),
        None,
    )
    if match is None:
        available = "、".join(p.name for p in playlists[:10]) or "（なし）"
        return f"「{name}」というプレイリストは見つかりません。候補: {available}"

    ok, tracks = await _run(
        _client.playlist_tracks(match.id), "music_play_playlist/tracks"
    )
    if not ok:
        return tracks
    if not tracks:
        return f"「{match.name}」は空でした。"
    ok, message = await _run(
        _play_song(random.choice(tracks), volume), "music_play_playlist/fetch"
    )
    return f"{message}（{match.name} から）" if ok else message


@mcp.tool()
async def music_now_playing() -> str:
    """What is playing on the user's speakers right now, if anything."""
    state = ncm_player.status()
    if state is None:
        return "今は何も再生していません。"
    elapsed = state.get("playing_for", 0)
    loop = "（繰り返し中）" if state.get("loop") else ""
    return f"再生中: {state.get('label', '?')}{loop} — {elapsed}秒経過"


@mcp.tool()
async def music_stop() -> str:
    """Stop whatever is playing on the user's speakers. This is also how an
    alarm that woke the user up gets silenced."""
    return "再生を止めました。" if ncm_player.stop() else "何も再生していません。"


if __name__ == "__main__":
    mcp.run()
