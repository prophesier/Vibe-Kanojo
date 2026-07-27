"""Pre-download songs so a wake alarm has something to play offline.

    uv run python experiments/netease_music/warm_cache.py "起床" 5

A wake alarm downloads its song when it fires, which is fine with a working
network — but the whole point of the alarm is that it must ring regardless.
Running this once after login (and occasionally after that) fills the cache so
there is always a fallback.

Args: playlist name (partial match, defaults to whatever ``wake_playlist`` is
set to in conf.yaml) and how many songs to fetch (default 5).
"""

from __future__ import annotations

import asyncio
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ncm_client import (  # noqa: E402
    NeteaseClient,
    NeteaseUnavailable,
    cached_songs,
    find_playlist,
)


def _playlist_from_conf() -> str:
    """Read wake_playlist from conf.yaml, if the project config is loadable."""
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        sys.path.insert(0, str(root))
        from src.open_llm_vtuber.config_manager import validate_config  # noqa: PLC0415
        from src.open_llm_vtuber.config_manager.utils import read_yaml  # noqa: PLC0415

        config = validate_config(read_yaml(str(root / "conf.yaml")))
        agent = config.character_config.agent_config.agent_settings
        return str(getattr(agent.basic_memory_agent, "wake_playlist", "") or "")
    except Exception:
        return ""


async def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else _playlist_from_conf()
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if not name:
        print("プレイリスト名を指定するか、conf.yaml の wake_playlist を設定して。")
        return 1

    client = NeteaseClient()
    try:
        who = await client.account()
        if not who:
            print("ログインしていません。先に login.py を実行して。")
            return 1
        print(f"ログイン中: {who}")

        playlists = await client.user_playlists()
        match = find_playlist(playlists, name)
        if match is None:
            print(f"「{name}」が見つかりません。自分のプレイリスト:")
            for p in playlists:
                if p.owned:
                    print(f"  - {p.name}（{p.count}曲）" + ("  ♥" if p.liked else ""))
            return 1

        tracks = await client.playlist_tracks(match.id)
        print(f"「{match.name}」: {len(tracks)}曲")
        if not tracks:
            return 1

        picks = random.sample(tracks, min(count, len(tracks)))
        ok = 0
        for song in picks:
            try:
                path = await client.fetch_audio(song)
                size = path.stat().st_size // 1024
                print(f"  OK   {song.label()}  ({size} KB)")
                ok += 1
            except NeteaseUnavailable as e:
                print(f"  SKIP {song.label()}: {e}")
        print(f"\n{ok}/{len(picks)} 曲をキャッシュしました。")
        print(f"キャッシュ内の曲数: {len(cached_songs())}")
        return 0 if ok else 1
    except NeteaseUnavailable as e:
        print(f"失敗: {e}")
        return 1
    finally:
        await client.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
