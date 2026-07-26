"""Tests for the NetEase music module and wake-alarm audio (2026-07-27).

Everything here is offline: the weapi crypto is checked for shape and
determinism, not against the live service, and playback is exercised through
a stubbed player.
"""

import asyncio
import base64
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import strip_tool_markers

_MUSIC_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "experiments" / "netease_music"
)
sys.path.insert(0, str(_MUSIC_DIR))

import ncm_client  # noqa: E402
from ncm_crypto import _PUB_MODULUS, weapi_encrypt  # noqa: E402

from src.open_llm_vtuber.alarms import wake_audio  # noqa: E402


class WeapiCryptoTests(unittest.TestCase):
    def test_payload_shape(self):
        out = weapi_encrypt({"s": "test", "limit": 3})
        self.assertEqual(set(out), {"params", "encSecKey"})
        # encSecKey is a fixed-width hex string: the modulus is 1024-bit, and
        # a short key here means a leading zero got trimmed.
        self.assertEqual(len(out["encSecKey"]), 256)
        int(out["encSecKey"], 16)  # raises if not hex
        # params is base64 of an AES-CBC blob, so it decodes and is block-sized.
        raw = base64.b64decode(out["params"])
        self.assertEqual(len(raw) % 16, 0)

    def test_random_per_call(self):
        a = weapi_encrypt({"s": "x"})
        b = weapi_encrypt({"s": "x"})
        self.assertNotEqual(a["encSecKey"], b["encSecKey"])

    def test_modulus_is_1024_bit(self):
        self.assertEqual(_PUB_MODULUS.bit_length(), 1024)


class CdnUnblockTests(unittest.TestCase):
    def test_rewrites_region_locked_host(self):
        self.assertEqual(
            ncm_client._unblock_cdn("http://m701.music.126.net/a/b.mp3"),
            "http://m701c.music.126.net/a/b.mp3",
        )

    def test_already_rewritten_is_untouched(self):
        url = "http://m701c.music.126.net/a/b.mp3"
        self.assertEqual(ncm_client._unblock_cdn(url), url)

    def test_unrelated_host_untouched(self):
        url = "https://p3.music.126.net/cover.jpg"
        self.assertEqual(ncm_client._unblock_cdn(url), url)


class CacheSidecarTests(unittest.TestCase):
    """The metadata sidecar sits next to the audio; a glob that picks it up
    hands a .json file to the player, which fails in a way that looks like a
    playback bug rather than a lookup bug."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = pathlib.Path(self._tmp.name)
        patcher = mock.patch.object(ncm_client, "CACHE_DIR", self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_picks_audio_not_metadata(self):
        (self.cache / "123.json").write_text("{}", encoding="utf-8")
        (self.cache / "123.mp3").write_bytes(b"ID3fake")
        found = ncm_client.cached_audio_path(123)
        self.assertIsNotNone(found)
        self.assertEqual(found.suffix, ".mp3")

    def test_ignores_partial_downloads(self):
        (self.cache / "77.mp3.part").write_bytes(b"half")
        self.assertIsNone(ncm_client.cached_audio_path(77))

    def test_missing_song(self):
        self.assertIsNone(ncm_client.cached_audio_path(999))

    def test_cached_songs_lists_only_downloaded(self):
        (self.cache / "5.mp3").write_bytes(b"audio")
        (self.cache / "5.json").write_text(
            json.dumps({"id": 5, "name": "S", "artists": "A", "file": "5.mp3"}),
            encoding="utf-8",
        )
        # Metadata whose audio was pruned away must not be offered.
        (self.cache / "6.json").write_text(
            json.dumps({"id": 6, "name": "Gone", "artists": "A", "file": "6.mp3"}),
            encoding="utf-8",
        )
        songs = ncm_client.cached_songs()
        self.assertEqual([s["id"] for s in songs], [5])


class _StubPlayer:
    def __init__(self):
        self.calls = []
        self.playing = False

    def play(self, path, **kwargs):
        self.calls.append((str(path), kwargs))
        self.playing = True
        return {"pid": 1}

    def stop(self):
        was, self.playing = self.playing, False
        return was

    def status(self):
        return {"label": "x"} if self.playing else None


class _StubClient:
    def __init__(self, songs):
        self._songs = songs

    def cached_songs(self):
        return list(self._songs)


async def _never_returns(*_args, **_kwargs):
    await asyncio.sleep(3600)


class _ExplodingPlayer(_StubPlayer):
    """Every OS-touching call fails — the state file is locked, ffplay is
    gone, whatever. Nothing may escape to the caller."""

    def play(self, path, **kwargs):
        raise OSError("boom")

    def stop(self):
        raise OSError("boom")

    def status(self):
        raise OSError("boom")


class WakeAudioTests(unittest.IsolatedAsyncioTestCase):
    def _install(self, songs, player=None):
        player = player or _StubPlayer()
        patcher = mock.patch.object(
            wake_audio, "_load", lambda: (player, _StubClient(songs))
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return player

    async def test_plays_a_cached_song_on_repeat(self):
        player = self._install(
            [{"id": 1, "name": "朝", "artists": "誰か", "path": "C:/x/1.mp3"}]
        )
        label = await wake_audio.start(volume=90)
        self.assertEqual(label, "朝 / 誰か")
        ((path, kwargs),) = player.calls
        self.assertEqual(path, "C:/x/1.mp3")
        self.assertTrue(kwargs["loop"])  # a wake alarm must not play once
        self.assertEqual(kwargs["volume"], 90)

    async def test_playlist_song_preferred_over_cache(self):
        player = self._install(
            [{"id": 9, "name": "古い", "artists": "誰か", "path": "C:/x/9.mp3"}]
        )
        with mock.patch.object(
            wake_audio,
            "_from_playlist",
            mock.AsyncMock(return_value=("C:/x/new.mp3", "新曲 / 誰か")),
        ):
            label = await wake_audio.start(playlist="起床")
        self.assertEqual(label, "新曲 / 誰か")
        self.assertEqual(player.calls[0][0], "C:/x/new.mp3")

    async def test_falls_back_to_cache_when_playlist_fetch_fails(self):
        player = self._install(
            [{"id": 9, "name": "古い", "artists": "誰か", "path": "C:/x/9.mp3"}]
        )
        with mock.patch.object(
            wake_audio,
            "_from_playlist",
            mock.AsyncMock(side_effect=RuntimeError("net")),
        ):
            label = await wake_audio.start(playlist="起床")
        self.assertEqual(label, "古い / 誰か")
        self.assertEqual(player.calls[0][0], "C:/x/9.mp3")

    async def test_falls_back_to_cache_on_fetch_timeout(self):
        player = self._install(
            [{"id": 9, "name": "古い", "artists": "誰か", "path": "C:/x/9.mp3"}]
        )
        with (
            mock.patch.object(wake_audio, "_FETCH_TIMEOUT_S", 0.01),
            mock.patch.object(wake_audio, "_from_playlist", _never_returns),
        ):
            label = await wake_audio.start(playlist="起床")
        self.assertEqual(label, "古い / 誰か")
        self.assertEqual(player.calls[0][0], "C:/x/9.mp3")

    async def test_silent_when_cache_is_empty(self):
        player = self._install([])
        self.assertIsNone(await wake_audio.start())
        self.assertEqual(player.calls, [])
        self.assertFalse(wake_audio.available())

    async def test_stop_reports_whether_it_was_playing(self):
        self._install([{"id": 1, "name": "n", "artists": "a", "path": "p"}])
        await wake_audio.start()
        self.assertTrue(wake_audio.is_playing())
        self.assertTrue(wake_audio.stop())
        self.assertFalse(wake_audio.stop())

    async def test_missing_music_module_degrades_quietly(self):
        with mock.patch.object(wake_audio, "_load", lambda: None):
            self.assertIsNone(await wake_audio.start())
            self.assertFalse(wake_audio.stop())
            self.assertFalse(wake_audio.available())
            self.assertFalse(wake_audio.is_playing())

    async def test_player_errors_never_escape(self):
        # あさひ is usually away from the machine: nothing about the music may
        # take down the alarm path or the message handler.
        self._install(
            [{"id": 1, "name": "n", "artists": "a", "path": "p"}], _ExplodingPlayer()
        )
        self.assertIsNone(await wake_audio.start())
        self.assertFalse(wake_audio.stop())
        self.assertFalse(wake_audio.is_playing())

    async def test_broken_cache_never_escapes(self):
        class _BadClient:
            def cached_songs(self):
                raise OSError("cache unreadable")

        with mock.patch.object(
            wake_audio, "_load", lambda: (_StubPlayer(), _BadClient())
        ):
            self.assertFalse(wake_audio.available())
            self.assertIsNone(await wake_audio.start())


class MusicMarkerTests(unittest.TestCase):
    """Every tool path must show a marker (あさひ audits usage from chat), and
    the music markers carry their result the way the time markers do."""

    def _marker(self, tool, content):
        return BasicMemoryAgent._deferred_tool_marker(tool, content)

    def test_no_pre_marker_for_music_tools(self):
        # The generic 🔧 tag would fire before the result is known.
        self.assertEqual(BasicMemoryAgent._mcp_tool_marker("music_play"), "")
        self.assertEqual(BasicMemoryAgent._mcp_tool_marker("music_stop"), "")

    def test_unknown_tool_still_gets_the_generic_marker(self):
        self.assertIn("🔧", BasicMemoryAgent._mcp_tool_marker("something_else"))

    def test_play_marker_names_the_song(self):
        out = self._marker("music_play", "再生開始: Lemon / 米津玄師")
        self.assertEqual(out, "\n🎵 *再生: Lemon / 米津玄師*\n")

    def test_play_marker_from_mcp_content_blocks(self):
        out = self._marker(
            "music_play_playlist",
            [{"type": "text", "text": "再生開始: 朝 / 誰か（起床 から）"}],
        )
        self.assertIn("🎵", out)
        self.assertIn("朝 / 誰か", out)

    def test_play_failure_shows_the_reason(self):
        out = self._marker(
            "music_play", "「xyzzy」に一致する曲は見つかりませんでした。"
        )
        self.assertIn("🎵", out)
        self.assertIn("見つかりません", out)

    def test_search_marker_carries_the_keyword(self):
        out = self._marker("music_search", "「YOASOBI」の検索結果:\n1. 夜に駆ける")
        self.assertEqual(out, "\n🎵 *曲を検索: YOASOBI*\n")

    def test_stop_and_status_markers(self):
        self.assertEqual(
            self._marker("music_stop", "再生を止めました。"), "\n🎵 *再生停止*\n"
        )
        self.assertEqual(
            self._marker("music_now_playing", "今は何も再生していません。"),
            "\n🎵 *再生状況*\n",
        )

    def test_garbage_result_still_yields_a_marker(self):
        for junk in (None, 12345, [], [{"type": "image"}]):
            self.assertIn("🎵", self._marker("music_stop", junk))

    def test_time_marker_still_works_through_the_dispatcher(self):
        out = self._marker(
            "get_current_time",
            '{"datetime": "2026-07-27T08:30:00+09:00", "day_of_week": "Monday"}',
        )
        self.assertIn("🕐", out)
        self.assertIn("2026-07-27 08:30", out)

    def test_music_markers_are_stripped_from_model_visible_text(self):
        self.assertEqual(
            strip_tool_markers("🎵 *再生: Lemon / 米津玄師*かけたよ。"), "かけたよ。"
        )
        self.assertEqual(strip_tool_markers("🎵 *再生停止*止めた。"), "止めた。")


if __name__ == "__main__":
    unittest.main()
