"""Tests for the NetEase music module and wake-alarm audio (2026-07-27).

Everything here is offline: the weapi crypto is checked for shape and
determinism, not against the live service, and playback is exercised through
a stubbed player.
"""

import base64
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

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


class WakeAudioTests(unittest.TestCase):
    def _install(self, songs):
        player = _StubPlayer()
        patcher = mock.patch.object(
            wake_audio, "_load", lambda: (player, _StubClient(songs))
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return player

    def test_plays_a_cached_song_on_repeat(self):
        player = self._install(
            [{"id": 1, "name": "朝", "artists": "誰か", "path": "C:/x/1.mp3"}]
        )
        label = wake_audio.start(volume=90)
        self.assertEqual(label, "朝 / 誰か")
        ((path, kwargs),) = player.calls
        self.assertEqual(path, "C:/x/1.mp3")
        self.assertTrue(kwargs["loop"])  # a wake alarm must not play once
        self.assertEqual(kwargs["volume"], 90)

    def test_silent_when_cache_is_empty(self):
        player = self._install([])
        self.assertIsNone(wake_audio.start())
        self.assertEqual(player.calls, [])
        self.assertFalse(wake_audio.available())

    def test_stop_reports_whether_it_was_playing(self):
        self._install([{"id": 1, "name": "n", "artists": "a", "path": "p"}])
        wake_audio.start()
        self.assertTrue(wake_audio.is_playing())
        self.assertTrue(wake_audio.stop())
        self.assertFalse(wake_audio.stop())

    def test_missing_music_module_degrades_quietly(self):
        with mock.patch.object(wake_audio, "_load", lambda: None):
            self.assertIsNone(wake_audio.start())
            self.assertFalse(wake_audio.stop())
            self.assertFalse(wake_audio.available())
            self.assertFalse(wake_audio.is_playing())


if __name__ == "__main__":
    unittest.main()
