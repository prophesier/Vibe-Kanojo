"""Tests for the NetEase music module and wake-alarm audio (2026-07-27).

Everything here is offline: the weapi crypto is checked for shape and
determinism, not against the live service, and playback is exercised through
a stubbed player.
"""

import asyncio
import base64
import contextlib
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import strip_tool_markers

_MUSIC_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "experiments" / "netease_music"
)
sys.path.insert(0, str(_MUSIC_DIR))

import ncm_client  # noqa: E402
import ncm_player  # noqa: E402
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


class CdnHostTests(unittest.TestCase):
    def test_rewrites_to_the_preferred_host(self):
        self.assertEqual(
            ncm_client._preferred_cdn_host("http://m701.music.126.net/a/b.mp3"),
            "http://m701c.music.126.net/a/b.mp3",
        )

    def test_already_rewritten_is_untouched(self):
        url = "http://m701c.music.126.net/a/b.mp3"
        self.assertEqual(ncm_client._preferred_cdn_host(url), url)

    def test_unrelated_host_untouched(self):
        url = "https://p3.music.126.net/cover.jpg"
        self.assertEqual(ncm_client._preferred_cdn_host(url), url)


class CookieConflictTests(unittest.TestCase):
    """NetEase sets the same cookie name under several domains at once. httpx's
    mapping interface raises CookieConflict on an ambiguous name — and it does
    so at exactly the worst moment, the instant a QR login is confirmed, which
    is what silently broke the first two login attempts."""

    def _client(self):
        import httpx

        client = httpx.AsyncClient()
        client.cookies.set("MUSIC_A_T", "aaa", domain=".163.com")
        client.cookies.set("MUSIC_A_T", "bbb", domain="music.163.com")
        client.cookies.set("__csrf", "tok", domain="music.163.com")
        return client

    def test_plain_dict_would_raise(self):
        import httpx

        with self.assertRaises(httpx.CookieConflict):
            dict(self._client().cookies)

    def test_cookie_dict_survives_duplicates(self):
        jar = ncm_client.cookie_dict(self._client())
        self.assertEqual(jar["MUSIC_A_T"], "bbb")  # music.163.com wins
        self.assertEqual(jar["__csrf"], "tok")

    def test_cookie_value_helper(self):
        client = self._client()
        self.assertEqual(ncm_client.cookie_value(client, "__csrf"), "tok")
        self.assertEqual(ncm_client.cookie_value(client, "nope", "dflt"), "dflt")


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


class PlaylistMatchTests(unittest.TestCase):
    """The playlist list holds other people's playlists too, and NetEase names
    every ♥ collection "<nickname>喜欢的音乐" — so a substring search for that
    matches several, and only ownership says which one is his."""

    def _lists(self):
        P = ncm_client.Playlist
        return [
            P(1, "_init__喜欢的音乐", 1635, owned=True, liked=True),
            P(2, "精选", 492, owned=True),
            P(3, "纯音乐", 7, owned=True),
            P(4, "nonosama喜欢的音乐", 2428),  # collected: someone else's ♥
            P(5, "又失眠的猫喜欢的音乐", 1048),
            P(6, "精选集：夜", 88),  # collected, and a superstring of "精选"
        ]

    def _find(self, name):
        return ncm_client.find_playlist(self._lists(), name)

    def test_his_own_hearts_win_over_a_strangers(self):
        self.assertEqual(self._find("喜欢的音乐").id, 1)

    def test_an_alias_reaches_the_hearts_without_naming_them(self):
        for alias in ("小红心", "红心", "ハート", "liked"):
            self.assertEqual(self._find(alias).id, 1, alias)

    def test_an_exact_name_beats_a_longer_one_that_contains_it(self):
        self.assertEqual(self._find("精选").id, 2)

    def test_ownership_breaks_ties_at_every_tier(self):
        # "音乐" is a substring of one owned and two collected names.
        self.assertTrue(self._find("音乐").owned)

    def test_a_collected_playlist_is_still_reachable_by_name(self):
        self.assertEqual(self._find("nonosama").id, 4)

    def test_no_match_and_empty_name(self):
        self.assertIsNone(self._find("存在しない"))
        self.assertIsNone(self._find(""))
        self.assertIsNone(self._find("   "))


class _StubPlayer:
    """Models the two distinctions the real player makes: a ringing alarm and a
    song the character put on are both "playing" but only the first answers to
    ``stop(only_wake=True)``, and "nothing was playing" is a different answer
    from "it would not stop"."""

    STOPPED, IDLE, FAILED = ncm_player.STOPPED, ncm_player.IDLE, ncm_player.FAILED

    def __init__(self):
        self.calls = []
        self.state = None
        self.unstoppable = False

    def play(self, path, **kwargs):
        self.calls.append((str(path), kwargs))
        self.state = {
            "label": kwargs.get("label", "x"),
            "wake": bool(kwargs.get("wake")),
            "loop": bool(kwargs.get("loop")),
        }
        return dict(self.state, pid=1)

    def stop(self, *, only_wake=False):
        if self.state is None or (only_wake and not self.state["wake"]):
            return self.IDLE
        if self.unstoppable:
            return self.FAILED
        self.state = None
        return self.STOPPED

    def status(self):
        return dict(self.state) if self.state else None


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

    def stop(self, *, only_wake=False):
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
        self.assertTrue(kwargs["wake"])  # ...and must be stoppable as an alarm
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

    async def test_answering_does_not_kill_music_the_character_put_on(self):
        # The server stops wake audio on EVERY incoming message. If that stop
        # didn't distinguish the two, asking for a song and then saying one
        # more word would cut it off mid-chorus.
        player = self._install([])
        player.play("C:/x/song.mp3", label="Lemon / 米津玄師", loop=False, wake=False)
        self.assertFalse(wake_audio.stop())
        self.assertIsNotNone(player.status())
        self.assertIsNone(wake_audio.ringing())
        self.assertFalse(wake_audio.is_playing())

    async def test_an_alarm_that_will_not_stop_is_not_reported_as_stopped(self):
        player = self._install([{"id": 1, "name": "n", "artists": "a", "path": "p"}])
        await wake_audio.start()
        player.unstoppable = True
        self.assertFalse(wake_audio.stop())
        self.assertTrue(wake_audio.is_playing())  # and it is still ringing

    async def test_a_residual_player_is_not_mistaken_for_the_alarm(self):
        # It is audible, so status() reports it — but it is a leftover nobody
        # chose. Counting it as the alarm would short-circuit the next real one.
        player = self._install(
            [{"id": 1, "name": "朝", "artists": "誰か", "path": "p"}]
        )
        await wake_audio.start()
        self.assertIsNotNone(wake_audio.ringing())
        player.state["residual"] = True
        self.assertIsNone(wake_audio.ringing())

    async def test_a_stale_ringing_record_does_not_suppress_the_next_alarm(self):
        # The player answers "still running" when the OS won't say — safe when
        # deciding whether to keep trying to stop something, dangerous here,
        # where believing it forever would mean no alarm ever rings again.
        player = self._install(
            [{"id": 1, "name": "朝", "artists": "誰か", "path": "p"}]
        )
        await wake_audio.start()
        self.assertIsNotNone(wake_audio.ringing())
        player.state["playing_for"] = wake_audio.MAX_RING_SECONDS + 1
        self.assertIsNone(wake_audio.ringing())

    async def test_ringing_reports_the_alarm_song(self):
        self._install([{"id": 1, "name": "朝", "artists": "誰か", "path": "p"}])
        self.assertIsNone(wake_audio.ringing())
        await wake_audio.start()
        self.assertEqual(wake_audio.ringing(), "朝 / 誰か")

    async def test_pick_and_play_are_separable(self):
        # The server needs a point between "song in hand" and "sound starts"
        # to bail out at, because the download in between can take seconds.
        player = self._install(
            [{"id": 1, "name": "朝", "artists": "誰か", "path": "C:/x/1.mp3"}]
        )
        choice = await wake_audio.pick()
        self.assertEqual(choice, ("C:/x/1.mp3", "朝 / 誰か"))
        self.assertEqual(player.calls, [])  # nothing rang yet
        self.assertEqual(wake_audio.play(*choice), "朝 / 誰か")
        self.assertEqual(len(player.calls), 1)

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


class PlayerStateTests(unittest.TestCase):
    """The state file is the only thing that makes an alarm started by the
    server stoppable by the MCP server. Everything here is about not lying in
    it — a stop we didn't achieve, or a pid that isn't ours any more."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)
        for name, value in (
            ("STATE_FILE", self.dir / "playback.json"),
            ("LOCK_FILE", self.dir / "playback.lock"),
        ):
            patcher = mock.patch.object(ncm_player, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.killed = []
        patcher = mock.patch.object(ncm_player, "_kill", self.killed.append)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_state(self, **extra):
        state = {"pid": 4242, "exe": "ffplay.exe", "label": "song", "wake": False}
        state.update(extra)
        ncm_player.STATE_FILE.write_text(json.dumps(state), encoding="utf-8")

    def _alive(self, *answers):
        """Answer the liveness probe with a fixed sequence (before the kill,
        then after it)."""
        seq = iter(answers)
        patcher = mock.patch.object(
            ncm_player,
            "_probe_process",
            lambda state: ncm_player.ALIVE if next(seq) else ncm_player.DEAD,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _spawning(self, pid=999):
        exe = str(self.dir / "ffplay.exe")
        return (
            mock.patch.object(ncm_player, "ffplay_path", lambda: exe),
            mock.patch.object(
                ncm_player.subprocess, "Popen", return_value=mock.Mock(pid=pid)
            ),
        )

    def test_only_wake_leaves_ordinary_music_alone(self):
        self._write_state(wake=False)
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.IDLE)
        self.assertEqual(self.killed, [])
        self.assertTrue(ncm_player.STATE_FILE.exists())

    def test_only_wake_stops_a_ringing_alarm(self):
        self._write_state(wake=True)
        self._alive(True, False)
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.STOPPED)
        self.assertEqual(self.killed, [4242])
        self.assertFalse(ncm_player.STATE_FILE.exists())

    def test_plain_stop_stops_ordinary_music_too(self):
        # music_stop, called by the character, means "stop whatever is on".
        self._write_state(wake=False)
        self._alive(True, False)
        self.assertEqual(ncm_player.stop(), ncm_player.STOPPED)
        self.assertEqual(self.killed, [4242])

    def test_a_kill_that_failed_is_not_reported_as_a_stop(self):
        # FAILED, not IDLE: the music is still audible, and the state stays so
        # a later stop can reach the process again.
        self._write_state(wake=True)
        self._alive(True, True)
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.FAILED)
        self.assertTrue(ncm_player.STATE_FILE.exists())

    def test_a_player_that_already_exited_clears_the_state(self):
        self._write_state(wake=True)
        self._alive(False)
        self.assertEqual(ncm_player.stop(), ncm_player.IDLE)
        self.assertEqual(self.killed, [])
        self.assertFalse(ncm_player.STATE_FILE.exists())

    def test_nothing_playing_costs_nothing(self):
        # This runs on every incoming message: no lock, no subprocess.
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.IDLE)
        self.assertFalse(ncm_player.LOCK_FILE.exists())

    def test_corrupt_state_never_escapes(self):
        ncm_player.STATE_FILE.write_text("{not json", encoding="utf-8")
        self.assertEqual(ncm_player.stop(), ncm_player.IDLE)
        self.assertIsNone(ncm_player.status())

    def test_status_does_not_drop_a_record_written_while_it_probed(self):
        # status() probes without the lock, and probing takes a moment. If a
        # new player was started in that moment, clearing "the" state would
        # orphan music that is genuinely playing.
        self._write_state(pid=111, started_at=1.0)
        stale = json.loads(ncm_player.STATE_FILE.read_text("utf-8"))

        def _probe(entry):
            if ncm_player._state_pid(entry) == 111:
                # The other process gets in while we are asking about the old one.
                self._write_state(pid=222, started_at=2.0)
                return False
            return True

        with mock.patch.object(
            ncm_player,
            "_probe_process",
            lambda entry: (ncm_player.ALIVE if _probe(entry) else ncm_player.DEAD),
        ):
            reported = ncm_player.status()
        self.assertIsNotNone(reported)
        self.assertEqual(reported["pid"], 222)
        survived = json.loads(ncm_player.STATE_FILE.read_text("utf-8"))
        self.assertEqual(survived["pid"], 222)
        self.assertNotEqual(survived["pid"], stale["pid"])

    def test_an_unkillable_player_is_carried_into_the_next_record(self):
        # Overwriting its pid is exactly how music becomes unstoppable.
        self._write_state(pid=111, wake=True)
        audio = self.dir / "s.mp3"
        audio.write_bytes(b"x")
        self._alive(True, True)  # refuses to die, before and after the kill
        exe, popen = self._spawning(pid=222)
        with exe, popen:
            state = ncm_player.play(audio, label="次")
        self.assertEqual(state["pid"], 222)
        self.assertEqual([o["pid"] for o in state["orphans"]], [111])

    def test_a_later_stop_reaps_the_orphan_too(self):
        self._write_state(pid=222, orphans=[{"pid": 111, "exe": "ffplay.exe"}])
        self._alive(True, False, True, False)  # current, then the orphan
        self.assertEqual(ncm_player.stop(), ncm_player.STOPPED)
        self.assertEqual(sorted(self.killed), [111, 222])
        self.assertFalse(ncm_player.STATE_FILE.exists())

    def test_status_promotes_a_live_orphan_when_the_current_song_ends(self):
        self._write_state(
            pid=222,
            started_at=2.0,
            orphans=[
                {
                    "pid": 111,
                    "exe": "ffplay.exe",
                    "started_at": 1.0,
                    "label": "residual",
                    "wake": True,
                }
            ],
        )

        def _probe(entry):
            return (
                ncm_player.ALIVE
                if ncm_player._state_pid(entry) == 111
                else ncm_player.DEAD
            )

        with mock.patch.object(ncm_player, "_probe_process", _probe):
            state = ncm_player.status()
        self.assertIsNotNone(state)
        self.assertEqual(state["pid"], 111)
        saved = json.loads(ncm_player.STATE_FILE.read_text("utf-8"))
        self.assertEqual(saved["pid"], 111)
        self.assertTrue(saved["wake"])
        # Promoted, but still a leftover: whoever reads this must be able to
        # tell it apart from an alarm somebody actually started.
        self.assertTrue(saved["residual"])

    def test_a_promoted_leftover_is_not_labelled_an_alarm(self):
        # A residual from an ordinary song must not inherit "wake". Marking it
        # so would make the next real alarm see "already ringing" and stay
        # silent — the one outcome this whole feature exists to prevent.
        self._write_state(
            pid=222,
            wake=False,
            label="Lemon",
            started_at=2.0,
            orphans=[{"pid": 111, "exe": "ffplay.exe", "started_at": 1.0}],
        )

        def _probe(entry):
            return (
                ncm_player.ALIVE
                if ncm_player._state_pid(entry) == 111
                else ncm_player.DEAD
            )

        with mock.patch.object(ncm_player, "_probe_process", _probe):
            state = ncm_player.status()
        self.assertEqual(state["pid"], 111)
        self.assertFalse(state.get("wake"))
        self.assertTrue(state["residual"])

    def test_a_promoted_leftover_is_still_reaped_by_an_incoming_message(self):
        # The other half of the split: not an alarm, but nobody asked for it
        # either, so answering must still silence it.
        self._write_state(pid=111, wake=False, residual=True, started_at=1.0)
        self._alive(True, False)
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.STOPPED)
        self.assertEqual(self.killed, [111])

    def test_orphans_are_reaped_even_on_the_wake_only_path(self):
        # Nobody chose to keep a leftover playing, whatever it was.
        self._write_state(pid=222, wake=False, orphans=[{"pid": 111}])
        self._alive(True, False)
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.STOPPED)
        self.assertEqual(self.killed, [111])

    def test_wake_only_reaps_orphan_but_preserves_current_requested_song(self):
        self._write_state(
            pid=222,
            wake=False,
            started_at=2.0,
            orphans=[{"pid": 111, "exe": "ffplay.exe", "started_at": 1.0}],
        )
        self._alive(True, False)  # orphan before/after kill; current is untouched
        self.assertEqual(ncm_player.stop(only_wake=True), ncm_player.STOPPED)
        self.assertEqual(self.killed, [111])
        saved = json.loads(ncm_player.STATE_FILE.read_text("utf-8"))
        self.assertEqual(saved["pid"], 222)
        self.assertFalse(saved["wake"])

    def test_play_records_what_a_later_stop_will_need(self):
        audio = self.dir / "s.mp3"
        audio.write_bytes(b"x")
        exe, popen = self._spawning()
        with exe, popen:
            state = ncm_player.play(audio, label="朝", loop=True, volume=90, wake=True)
        self.assertEqual(state["pid"], 999)
        self.assertEqual(state["exe"], "ffplay.exe")  # guards against pid reuse
        self.assertTrue(state["wake"])
        saved = json.loads(ncm_player.STATE_FILE.read_text("utf-8"))
        self.assertEqual((saved["pid"], saved["wake"]), (999, True))

    def test_play_replaces_the_previous_player(self):
        self._write_state(pid=111, wake=True)
        audio = self.dir / "s.mp3"
        audio.write_bytes(b"x")
        self._alive(True, False)
        exe, popen = self._spawning()
        with exe, popen:
            ncm_player.play(audio, label="次", wake=False)
        self.assertEqual(self.killed, [111])
        saved = json.loads(ncm_player.STATE_FILE.read_text("utf-8"))
        self.assertEqual((saved["pid"], saved["wake"]), (999, False))

    def test_play_never_drops_survivors_to_fit_an_orphan_cap(self):
        self._write_state(
            pid=100,
            started_at=1.0,
            orphans=[
                {"pid": pid, "exe": "ffplay.exe", "started_at": float(pid)}
                for pid in range(101, 105)
            ],
        )
        audio = self.dir / "s.mp3"
        audio.write_bytes(b"x")
        self._alive(*([True, True] * 5))
        exe, popen = self._spawning(pid=999)
        with exe, popen:
            state = ncm_player.play(audio, label="次")
        self.assertEqual(
            [entry["pid"] for entry in state["orphans"]], list(range(100, 105))
        )

    def test_play_fails_closed_when_the_state_lock_is_unavailable(self):
        audio = self.dir / "s.mp3"
        audio.write_bytes(b"x")

        @contextlib.contextmanager
        def _unavailable():
            yield False

        exe, popen = self._spawning(pid=999)
        with (
            exe,
            popen as popen_mock,
            mock.patch.object(ncm_player, "_state_lock", _unavailable),
            self.assertRaises(ncm_player.PlaybackError),
        ):
            ncm_player.play(audio, label="次")
        popen_mock.assert_not_called()

    def test_unpersisted_new_player_is_terminated_through_its_handle(self):
        audio = self.dir / "s.mp3"
        audio.write_bytes(b"x")
        proc = mock.Mock(pid=999)
        exe = mock.patch.object(
            ncm_player, "ffplay_path", lambda: str(self.dir / "ffplay.exe")
        )
        popen = mock.patch.object(ncm_player.subprocess, "Popen", return_value=proc)
        with (
            exe,
            popen,
            mock.patch.object(ncm_player, "_write_state", return_value=False),
            self.assertRaises(ncm_player.PlaybackError),
        ):
            ncm_player.play(audio, label="次")
        proc.kill.assert_called_once_with()
        proc.wait.assert_called_once()


class ProcessIdentityTests(unittest.TestCase):
    """stop() kills a whole process *tree*. Pids get recycled, so matching on
    the pid alone could one day take down whatever inherited it."""

    def test_right_pid_wrong_program_is_not_our_player(self):
        # tasklist succeeded but its image-name filter found no matching row.
        state = {"pid": os.getpid(), "exe": "ffplay.exe"}
        no_match = mock.Mock(returncode=0, stdout="INFO: No tasks are running...\n")
        with mock.patch.object(ncm_player.subprocess, "run", return_value=no_match):
            self.assertEqual(ncm_player._probe_process(state), ncm_player.DEAD)

    def test_no_pid_is_not_alive(self):
        self.assertEqual(ncm_player._probe_process({}), ncm_player.DEAD)
        self.assertEqual(ncm_player._probe_process({"pid": "junk"}), ncm_player.DEAD)

    @unittest.skipUnless(sys.platform == "win32", "the returncode path is Windows")
    def test_a_failing_tasklist_is_not_read_as_a_dead_process(self):
        # tasklist prints nothing and returns non-zero when it fails (bad
        # filter, Access Denied) — empty output alone looks exactly like "no
        # such process", and reading it that way is how a stop that never
        # happened gets reported as done.
        failed = mock.Mock(returncode=1, stdout="")
        with mock.patch.object(ncm_player.subprocess, "run", return_value=failed):
            self.assertEqual(ncm_player._probe_process({"pid": 5}), ncm_player.UNKNOWN)

    @unittest.skipUnless(sys.platform == "win32", "the returncode path is Windows")
    def test_an_unknown_pid_is_preserved_but_never_killed(self):
        failed = mock.Mock(returncode=1, stdout="")
        killed = []
        with (
            mock.patch.object(ncm_player.subprocess, "run", return_value=failed),
            mock.patch.object(ncm_player, "_kill", killed.append),
        ):
            did_kill, survivors = ncm_player._reap_entries([{"pid": 5}])
        self.assertFalse(did_kill)
        self.assertEqual([entry["pid"] for entry in survivors], [5])
        self.assertEqual(killed, [])

    @unittest.skipUnless(sys.platform == "win32", "the returncode path is Windows")
    def test_a_stop_after_a_failing_probe_is_not_claimed_as_done(self):
        # The consequence of the rule above, at the level that matters.
        failed = mock.Mock(returncode=1, stdout="")
        with (
            mock.patch.object(ncm_player.subprocess, "run", return_value=failed),
            mock.patch.object(ncm_player, "STATE_FILE", self._state_file()),
            mock.patch.object(ncm_player, "LOCK_FILE", self._state_file(".lock")),
        ):
            ncm_player.STATE_FILE.write_text(
                json.dumps({"pid": 4242, "exe": "ffplay.exe", "wake": True}),
                encoding="utf-8",
            )
            self.assertEqual(ncm_player.stop(), ncm_player.FAILED)
            self.assertTrue(ncm_player.STATE_FILE.exists())

    def _state_file(self, suffix=".json"):
        tmp = getattr(self, "_tmpdir", None)
        if tmp is None:
            tmp = self._tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
        return pathlib.Path(tmp.name) / f"playback{suffix}"

    @unittest.skipUnless(sys.platform == "win32", "the returncode path is Windows")
    def test_a_clean_no_match_still_means_dead(self):
        # The real thing: exit code 0 with an informational line, no CSV row.
        empty = mock.Mock(returncode=0, stdout="INFO: No tasks are running...\n")
        with mock.patch.object(ncm_player.subprocess, "run", return_value=empty):
            self.assertEqual(ncm_player._probe_process({"pid": 5}), ncm_player.DEAD)

    @unittest.skipUnless(sys.platform == "win32", "tasklist is Windows-only")
    def test_a_live_process_is_recognised_by_its_own_image_name(self):
        pid = os.getpid()
        state = {"pid": pid, "exe": pathlib.Path(sys.executable).name}
        found = mock.Mock(returncode=0, stdout=f'"python.exe","{pid}","Console"\n')
        with mock.patch.object(ncm_player.subprocess, "run", return_value=found):
            self.assertEqual(ncm_player._probe_process(state), ncm_player.ALIVE)


class PlayerLockTests(unittest.TestCase):
    """Two processes share this player: the server starts the alarm, the MCP
    server stops it. Without the lock they can each spawn one and orphan the
    other's — music at wake volume with no pid on record to stop it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(
            ncm_player, "LOCK_FILE", pathlib.Path(tmp.name) / "playback.lock"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_lock_is_taken_and_released(self):
        with ncm_player._state_lock() as acquired:
            self.assertTrue(acquired)
            self.assertTrue(ncm_player.LOCK_FILE.exists())
        # The inode persists; only the OS lock is released.
        with ncm_player._state_lock() as acquired_again:
            self.assertTrue(acquired_again)

    def test_contention_reports_failure_instead_of_pretending_to_hold_the_lock(self):
        with ncm_player._state_lock() as outer:
            self.assertTrue(outer)
            with mock.patch.object(ncm_player, "_LOCK_WAIT_S", 0.05):
                with ncm_player._state_lock() as inner:
                    self.assertFalse(inner)
        self.assertTrue(ncm_player.LOCK_FILE.exists())

    def test_a_stale_lock_file_without_an_os_owner_is_harmless(self):
        ncm_player.LOCK_FILE.write_text("crashed holding it", encoding="utf-8")
        with ncm_player._state_lock() as acquired:
            self.assertTrue(acquired)

    def test_an_error_inside_still_releases_it(self):
        with self.assertRaises(ValueError):
            with ncm_player._state_lock() as acquired:
                self.assertTrue(acquired)
                raise ValueError("boom")
        with ncm_player._state_lock() as acquired_again:
            self.assertTrue(acquired_again)


class MusicToolContractTests(unittest.IsolatedAsyncioTestCase):
    """The MCP client gives a tool call 30 seconds total, so the server's
    budget has to cover the whole call rather than each step inside it."""

    @classmethod
    def setUpClass(cls):
        import server as ncm_server  # noqa: PLC0415 — sys.path set at import

        cls.server = ncm_server

    async def test_every_tool_survived_the_timeout_wrapper(self):
        tools = {t.name: t for t in await self.server.mcp.list_tools()}
        self.assertEqual(
            set(tools),
            {
                "music_search",
                "music_play",
                "music_playlists",
                "music_play_playlist",
                "music_now_playing",
                "music_stop",
            },
        )
        # Wrapping must not have eaten the signature or the docstring the
        # model reads to decide how to call it.
        self.assertEqual(
            set(tools["music_play"].inputSchema["properties"]),
            {"keyword", "song_id", "volume"},
        )
        self.assertIn("song_id", tools["music_play"].description)

    async def test_one_budget_covers_the_whole_call(self):
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(30)

        # music_play_playlist used to be three separate 25s steps — 75s worth
        # of patience against a client that gives up at 30.
        with (
            mock.patch.object(self.server, "_TOOL_TIMEOUT", 0.05),
            mock.patch.object(self.server._client, "user_playlists", _hang),
        ):
            started = time.monotonic()
            out = await self.server.music_play_playlist("any")
            elapsed = time.monotonic() - started
        self.assertIn("タイムアウト", out)
        self.assertLess(elapsed, 1.0)

    async def test_failures_come_back_as_something_sayable(self):
        async def _boom(*_args, **_kwargs):
            raise RuntimeError("network on fire")

        with mock.patch.object(self.server._client, "search", _boom):
            out = await self.server.music_search("x")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip())

    async def test_music_stop_does_not_confuse_idle_with_stuck(self):
        for outcome, expected in (
            (ncm_player.STOPPED, "止めました"),
            (ncm_player.IDLE, "何も再生していません"),
            (ncm_player.FAILED, "まだ鳴っている"),
        ):
            with mock.patch.object(ncm_player, "stop", lambda **_k: outcome):
                out = await self.server.music_stop()
            self.assertIn(expected, out, f"for {outcome}")

    async def test_a_player_that_will_not_confirm_is_not_called_started(self):
        # The player start runs in a thread, and a thread already inside a
        # blocking OS call cannot be cancelled. Claiming either outcome would
        # be a guess; the tool says so instead.
        class _Song:
            id = 42

            def label(self):
                return "X / Y"

        async def _search(*_args, **_kwargs):
            return [_Song()]

        async def _fetch(_song):
            return "C:/x/1.mp3"

        def _slow_play(*_args, **_kwargs):
            time.sleep(0.5)

        with (
            mock.patch.object(self.server, "_PLAYER_BUDGET_S", 0.05),
            mock.patch.object(self.server._client, "search", _search),
            mock.patch.object(self.server._client, "fetch_audio", _fetch),
            mock.patch.object(ncm_player, "play", _slow_play),
        ):
            out = await self.server.music_play(keyword="x")
        self.assertIn("確認できませんでした", out)
        self.assertNotIn("再生開始", out)

    async def test_slow_lookup_cannot_let_outer_timeout_land_mid_player(self):
        class _Song:
            id = 42

            def label(self):
                return "X / Y"

        async def _slow_search(*_args, **_kwargs):
            await asyncio.sleep(0.07)
            return [_Song()]

        fetched = []
        played = []

        async def _fetch(song):
            fetched.append(song.id)
            return "C:/x/1.mp3"

        def _play(*_args, **_kwargs):
            played.append(True)

        with (
            mock.patch.object(self.server, "_TOOL_TIMEOUT", 0.12),
            mock.patch.object(self.server, "_PLAYER_BUDGET_S", 0.08),
            mock.patch.object(self.server, "_PLAYER_RETURN_MARGIN_S", 0.01),
            mock.patch.object(self.server._client, "search", _slow_search),
            mock.patch.object(self.server._client, "fetch_audio", _fetch),
            mock.patch.object(ncm_player, "play", _play),
        ):
            out = await self.server.music_play(keyword="x")
        self.assertIn("開始しませんでした", out)
        self.assertEqual(fetched, [])
        self.assertEqual(played, [])


class WakeRaceTests(unittest.IsolatedAsyncioTestCase):
    """Choosing the song takes seconds — a download, sometimes. If あさひ
    answers during them he is already awake, and starting the music at that
    point would be noise arriving after the conversation moved on."""

    def _handler(self):
        from src.open_llm_vtuber.websocket_handler import (  # noqa: PLC0415
            WebSocketHandler,
        )

        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler._wake_epoch = 0
        handler._wake_timeout_task = None
        handler._wake_stop_tasks = set()
        # Belt and braces: nothing in these tests may reach the real player
        # and touch あさひ's actual playback state.
        patcher = mock.patch.object(
            wake_audio, "_load", lambda: (_StubPlayer(), _StubClient([]))
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return handler

    def _patched(self, pick, played, ringing=None, stopped=None):
        stopped = [] if stopped is None else stopped
        return (
            mock.patch.object(wake_audio, "pick", pick),
            mock.patch.object(wake_audio, "ringing", lambda: ringing),
            mock.patch.object(wake_audio, "stop", lambda: bool(stopped.append("stop"))),
            mock.patch.object(
                wake_audio,
                "play",
                lambda path, label, *a, **k: (played.append(label), label)[1],
            ),
        )

    async def _drain(self, handler):
        """Let the backgrounded stop finish before the patches come off."""
        if handler._wake_stop_tasks:
            await asyncio.gather(*list(handler._wake_stop_tasks))

    async def test_answering_during_the_download_cancels_the_ringing(self):
        handler = self._handler()
        played = []

        async def _slow_pick(playlist=""):
            handler.stop_wake_audio("user replied")  # he answers mid-download
            return ("C:/x/1.mp3", "朝 / 誰か")

        a, b, c, d = self._patched(_slow_pick, played)
        with a, b, c, d:
            self.assertIsNone(await handler._start_wake_audio())
            await self._drain(handler)
        self.assertEqual(played, [])

    async def test_answering_as_it_starts_silences_it_again(self):
        # The stop his message scheduled can run before there is anything to
        # stop; the start has to notice and undo itself.
        handler = self._handler()
        played, stopped = [], []

        async def _pick(playlist=""):
            return ("C:/x/1.mp3", "朝 / 誰か")

        def _play_then_interrupt(path, label, *a, **k):
            played.append(label)
            handler._wake_epoch += 1  # his message lands during the start
            return label

        a, b, c, d = self._patched(_pick, played, stopped=stopped)
        with a, b, c, d, mock.patch.object(wake_audio, "play", _play_then_interrupt):
            self.assertIsNone(await handler._start_wake_audio())
            await self._drain(handler)
        self.assertEqual(played, ["朝 / 誰か"])
        self.assertEqual(stopped, ["stop"])  # started, then taken back

    async def test_an_undisturbed_alarm_still_rings(self):
        handler = self._handler()
        played = []

        async def _pick(playlist=""):
            return ("C:/x/1.mp3", "朝 / 誰か")

        a, b, c, d = self._patched(_pick, played)
        with a, b, c, d:
            self.assertEqual(await handler._start_wake_audio(), "朝 / 誰か")
        self.assertEqual(played, ["朝 / 誰か"])
        self.assertIsNotNone(handler._wake_timeout_task)
        handler._wake_timeout_task.cancel()

    async def test_a_retry_does_not_restart_a_song_already_ringing(self):
        # Re-rolling the song would also push the 15-minute backstop out.
        handler = self._handler()
        played = []

        async def _pick(playlist=""):
            raise AssertionError("should not have looked for a new song")

        a, b, c, d = self._patched(_pick, played, ringing="朝 / 誰か")
        with a, b, c, d:
            self.assertEqual(await handler._start_wake_audio(), "朝 / 誰か")
        self.assertEqual(played, [])

    async def test_stopping_the_alarm_never_blocks_the_socket_loop(self):
        # tasklist and taskkill are not instant, and this runs on every single
        # message he sends.
        handler = self._handler()

        def _slow_stop():
            time.sleep(0.4)
            return True

        with mock.patch.object(wake_audio, "stop", _slow_stop):
            started = time.monotonic()
            handler.stop_wake_audio("user replied")
            self.assertLess(time.monotonic() - started, 0.1)
            await self._drain(handler)


class _FakeAlarmStore:
    def __init__(self, pending=(), near=None):
        self._pending = list(pending)
        self._near = near

    async def list_pending(self):
        return list(self._pending)

    async def find_near(self, fire_at_utc, within_seconds=300):
        return self._near

    async def add(self, *, fire_at_utc, note, wake=False):
        return {
            "id": "new",
            "fire_at_utc": fire_at_utc.isoformat(),
            "note": note,
            "wake": wake,
        }


class AlarmWakeVisibilityTests(unittest.IsolatedAsyncioTestCase):
    """``wake`` is persisted correctly, but the model can only answer "which
    of these will actually wake me" if the tool results carry it back."""

    def _agent(self, store):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._alarm_store = store
        agent._turn_inproc_calls = []
        return agent

    async def test_list_alarms_says_which_ones_ring(self):
        store = _FakeAlarmStore(
            [
                {"id": "a", "fire_at_utc": "2026-07-27T22:00:00+00:00", "note": "薬"},
                {
                    "id": "b",
                    "fire_at_utc": "2026-07-27T23:00:00+00:00",
                    "note": "起床",
                    "wake": True,
                },
            ]
        )
        _, result = await self._agent(store)._run_alarm_tool("list_alarms", {})
        self.assertEqual([a["wake"] for a in result["alarms"]], [False, True])

    async def test_a_silent_neighbour_does_not_quietly_block_a_wake_alarm(self):
        store = _FakeAlarmStore(
            near={"id": "old", "fire_at_utc": "2026-07-28T22:00:00+00:00", "note": "薬"}
        )
        _, result = await self._agent(store)._run_alarm_tool(
            "set_alarm", {"note": "起きる", "at": "07:00", "wake": True}
        )
        self.assertEqual(result["status"], "duplicate_nearby")
        self.assertIs(result["existing"]["wake"], False)
        self.assertIn("音楽を鳴らさない", result["message"])

    async def test_two_wake_alarms_need_no_extra_nudge(self):
        store = _FakeAlarmStore(
            near={
                "id": "old",
                "fire_at_utc": "2026-07-28T22:00:00+00:00",
                "note": "起床",
                "wake": True,
            }
        )
        _, result = await self._agent(store)._run_alarm_tool(
            "set_alarm", {"note": "起きる", "at": "07:00", "wake": True}
        )
        self.assertIs(result["existing"]["wake"], True)
        self.assertNotIn("音楽を鳴らさない", result["message"])


class LegacyAlarmTests(unittest.IsolatedAsyncioTestCase):
    """Alarms created before the wake feature have no ``wake`` key at all.
    They must fire exactly as they always did — silent, spoken only."""

    async def test_record_without_wake_key_does_not_ring(self):
        legacy = {"id": "abc", "note": "薬を飲んだか聞く", "status": "pending"}
        self.assertFalse(any(a.get("wake") for a in [legacy]))

    async def test_mixed_batch_rings_only_for_the_wake_one(self):
        legacy = {"id": "old", "note": "旧アラーム"}
        waking = {"id": "new", "note": "起きる時間", "wake": True}
        self.assertTrue(any(a.get("wake") for a in [legacy, waking]))
        self.assertFalse(any(a.get("wake") for a in [legacy, {"wake": False}]))

    async def test_store_writes_the_key_for_new_alarms(self):
        import datetime
        import tempfile
        from unittest import mock as _mock

        from src.open_llm_vtuber.alarms.store import AlarmStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AlarmStore("test_uid")
            with (
                _mock.patch.object(store, "_dir", tmp),
                _mock.patch.object(store, "_path", f"{tmp}/alarms.json"),
            ):
                when = datetime.datetime.now(datetime.timezone.utc)
                plain = await store.add(fire_at_utc=when, note="n")
                waking = await store.add(fire_at_utc=when, note="n", wake=True)
        self.assertIs(plain["wake"], False)
        self.assertIs(waking["wake"], True)


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

    def test_play_marker_ignores_the_candidates_line(self):
        out = self._marker(
            "music_play",
            "再生開始: Lemon / 米津玄師\n（同名の他候補: Lemon / KBShinya (id=9)…）",
        )
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
