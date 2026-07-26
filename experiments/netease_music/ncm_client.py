"""NetEase Cloud Music API client — search, playlists, and playable audio.

Talks to ``music.163.com/weapi/*`` directly (the same calls the web player
makes, see :mod:`ncm_crypto`). No browser, no desktop client, no UI
automation: a stale cookie or a slow page can't wedge it, and anything that
breaks breaks as a JSON error we can read.

Two settings this client always sends, both required for the API to return
usable results, and both taken from the open-source NetEaseMusicWorldNext
extension:

  * ``X-Real-IP`` — NetEase reads this request header when deciding which
    catalogue the caller gets, including whether a play URL is returned at
    all. ``_REAL_IP`` holds the address we report.
  * a CDN hostname rewrite — play URLs are handed back on
    ``mNNN.music.126.net``; the parallel ``mNNNc.music.126.net`` hosts serve
    the same bytes and are the ones that answer reliably.

Login is by QR code (``login.py``), which yields a ``MUSIC_U`` cookie that
lasts months. Everything the character can reach is read-only: search, list,
and fetch audio. There is no tool that can post, follow, or modify the
account.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ncm_crypto import weapi_encrypt

HERE = pathlib.Path(__file__).resolve().parent
SESSION_FILE = HERE / "ncm_session.json"
CACHE_DIR = HERE / "music_cache"

_BASE = "https://music.163.com/weapi"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# The address reported in X-Real-IP (see the module docstring). Overridable
# via env in case this one ever stops working.
_REAL_IP = os.environ.get("NCM_REAL_IP", "211.161.244.70")
_TIMEOUT = 15.0
_DOWNLOAD_TIMEOUT = 90.0
_CACHE_KEEP = 40  # songs to keep on disk; the newest are also the alarm fallback
# Non-audio files living in the cache next to each song. Anything globbing the
# cache must skip these — a ".json" sidecar handed to the player is not a
# mysterious failure, it just isn't audio.
_NON_AUDIO_SUFFIXES = (".json", ".part")

# Play URLs arrive on mNNN.music.126.net; the mNNNc.* hosts carry the same
# bytes and are the ones that answer reliably.
_CDN_RE = re.compile(r"(m\d+?)(?!c)\.music\.126\.net")


class NeteaseUnavailable(Exception):
    """Any failure to reach or use NetEase. ``str(e)`` is safe to show."""


@dataclass
class Song:
    id: int
    name: str
    artists: str
    album: str
    duration_ms: int

    def label(self) -> str:
        return f"{self.name} / {self.artists}" if self.artists else self.name


@dataclass
class Playlist:
    id: int
    name: str
    count: int


def _preferred_cdn_host(url: str) -> str:
    return _CDN_RE.sub(r"\1c.music.126.net", url)


def cookie_dict(client: httpx.AsyncClient) -> Dict[str, str]:
    """Flatten a client's cookie jar to ``{name: value}``.

    NetEase sets several cookies (MUSIC_A_T, MUSIC_U, __csrf) under more than
    one domain at once, and httpx's mapping interface raises CookieConflict
    the moment a name is ambiguous — which is exactly what happens at the
    instant a QR login succeeds. Reading the jar directly sidesteps that;
    where a name repeats, the music.163.com copy wins.
    """
    out: Dict[str, str] = {}
    preferred: Dict[str, bool] = {}
    for cookie in client.cookies.jar:
        specific = "music.163.com" in (cookie.domain or "")
        if cookie.name in out and preferred.get(cookie.name) and not specific:
            continue
        out[cookie.name] = cookie.value or ""
        preferred[cookie.name] = specific
    return out


def cookie_value(client: httpx.AsyncClient, name: str, default: str = "") -> str:
    """One cookie by name, conflict-safe (see :func:`cookie_dict`)."""
    return cookie_dict(client).get(name, default)


def _song_from(raw: Dict[str, Any]) -> Song:
    artists = raw.get("ar") or raw.get("artists") or []
    album = raw.get("al") or raw.get("album") or {}
    return Song(
        id=int(raw.get("id", 0)),
        name=str(raw.get("name", "")).strip(),
        artists="、".join(a.get("name", "") for a in artists if a.get("name")),
        album=str(album.get("name", "")).strip(),
        duration_ms=int(raw.get("dt") or raw.get("duration") or 0),
    )


class NeteaseClient:
    """Async NetEase client. One instance per process; safe to share."""

    def __init__(self, session_file: pathlib.Path = SESSION_FILE) -> None:
        self._session_file = session_file
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- plumbing
    def _load_cookies(self) -> Dict[str, str]:
        cookies = {"os": "pc", "appver": "8.9.70"}
        try:
            saved = json.loads(self._session_file.read_text("utf-8"))
            cookies.update(saved.get("cookies", {}))
        except FileNotFoundError:
            pass
        except Exception as e:  # corrupt file shouldn't be fatal — just log out
            raise NeteaseUnavailable(f"ログイン情報が壊れています: {e}") from e
        return cookies

    def save_cookies(self, cookies: Dict[str, str]) -> None:
        keep = {k: v for k, v in cookies.items() if k not in ("os", "appver")}
        self._session_file.write_text(
            json.dumps({"cookies": keep, "saved_at": time.time()}, indent=2),
            encoding="utf-8",
        )
        try:
            self._session_file.chmod(0o600)
        except OSError:
            pass  # tightening permissions is a nicety, not worth losing a login

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_TIMEOUT,
                follow_redirects=True,
                cookies=self._load_cookies(),
                headers={
                    "User-Agent": _UA,
                    "Referer": "https://music.163.com/",
                    "Origin": "https://music.163.com",
                    "X-Real-IP": _REAL_IP,
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """One weapi call. Raises :class:`NeteaseUnavailable` on any failure."""
        client = await self.client()
        body = dict(payload)
        body.setdefault("csrf_token", cookie_value(client, "__csrf"))
        try:
            resp = await client.post(f"{_BASE}/{path}", data=weapi_encrypt(body))
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as e:
            raise NeteaseUnavailable("音楽サービスの応答が遅すぎます。") from e
        except httpx.HTTPError as e:
            raise NeteaseUnavailable("音楽サービスに接続できません。") from e
        except ValueError as e:
            raise NeteaseUnavailable("音楽サービスの応答を解釈できません。") from e
        code = data.get("code", 200)
        if code == 301:
            raise NeteaseUnavailable("ログインが切れています。再ログインが必要。")
        if code != 200:
            raise NeteaseUnavailable(
                f"音楽サービスがエラーを返しました（code={code}）。"
            )
        return data

    # ---------------------------------------------------------------- reads
    async def account(self) -> Optional[str]:
        """Logged-in nickname, or ``None`` when anonymous."""
        try:
            data = await self.post("w/nuser/account/get", {})
        except NeteaseUnavailable:
            return None
        profile = data.get("profile") or {}
        return profile.get("nickname") or None

    async def search(self, keyword: str, limit: int = 10) -> List[Song]:
        # search/get, not the newer cloudsearch/get/web: cloudsearch answers
        # code=50000005 unless the session is logged in, and we want search to
        # keep working even when the cookie lapses.
        data = await self.post(
            "search/get",
            {"s": keyword, "type": 1, "offset": 0, "limit": max(1, min(limit, 50))},
        )
        songs = (data.get("result") or {}).get("songs") or []
        return [_song_from(s) for s in songs]

    async def user_playlists(self, limit: int = 50) -> List[Playlist]:
        uid = await self._uid()
        data = await self.post(
            "user/playlist", {"uid": uid, "limit": limit, "offset": 0}
        )
        return [
            Playlist(
                id=int(p["id"]),
                name=str(p.get("name", "")),
                count=int(p.get("trackCount", 0)),
            )
            for p in data.get("playlist") or []
        ]

    async def playlist_tracks(self, playlist_id: int, limit: int = 1000) -> List[Song]:
        data = await self.post(
            "v6/playlist/detail", {"id": playlist_id, "n": limit, "s": 0}
        )
        playlist = data.get("playlist") or {}
        tracks = playlist.get("tracks") or []
        if tracks:
            return [_song_from(t) for t in tracks]
        # Long playlists come back as ids only; hydrate them in one call.
        ids = [t["id"] for t in (playlist.get("trackIds") or [])][:limit]
        return await self.songs_by_id(ids) if ids else []

    async def songs_by_id(self, ids: List[int]) -> List[Song]:
        data = await self.post(
            "v3/song/detail",
            {"c": json.dumps([{"id": int(i)} for i in ids], separators=(",", ":"))},
        )
        return [_song_from(s) for s in data.get("songs") or []]

    async def _uid(self) -> int:
        data = await self.post("w/nuser/account/get", {})
        account = data.get("account") or {}
        uid = account.get("id")
        if not uid:
            raise NeteaseUnavailable("ログインしていません。")
        return int(uid)

    # -------------------------------------------------------------- audio
    async def play_url(self, song_id: int, bitrate: int = 320000) -> str:
        data = await self.post(
            "song/enhance/player/url",
            {"ids": json.dumps([int(song_id)]), "br": bitrate},
        )
        entries = data.get("data") or []
        url = entries[0].get("url") if entries else None
        if not url:
            raise NeteaseUnavailable("この曲は再生できません（版権制限の可能性）。")
        return _preferred_cdn_host(url)

    async def fetch_audio(self, song: Song) -> pathlib.Path:
        """Download a song into the cache and return its path. A cached copy
        is reused, which is also what makes playback survive a dead network."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        existing = cached_audio_path(song.id)
        if existing:
            existing.touch()
            return existing

        url = await self.play_url(song.id)
        suffix = pathlib.Path(url.split("?")[0]).suffix or ".mp3"
        target = CACHE_DIR / f"{song.id}{suffix}"
        partial = target.with_suffix(target.suffix + ".part")
        client = await self.client()
        try:
            async with client.stream("GET", url, timeout=_DOWNLOAD_TIMEOUT) as response:
                response.raise_for_status()
                with partial.open("wb") as fh:
                    async for chunk in response.aiter_bytes(65536):
                        fh.write(chunk)
        except httpx.HTTPError as e:
            partial.unlink(missing_ok=True)
            raise NeteaseUnavailable("曲の取得に失敗しました。") from e
        if partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise NeteaseUnavailable("空のファイルが返されました。")
        partial.replace(target)
        _write_meta(song, target)
        _prune_cache()
        return target

    # -------------------------------------------------------------- login
    async def qr_key(self) -> str:
        data = await self.post("login/qrcode/unikey", {"type": 1})
        key = data.get("unikey")
        if not key:
            raise NeteaseUnavailable("QRコードを取得できません。")
        return str(key)

    @staticmethod
    def qr_login_url(unikey: str) -> str:
        return f"https://music.163.com/login?codekey={unikey}"

    async def qr_check(self, unikey: str) -> Dict[str, Any]:
        """Poll a pending QR login. Codes: 800 expired, 801 waiting,
        802 scanned, 803 confirmed (cookies are saved on 803)."""
        client = await self.client()
        body = {
            "key": unikey,
            "type": 1,
            "csrf_token": cookie_value(client, "__csrf"),
        }
        try:
            resp = await client.post(
                f"{_BASE}/login/qrcode/client/login", data=weapi_encrypt(body)
            )
        except httpx.HTTPError as e:
            # A hiccup while polling shouldn't end the login attempt — report
            # "still waiting" and let the caller poll again.
            return {"code": 801, "message": f"polling failed: {e}"}
        try:
            data = resp.json()
        except ValueError as e:
            return {"code": 801, "message": f"unreadable poll response: {e}"}
        if data.get("code") == 803:
            self.save_cookies(cookie_dict(client))
        return data


def _write_meta(song: Song, path: pathlib.Path) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "id": song.id,
                "name": song.name,
                "artists": song.artists,
                "album": song.album,
                "file": path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def cached_audio_path(song_id: int) -> Optional[pathlib.Path]:
    """The cached audio file for a song, skipping its metadata sidecar."""
    for path in CACHE_DIR.glob(f"{int(song_id)}.*"):
        if path.suffix not in _NON_AUDIO_SUFFIXES and path.stat().st_size > 0:
            return path
    return None


def _prune_cache() -> None:
    audio = sorted(
        (p for p in CACHE_DIR.glob("*") if p.suffix not in _NON_AUDIO_SUFFIXES),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in audio[_CACHE_KEEP:]:
        stale.unlink(missing_ok=True)
        stale.with_suffix(".json").unlink(missing_ok=True)


def cached_songs() -> List[Dict[str, Any]]:
    """Cached songs, newest first — the offline fallback for a wake alarm."""
    if not CACHE_DIR.exists():
        return []
    out = []
    for meta in sorted(
        CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            data = json.loads(meta.read_text("utf-8"))
        except Exception:
            continue
        audio = CACHE_DIR / data.get("file", "")
        if audio.exists():
            data["path"] = str(audio)
            out.append(data)
    return out


def random_cached_song() -> Optional[Dict[str, Any]]:
    songs = cached_songs()
    return random.choice(songs) if songs else None
