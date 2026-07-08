"""Steam library snapshot manager.

Builds and maintains a local JSON snapshot of the user's Steam presence
(owned games, recent playtime, wishlist, followed games, popular-tag
table) plus slower "enrichment" data (per-game achievements and
store-page tags), and renders a compact Japanese digest suitable for
injecting into the character's context at startup.

Refresh is split into two phases so startup stays fast:

- :meth:`SnapshotManager.refresh_core` — one concurrent burst of cheap
  calls. Owned/recent need a Web API key; wishlist/followed/tag table
  are keyless. A missing key or private profile just flips
  ``library_ok`` to ``False`` and the keyless parts still populate.
- :meth:`SnapshotManager.refresh_enrichment` — slow, rate-limited,
  per-appid fetches (achievements + store-page tags) for the top played
  games. Results are cached inside the snapshot so later runs only
  touch appids not seen before. This method never raises.

The snapshot lives at ``<cache_dir>/steam_snapshot.json`` and is saved
atomically (tmp file + ``os.replace``). Because it round-trips through
JSON, the ``game_tags`` / ``achievements`` maps are keyed by *string*
appids; all helpers here look up with ``str(appid)``.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .client import SteamClient, SteamUnavailable

# Delay between per-appid enrichment fetches (appdetails-family endpoints
# are rate limited to roughly 200 requests / 5 min per IP).
_ENRICH_SLEEP_S = 0.5

# steamid64 for individual accounts always starts with 7656 (17 digits).
_VDF_USER_BLOCK = re.compile(r'"(7656\d{13})"\s*\{([^}]*)\}')
_VDF_MOST_RECENT = re.compile(r'"MostRecent"\s+"1"', re.IGNORECASE)


def _now_iso() -> str:
    """Local time as a compact ISO string (snapshot timestamps)."""
    return datetime.now().isoformat(timespec="seconds")


def _norm(text: str) -> str:
    """Normalize for matching: NFKC (full/half width) + casefold."""
    return unicodedata.normalize("NFKC", text or "").casefold()


def _hours(minutes: Any) -> float:
    """Steam playtime minutes -> hours with one decimal."""
    try:
        return round(int(minutes or 0) / 60, 1)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class SnapshotManager:
    """Owns the on-disk Steam snapshot and pure query helpers over it.

    Network work is delegated to a :class:`SteamClient`; everything else
    (persistence, matching, digest rendering) is local and synchronous.
    """

    def __init__(
        self,
        client: SteamClient,
        cache_dir: str | Path = "cache",
        achievements_top_n: int = 40,
        tags_top_n: int = 40,
    ) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)
        self._achievements_top_n = achievements_top_n
        self._tags_top_n = tags_top_n
        self._snapshot_path = self._cache_dir / "steam_snapshot.json"
        self._applist_path = self._cache_dir / "steam_applist.json"

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def load(self) -> dict | None:
        """Load the snapshot from disk. Missing/corrupt file -> None."""
        try:
            with open(self._snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Steam snapshot unreadable ({self._snapshot_path}): {e}")
            return None

    def _save(self, snapshot: dict) -> None:
        """Atomically write the snapshot (tmp file then os.replace)."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._snapshot_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self._snapshot_path)
        except OSError as e:
            logger.warning(f"Failed to save Steam snapshot: {e}")

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------

    async def refresh_core(self) -> dict:
        """Fetch the cheap account data concurrently and save a snapshot.

        Owned/recent (Web API, key required) may fail with
        :class:`SteamUnavailable` — that sets ``library_ok: False`` and
        the keyless parts (wishlist / followed / tag table) still go
        through. Enrichment fields from a previous snapshot
        (``game_tags`` / ``achievements`` / ``enriched_at``) are
        preserved.
        """
        prior = self.load() or {}

        owned_res, recent_res, wish_res, followed_res, tags_res = await asyncio.gather(
            self._client.owned_games(),
            self._client.recently_played(),
            self._client.wishlist(),
            self._client.followed_games(),
            self._client.popular_tags(),
            return_exceptions=True,
        )

        library_ok = not isinstance(owned_res, BaseException)
        if not library_ok:
            if isinstance(owned_res, SteamUnavailable):
                logger.info(f"Steam library not available: {owned_res}")
            else:
                logger.warning(f"Steam owned_games failed: {owned_res!r}")
            owned_res = []
        if isinstance(recent_res, BaseException):
            logger.debug(f"Steam recently_played failed: {recent_res!r}")
            recent_res = []
        if isinstance(wish_res, BaseException):
            logger.warning(f"Steam wishlist failed: {wish_res!r}")
            wish_res = []
        if isinstance(followed_res, BaseException):
            logger.warning(f"Steam followed_games failed: {followed_res!r}")
            followed_res = []
        if isinstance(tags_res, BaseException):
            logger.warning(f"Steam popular_tags failed: {tags_res!r}")
            tags_res = []

        # Wishlist entries arrive as bare appids; enrich names via the
        # (disk-cached) global app list. Unknown appids keep a
        # recognizable placeholder.
        app_names: dict[int, str] = {}
        if wish_res:
            try:
                app_names = await self._client.app_list(self._applist_path)
            except Exception as e:
                logger.warning(f"Steam app_list unavailable for wishlist names: {e}")
        wishlist = [
            {
                "appid": _int(item.get("appid")),
                "name": app_names.get(_int(item.get("appid")))
                or f"appid:{_int(item.get('appid'))}",
                "date_added": item.get("date_added"),
            }
            for item in wish_res
        ]

        owned = [self._project_game(g) for g in owned_res]
        recent = [self._project_game(g) for g in recent_res]
        followed = [
            {"appid": _int(g.get("appid")), "name": g.get("name") or ""}
            for g in followed_res
        ]
        tag_table = [
            {"tagid": _int(t.get("tagid")), "name": t.get("name") or ""}
            for t in tags_res
        ]

        snapshot = {
            "generated_at": _now_iso(),
            "steamid": str(
                getattr(self._client, "steamid", "")
                or getattr(self._client, "_steamid", "")
                or prior.get("steamid", "")
            ),
            "library_ok": library_ok,
            "owned": owned,
            "recent": recent,
            "wishlist": wishlist,
            "followed": followed,
            "tag_table": tag_table,
            # preserved enrichment (string-keyed appid maps)
            "game_tags": prior.get("game_tags") or {},
            "achievements": prior.get("achievements") or {},
            "enriched_at": prior.get("enriched_at"),
        }
        self._save(snapshot)
        logger.info(
            f"Steam snapshot refreshed: owned={len(owned)} recent={len(recent)} "
            f"wishlist={len(wishlist)} followed={len(followed)} "
            f"tags={len(tag_table)} library_ok={library_ok}"
        )
        return snapshot

    async def refresh_enrichment(self) -> None:
        """Fetch achievements + store-page tags for the top played games.

        Only appids not already present in ``achievements`` /
        ``game_tags`` are fetched (sequentially, with a small sleep, to
        respect store rate limits). A ``None`` achievements result (game
        has no achievements, or profile private) is cached as ``null``
        so it is not refetched; an empty tag list is treated as a
        transient failure and NOT cached, so it will be retried on the
        next run. This method never raises.
        """
        try:
            snapshot = self.load()
            if not snapshot:
                logger.debug("No Steam snapshot on disk; skipping enrichment")
                return

            top = self.top_played(
                snapshot, max(self._achievements_top_n, self._tags_top_n)
            )
            achievements: dict[str, Any] = dict(snapshot.get("achievements") or {})
            game_tags: dict[str, list[str]] = dict(snapshot.get("game_tags") or {})

            ach_targets = [
                g
                for g in top[: self._achievements_top_n]
                if str(g["appid"]) not in achievements
            ]
            tag_targets = [
                g for g in top[: self._tags_top_n] if str(g["appid"]) not in game_tags
            ]
            if not ach_targets and not tag_targets:
                logger.debug("Steam enrichment already up to date")
                return

            for game in ach_targets:
                appid = game["appid"]
                try:
                    achievements[str(appid)] = await self._client.player_achievements(
                        appid
                    )
                except Exception as e:
                    logger.debug(f"Achievements fetch failed for {appid}: {e}")
                await asyncio.sleep(_ENRICH_SLEEP_S)

            for game in tag_targets:
                appid = game["appid"]
                try:
                    tags = await self._client.store_page_tags(appid)
                    if tags:
                        game_tags[str(appid)] = tags
                except Exception as e:
                    logger.debug(f"Store tags fetch failed for {appid}: {e}")
                await asyncio.sleep(_ENRICH_SLEEP_S)

            snapshot["achievements"] = achievements
            snapshot["game_tags"] = game_tags
            snapshot["enriched_at"] = _now_iso()
            self._save(snapshot)
            logger.info(
                f"Steam enrichment refreshed: +{len(ach_targets)} achievement / "
                f"+{len(tag_targets)} tag targets"
            )
        except Exception as e:
            logger.warning(f"Steam enrichment failed (skipped): {e}")

    # ------------------------------------------------------------------
    # digest
    # ------------------------------------------------------------------

    def build_digest(self, snapshot: dict) -> str:
        """Render a compact Japanese digest (~12 lines) of the snapshot.

        Framed as data, not character speech, so it can be dropped into
        a prompt verbatim.
        """
        lines = ["【Steamライブラリ概況（起動時スナップショット）】"]
        if not snapshot:
            lines.append("ライブラリ未取得（APIキー未設定/プロフィール非公開）")
            return "\n".join(lines)

        wishlist = snapshot.get("wishlist") or []
        followed = snapshot.get("followed") or []

        if not snapshot.get("library_ok"):
            lines.append("ライブラリ未取得（APIキー未設定/プロフィール非公開）")
            lines.append(f"ウィッシュリスト: {len(wishlist)}件")
            if followed:
                lines.append(f"フォロー中: {len(followed)}件")
            return "\n".join(lines)

        owned = snapshot.get("owned") or []
        total_hours = round(sum(_int(g.get("playtime_forever")) for g in owned) / 60, 1)
        lines.append(
            f"所持: {len(owned)}本 / 総プレイ: {total_hours}時間 / "
            f"ウィッシュリスト: {len(wishlist)}件"
        )

        top5 = self.top_played(snapshot, 5)
        if top5:
            lines.append("プレイ時間トップ5:")
            for g in top5:
                lines.append(f"- {g['name']}: {g['hours']}時間")

        recent = [g for g in self.recent_detail(snapshot) if g["hours_2weeks"] > 0]
        if recent:
            parts = "、".join(f"{g['name']}({g['hours_2weeks']}h)" for g in recent[:5])
            lines.append(f"直近2週間: {parts}")
        else:
            lines.append("直近2週間: プレイなし")

        completed = self._completed_titles(snapshot)
        if completed:
            lines.append(f"全実績コンプ済み: {'、'.join(completed[:6])}")

        top_tags = self._top_tags(snapshot, 8)
        if top_tags:
            lines.append(f"高プレイ時間帯の人気タグ: {'、'.join(top_tags)}")

        return "\n".join(lines)

    def _completed_titles(self, snapshot: dict) -> list[str]:
        """Names of owned games whose achievements are 100% unlocked."""
        names = {
            _int(g.get("appid")): g.get("name") or ""
            for g in snapshot.get("owned") or []
        }
        completed = []
        for appid_str, entry in (snapshot.get("achievements") or {}).items():
            if not isinstance(entry, dict):
                continue  # null marker: no achievements / private
            total = _int(entry.get("total"))
            if total > 0 and _int(entry.get("unlocked")) >= total:
                completed.append(names.get(_int(appid_str)) or f"appid:{appid_str}")
        return completed

    def _top_tags(self, snapshot: dict, n: int) -> list[str]:
        """Most frequent store tags across the top played games."""
        game_tags = snapshot.get("game_tags") or {}
        counter: Counter[str] = Counter()
        for g in self.top_played(snapshot, self._tags_top_n):
            for tag in game_tags.get(str(g["appid"]), []):
                counter[tag] += 1
        return [tag for tag, _ in counter.most_common(n)]

    # ------------------------------------------------------------------
    # steamid autodetection (Windows)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_steamid64() -> str | None:
        """Detect the local Steam account's steamid64 (Windows only).

        Reads HKCU\\Software\\Valve\\Steam -> SteamPath, then parses
        ``<SteamPath>/config/loginusers.vdf`` with a regex (no vdf
        dependency): picks the user block with ``"MostRecent" "1"``, or
        the only user if there is exactly one. Returns None on any
        failure (non-Windows, no Steam, unreadable file, ...).
        """
        try:
            import winreg
        except ImportError:
            return None
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"
            ) as key:
                steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
        except OSError:
            return None

        # SteamPath uses forward slashes / lowercase drive; Path copes.
        vdf_path = Path(str(steam_path)) / "config" / "loginusers.vdf"
        try:
            text = vdf_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        blocks = _VDF_USER_BLOCK.findall(text)
        if not blocks:
            return None
        for steamid, body in blocks:
            if _VDF_MOST_RECENT.search(body):
                return steamid
        if len(blocks) == 1:
            return blocks[0][0]
        return None

    # ------------------------------------------------------------------
    # local query helpers (pure, no network)
    # ------------------------------------------------------------------

    def owned_set(self, snapshot: dict) -> set[int]:
        """Appids of all owned games."""
        return {_int(g.get("appid")) for g in (snapshot or {}).get("owned") or []}

    def find_game(self, snapshot: dict, query: str) -> list[dict]:
        """Fuzzy-match a game name over owned + wishlist + followed.

        Matching is NFKC width-normalized and casefolded: exact and
        substring hits first (exact, then by playtime), falling back to
        difflib close matches when nothing contains the query. Each hit
        carries where it was found and the owned playtime.
        """
        q = _norm(query).strip()
        if not q or not snapshot:
            return []

        candidates: dict[int, dict] = {}
        for where in ("owned", "wishlist", "followed"):
            for g in snapshot.get(where) or []:
                appid = _int(g.get("appid"))
                if appid in candidates:
                    continue  # owned wins over wishlist wins over followed
                candidates[appid] = {
                    "appid": appid,
                    "name": g.get("name") or "",
                    "where": where,
                    "playtime_forever": _int(g.get("playtime_forever")),
                }

        subs = [c for c in candidates.values() if q in _norm(c["name"])]
        if subs:
            subs.sort(
                key=lambda c: (
                    0 if _norm(c["name"]) == q else 1,
                    -c["playtime_forever"],
                )
            )
            return subs[:10]

        by_norm: dict[str, list[dict]] = {}
        for c in candidates.values():
            by_norm.setdefault(_norm(c["name"]), []).append(c)
        matches: list[dict] = []
        for norm_name in difflib.get_close_matches(q, list(by_norm), n=5, cutoff=0.5):
            matches.extend(by_norm[norm_name])
        return matches[:10]

    def top_played(self, snapshot: dict, n: int) -> list[dict]:
        """Top-n owned games by total playtime, with an ``hours`` field."""
        owned = (snapshot or {}).get("owned") or []
        top = sorted(
            owned, key=lambda g: _int(g.get("playtime_forever")), reverse=True
        )[:n]
        return [{**g, "hours": _hours(g.get("playtime_forever"))} for g in top]

    def recent_detail(self, snapshot: dict) -> list[dict]:
        """Recently played games with 2-week and lifetime hours added."""
        return [
            {
                **g,
                "hours_2weeks": _hours(g.get("playtime_2weeks")),
                "hours_forever": _hours(g.get("playtime_forever")),
            }
            for g in (snapshot or {}).get("recent") or []
        ]

    def resolve_tag(self, snapshot: dict, text: str) -> list[dict]:
        """Resolve free text to store tag candidates ({tagid, name}).

        Tries exact (normalized) match, then substring, then difflib
        close matches over the snapshot's tag table. Returns up to 5
        candidates, best first.
        """
        t = _norm(text).strip()
        table = (snapshot or {}).get("tag_table") or []
        if not t or not table:
            return []

        def _entry(row: dict) -> dict:
            return {"tagid": _int(row.get("tagid")), "name": row.get("name") or ""}

        exact = [row for row in table if _norm(row.get("name")) == t]
        if exact:
            return [_entry(row) for row in exact[:5]]

        subs = [row for row in table if t in _norm(row.get("name"))]
        if subs:
            subs.sort(key=lambda row: len(row.get("name") or ""))
            return [_entry(row) for row in subs[:5]]

        by_norm: dict[str, dict] = {}
        for row in table:
            by_norm.setdefault(_norm(row.get("name")), row)
        close = difflib.get_close_matches(t, list(by_norm), n=5, cutoff=0.5)
        return [_entry(by_norm[name]) for name in close]

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _project_game(g: dict) -> dict:
        """Reduce a Web-API game dict to the snapshot's stable shape."""
        appid = _int(g.get("appid"))
        return {
            "appid": appid,
            "name": g.get("name") or f"appid:{appid}",
            "playtime_forever": _int(g.get("playtime_forever")),
            "playtime_2weeks": _int(g.get("playtime_2weeks")),
            "rtime_last_played": _int(g.get("rtime_last_played")),
        }
