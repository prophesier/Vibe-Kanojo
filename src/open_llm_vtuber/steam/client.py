"""Steam storefront / Web API / community client (keyless where possible).

Endpoint shapes verified live 2026-07-08 (cc=jp, l=japanese unless noted):

- ``/api/storesearch/`` → ``items[]`` with ``price.final`` in MINOR units
  (902000 == ¥9,020 — always ÷100). Free games omit ``price``.
- ``/api/appdetails?appids=ID`` → ``{"ID": {success, data}}``. ONE appid per
  call (multi-appid unreliable); rate limit ~200 req / 5 min per IP.
- ``/appreviews/ID?json=1&num_per_page=0`` → cheap ``query_summary`` only.
- ``/search/suggest`` → HTML fragment; ``/search/results/?...&json=1`` →
  ``{success, results_html, total_count}`` (``sort_by=Reviews_DESC`` variant
  re-verified live 2026-07-08); ``/recommended/morelike/app/ID/`` → HTML page
  where titles exist ONLY as URL slugs (lossy for non-ASCII titles).
- Store pages need age-gate cookies (``birthtime`` etc.) for mature content;
  community tags live in the ``InitAppTagModal( appid, [...]`` JSON blob.
- Web API (api.steampowered.com) needs ``key`` except IWishlistService.
  A PRIVATE profile yields an empty ``response`` — indistinguishable from an
  empty wishlist / empty library; we return ``[]`` and document it.
- Anonymous ``ISteamApps/GetAppList/v2`` was REMOVED by Valve (404 "Method
  not found", verified 2026-07-08). ``app_list()`` therefore falls back to the
  key-gated, paginated ``IStoreService/GetAppList/v1`` when a key is set, and
  serves a stale disk cache as a last resort.

Error policy (consistent across methods):

- Network / HTTP / JSON-decode failure → :class:`SteamUnavailable` with a
  short human-readable message. The Web API key is NEVER included in messages,
  logs, or exception chains (keyed failures are raised ``from None`` and any
  accidental key occurrence is scrubbed).
- "Soft" emptiness (unknown appid, private profile, no reviews, privacy wall)
  → ``None`` / ``[]`` per the method signature, never an exception.
- :meth:`store_page_tags` returns ``[]`` on ANY failure (enrichment is
  best-effort by contract).
"""

from __future__ import annotations

import json
import logging
import re
import time
from html import unescape
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

import httpx
from loguru import logger

# httpx logs 'HTTP Request: GET <full URL>' at INFO via stdlib logging; for
# keyed Web-API calls that URL contains the key. Clamp so that no logging
# bridge / verbose mode can ever print it.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_STORE = "https://store.steampowered.com"
_WEB_API = "https://api.steampowered.com"
_COMMUNITY = "https://steamcommunity.com"

# Browser-ish UA: Steam serves privacy walls / degraded HTML to bare clients.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Bypass the mature-content age gate on store pages (scoped to the store host).
_AGE_GATE_COOKIES = {
    "birthtime": "568022401",
    "wants_mature_content": "1",
    "lastagecheckage": "1-January-1988",
}

_APP_LIST_TTL_SECONDS = 7 * 24 * 3600  # 7-day disk cache for the huge app list

# --- HTML row patterns (written against live HTML captured 2026-07-08) -----

# suggest row: <a class="match match_app ..." data-ds-appid="367520" ...>
#   <div class="match_name ">Hollow Knight</div>...
#   <div class="match_subtitle">¥ 850</div></a>
_SUGGEST_ROW_RE = re.compile(
    r'<a class="match[^>]*?data-ds-appid="(\d+)".*?'
    r'<div class="match_name[^"]*">(.*?)</div>.*?'
    r'<div class="match_subtitle">(.*?)</div>',
    re.S,
)

# search result row: <a href="..." data-ds-appid="4704690" ...
#   class="search_result_row ...> ... </a>
_SEARCH_ROW_RE = re.compile(
    r'<a\s[^>]*?data-ds-appid="(\d+)"[^>]*?class="search_result_row.*?</a>',
    re.S,
)
# inside a row: <span class="title">めっちゃカメレオン</span>
_SEARCH_TITLE_RE = re.compile(r'<span class="title">(.*?)</span>', re.S)
# <div class="search_released responsive_secondrow">\n  2026年6月9日  </div>
_SEARCH_RELEASED_RE = re.compile(
    r'<div class="search_released[^"]*">\s*(.*?)\s*</div>', re.S
)
# <span class="search_review_summary positive" data-tooltip-html="非常に好評
#   &lt;br&gt;...490件中88%が好評です...">
_SEARCH_REVIEW_RE = re.compile(
    r'class="search_review_summary[^"]*"\s+data-tooltip-html="(.*?)"', re.S
)
_PRICE_FINAL_RE = re.compile(r'data-price-final="(\d+)"')
_DISCOUNT_PCT_RE = re.compile(r'data-discount="(\d+)"')

# morelike capsule: <a ... class="similar_grid_capsule"  data-ds-appid="374320"
#   ... href="https://store.steampowered.com/app/374320/DARK_SOULS_III/?snr=...">
_MORELIKE_CAPSULE_RE = re.compile(
    r'class="similar_grid_capsule"[^>]*?data-ds-appid="(\d+)"'
)
_MORELIKE_HREF_RE = re.compile(
    r'href="https?://store\.steampowered\.com/app/\d+/([^/?"]*)'
)
# <div class="regular_price price">\n ¥ 5,940&nbsp; </div>
_MORELIKE_REGULAR_PRICE_RE = re.compile(r'regular_price price">\s*¥\s*([\d,]+)')

# store page tag blob: InitAppTagModal( 1245620,
#   [{"tagid":29482,"name":"ソウル...","count":7522,...}, ...],
_APP_TAG_MODAL_MARKER = "InitAppTagModal("

# followed games row: <div ... class="gameListRow notfollowed" data-appid="1675200">
#   ... <div class="gameListRowItemName"><a href="...">Steam Deck</a></div>
_FOLLOWED_ROW_RE = re.compile(
    r'class="gameListRow[^"]*"\s+data-appid="(\d+)".*?'
    r'gameListRowItemName"><a[^>]*>(.*?)</a>',
    re.S,
)


class SteamUnavailable(Exception):
    """Steam endpoint unreachable / auth failed / unparseable.

    ``str(e)`` is short, human-readable, and never contains the Web API key.
    """


def _yen_from_minor(value: Any) -> Optional[int]:
    """Minor currency units (JPY*100 as served by Steam) → integer yen."""
    try:
        return int(value) // 100
    except (TypeError, ValueError):
        return None


def _yen_from_text(text: str) -> Optional[int]:
    """Formatted price text like ``¥ 5,940`` → 5940. No digits → None."""
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


class SteamClient:
    """Async client for Steam storefront, Web API, and community pages.

    Storefront methods are keyless; Web-API methods need ``web_api_key`` and
    ``steamid`` (steamid64). All prices are normalized to integer yen at this
    boundary. See the module docstring for the error policy.
    """

    def __init__(
        self,
        web_api_key: str = "",
        steamid: str = "",
        cc: str = "jp",
        lang: str = "japanese",
        timeout: float = 15.0,
    ) -> None:
        self._web_api_key = web_api_key
        self._steamid = steamid
        self._cc = cc
        self._lang = lang
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------ http

    def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the shared AsyncClient (age-gate cookies scoped to
        the store host so they never leak to api.steampowered.com)."""
        if self._client is None or self._client.is_closed:
            cookies = httpx.Cookies()
            for name, value in _AGE_GATE_COOKIES.items():
                cookies.set(name, value, domain="store.steampowered.com")
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "User-Agent": _UA,
                    "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
                },
                cookies=cookies,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client (safe to call multiple times)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _scrub(self, text: str) -> str:
        """Remove the API key from any string destined for logs/messages."""
        if self._web_api_key and self._web_api_key in text:
            text = text.replace(self._web_api_key, "***")
        return text

    async def _get(
        self,
        url: str,
        params: Optional[dict] = None,
        *,
        keyed: bool = False,
        what: str = "Steam",
    ) -> httpx.Response:
        """One GET. Raises SteamUnavailable on transport errors. For keyed
        requests the original exception is dropped (``from None``) because
        httpx exception texts embed the full URL including the key."""
        client = self._ensure_client()
        try:
            return await client.get(url, params=params)
        except Exception as e:
            msg = self._scrub(f"{what}に接続できません ({type(e).__name__})")
            if keyed:
                raise SteamUnavailable(msg) from None
            raise SteamUnavailable(msg) from e

    async def _get_store_json(self, url: str, params: Optional[dict] = None) -> Any:
        resp = await self._get(url, params, what="Steamストア")
        if resp.status_code != 200:
            raise SteamUnavailable(f"SteamストアHTTP {resp.status_code}: {url}")
        try:
            return resp.json()
        except Exception as e:
            raise SteamUnavailable(f"Steamストア応答が不正です: {url}") from e

    async def _get_store_text(self, url: str, params: Optional[dict] = None) -> str:
        resp = await self._get(url, params, what="Steamストア")
        if resp.status_code != 200:
            raise SteamUnavailable(f"SteamストアHTTP {resp.status_code}: {url}")
        return resp.text

    async def _get_webapi_json(
        self, path: str, params: dict, *, keyed: bool = True
    ) -> Any:
        """GET api.steampowered.com/<path>. Error messages reference only the
        path (never the query string, which may carry the key)."""
        url = f"{_WEB_API}/{path.lstrip('/')}"
        resp = await self._get(url, params, keyed=keyed, what="Steam Web API")
        if resp.status_code in (401, 403):
            raise SteamUnavailable(
                f"Steam Web API: APIキー未設定/無効 ({resp.status_code})"
            )
        if resp.status_code != 200:
            raise SteamUnavailable(f"Steam Web API HTTP {resp.status_code}: {path}")
        try:
            return resp.json()
        except Exception:
            raise SteamUnavailable(f"Steam Web API応答が不正です: {path}") from None

    def _require_key(self) -> None:
        if not self._web_api_key:
            raise SteamUnavailable("Steam Web APIキー未設定")

    def _require_steamid(self) -> None:
        if not self._steamid:
            raise SteamUnavailable("SteamID64未設定")

    # ----------------------------------------------------- storefront (keyless)

    async def store_search(self, term: str, limit: int = 10) -> list[dict]:
        """Search the store. Returns up to ``limit`` items:
        ``{appid, name, type, price_yen (None if free/unpriced),
        discount_percent, metascore}``."""
        data = await self._get_store_json(
            f"{_STORE}/api/storesearch/",
            {"term": term, "cc": self._cc, "l": self._lang},
        )
        out: list[dict] = []
        for item in (data or {}).get("items") or []:
            if not isinstance(item, dict):
                continue
            price = item.get("price") or {}
            final = price.get("final")
            initial = price.get("initial")
            discount = 0
            if isinstance(final, int) and isinstance(initial, int) and initial > final:
                discount = round((1 - final / initial) * 100)
            out.append(
                {
                    "appid": int(item.get("id", 0)),
                    "name": str(item.get("name", "")),
                    "type": str(item.get("type", "")),
                    "price_yen": _yen_from_minor(final),
                    "discount_percent": discount,
                    "metascore": item.get("metascore"),
                }
            )
            if len(out) >= limit:
                break
        return out

    async def app_details(self, appid: int) -> Optional[dict]:
        """Normalized appdetails for ONE appid (multi-appid is unreliable).

        When the home store (``cc``) doesn't carry the title (region-locked /
        not sold in JP — e.g. CERO-refused games), falls back to the US store:
        the result then carries ``region``/``region_note``, keeps the JP-
        localized text (``l`` is independent of ``cc``), and ``price_yen`` is
        None because the price isn't JPY (``final_formatted`` still shows it,
        e.g. "$59.99"). Returns None when no region works (unknown appid,
        fully delisted). Rate limit ~200 req / 5 min per IP."""
        for cc in dict.fromkeys((self._cc, "us")):
            data = await self._get_store_json(
                f"{_STORE}/api/appdetails",
                {"appids": appid, "cc": cc, "l": self._lang},
            )
            entry = (data or {}).get(str(appid)) or {}
            if not entry.get("success"):
                continue
            d = entry.get("data") or {}
            po = d.get("price_overview") or {}
            rd = d.get("release_date") or {}
            is_free = bool(d.get("is_free"))
            currency = str(po.get("currency", ""))
            price_yen = (
                _yen_from_minor(po.get("final")) if currency in ("", "JPY") else None
            )
            if price_yen is None and is_free:
                price_yen = 0
            out = {
                "appid": int(d.get("steam_appid", appid)),
                "name": str(d.get("name", "")),
                "short_description": str(d.get("short_description", "")),
                "genres": [
                    str(g.get("description", ""))
                    for g in d.get("genres") or []
                    if isinstance(g, dict)
                ],
                "price_yen": price_yen,
                "currency": currency,
                "final_formatted": str(po.get("final_formatted", "")),
                "discount_percent": int(po.get("discount_percent") or 0),
                "is_free": is_free,
                "release_date": str(rd.get("date", "")),
                "coming_soon": bool(rd.get("coming_soon")),
                "metacritic": (d.get("metacritic") or {}).get("score"),
                "developers": d.get("developers") or [],
                "publishers": d.get("publishers") or [],
                "header_image": str(d.get("header_image", "")),
            }
            if cc != self._cc:
                out["region"] = cc
                out["region_note"] = (
                    "日本ストア未掲載のため他リージョンの情報"
                    "（価格は円ではない・購入不可の可能性）。"
                )
            return out
        return None

    async def app_reviews_summary(
        self, appid: int, language: str = "japanese"
    ) -> Optional[dict]:
        """Cheap review summary (num_per_page=0 → query_summary only).
        Returns ``{review_score, review_score_desc, total_positive,
        total_negative, total_reviews}`` or None if Steam says no."""
        data = await self._get_store_json(
            f"{_STORE}/appreviews/{appid}",
            {
                "json": 1,
                "language": language,
                "purchase_type": "all",
                "num_per_page": 0,
            },
        )
        if not data or data.get("success") != 1:
            return None
        qs = data.get("query_summary") or {}
        return {
            "review_score": int(qs.get("review_score") or 0),
            "review_score_desc": str(qs.get("review_score_desc", "")),
            "total_positive": int(qs.get("total_positive") or 0),
            "total_negative": int(qs.get("total_negative") or 0),
            "total_reviews": int(qs.get("total_reviews") or 0),
        }

    async def search_suggest(self, term: str) -> list[dict]:
        """Prefix/substring title matcher (NO typo correction). Returns
        ``{appid, name, price_yen}``; price_yen is None for free/unpriced.

        Parses rows like: ``<a class="match match_app ..."
        data-ds-appid="367520" ...><div class="match_name ">Hollow Knight
        </div>...<div class="match_subtitle">¥ 850</div></a>``."""
        html = await self._get_store_text(
            f"{_STORE}/search/suggest",
            {
                "term": term,
                "f": "games",
                "cc": self._cc,
                "l": self._lang,
                "use_store_query": 1,
            },
        )
        out: list[dict] = []
        for m in _SUGGEST_ROW_RE.finditer(html):
            appid, name_html, subtitle_html = m.groups()
            out.append(
                {
                    "appid": int(appid),
                    "name": unescape(re.sub(r"<[^>]+>", "", name_html)).strip(),
                    # subtitle is ALREADY formatted yen ("¥ 850"), not minor units
                    "price_yen": _yen_from_text(unescape(subtitle_html)),
                }
            )
        return out

    async def search_results(
        self,
        tags: Optional[list[int]] = None,
        filter: Optional[str] = None,
        sort_by: Optional[str] = None,
        count: int = 10,
        start: int = 0,
    ) -> tuple[int, list[dict]]:
        """Store search-results browse (tag browse / top lists). ``filter`` is
        one of popularnew|topsellers|comingsoon|popularwishlist; ``sort_by``
        e.g. Reviews_DESC|Released_DESC (both families verified live).
        Returns ``(total_count, rows)`` where rows are
        ``{appid, name, price_yen, discount_percent, release_date,
        review_desc, review_percent}``. Non-app rows (pure bundles) and
        unparseable rows are skipped. Steam ignores small ``count`` values
        (count=5 still returned 25 rows, verified live), so rows are trimmed
        to ``count`` here.

        Row anchor: ``<a href="..." data-ds-appid="4704690" ...
        class="search_result_row ...>``; price sits in
        ``data-price-final="79000"`` (minor units); review tooltip is
        ``data-tooltip-html="非常に好評&lt;br&gt;...88%が好評です..."``."""
        params: dict[str, Any] = {
            "cc": self._cc,
            "l": self._lang,
            "infinite": 1,
            "count": count,
            "start": start,
            "json": 1,
        }
        if tags:
            params["tags"] = ",".join(str(t) for t in tags)
        if filter:
            params["filter"] = filter
        if sort_by:
            params["sort_by"] = sort_by
        data = await self._get_store_json(f"{_STORE}/search/results/", params)
        if not data or data.get("success") != 1:
            raise SteamUnavailable("Steamストア検索が失敗しました (success!=1)")
        total_count = int(data.get("total_count") or 0)
        rows: list[dict] = []
        for m in _SEARCH_ROW_RE.finditer(data.get("results_html") or ""):
            block = m.group(0)
            title_m = _SEARCH_TITLE_RE.search(block)
            released_m = _SEARCH_RELEASED_RE.search(block)
            review_m = _SEARCH_REVIEW_RE.search(block)
            price_m = _PRICE_FINAL_RE.search(block)
            discount_m = _DISCOUNT_PCT_RE.search(block)
            review_desc = ""
            review_percent: Optional[int] = None
            if review_m:
                tooltip = unescape(review_m.group(1))
                review_desc = tooltip.split("<br>", 1)[0].strip()
                pct_m = re.search(r"(\d+)%", tooltip)
                if pct_m:
                    review_percent = int(pct_m.group(1))
            rows.append(
                {
                    "appid": int(m.group(1)),
                    "name": unescape(title_m.group(1)).strip() if title_m else "",
                    "price_yen": _yen_from_minor(price_m.group(1)) if price_m else None,
                    "discount_percent": int(discount_m.group(1)) if discount_m else 0,
                    "release_date": unescape(released_m.group(1)).strip()
                    if released_m
                    else "",
                    "review_desc": review_desc,
                    "review_percent": review_percent,
                }
            )
            if count > 0 and len(rows) >= count:
                break
        return total_count, rows

    async def morelike(self, appid: int) -> list[dict]:
        """Games similar to ``appid`` (anonymous recommendation page).
        Returns ``{appid, name, price_yen}``. NOTE: this page carries titles
        only inside URL slugs, so ``name`` is slug-derived — non-ASCII titles
        come back lossy or empty (verified live: a JP title's slug was "3").

        Capsule anchor: ``class="similar_grid_capsule"  data-ds-appid="374320"
        ... href="https://store.steampowered.com/app/374320/DARK_SOULS_III/?..."``;
        price is either ``<div class="regular_price price"> ¥ 5,940&nbsp;</div>``
        (formatted yen) or a discount block with ``data-price-final="677600"``
        (minor units)."""
        html = await self._get_store_text(
            f"{_STORE}/recommended/morelike/app/{appid}/",
            {"cc": self._cc, "l": self._lang},
        )
        matches = list(_MORELIKE_CAPSULE_RE.finditer(html))
        out: list[dict] = []
        seen: set[int] = set()
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else m.start() + 4000
            block = html[m.start() : end]
            item_appid = int(m.group(1))
            if item_appid in seen:
                continue
            seen.add(item_appid)
            href_m = _MORELIKE_HREF_RE.search(block)
            name = ""
            if href_m:
                name = unquote(href_m.group(1)).replace("_", " ").strip()
            price_yen: Optional[int] = None
            price_m = _PRICE_FINAL_RE.search(block)
            if price_m:
                price_yen = _yen_from_minor(price_m.group(1))
            else:
                regular_m = _MORELIKE_REGULAR_PRICE_RE.search(block)
                if regular_m:
                    price_yen = _yen_from_text(regular_m.group(1))
            out.append({"appid": item_appid, "name": name, "price_yen": price_yen})
        return out

    async def featured_categories(self) -> dict:
        """Generic (non-personalized) featured sections. Returns
        ``{section_key: [{appid, name, price_yen, discount_percent}]}`` for
        every section that carries an ``items`` list (specials, top_sellers,
        new_releases, coming_soon — 'genres'/'trailerslideshow' have no items
        and are skipped)."""
        data = await self._get_store_json(
            f"{_STORE}/api/featuredcategories",
            {"cc": self._cc, "l": self._lang},
        )
        out: dict[str, list[dict]] = {}
        for key, section in (data or {}).items():
            if not isinstance(section, dict):
                continue
            items = section.get("items")
            if not isinstance(items, list):
                continue
            rows: list[dict] = []
            for item in items:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                rows.append(
                    {
                        "appid": int(item.get("id", 0)),
                        "name": str(item.get("name", "")),
                        "price_yen": _yen_from_minor(item.get("final_price")),
                        "discount_percent": int(item.get("discount_percent") or 0),
                    }
                )
            out[str(key)] = rows
        return out

    async def popular_tags(self) -> list[dict]:
        """The ~430-entry popular tag table with localized names.
        Returns ``[{tagid: int, name: str}]``."""
        data = await self._get_store_json(f"{_STORE}/tagdata/populartags/{self._lang}")
        out: list[dict] = []
        for row in data or []:
            if not isinstance(row, dict):
                continue
            try:
                tagid = int(row.get("tagid", 0))
            except (TypeError, ValueError):
                continue
            out.append({"tagid": tagid, "name": str(row.get("name", ""))})
        return out

    async def store_page_tags(self, appid: int) -> list[str]:
        """Community 人気タグ from the store page, localized. Returns ``[]``
        on ANY failure (never raises) — enrichment is best-effort.

        Parses the JS blob ``InitAppTagModal( 1245620, [{"tagid":29482,
        "name":"\\u30bd\\u30a6\\u30eb\\u30e9\\u30a4\\u30af",...},...],`` via a
        raw JSON decode starting at the array's ``[``. Age-gate cookies are
        pre-set on the shared client for mature-gated pages."""
        for cc in dict.fromkeys((self._cc, "us")):
            try:
                html = await self._get_store_text(
                    f"{_STORE}/app/{appid}/", {"cc": cc, "l": self._lang}
                )
                marker = html.find(_APP_TAG_MODAL_MARKER)
                if marker < 0:
                    # Region-locked pages redirect to the store front — retry
                    # once via the US store (tags are community-global; names
                    # still follow ``l``).
                    logger.debug(
                        f"store_page_tags({appid}, cc={cc}): InitAppTagModal not found"
                    )
                    continue
                bracket = html.find("[", marker)
                if bracket < 0:
                    continue
                tags, _ = json.JSONDecoder().raw_decode(html, bracket)
                return [
                    str(t["name"]) for t in tags if isinstance(t, dict) and "name" in t
                ]
            except Exception as e:
                logger.debug(
                    f"store_page_tags({appid}, cc={cc}) failed: {self._scrub(str(e))}"
                )
        return []

    # -------------------------------------------------------- web api (keyed)

    async def owned_games(self) -> list[dict]:
        """Full library via IPlayerService/GetOwnedGames. Playtime is in
        MINUTES; ``rtime_last_played`` is a unix timestamp. A PRIVATE profile
        returns an empty response → ``[]`` (indistinguishable from an empty
        library). Raises SteamUnavailable on auth/network failure."""
        self._require_key()
        self._require_steamid()
        data = await self._get_webapi_json(
            "IPlayerService/GetOwnedGames/v1/",
            {
                "key": self._web_api_key,
                "steamid": self._steamid,
                "include_appinfo": 1,
                "include_played_free_games": 1,
                "format": "json",
            },
        )
        games = ((data or {}).get("response") or {}).get("games") or []
        return [self._normalize_game(g) for g in games if isinstance(g, dict)]

    async def recently_played(self, count: int = 20) -> list[dict]:
        """Games played in the last 2 weeks (same shape as owned_games rows,
        ``playtime_2weeks`` populated). Private profile → ``[]``."""
        self._require_key()
        self._require_steamid()
        data = await self._get_webapi_json(
            "IPlayerService/GetRecentlyPlayedGames/v1/",
            {"key": self._web_api_key, "steamid": self._steamid, "count": count},
        )
        games = ((data or {}).get("response") or {}).get("games") or []
        return [self._normalize_game(g) for g in games if isinstance(g, dict)]

    @staticmethod
    def _normalize_game(g: dict) -> dict:
        return {
            "appid": int(g.get("appid", 0)),
            "name": str(g.get("name", "")),
            "playtime_forever": int(g.get("playtime_forever") or 0),
            "playtime_2weeks": int(g.get("playtime_2weeks") or 0),
            "rtime_last_played": int(g.get("rtime_last_played") or 0),
        }

    async def player_summary(self) -> Optional[dict]:
        """First player from GetPlayerSummaries or None. ``gameid`` /
        ``gameextrainfo`` keys are present ONLY while the user is in-game."""
        self._require_key()
        self._require_steamid()
        data = await self._get_webapi_json(
            "ISteamUser/GetPlayerSummaries/v2/",
            {"key": self._web_api_key, "steamids": self._steamid},
        )
        players = ((data or {}).get("response") or {}).get("players") or []
        return players[0] if players else None

    async def player_achievements(self, appid: int) -> Optional[dict]:
        """Achievement progress ``{total, unlocked}`` for one game, or None
        when the game has no achievements / stats are private (Steam replies
        HTTP 400 or ``success: false`` — handled, never raised). Auth and
        network failures still raise SteamUnavailable."""
        self._require_key()
        self._require_steamid()
        url = f"{_WEB_API}/ISteamUserStats/GetPlayerAchievements/v1/"
        resp = await self._get(
            url,
            {
                "key": self._web_api_key,
                "steamid": self._steamid,
                "appid": appid,
                "l": self._lang,
            },
            keyed=True,
            what="Steam Web API",
        )
        if resp.status_code in (401, 403):
            raise SteamUnavailable(
                f"Steam Web API: APIキー未設定/無効 ({resp.status_code})"
            )
        if resp.status_code == 400:
            # "Requested app has no stats" — normal for achievement-less games
            return None
        if resp.status_code != 200:
            raise SteamUnavailable(
                f"Steam Web API HTTP {resp.status_code}: GetPlayerAchievements"
            )
        try:
            stats = resp.json().get("playerstats") or {}
        except Exception:
            return None
        if not stats.get("success"):
            return None
        achievements = stats.get("achievements")
        if not isinstance(achievements, list) or not achievements:
            return None
        unlocked = sum(
            1 for a in achievements if isinstance(a, dict) and a.get("achieved")
        )
        return {"total": len(achievements), "unlocked": unlocked}

    # ---------------------------------------------------- keyless account data

    async def wishlist(self) -> list[dict]:
        """Wishlist via keyless IWishlistService. Returns
        ``[{appid, date_added}]``. A PRIVATE profile yields
        ``{"response": {}}`` — indistinguishable from an empty wishlist, so
        both come back as ``[]``."""
        self._require_steamid()
        data = await self._get_webapi_json(
            "IWishlistService/GetWishlist/v1/",
            {"steamid": self._steamid},
            keyed=False,
        )
        items = ((data or {}).get("response") or {}).get("items") or []
        return [
            {
                "appid": int(it.get("appid", 0)),
                "date_added": int(it.get("date_added") or 0),
            }
            for it in items
            if isinstance(it, dict)
        ]

    async def app_list(self, cache_path) -> dict[int, str]:
        """appid → English name for the whole store, disk-cached 7 days
        (the payload is 10 MB+). Source order:

        1. fresh disk cache;
        2. anonymous ``ISteamApps/GetAppList/v2`` — Valve REMOVED this
           (404 "Method not found", verified 2026-07-08) but it is tried
           first in case it returns;
        3. key-gated paginated ``IStoreService/GetAppList/v1`` (needs
           ``web_api_key``);
        4. stale disk cache (with a warning);
        5. otherwise raises SteamUnavailable.
        """
        path = Path(cache_path)
        cached = self._load_app_list_cache(path)
        if cached is not None:
            fetched_at, apps = cached
            if time.time() - fetched_at < _APP_LIST_TTL_SECONDS:
                return apps

        apps_fresh: Optional[dict[int, str]] = None
        try:
            apps_fresh = await self._fetch_app_list_anonymous()
        except SteamUnavailable as e:
            logger.debug(f"anonymous GetAppList failed (expected since removal): {e}")
        if apps_fresh is None and self._web_api_key:
            try:
                apps_fresh = await self._fetch_app_list_storeservice()
            except SteamUnavailable as e:
                logger.warning(f"IStoreService/GetAppList failed: {e}")

        if apps_fresh:
            self._save_app_list_cache(path, apps_fresh)
            return apps_fresh
        if cached is not None:
            logger.warning("app list refresh failed; serving stale disk cache")
            return cached[1]
        raise SteamUnavailable(
            "Steamアプリ一覧を取得できません（匿名GetAppList廃止済み・APIキー必要）"
        )

    async def _fetch_app_list_anonymous(self) -> dict[int, str]:
        data = await self._get_webapi_json(
            "ISteamApps/GetAppList/v2/", {"format": "json"}, keyed=False
        )
        apps = ((data or {}).get("applist") or {}).get("apps")
        if not isinstance(apps, list) or not apps:
            raise SteamUnavailable("GetAppList応答が空です")
        return {
            int(a["appid"]): str(a.get("name", ""))
            for a in apps
            if isinstance(a, dict) and "appid" in a
        }

    async def _fetch_app_list_storeservice(self) -> dict[int, str]:
        """Paginated IStoreService/GetAppList/v1 (documented shape:
        response.{apps[], have_more_results, last_appid}). Not live-testable
        without a key — coded to the documented shape, defensively."""
        out: dict[int, str] = {}
        last_appid = 0
        for _ in range(40):  # safety cap: 40 pages * 50k = 2M appids
            params: dict[str, Any] = {
                "key": self._web_api_key,
                "include_games": "true",
                "max_results": 50000,
            }
            if last_appid:
                params["last_appid"] = last_appid
            data = await self._get_webapi_json("IStoreService/GetAppList/v1/", params)
            resp = (data or {}).get("response") or {}
            for a in resp.get("apps") or []:
                if isinstance(a, dict) and "appid" in a:
                    out[int(a["appid"])] = str(a.get("name", ""))
            if not resp.get("have_more_results"):
                break
            last_appid = int(resp.get("last_appid") or 0)
            if not last_appid:
                break
        if not out:
            raise SteamUnavailable("IStoreService/GetAppList応答が空です")
        return out

    @staticmethod
    def _load_app_list_cache(path: Path) -> Optional[tuple[float, dict[int, str]]]:
        try:
            if not path.is_file():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = float(raw.get("fetched_at") or 0)
            apps = {int(k): str(v) for k, v in (raw.get("apps") or {}).items()}
            if not apps:
                return None
            return fetched_at, apps
        except Exception as e:
            logger.debug(f"app list cache unreadable ({path}): {e}")
            return None

    @staticmethod
    def _save_app_list_cache(path: Path, apps: dict[int, str]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fetched_at": int(time.time()),
                "apps": {str(k): v for k, v in apps.items()},
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"could not write app list cache ({path}): {e}")

    async def followed_games(self) -> list[dict]:
        """Followed games from the public community profile. Returns
        ``[{appid, name}]``; a privacy wall simply yields no rows → ``[]``
        (documented — not distinguishable without login).

        Row pattern: ``<div ... class="gameListRow notfollowed"
        data-appid="1675200"> ... <div class="gameListRowItemName">
        <a href="...">Steam Deck</a></div>``."""
        self._require_steamid()
        resp = await self._get(
            f"{_COMMUNITY}/profiles/{self._steamid}/followedgames/",
            what="Steamコミュニティ",
        )
        if resp.status_code != 200:
            raise SteamUnavailable(
                f"SteamコミュニティHTTP {resp.status_code}: followedgames"
            )
        return [
            {"appid": int(appid), "name": unescape(name).strip()}
            for appid, name in _FOLLOWED_ROW_RE.findall(resp.text)
        ]
