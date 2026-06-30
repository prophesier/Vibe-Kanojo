"""Uber Eats read-only browse client (Stage 1).

Drives a persistent, logged-in Playwright browser context but pulls data by
calling Uber's own ``/_p/api/*`` JSON endpoints directly via the context's
APIRequestContext (which shares the profile's cookies). No DOM scraping, no
clicking — so ads / popups / markup changes can't break it (the brittleness we
hit during the spike).

Auth recipe (set up by ``login.py``): the Chrome profile holds the session
cookies; a small header template (csrf + ``x-uber-*`` + delivery location) is
saved to ``uber_session.json``. We replay those headers with the profile's
cookies on every call.

Everything degrades gracefully: any failure raises :class:`UberUnavailable`
with a short, user-safe Japanese message. Nothing here blocks forever — every
network call has a timeout, and a process-wide lock serialises access to the
single browser context.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re
import uuid
from typing import Any, Dict, List

from playwright.async_api import async_playwright

HERE = pathlib.Path(__file__).parent
PROFILE = HERE / "uber_profile"
SESSION_FILE = HERE / "uber_session.json"

_BASE = "https://www.ubereats.com/_p/api"
_LOCALE = "jp"
_CALL_TIMEOUT_MS = 15000  # per API call; well under the MCP client's read timeout

# Header keys worth replaying (stable auth/session/location). x-uber-request-id
# is regenerated per call; cookies are supplied by the context, not here.
_REPLAY_HEADER_KEYS = {
    "x-csrf-token",
    "x-uber-ciid",
    "x-uber-client-gitref",
    "x-uber-session-id",
    "x-uber-device-location-latitude",
    "x-uber-device-location-longitude",
    "x-uber-target-location-latitude",
    "x-uber-target-location-longitude",
    "user-agent",
    "accept-language",
    "content-type",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
}


class UberUnavailable(Exception):
    """Any failure to browse Uber. ``str(e)`` is safe to show the user."""


def filter_replay_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Keep only the stable headers worth replaying (used by login.py too)."""
    return {k: v for k, v in headers.items() if k.lower() in _REPLAY_HEADER_KEYS}


def _tidy_eta(text: str) -> str:
    """Uber returns equal-bound ranges like "20～20 分"; collapse to "約20分".
    Genuine ranges ("20～30 分") are left as-is."""
    if not text:
        return ""
    m = re.match(r"^\s*(\d+)\s*[～~\-]\s*(\d+)\s*(.*)$", text)
    if m and m.group(1) == m.group(2):
        return f"約{m.group(1)}{m.group(3).strip()}"
    return text.strip()


def _strip_xssi(text: str) -> str:
    """Strip the ``)]}'`` anti-JSON-hijacking prefix some Uber responses carry."""
    s = text.lstrip()
    if s.startswith(")]}'"):
        return s[s.index("\n") + 1 :] if "\n" in s else s[4:]
    return text


def _item_endorsement(it: Dict[str, Any]) -> Dict[str, Any]:
    """Pull a catalog item's eater endorsement — the 👍 figure shown under items
    in the Uber app. Two shapes from Uber's ``endorsementMetadata``:
      - ``ratings``    → a like/positive-rating rate, e.g. rating="94%", numRatings=879
      - ``most_liked`` → a top seller ranked by like count (the "「ライク」数 #N"
        overlay badge); no percentage.
    This is a 好評率 (like rate), NOT a repeat/reorder rate.
    Returns ``{"like_rate": "94%"|"", "num_ratings": int, "top_liked_rank": int}``.
    """
    endo = (it.get("catalogItemAnalyticsData") or {}).get("endorsementMetadata") or {}
    etype = endo.get("endorsementType")
    if etype == "ratings":
        try:
            n = int(endo.get("numRatings") or 0)
        except (TypeError, ValueError):
            n = 0
        return {
            "like_rate": str(endo.get("rating") or ""),
            "num_ratings": n,
            "top_liked_rank": 0,
        }
    if etype == "most_liked":
        rank = 0
        for ov in it.get("imageOverlayElements") or []:
            tags = (
                ((ov.get("element") or {}).get("payload") or {}).get("tagsPayload")
                or {}
            ).get("tags") or []
            for t in tags:
                m = re.search(r"#\s*(\d+)", str(t.get("text") or ""))
                if m:
                    rank = int(m.group(1))
                    break
            if rank:
                break
        return {"like_rate": "", "num_ratings": 0, "top_liked_rank": rank}
    return {"like_rate": "", "num_ratings": 0, "top_liked_rank": 0}


def _parse_store_card(st: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the search-card fields out of a getSearchFeedV1 store object: name,
    rating (+ review count), delivery fee, ETA, sponsored flag, and promos."""
    title = st.get("title")
    name = title.get("text", "") if isinstance(title, dict) else (title or "")
    r = st.get("rating") or {}
    rating = r.get("text", "") if isinstance(r, dict) else str(r or "")
    count = ""
    if isinstance(r, dict):
        m = re.search(r"([\d,]+)\s*件(以上)?", r.get("accessibilityText", "") or "")
        if m:
            count = m.group(1).replace(",", "") + ("+" if m.group(2) else "")

    def _yen(s: str) -> str:
        m = re.search(r"[¥￥]\s*([\d,]+)", s or "")
        return f"¥{m.group(1)}" if m else (s or "").strip()

    fee = eta = original_fee = ""
    uber_one = surge = False
    sponsored = bool(st.get("storeAd"))
    for b in st.get("meta") or []:
        if not isinstance(b, dict):
            continue
        bt, t = b.get("badgeType"), (b.get("text") or "").strip()
        if bt == "MembershipBenefit" and not fee:
            # Uber One member benefit. text = member price; the non-member price
            # is in originalDeliveryFee. The user HAS Uber One, so the member
            # price is what they actually pay.
            uber_one = True
            fee = _yen(t)
            mem = (b.get("badgeDataWithFallback") or {}).get("membership") or {}
            original_fee = _yen(str(mem.get("originalDeliveryFee", "")))
        elif bt == "FARE" and not fee:
            fee = _yen(t)
            surge = bool(((b.get("badgeData") or {}).get("fare") or {}).get("isSurge"))
        elif bt == "ETD" and not eta:
            eta = t.replace(" ", "")
        elif bt == "SPONSORED":
            sponsored = True
    promos = [
        (x.get("text") or "").strip()
        for x in (st.get("signposts") or [])
        if isinstance(x, dict) and (x.get("text") or "").strip()
    ]
    return {
        "name": name,
        "store_uuid": st.get("storeUuid") or st.get("uuid", ""),
        "rating": str(rating or ""),
        "rating_count": count,
        "fee": fee,
        "original_fee": original_fee,
        "uber_one": uber_one,
        "surge": surge,
        "eta": eta,
        "sponsored": sponsored,
        "promos": promos,
    }


# Cross-promo sections mix in OTHER stores' products ("【ドリンクもどうぞ】",
# "【その他の商品もどうぞ】" etc.); exclude them from a store's own menu by title.
_CROSS_PROMO_RE = re.compile(
    r"もどうぞ|その他の商品|他のお店|別のお店|他店|一緒に(頼|注文)"
)


def _opt_price(cents: Any) -> str:
    """Customization option prices are in 1/100 yen (27000 → ¥270)."""
    try:
        c = int(cents or 0)
    except (TypeError, ValueError):
        return ""
    return f"+¥{c // 100}" if c > 0 else ""


def _parse_customizations(
    cl: Any, depth: int = 0, max_depth: int = 2
) -> List[Dict[str, Any]]:
    """Flatten a customizationsList into groups → options (one level of nesting,
    e.g. McDonald's 'Drink M' → the actual drink choices)."""
    groups: List[Dict[str, Any]] = []
    for g in cl or []:
        if not isinstance(g, dict):
            continue
        options = []
        for o in g.get("options") or []:
            if not isinstance(o, dict):
                continue
            opt = {
                "name": o.get("title", ""),
                "extra": _opt_price(o.get("price")),
                "sold_out": bool(o.get("isSoldOut")),
            }
            child = o.get("childCustomizationList")
            if child and depth < max_depth:
                opt["children"] = _parse_customizations(child, depth + 1, max_depth)
            options.append(opt)
        groups.append(
            {
                "title": g.get("title", ""),
                "min": g.get("minPermitted"),
                "max": g.get("maxPermitted"),
                "options": options,
            }
        )
    return groups


def _richtext(rt: Any) -> str:
    """Flatten a getStoreV1 richTextElements structure to plain text."""
    if not isinstance(rt, dict):
        return ""
    out = []
    for el in rt.get("richTextElements") or []:
        if isinstance(el, dict) and el.get("type") == "text":
            t = ((el.get("text") or {}).get("text") or {}).get("text")
            if isinstance(t, str):
                out.append(t)
    return "".join(out).strip()


class UberEatsClient:
    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._headers: Dict[str, str] = {}
        # Serialise calls within this process so two concurrent tool calls don't
        # launch two browsers on the same profile (which would lock-conflict).
        self._lock = asyncio.Lock()

    def _load_session(self) -> Dict[str, str]:
        """Load + validate the saved auth headers (no browser). Cached."""
        if self._headers:
            return self._headers
        if not SESSION_FILE.exists() or not PROFILE.exists():
            raise UberUnavailable(
                "Uberのログイン情報がまだありません。login.py を一度実行してログインしてください。"
            )
        try:
            hdrs = filter_replay_headers(
                json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("headers", {})
            )
        except Exception as e:
            raise UberUnavailable(f"Uberセッション情報の読み込みに失敗しました: {e}")
        if not hdrs.get("x-csrf-token"):
            raise UberUnavailable(
                "Uberセッション情報が不完全です。login.py で再ログインしてください。"
            )
        self._headers = hdrs
        return hdrs

    async def _call(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Open a browser context, make ONE API call, close it. Opening per call
        (instead of keeping a context warm) means we never hold the profile lock
        between calls — no lingering browser, no zombie if the server is killed,
        and login.py / probes can run without first stopping OLV. Costs ~1s of
        launch per call, fine for occasional browsing. Uses bundled Chromium (not
        channel=chrome) so it never conflicts with the user's everyday Chrome;
        bot detection is irrelevant here since we only make API calls, no page
        render. The lock serialises concurrent calls in this process.
        """
        async with self._lock:
            headers = dict(self._load_session())
            headers["x-uber-request-id"] = str(uuid.uuid4())
            headers.setdefault("content-type", "application/json")
            url = f"{_BASE}/{endpoint}?localeCode={_LOCALE}"
            pw = ctx = None
            try:
                try:
                    pw = await async_playwright().start()
                    ctx = await pw.chromium.launch_persistent_context(
                        user_data_dir=str(PROFILE),
                        headless=self._headless,
                        locale="ja-JP",
                        timezone_id="Asia/Tokyo",
                        args=["--disable-blink-features=AutomationControlled"],
                        ignore_default_args=["--enable-automation"],
                    )
                except Exception as e:
                    raise UberUnavailable(
                        f"ブラウザの起動に失敗しました（profile使用中かもしれません）: {e}"
                    )
                try:
                    resp = await ctx.request.post(
                        url,
                        headers=headers,
                        data=json.dumps(body),
                        timeout=_CALL_TIMEOUT_MS,
                    )
                except Exception as e:
                    raise UberUnavailable(f"Uberへの接続に失敗しました: {e}")
                if resp.status in (401, 403):
                    raise UberUnavailable(
                        "Uberのセッションが切れました。login.py で再ログインしてください。"
                    )
                if resp.status != 200:
                    raise UberUnavailable(
                        f"Uber APIエラー (status {resp.status})。少し待って再試行してください。"
                    )
                try:
                    data = json.loads(_strip_xssi(await resp.text()))
                except Exception as e:
                    raise UberUnavailable(f"Uberの応答を解析できませんでした: {e}")
                inner = data.get("data") if isinstance(data, dict) else None
                if isinstance(inner, dict) and "challenge" in str(
                    inner.get("nextUrl", "")
                ):
                    raise UberUnavailable(
                        "Uberが本人確認(reCAPTCHA)を要求しました。login.py で再ログインしてください。"
                    )
                return data
            finally:
                # Always close — this is what frees the profile lock each call.
                try:
                    if ctx is not None:
                        await ctx.close()
                except Exception:
                    pass
                try:
                    if pw is not None:
                        await pw.stop()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    async def search(self, keyword: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Return stores matching ``keyword`` with the search-card details:
        name, store_uuid, rating (+ review count), delivery fee, ETA, sponsored
        flag, and any promos.

        Uses getSearchFeedV1 (the full results endpoint, ~dozens of stores) — NOT
        getSearchSuggestionsV1 (autocomplete, ~3). It's a direct API call, so no
        page navigation, no ad, and no reCAPTCHA (which only the /search page
        navigation triggers).
        """
        if not keyword or not keyword.strip():
            return []
        data = await self._call(
            "getSearchFeedV1",
            {
                "userQuery": keyword.strip(),
                "displayType": "SEARCH_RESULTS",
                "date": "",
                "startTime": 0,
                "endTime": 0,
                "sortAndFilters": [],
                "vertical": "",
                "searchSource": "",
                "searchType": "",
                "keyName": "",
                "cacheKey": "",
                "recaptchaToken": "",
            },
        )
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for fi in (data.get("data") or {}).get("feedItems") or []:
            if not isinstance(fi, dict):
                continue
            st = fi.get("store") or {}
            uid = st.get("storeUuid") or st.get("uuid")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            out.append(_parse_store_card(st))
            if len(out) >= limit:
                break
        return out

    async def store(self, store_uuid: str, max_items: int = 60) -> Dict[str, Any]:
        """Return a store's header (rating + review count, ETA, distance, delivery
        fee, open status/hours) and its menu (sectioned; each item has name, price,
        short description, and a sold-out flag).

        Cross-promo sections that splice in OTHER stores' products
        (【ドリンクもどうぞ】 etc.) are filtered out so the menu is the store's own.
        """
        if not store_uuid or not store_uuid.strip():
            raise UberUnavailable("store_uuid が指定されていません。")
        data = await self._call(
            "getStoreV1",
            {
                "storeUuid": store_uuid.strip(),
                "diningMode": "DELIVERY",
                "time": {"asap": True},
                "cbType": "EATER_ENDORSED",
            },
        )
        d = data.get("data") or {}
        if not d:
            raise UberUnavailable(
                "店舗情報が取得できませんでした（store_uuid が無効かもしれません）。"
            )
        sec_titles = {
            s.get("uuid"): (s.get("title") or "")
            for s in (d.get("sections") or [])
            if isinstance(s, dict)
        }
        menu: List[Dict[str, Any]] = []
        total = 0
        seen: set = set()  # de-dupe: Uber repeats items across sections (popular, etc.)
        for sec_uuid, entries in (d.get("catalogSectionsMap") or {}).items():
            sec_title = sec_titles.get(sec_uuid, "")
            if sec_title and _CROSS_PROMO_RE.search(sec_title):
                continue  # other stores' products spliced in — not this store's menu
            items: List[Dict[str, Any]] = []
            for entry in entries or []:
                payload = (entry or {}).get("payload") or {}
                catalog = (payload.get("standardItemsPayload") or {}).get(
                    "catalogItems"
                ) or []
                for it in catalog:
                    if not isinstance(it, dict):
                        continue
                    iid = it.get("uuid") or it.get("title", "")
                    if iid in seen:
                        continue
                    seen.add(iid)
                    price = it.get("priceTagline")
                    if isinstance(price, dict):
                        price = price.get("text", "")
                    items.append(
                        {
                            "name": it.get("title", ""),
                            "price": price or "",
                            "item_uuid": it.get("uuid", ""),
                            "desc": " ".join((it.get("itemDescription") or "").split())[
                                :80
                            ],
                            "sold_out": bool(it.get("isSoldOut")),
                            "customizable": bool(it.get("hasCustomizations")),
                            **_item_endorsement(it),
                        }
                    )
                    total += 1
                    if total >= max_items:
                        break
                if total >= max_items:
                    break
            if items:
                menu.append({"section": sec_title, "items": items})
            if total >= max_items:
                break

        r = d.get("rating") or {}
        rating = (r.get("text") or r.get("ratingValue")) if isinstance(r, dict) else r
        count = r.get("reviewCount", "") if isinstance(r, dict) else ""
        fee = ""
        for mo in (d.get("modalityInfo") or {}).get("modalityOptions") or []:
            if isinstance(mo, dict) and mo.get("diningMode") == "DELIVERY":
                fee = _richtext(mo.get("priceTitleRichText"))
                break
        meta = d.get("storeInfoMetadata") or {}
        return {
            "name": d.get("title", ""),
            "rating": str(rating or ""),
            "rating_count": str(count or ""),
            "cuisines": [c for c in (d.get("categories") or []) if isinstance(c, str)],
            "eta": _tidy_eta(
                (d.get("etaRange") or {}).get("text", "")
                if isinstance(d.get("etaRange"), dict)
                else ""
            ),
            "distance": (d.get("distanceBadge") or {}).get("text", "")
            if isinstance(d.get("distanceBadge"), dict)
            else "",
            "fee": fee,
            "is_open": bool(d.get("isOpen")),
            "hours": str(meta.get("workingHoursTagline", "") or ""),
            "closed_message": str(d.get("closedMessage") or ""),
            "menu": menu,
            "truncated": total >= max_items,
        }

    async def item(self, store_uuid: str, item_uuid: str) -> Dict[str, Any]:
        """Return one item's detail incl. customization groups (toppings, set
        choices). getMenuItemV1 needs the item's section + subsection, so we look
        those up from getStoreV1 first (the item carries subsectionUuid; the
        section is its catalogSectionsMap key)."""
        if (
            not store_uuid
            or not store_uuid.strip()
            or not item_uuid
            or not item_uuid.strip()
        ):
            raise UberUnavailable("store_uuid と item_uuid が必要です。")
        store_uuid, item_uuid = store_uuid.strip(), item_uuid.strip()
        sd = (
            await self._call(
                "getStoreV1",
                {
                    "storeUuid": store_uuid,
                    "diningMode": "DELIVERY",
                    "time": {"asap": True},
                    "cbType": "EATER_ENDORSED",
                },
            )
        ).get("data") or {}
        section_uuid = subsection_uuid = None
        for sec_uuid, entries in (sd.get("catalogSectionsMap") or {}).items():
            for entry in entries or []:
                for it in ((entry or {}).get("payload") or {}).get(
                    "standardItemsPayload", {}
                ).get("catalogItems") or []:
                    if isinstance(it, dict) and it.get("uuid") == item_uuid:
                        section_uuid = it.get("sectionUuid") or sec_uuid
                        subsection_uuid = it.get("subsectionUuid")
                        break
                if section_uuid:
                    break
            if section_uuid:
                break
        if not section_uuid:
            raise UberUnavailable(
                "指定された item_uuid がこの店舗に見つかりませんでした。"
            )

        body = {
            "storeUuid": store_uuid,
            "menuItemUuid": item_uuid,
            "sectionUuid": section_uuid,
        }
        if subsection_uuid:
            body["subsectionUuid"] = subsection_uuid
        d = (await self._call("getMenuItemV1", body)).get("data") or {}
        if not d:
            raise UberUnavailable("商品の詳細が取得できませんでした。")
        price = d.get("price")
        try:
            price = f"¥{int(price) // 100}" if price else ""
        except (TypeError, ValueError):
            price = ""
        tags = [
            str(t.get("text")).strip()
            for t in (d.get("endorsementTags") or [])
            if isinstance(t, dict) and t.get("text")
        ]
        dietary = [
            str(x)
            for x in ((d.get("itemAttributeInfo") or {}).get("dietaryLabels") or [])
            if x
        ]
        return {
            "name": d.get("title", ""),
            "price": price,
            "desc": " ".join(
                (d.get("itemDescription") or "").split()
            ),  # full, untruncated
            "sold_out": bool(d.get("isSoldOut")),
            "tags": tags + dietary,
            "customizations": _parse_customizations(d.get("customizationsList")),
        }

    async def close(self) -> None:
        """No-op: each call already opens and closes its own browser context, so
        there is nothing long-lived to tear down. Kept for API compatibility."""
        return
