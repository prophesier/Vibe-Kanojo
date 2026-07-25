"""Diagnostic probe round 5: getSearchFeedV1 body-parameter variants.

Home feed shows restaurants with the pinned cookie; search shows only retail.
Try vertical/searchSource/searchType variants to find what brings the
restaurant segment back. Also check getSearchSuggestionsV1 (does the
restaurant index itself still answer?).
"""

import asyncio
import json
import pathlib
import sys
import uuid
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from playwright.async_api import async_playwright  # noqa: E402

from uber_client import (  # noqa: E402
    PROFILE,
    SESSION_FILE,
    _strip_xssi,
    filter_replay_headers,
)

BASE_BODY = {
    "userQuery": "ラーメン",
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
}

VARIANTS = [
    ("baseline", {}),
    ("vertical=ALL", {"vertical": "ALL"}),
    ("vertical=RESTAURANT", {"vertical": "RESTAURANT"}),
    ("vertical=RESTAURANTS", {"vertical": "RESTAURANTS"}),
    ("searchSource=SEARCH_BAR", {"searchSource": "SEARCH_BAR"}),
    ("searchType=RESTAURANTS", {"searchType": "RESTAURANTS"}),
    (
        "SEARCH_BAR+USER_QUERY",
        {"searchSource": "SEARCH_BAR", "searchType": "USER_QUERY"},
    ),
]


def _classify(names):
    """Rough retail-vs-restaurant split by name keywords."""
    retail_kw = (
        "ローソン", "ファミリーマート", "セブン", "ミニストップ", "まいばすけっと",
        "ストア", "スーパー", "ドラッグ", "薬局", "マート", "酒", "リカー",
        "コストコ", "ポプラ", "ダイエー", "イオン",
    )
    retail = sum(1 for n in names if any(k in n for k in retail_kw))
    return retail, len(names) - retail


async def main() -> None:
    headers = filter_replay_headers(
        json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("headers", {})
    )
    headers.setdefault("content-type", "application/json")
    pw = await async_playwright().start()
    ctx = await pw.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE),
        headless=True,
        locale="ja-JP",
        timezone_id="Asia/Tokyo",
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
    )
    try:

        async def call(endpoint: str, body: dict):
            h = dict(headers)
            h["x-uber-request-id"] = str(uuid.uuid4())
            return await ctx.request.post(
                f"https://www.ubereats.com/_p/api/{endpoint}?localeCode=jp",
                headers=h,
                data=json.dumps(body),
                timeout=15000,
            )

        for label, over in VARIANTS:
            body = dict(BASE_BODY)
            body.update(over)
            r = await call("getSearchFeedV1", body)
            if r.status != 200:
                print(f"[{label}] status={r.status}")
                continue
            d = json.loads(_strip_xssi(await r.text()))
            names = []
            types: Counter = Counter()
            for fi in (d.get("data") or {}).get("feedItems") or []:
                if not isinstance(fi, dict):
                    continue
                types[fi.get("type") or "?"] += 1
                st = fi.get("store")
                if not isinstance(st, dict):
                    st = (fi.get("miniStoreWithItems") or {}).get("store")
                if isinstance(st, dict):
                    t = st.get("title")
                    names.append(t.get("text", "") if isinstance(t, dict) else (t or ""))
            retail, rest = _classify(names)
            print(
                f"[{label}] stores={len(names)} retail≈{retail} "
                f"non-retail≈{rest} types={dict(types)}"
            )
            if rest > 5:
                print("   sample non-retail:")
                shown = 0
                for n in names:
                    if not any(
                        k in n
                        for k in ("ローソン", "ストア", "ドラッグ", "まいばすけっと")
                    ):
                        print("    -", n)
                        shown += 1
                        if shown >= 6:
                            break

        r = await call("getSearchSuggestionsV1", {"userQuery": "ラーメン"})
        print(f"[getSearchSuggestionsV1] status={r.status}")
        if r.status == 200:
            d = json.loads(_strip_xssi(await r.text()))
            blob = json.dumps(d, ensure_ascii=False)
            print("  contains 東京駅?", "東京駅" in blob, " length:", len(blob))
            data = d.get("data")
            if isinstance(data, list):
                for s in data[:8]:
                    if isinstance(s, dict):
                        title = s.get("title") or {}
                        txt = (
                            title.get("text")
                            if isinstance(title, dict)
                            else str(title)
                        )
                        print(f"   - [{s.get('type')}] {txt}")
    finally:
        await ctx.close()
        await pw.stop()


asyncio.run(main())
