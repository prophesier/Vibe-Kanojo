"""Diagnostic probe round 3 (2026-07-25): rebuild a FULL-fidelity uev2.loc.

Round 2 showed a fabricated minimal cookie restores the area but not
restaurants — likely because the real cookie carries a resolvable place
reference. This round asks Uber's own location APIs for the 東京駅 address:
  1. getLocationAutocompleteV1("東京駅") → candidates with references;
  2. getLocationDetailsV1(best candidate) → the full location payload;
  3. write that payload verbatim into uev2.loc, then search ラーメン again.
"""

import asyncio
import json
import pathlib
import sys
import uuid
from collections import Counter
from urllib.parse import quote

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from playwright.async_api import async_playwright  # noqa: E402

from uber_client import (  # noqa: E402
    PROFILE,
    SESSION_FILE,
    _strip_xssi,
    filter_replay_headers,
)

QUERY = "東京駅"


def _summarize_feed(data: dict, label: str) -> None:
    fis = (data.get("data") or {}).get("feedItems") or []
    types: Counter = Counter()
    names = []
    for fi in fis:
        if not isinstance(fi, dict):
            continue
        t = fi.get("type") or "?"
        types[t] += 1
        st = fi.get("store")
        if not isinstance(st, dict):
            st = (fi.get("miniStoreWithItems") or {}).get("store")
        if isinstance(st, dict):
            title = st.get("title")
            name = title.get("text", "") if isinstance(title, dict) else (title or "")
            names.append(f"{name} [{t}]")
    print(f"[{label}] types={dict(types)} stores={len(names)}")
    for n in names[:12]:
        print("   -", n)


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

        r = await call("getLocationAutocompleteV1", {"query": QUERY})
        print(f"[getLocationAutocompleteV1 {QUERY!r}] status={r.status}")
        if r.status != 200:
            return
        d = json.loads(_strip_xssi(await r.text()))
        preds = (d.get("data") or {}).get("predictions") or (d.get("data") or [])
        if isinstance(preds, dict):
            preds = preds.get("predictions") or []
        cands = []
        for p in preds if isinstance(preds, list) else []:
            if not isinstance(p, dict):
                continue
            cands.append(
                {
                    "title": p.get("mainText")
                    or (p.get("structuredFormatting") or {}).get("mainText")
                    or p.get("description")
                    or "?",
                    "reference": p.get("reference") or p.get("placeId") or p.get("id"),
                    "referenceType": p.get("referenceType") or p.get("provider") or "",
                    "raw_keys": sorted(p.keys()),
                }
            )
        print(f"  candidates={len(cands)}")
        for c in cands[:5]:
            print(
                f"   - {c['title']}  ref={str(c['reference'])[:40]}  "
                f"type={c['referenceType']}"
            )
        if cands and cands[0]["reference"] is None:
            print("  raw keys of first prediction:", cands[0]["raw_keys"])
        target = next((c for c in cands if c["reference"]), None)
        if not target:
            print("  ✗ no usable reference; dumping first prediction raw:")
            if preds:
                print(json.dumps(preds[0], ensure_ascii=False)[:600])
            return

        r = await call(
            "getLocationDetailsV1",
            {
                "reference": target["reference"],
                "referenceType": target["referenceType"] or "google_places",
            },
        )
        print(f"[getLocationDetailsV1] status={r.status}")
        if r.status != 200:
            return
        detail = (json.loads(_strip_xssi(await r.text())) or {}).get("data") or {}
        print(
            "  detail keys:",
            sorted(detail.keys())[:12],
            " lat/lng:",
            detail.get("latitude"),
            detail.get("longitude"),
        )
        if not detail.get("latitude"):
            print("  ✗ detail has no coordinates; raw:", json.dumps(detail)[:400])
            return

        await ctx.add_cookies(
            [
                {
                    "name": "uev2.loc",
                    "value": quote(json.dumps(detail, ensure_ascii=False)),
                    "domain": ".ubereats.com",
                    "path": "/",
                    "expires": 2000000000,
                    "httpOnly": False,
                    "secure": True,
                    "sameSite": "None",
                }
            ]
        )
        print("[pin] full-fidelity uev2.loc written from getLocationDetailsV1")

        r = await call(
            "getSearchFeedV1",
            {
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
            },
        )
        print(f"[search after full pin] status={r.status}")
        if r.status == 200:
            _summarize_feed(
                json.loads(_strip_xssi(await r.text())), "search with real reference"
            )
    finally:
        await ctx.close()
        await pw.stop()


asyncio.run(main())
