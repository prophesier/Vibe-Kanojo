"""Diagnostic probe (2026-07-25): where does Uber think we are?

Symptoms: delivery location shows 中目黒 instead of the configured 東京駅, and
food searches return only convenience stores/supermarkets. The replayed
headers in uber_session.json still carry the 東京駅 coordinates, so this probe
checks the two other location carriers:
  1. the profile's ``uev2.loc`` cookie (the address selection the site stores);
  2. what getSearchFeedV1 actually returns (card types + store names), to tell
     "server returns only grocery" apart from "our parser drops restaurants".

Read-only, one API call. Prints NO tokens — only location data and store names.
"""

import asyncio
import json
import pathlib
import sys
import uuid
from collections import Counter
from urllib.parse import unquote

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from playwright.async_api import async_playwright  # noqa: E402

from uber_client import (  # noqa: E402
    PROFILE,
    SESSION_FILE,
    _strip_xssi,
    filter_replay_headers,
)

ARGS = [a for a in sys.argv[1:] if a != "--pin"]
PIN_LOCATION = "--pin" in sys.argv[1:]
KEYWORD = ARGS[0] if ARGS else "ラーメン"


def _fabricated_loc_cookie(lat: str, lng: str) -> dict:
    """Minimal uev2.loc built from the saved coordinates — probing whether the
    API honors a client-fabricated address selection."""
    from urllib.parse import quote

    payload = {
        "address": {
            "address1": "東京駅",
            "title": "東京駅",
            "subtitle": "東京都千代田区",
        },
        "latitude": float(lat),
        "longitude": float(lng),
        "reference": "",
        "referenceType": "",
        "type": "MANUAL",
        "source": "manual_auto_complete",
    }
    return {
        "name": "uev2.loc",
        "value": quote(json.dumps(payload, ensure_ascii=False)),
        "domain": ".ubereats.com",
        "path": "/",
        "expires": 2000000000,
        "httpOnly": False,
        "secure": True,
        "sameSite": "None",
    }


async def main() -> None:
    headers = filter_replay_headers(
        json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("headers", {})
    )
    headers["x-uber-request-id"] = str(uuid.uuid4())
    headers.setdefault("content-type", "application/json")
    print(
        "[headers] target-location:",
        headers.get("x-uber-target-location-latitude"),
        headers.get("x-uber-target-location-longitude"),
    )

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
        found = False
        for c in await ctx.cookies("https://www.ubereats.com"):
            if c["name"] != "uev2.loc":
                continue
            found = True
            try:
                loc = json.loads(unquote(c["value"]))
                addr = loc.get("address") or {}
                print("[cookie uev2.loc]")
                print(
                    "  address:",
                    json.dumps(
                        {
                            k: addr.get(k)
                            for k in ("address1", "title", "subtitle", "fullAddress")
                            if addr.get(k)
                        },
                        ensure_ascii=False,
                    ),
                )
                print(
                    "  lat/lng:",
                    loc.get("latitude"),
                    loc.get("longitude"),
                )
            except Exception as e:
                print(f"[cookie uev2.loc] decode failed ({e}); raw length "
                      f"{len(c['value'])}")
        if not found:
            print("[cookie uev2.loc] NOT FOUND in profile")

        if PIN_LOCATION:
            ck = _fabricated_loc_cookie(
                headers.get("x-uber-target-location-latitude", "0"),
                headers.get("x-uber-target-location-longitude", "0"),
            )
            await ctx.add_cookies([ck])
            print("[pin] injected fabricated uev2.loc with saved coordinates")

        body = {
            "userQuery": KEYWORD,
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
        resp = await ctx.request.post(
            "https://www.ubereats.com/_p/api/getSearchFeedV1?localeCode=jp",
            headers=headers,
            data=json.dumps(body),
            timeout=15000,
        )
        print(f"[getSearchFeedV1 {KEYWORD!r}] status: {resp.status}")
        if resp.status != 200:
            return
        data = json.loads(_strip_xssi(await resp.text()))
        fis = (data.get("data") or {}).get("feedItems") or []
        types: Counter = Counter()
        names = []
        unparsed_samples = {}
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
                name = (
                    title.get("text", "") if isinstance(title, dict) else (title or "")
                )
                names.append(f"{name}  [{t}]")
            elif t not in unparsed_samples:
                unparsed_samples[t] = sorted(fi.keys())
        print("[feedItem types]", dict(types))
        print(f"[stores our parser sees: {len(names)}] first 15:")
        for n in names[:15]:
            print("  -", n)
        for t, keys in unparsed_samples.items():
            print(f"[unparsed type {t}] keys: {keys}")
    finally:
        await ctx.close()
        await pw.stop()


if __name__ == "__main__":
    asyncio.run(main())
