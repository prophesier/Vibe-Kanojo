"""Diagnostic probe round 2 (2026-07-25).

Round 1 found: uev2.loc cookie gone; fabricated one fixes the AREA of results
but a food query still returns only retail/grocery — no restaurants.

This round:
  1. summarize the OLD working search_response.json (card types + names);
  2. list surviving ubereats.com cookie NAMES (no values printed);
  3. search again with fabricated uev2.loc + uev2.diningMode=DELIVERY;
  4. probe candidate address-book endpoints to find the server-side saved
     addresses (so the 東京駅 address can be re-selected server-side).

Read-only except the address-book probes, which only LIST addresses.
"""

import asyncio
import json
import pathlib
import sys
import uuid
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from playwright.async_api import async_playwright  # noqa: E402

from _diag_loc_probe import _fabricated_loc_cookie  # noqa: E402
from uber_client import (  # noqa: E402
    PROFILE,
    SESSION_FILE,
    _strip_xssi,
    filter_replay_headers,
)

HERE = pathlib.Path(__file__).resolve().parent
KEYWORD = "ラーメン"


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
    for n in names[:10]:
        print("   -", n)


async def main() -> None:
    old = HERE / "search_probe_out" / "search_response.json"
    if old.exists():
        try:
            _summarize_feed(
                json.loads(_strip_xssi(old.read_text(encoding="utf-8"))),
                "OLD working response",
            )
        except Exception as e:
            print("[OLD response] parse failed:", e)

    headers = filter_replay_headers(
        json.loads(SESSION_FILE.read_text(encoding="utf-8")).get("headers", {})
    )
    headers["x-uber-request-id"] = str(uuid.uuid4())
    headers.setdefault("content-type", "application/json")
    lat = headers.get("x-uber-target-location-latitude", "0")
    lng = headers.get("x-uber-target-location-longitude", "0")

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
        cookie_names = sorted(
            {c["name"] for c in await ctx.cookies("https://www.ubereats.com")}
        )
        print("[cookie names]", cookie_names)

        await ctx.add_cookies(
            [
                _fabricated_loc_cookie(lat, lng),
                {
                    "name": "uev2.diningMode",
                    "value": "DELIVERY",
                    "domain": ".ubereats.com",
                    "path": "/",
                    "expires": 2000000000,
                    "httpOnly": False,
                    "secure": True,
                    "sameSite": "None",
                },
            ]
        )

        async def call(endpoint: str, body: dict):
            h = dict(headers)
            h["x-uber-request-id"] = str(uuid.uuid4())
            return await ctx.request.post(
                f"https://www.ubereats.com/_p/api/{endpoint}?localeCode=jp",
                headers=h,
                data=json.dumps(body),
                timeout=15000,
            )

        resp = await call(
            "getSearchFeedV1",
            {
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
            },
        )
        print(f"[search+loc+diningMode] status={resp.status}")
        if resp.status == 200:
            _summarize_feed(
                json.loads(_strip_xssi(await resp.text())), "search with both cookies"
            )

        # Find the server-side address book. Only LIST — no mutation here.
        for ep in (
            "getDeliveryLocationsV1",
            "getEaterAddressesV1",
            "getAddressesV1",
            "getDeliveryLocationV1",
        ):
            try:
                r = await call(ep, {})
                info = f"[{ep}] status={r.status}"
                if r.status == 200:
                    try:
                        d = json.loads(_strip_xssi(await r.text()))
                        inner = d.get("data")
                        if isinstance(inner, dict):
                            info += f" data keys={sorted(inner.keys())[:8]}"
                        elif isinstance(inner, list):
                            titles = []
                            for a in inner[:5]:
                                if isinstance(a, dict):
                                    addr = a.get("address") or a
                                    titles.append(
                                        str(
                                            addr.get("title")
                                            or addr.get("address1")
                                            or "?"
                                        )
                                    )
                            info += f" list len={len(inner)} titles={titles}"
                        else:
                            info += f" data type={type(inner).__name__}"
                    except Exception as e:
                        info += f" (parse fail: {e})"
                print(info)
            except Exception as e:
                print(f"[{ep}] error: {e}")
    finally:
        await ctx.close()
        await pw.stop()


asyncio.run(main())
