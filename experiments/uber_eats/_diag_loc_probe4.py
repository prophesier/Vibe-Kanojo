"""Diagnostic probe round 4: with the fabricated 東京駅 uev2.loc pinned,
does the HOME feed (getFeedV1) still contain restaurants?

If yes → only search is address-fidelity-gated. If the home feed is also
retail-only → restaurant serviceability needs a resolvable address, and a
fresh site-minted cookie (login.py re-run) is the only reliable fix.
Also tries two more autocomplete endpoint name guesses.
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

        r = await call("getFeedV1", {"cacheKey": ""})
        print(f"[getFeedV1] status={r.status}")
        if r.status == 200:
            d = json.loads(_strip_xssi(await r.text()))
            fis = (d.get("data") or {}).get("feedItems") or []
            types: Counter = Counter()
            names = []
            for fi in fis:
                if not isinstance(fi, dict):
                    continue
                types[fi.get("type") or "?"] += 1
                st = fi.get("store")
                if not isinstance(st, dict):
                    st = (fi.get("miniStoreWithItems") or {}).get("store")
                if isinstance(st, dict):
                    t = st.get("title")
                    names.append(t.get("text", "") if isinstance(t, dict) else (t or ""))
                car = (fi.get("carousel") or {}).get("stores") or []
                for cs in car[:6]:
                    if isinstance(cs, dict):
                        t = cs.get("title")
                        names.append(
                            "(carousel) "
                            + (t.get("text", "") if isinstance(t, dict) else (t or ""))
                        )
            print("  types:", dict(types))
            print(f"  first 15 of {len(names)} store names:")
            for n in names[:15]:
                print("   -", n)

        for ep in ("getLocationAutocompleteV2", "getMapsAutocompleteV1"):
            r = await call(ep, {"query": "東京駅"})
            print(f"[{ep}] status={r.status}")
    finally:
        await ctx.close()
        await pw.stop()


asyncio.run(main())
