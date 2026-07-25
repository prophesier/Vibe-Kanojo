"""End-to-end verification of the 2026-07-25 fixes via the production client:
cookie pinning in _call + dual-vertical merged search."""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from uber_client import UberEatsClient  # noqa: E402


async def main() -> None:
    c = UberEatsClient(headless=True)
    for kw, v in (
        ("ラーメン", "RESTAURANTS"),
        ("ローソン", "RETAIL"),
        ("ラーメン", "ALL"),  # trap value — must fall back to RESTAURANTS
    ):
        stores = await c.search(kw, limit=8, vertical=v)
        print(f"[search {kw!r} vertical={v}] {len(stores)} stores:")
        for s in stores:
            bits = [s["name"]]
            if s.get("rating"):
                bits.append(f"★{s['rating']}")
            if s.get("eta"):
                bits.append(s["eta"])
            print("  -", "  ".join(bits))


asyncio.run(main())
