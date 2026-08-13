"""uber_store big-menu digest + section drill (あさひ 08-12: two large menus
in one round cost 24k fresh tokens; big stores now return a per-category
popularity digest, with section= for one category in full).

Same import guard as test_uber_item_promo: uber_client needs playwright,
which only the Anaconda env has — this venv skips, never calls a real API.
"""

import pathlib
import sys
import unittest
from unittest.mock import AsyncMock

_UBER_DIR = str(
    pathlib.Path(__file__).resolve().parents[1] / "experiments" / "uber_eats"
)
sys.path.insert(0, _UBER_DIR)
try:
    import uber_client
except ImportError as e:
    raise unittest.SkipTest(f"uber_client unimportable here: {e}")
finally:
    try:
        sys.path.remove(_UBER_DIR)
    except ValueError:
        pass


def _payload(n_per_section, sections=("丼もの", "サイド")):
    # Real shape (08-13 probe): categories are ENTRY-level carousels inside
    # one section, each with its own standardItemsPayload.title.
    entries = []
    for si, title in enumerate(sections):
        items = []
        for i in range(n_per_section):
            items.append(
                {
                    "uuid": f"it{si}-{i}",
                    "title": f"{title}の品{i}",
                    "itemDescription": "説明テキスト",
                    "priceTagline": {"text": f"￥{500 + i}"},
                    "isSoldOut": False,
                    "hasCustomizations": True,
                    "catalogItemAnalyticsData": {
                        "endorsementMetadata": {
                            "endorsementType": "ratings",
                            "rating": "90%",
                            "numRatings": 100 - i,
                        }
                    },
                }
            )
        entries.append(
            {
                "payload": {
                    "standardItemsPayload": {
                        "title": {"text": title},
                        "catalogItems": items,
                    }
                }
            }
        )
    return {
        "data": {
            "title": "テスト店",
            "sections": [{"uuid": "sec0", "title": "주문"}],
            "catalogSectionsMap": {"sec0": entries},
            "isOpen": True,
        }
    }


def _client(payload):
    c = uber_client.UberEatsClient(headless=True)
    c._call = AsyncMock(return_value=payload)
    return c


class MenuDigestTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_store_full_menu_unchanged(self):
        data = await _client(_payload(10)).store("u")
        self.assertEqual(data["digest"], [])
        self.assertEqual(len(data["menu"]), 2)
        self.assertEqual(len(data["menu"][0]["items"]), 10)
        self.assertIn("desc", data["menu"][0]["items"][0])
        self.assertFalse(data["truncated"])

    async def test_big_store_returns_digest_with_visible_totals(self):
        data = await _client(_payload(30)).store("u")  # 60 items > 35
        self.assertEqual(data["menu"], [])
        self.assertEqual(data["total_items"], 60)
        self.assertEqual(len(data["digest"]), 2)
        sec = data["digest"][0]
        self.assertEqual(sec["section"], "丼もの")  # entry title, not 주문
        self.assertEqual(sec["total"], 30)
        self.assertEqual(len(sec["items"]), uber_client._DIGEST_TOP)
        # Popularity order: highest num_ratings first (品0 has 100).
        self.assertEqual(sec["items"][0]["name"], "丼ものの品0")
        # Full detail kept (あさひ 08-13: desc and uuid are load-bearing).
        for it in sec["items"]:
            self.assertIn("desc", it)
            self.assertIn("item_uuid", it)

    async def test_section_drill_returns_full_detail(self):
        data = await _client(_payload(30)).store("u", section="丼")
        self.assertEqual(data["section_view"], "丼もの")
        self.assertEqual(data["section_total"], 30)
        items = data["menu"][0]["items"]
        self.assertEqual(len(items), 30)  # under _SECTION_CAP
        self.assertIn("desc", items[0])
        self.assertIn("item_uuid", items[0])
        self.assertFalse(data["truncated"])

    async def test_section_cap_is_labelled(self):
        data = await _client(_payload(50)).store("u", section="サイド")
        self.assertEqual(data["section_total"], 50)
        self.assertEqual(len(data["menu"][0]["items"]), uber_client._SECTION_CAP)
        self.assertTrue(data["truncated"])

    async def test_unknown_section_lists_available(self):
        with self.assertRaises(uber_client.UberUnavailable) as ctx:
            await _client(_payload(30)).store("u", section="デザート")
        self.assertIn("丼もの", str(ctx.exception))

    async def test_personalized_and_discount_categories_dropped(self):
        # あさひ 08-13: the browsing account has no order history, so the
        # targeted carousels are dead weight, and 特定商品の割引 items are
        # duplicated in regular categories with identical promo/strike data.
        p = _payload(
            5,
            sections=("また注文する", "あなたへのおすすめ", "特定商品の割引", "丼もの"),
        )
        data = await _client(p).store("u")
        self.assertEqual([m["section"] for m in data["menu"]], ["丼もの"])

    async def test_strikethrough_original_price_extracted(self):
        p = _payload(2)
        it = p["data"]["catalogSectionsMap"]["sec0"][0]["payload"][
            "standardItemsPayload"
        ]["catalogItems"][0]
        it["priceTagline"] = {
            "text": "￥1,112",
            "textFormat": (
                '<span><span style="color:#05944F">￥1,112 </span>'
                '<span style="text-decoration:line-through;color:#757575">'
                "￥1,390</span></span>"
            ),
        }
        data = await _client(p).store("u")
        first = data["menu"][0]["items"][0]
        self.assertEqual(first["price"], "￥1,112")
        self.assertEqual(first["orig_price"], "￥1,390")
        self.assertEqual(data["menu"][0]["items"][1]["orig_price"], "")


if __name__ == "__main__":
    unittest.main()
