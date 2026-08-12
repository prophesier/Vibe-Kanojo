"""uber_client._item_promo — item-card promo badges reach the tool output.

あさひ 08-11: he saw a discount badge on a store page; ヒロ's menu showed
nothing. All three parse paths dropped promoInfo entirely. The specimen here
is the real payload shape captured that day (¥446 Uber Cash badge, 豚丼
TONTON) — see experiments/uber_eats/_diag_discount_probe.py.

uber_client imports playwright at module level; the MCP server runs in the
Anaconda env, so if this venv lacks playwright the whole file skips (never
call a real API from routine tests — hard rule).
"""

import pathlib
import sys
import unittest

_UBER_DIR = str(
    pathlib.Path(__file__).resolve().parents[1] / "experiments" / "uber_eats"
)
sys.path.insert(0, _UBER_DIR)
try:
    import uber_client
except ImportError as e:  # playwright missing in this venv
    raise unittest.SkipTest(f"uber_client unimportable here: {e}")
finally:
    # A bare-name dir left on sys.path leaks into OTHER tests' bare imports
    # (test_music_and_wake's `import server` grabbed uber_eats/server.py).
    # Remove immediately — the imported module stays cached in sys.modules.
    try:
        sys.path.remove(_UBER_DIR)
    except ValueError:
        pass


SPECIMEN = {
    "title": "国産ホワイトチキン　ささみかつ定食",
    "price": 230000,
    "promoInfo": {
        "promoBadge": {
            "content": {"type": "richText"},
            "hierarchy": "PRIMARY",
            "accessibilityText": "写真のアップロードで ￥446 分の Uber Cash を獲得",
        }
    },
}


class ItemPromoTests(unittest.TestCase):
    def test_specimen_badge_extracted(self):
        self.assertEqual(
            uber_client._item_promo(SPECIMEN),
            {"promo": "写真のアップロードで ￥446 分の Uber Cash を獲得"},
        )

    def test_plain_item_adds_nothing(self):
        self.assertEqual(uber_client._item_promo({"title": "素の丼"}), {})
        self.assertEqual(uber_client._item_promo({"promoInfo": {}}), {})
        self.assertEqual(uber_client._item_promo({"promoInfo": {"promoBadge": {}}}), {})

    def test_badge_without_text_adds_nothing(self):
        self.assertEqual(
            uber_client._item_promo(
                {"promoInfo": {"promoBadge": {"accessibilityText": "  "}}}
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
