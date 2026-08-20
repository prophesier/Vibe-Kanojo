"""Read-only Uber Eats browse MCP server (Stage 1).

Tools (browse only — there is deliberately NO cart / checkout / payment tool, so
the model cannot order even if it wanted to):
  - uber_search(keyword)   -> stores matching the keyword (name + store_uuid)
  - uber_store(store_uuid) -> a store's menu (sections, items, prices)

Robustness contract (per the user's requirements):
  - every tool ALWAYS returns a short string, never raises;
  - a hard per-tool timeout (< the MCP client's read timeout) guarantees a tool
    call can never hang the chat turn — the character can always keep talking;
  - all failures are logged to uber_mcp.log so problems are visible.

Run (registered in mcp_servers.json):  python experiments/uber_eats/server.py
Env: UBER_EATS_HEADLESS=0 to watch the browser (default headless).
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys

# Make sibling modules importable regardless of the launch cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from uber_client import (  # noqa: E402
    UberEatsClient,
    UberUnavailable,
    normalize_vertical,
)

HERE = pathlib.Path(__file__).resolve().parent
logger.remove()  # don't write to stdout — stdout is the MCP stdio channel!
logger.add(sys.stderr, level="INFO")
logger.add(str(HERE / "uber_mcp.log"), rotation="2 MB", level="INFO")

_HEADLESS = os.environ.get("UBER_EATS_HEADLESS", "1") != "0"
_TOOL_TIMEOUT = 25  # seconds — under the MCP client's 30s read timeout

# Appended to every result. Since the 08-13 per-round replay work, tool
# results DO persist into later turns, but trimmed hard (~500 tok-equiv from
# the top; a leading related-facts section survives whole) — so the character
# must still relay anything worth keeping in its actual reply. Since 08-15
# facts link to stores by store_uuid (memory_add's store_id), not by name.
_RESULT_NOTE = (
    "\n\n（メモ：この結果は次のターン以降、先頭のわずかな部分を除いて"
    "切り詰められる。気になった店・料理・価格などは、ユーザーへの返信の中で"
    "必ず自分の言葉で伝えること。また、店の感想を記憶（facts）に記録する"
    "ときは、その店の store_uuid を memory_add の store_id 引数に渡すこと。）"
)

mcp = FastMCP("uber-eats")
_client = UberEatsClient(headless=_HEADLESS)

# Short store-id registry (あさひ 08-15): the LLM sees only the first 8 hex
# chars of a store_uuid (a full uuid tokenizes to ~18-20 tok and repeats per
# result row); the map resolves shorts back to full uuids for the store /
# item / category calls and keeps the official title for the facts linkage.
# Persisted next to this file so replayed short ids survive a restart.
_STORE_IDS_PATH = HERE / "_store_ids.json"
_SHORT_LEN = 8


def _load_store_ids() -> dict:
    try:
        with open(_STORE_IDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_store_ids: dict = _load_store_ids()


def _save_store_ids() -> None:
    try:
        tmp = _STORE_IDS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_store_ids, f, ensure_ascii=False, indent=1)
        tmp.replace(_STORE_IDS_PATH)
    except Exception as e:
        logger.warning(f"store-id registry save failed: {e}")


def _register_store(uuid: str, name: str) -> str:
    """Book a store in the registry; returns the id to DISPLAY (the 8-char
    short, or the full uuid on the astronomically rare short collision)."""
    uuid = (uuid or "").strip()
    if len(uuid) < _SHORT_LEN:
        return uuid
    short = uuid[:_SHORT_LEN]
    known = _store_ids.get(short)
    if known and known.get("uuid") != uuid:
        return uuid  # collision — this store keeps its full uuid visible
    if not known or (name and known.get("name") != name):
        _store_ids[short] = {"uuid": uuid, "name": name or ""}
    return short


def _resolve_store_uuid(s: str) -> str | None:
    """Full uuid from a short display id (or a full uuid passed through)."""
    s = (s or "").strip()
    if len(s) >= 32:
        return s
    entry = _store_ids.get(s[:_SHORT_LEN])
    return entry.get("uuid") if entry else None


# Same short-id treatment for ITEM uuids (08-15). In-memory ONLY, unlike the
# store registry (あさひ: items are numerous, churn often, and there is no
# pre-fetch path — a stale short after a restart just means re-opening the
# menu, which is the natural flow anyway).
_item_ids: dict = {}


def _register_item(uuid: str) -> str:
    uuid = (uuid or "").strip()
    if len(uuid) < _SHORT_LEN:
        return uuid
    short = uuid[:_SHORT_LEN]
    known = _item_ids.get(short)
    if known and known != uuid:
        return uuid  # collision — this item keeps its full uuid visible
    _item_ids[short] = uuid
    return short


def _resolve_item_uuid(s: str) -> str | None:
    s = (s or "").strip()
    if len(s) >= 32:
        return s
    return _item_ids.get(s[:_SHORT_LEN])


# Appended once under menu/item listings — explains the row-tail id and the
# ⚙ mark in ONE place instead of per row (the old per-item arrow text and
# the orphaned bare ⚙ both died here, 08-15).
_MENU_FOOT = (
    "\n※ 商品行末の英数字がその商品の item_uuid（uber_item で選択肢・詳細）。"
    "⚙ はトッピング等の選択肢あり。名前だけの行は前のカテゴリに出た商品の再掲。"
)


async def _run(coro, what: str):
    """Run a client coroutine with a hard timeout; never raise. Returns
    ``(ok, value_or_message)``."""
    try:
        return True, await asyncio.wait_for(coro, timeout=_TOOL_TIMEOUT)
    except UberUnavailable as e:
        logger.warning(f"{what}: unavailable: {e}")
        return False, str(e)
    except asyncio.TimeoutError:
        logger.warning(f"{what}: timed out after {_TOOL_TIMEOUT}s")
        return (
            False,
            "Uberの応答が遅すぎます（タイムアウト）。少し待って試してください。",
        )
    except Exception:
        logger.exception(f"{what}: unexpected error")
        return False, "Uberの処理中に問題が発生しました。"


@mcp.tool()
async def uber_search(
    keyword: str, vertical: str = "RESTAURANTS", limit: int = 10
) -> str:
    """Search Uber Eats (Japan) for stores. Takes a keyword such as a dish name
    or store name and returns a list of deliverable stores (name, rating,
    store_uuid).
    limit is how many stores to return (default 10, max 30). Decide it UP
    FRONT from the user's intent: if he wants breadth ("いくつか候補",
    "比較したい", "他には？"), pass a larger limit on the FIRST call —
    re-searching with a higher limit later costs a whole extra round.
    vertical picks the search index and must be chosen to match the intent:
      - "RESTAURANTS" (default): restaurants and eateries — any food or dish
        query.
      - "RETAIL": convenience stores, supermarkets, drugstores, liquor shops —
        when the user wants conbini/grocery shopping.
    There is NO mixed mode: since 2026-07 Uber's API returns retail-only for
    an empty or "ALL" vertical, so always pass the vertical explicitly (an
    omitted/unknown value falls back to RESTAURANTS). If the intent spans
    both, call this tool once per vertical.
    Browse-only: you cannot order or pay. If a store looks
    interesting, pass its store_uuid to uber_store to see the menu.
    Stores marked ［PR］ are sponsored (ad slots).
    A store whose delivery fee is marked "Uber One" is showing the member price.
    The browsing account is not an Uber One member, but the user himself is, so
    that member price (cheaper than usual, or free) is what the user actually
    pays. "通常¥…" is the reference price for non-members."""
    v = normalize_vertical(vertical)
    try:
        limit = max(1, min(int(limit), 30))
    except (TypeError, ValueError):
        limit = 10
    ok, data = await _run(
        _client.search(keyword, limit=limit, vertical=v),
        f"uber_search({keyword!r},{v},limit={limit})",
    )
    if not ok:
        return f"検索できませんでした: {data}"
    v_label = "飲食店" if v == "RESTAURANTS" else "コンビニ・スーパー等の小売"
    if not data:
        return (
            f"「{keyword}」に一致する配達可能な店舗（{v_label}）は"
            "見つかりませんでした。"
        )
    trunc = (
        f"上位{len(data)}件のみ表示・limit引数で最大30件まで増やせる"
        if len(data) >= limit
        else f"全{len(data)}件"
    )
    lines = [f"「{keyword}」の検索結果（{trunc}・{v_label}）:"]
    for s in data:
        pr = "［PR］" if s.get("sponsored") else ""
        bits = []
        if s.get("rating"):
            cnt = f"({s['rating_count']}件)" if s.get("rating_count") else ""
            bits.append(f"★{s['rating']}{cnt}")
        if s.get("fee"):
            if s.get("uber_one"):
                orig = f"・通常{s['original_fee']}" if s.get("original_fee") else ""
                bits.append(f"配送{s['fee']}(Uber One{orig})")
            elif s.get("surge"):
                bits.append(f"配送{s['fee']}(混雑)")
            else:
                bits.append(f"配送{s['fee']}")
        if s.get("eta"):
            bits.append(s["eta"])
        # The store_uuid (8-char short) rides the info bits as the LAST field
        # (08-15 あさひ: the labelled "store_uuid:" line was pure overhead);
        # the footer explains it once, outside the repeated rows.
        bits.append(_register_store(s["store_uuid"], s["name"]))
        line = f"- {pr}{s['name']}  " + " · ".join(bits)
        if s.get("promos"):
            line += "\n    🎁 " + " / ".join(s["promos"][:2])
        lines.append(line)
    _save_store_ids()
    lines.append(
        "\n※ 各行末尾の英数字がその店の store_uuid。"
        "メニューを見るには uber_store に渡してください。"
    )
    return "\n".join(lines) + _RESULT_NOTE


@mcp.tool()
async def uber_store(store_uuid: str, section: str = "") -> str:
    """Get the menu of the store with the given store_uuid. Pass a store_uuid
    from the uber_search results. Returns the store name, rating, delivery
    estimate, and the menu (categories, dish names, prices, like rate).
    LARGE menus (over ~35 items) come back as a per-category DIGEST: every
    category (the same categories the app shows) with its top few items by
    popularity, full detail included. The digest is usually enough to
    recommend from. To see ALL items of ONE category, call this tool again
    with section="カテゴリ名" (fuzzy match on the category names shown).
    Only drill into a category when you actually need it — each extra call
    costs a round.
    Every item line carries its item_uuid — pass it to uber_item for
    toppings/set options and the full description. A trailing ⚙ on the
    uuid line means the item has options to choose.
    Each item's "好評率" is the share of "likes" (positive ratings), shown with
    the rating count — it is not a repeat-order rate.
    Browse-only: you cannot order or pay."""
    full_uuid = _resolve_store_uuid(store_uuid)
    if not full_uuid:
        return (
            f"store_uuid {store_uuid!r} が見つかりません。"
            "先に uber_search で店を検索してください。"
        )
    ok, data = await _run(
        _client.store(full_uuid, section=section),
        f"uber_store({store_uuid!r},section={section!r})",
    )
    if not ok:
        return f"メニューを取得できませんでした: {data}"
    _register_store(full_uuid, data.get("name", ""))
    _save_store_ids()
    head = data["name"]
    bits = []
    if data.get("rating"):
        cnt = f"({data['rating_count']}件)" if data.get("rating_count") else ""
        bits.append(f"★{data['rating']}{cnt}")
    if data.get("eta"):
        bits.append(str(data["eta"]))
    if data.get("distance"):
        bits.append(str(data["distance"]))
    if data.get("cuisines"):
        bits.append("/".join(data["cuisines"][:3]))
    lines = [f"【{head}】" + (("  " + " · ".join(bits)) if bits else "")]
    # second header line: open status / hours / delivery fee
    sub = ["営業中" if data.get("is_open") else "営業時間外"]
    if data.get("hours"):
        sub.append(str(data["hours"]))
    if data.get("fee"):
        sub.append(f"配送: {data['fee']}")
    lines.append("  " + " · ".join(sub))
    if data.get("closed_message"):
        lines.append(f"  ⚠ {data['closed_message']}")
    # Convenience/supermarket:商品数が多すぎるので商品は出さず、カテゴリ一覧だけ返す。
    # 具体的な商品は uber_category で カテゴリ→サブカテゴリ と絞り込んでから見る。
    if data.get("is_convenience"):
        cats = data.get("categories") or []
        if not cats:
            lines.append("（カテゴリ一覧を取得できませんでした）")
            return "\n".join(lines) + _RESULT_NOTE
        lines.append(
            f"\n■ コンビニ／スーパー（商品数が多いのでカテゴリのみ表示・{len(cats)}カテゴリ）"
        )
        for c in cats:
            cnt = f"（約{c['num_items']}点）" if c.get("num_items") else ""
            lines.append(f"- {c['title']}{cnt}\n    section_uuid: {c['section_uuid']}")
        lines.append(
            "\n※ カテゴリの中身は uber_category に store_uuid と section_uuid を渡す。"
            "サブカテゴリ（細分類）が出たら、その subsection_uuid を渡すと商品が表示される。"
        )
        return "\n".join(lines) + _RESULT_NOTE
    # Big-menu digest: every category, popularity top-N only, cuts labelled.
    # Items repeat across Uber's category carousels — a repeat keeps only its
    # name (あさひ 08-15: everything else is a byte-for-byte rerun).
    if data.get("digest"):
        seen_items: set = set()
        lines.append(
            f"\n■ メニューが大きい（全{data.get('total_items', '?')}品）ため、"
            "各カテゴリの人気上位のみ表示:"
        )
        for sec in data["digest"]:
            more = sec["total"] - len(sec["items"])
            more_s = f"　…他{more}品" if more > 0 else ""
            lines.append(f"\n▼ {sec['section'] or '(無題)'}（全{sec['total']}品）")
            for it in sec["items"]:
                key = it.get("item_uuid") or it.get("name")
                if key in seen_items:
                    lines.append(f"- {it['name']}")
                else:
                    seen_items.add(key)
                    lines.append(_fmt_menu_item(it))
            if more_s:
                lines.append(more_s)
        lines.append(
            "\n※ カテゴリの全品を見るには uber_store を store_uuid と "
            "section=カテゴリ名 で再呼び出し。必要なカテゴリだけ絞って呼ぶこと。"
        )
        return "\n".join(lines) + _MENU_FOOT + _RESULT_NOTE
    if not data.get("menu"):
        lines.append("（メニュー項目を取得できませんでした）")
    if data.get("section_view"):
        cap = (
            f"（全{data['section_total']}品中、先頭{len(data['menu'][0]['items'])}品"
            "のみ表示）"
            if data.get("truncated")
            else f"（全{data['section_total']}品）"
        )
        lines.append(f"\n▼ カテゴリ「{data['section_view']}」{cap}")
        for it in data["menu"][0]["items"]:
            lines.append(_fmt_menu_item(it))
        return "\n".join(lines) + _MENU_FOOT + _RESULT_NOTE
    seen_items = set()
    for sec in data["menu"]:
        if sec.get("section"):
            lines.append(f"\n■ {sec['section']}")
        for it in sec["items"]:
            key = it.get("item_uuid") or it.get("name")
            if key in seen_items:
                lines.append(f"- {it['name']}")
            else:
                seen_items.add(key)
                lines.append(_fmt_menu_item(it))
    if data.get("truncated"):
        lines.append("\n…（メニューは一部のみ表示）")
    return "\n".join(lines) + _MENU_FOOT + _RESULT_NOTE


def _fmt_menu_item(it: dict) -> str:
    """One menu line: name/price/likes with the short item id as the last
    field (⚙ suffix = customizable), then promo and description. Same shape
    in digest and full views — desc and id are load-bearing everywhere
    (あさひ 08-13)."""
    so = "[品切れ] " if it.get("sold_out") else ""
    price = f"  {it['price']}" if it.get("price") else ""
    if price and it.get("orig_price"):
        # price is already the DISCOUNTED figure; the original was dug out
        # of the strikethrough span client-side (08-13).
        price += f"（通常{it['orig_price']}）"
    # Eater endorsement (好評率 = 「ライク」率, NOT リピート率) — labelled
    # explicitly so it can't be misread as a repeat rate.
    if it.get("like_rate"):
        cnt = f"({it['num_ratings']}件)" if it.get("num_ratings") else ""
        endo = f"  👍好評率{it['like_rate']}{cnt}"
    elif it.get("top_liked_rank"):
        endo = f"  👍ライク数#{it['top_liked_rank']}"
    else:
        endo = ""
    line = f"- {so}{it['name']}{price}{endo}"
    # The item id rides the info line as the LAST field, like store rows
    # (08-15); _MENU_FOOT explains it and the ⚙ mark once per listing.
    if it.get("item_uuid"):
        gear = " ⚙" if it.get("customizable") else ""
        line += f" · {_register_item(it['item_uuid'])}{gear}"
    if it.get("promo"):
        line += f"\n    🎁 {it['promo']}"
    if it.get("desc"):
        line += f"\n    {it['desc']}"
    return line


def _fmt_catalog_item(it: dict) -> str:
    """One catalog line: [品切れ] 名前  ¥価格  👍好評率…（uber_store と同じ体裁）。"""
    so = "[品切れ] " if it.get("sold_out") else ""
    price = f"  {it['price']}" if it.get("price") else ""
    if it.get("like_rate"):
        cnt = f"({it['num_ratings']}件)" if it.get("num_ratings") else ""
        endo = f"  👍好評率{it['like_rate']}{cnt}"
    elif it.get("top_liked_rank"):
        endo = f"  👍ライク数#{it['top_liked_rank']}"
    else:
        endo = ""
    line = f"- {so}{it['name']}{price}{endo}"
    if it.get("item_uuid"):
        gear = " ⚙" if it.get("customizable") else ""
        line += f" · {_register_item(it['item_uuid'])}{gear}"
    if it.get("promo"):
        line += f"\n    🎁 {it['promo']}"
    return line


@mcp.tool()
async def uber_category(
    store_uuid: str, section_uuid: str, subsection_uuid: str = "", offset: int = 0
) -> str:
    """Look inside a category at stores with many items, such as convenience
    stores and supermarkets. Pass the section_uuid of a category returned by
    uber_store. You first get a list of subcategories; pass one of their
    subsection_uuid values to see that subcategory's items (name and price).
    Categories without subcategories show their items directly. Use offset to
    page further.
    Browse-only: you cannot order or pay."""
    full_uuid = _resolve_store_uuid(store_uuid)
    if not full_uuid:
        return (
            f"store_uuid {store_uuid!r} が見つかりません。"
            "先に uber_search で店を検索してください。"
        )
    ok, data = await _run(
        _client.catalog(full_uuid, section_uuid, subsection_uuid, offset),
        f"uber_category({section_uuid!r},{subsection_uuid!r})",
    )
    if not ok:
        return f"カテゴリを取得できませんでした: {data}"
    subs = data.get("subcategories") or []
    items = data.get("items") or []
    chose_sub = bool((subsection_uuid or "").strip())
    lines = []
    # No subcategory chosen yet and this category HAS subcategories → show the
    # 細分類 list only（ユーザー要望どおり、細分類を選んでから商品を出す）。
    if not chose_sub and subs:
        lines.append(
            f"サブカテゴリ（細分類・{len(subs)}件）。"
            "どれかの subsection_uuid を uber_category に渡すと商品が見られる："
        )
        for s in subs:
            lines.append(f"- {s['title']}\n    subsection_uuid: {s['subsection_uuid']}")
        return "\n".join(lines) + _RESULT_NOTE
    # Otherwise show items (a subcategory was chosen, or the category has none).
    if chose_sub:
        cur = next(
            (
                s["title"]
                for s in subs
                if s["subsection_uuid"] == subsection_uuid.strip()
            ),
            "",
        )
        if cur:
            lines.append(f"■ {cur}")
    if not items:
        lines.append("（このカテゴリの商品を取得できませんでした）")
    lines.extend(_fmt_catalog_item(it) for it in items)
    if data.get("has_more"):
        lines.append(
            f"\n…（続きあり。offset={data['next_offset']} を uber_category に渡すと続き）"
        )
    return "\n".join(lines) + _MENU_FOOT + _RESULT_NOTE


@mcp.tool()
async def uber_item(store_uuid: str, item_uuid: str) -> str:
    """Get an item's details. Pass store_uuid and item_uuid (the trailing id
    of an item row in the uber_store results). Returns the available options
    and their surcharges: toppings, set contents, sizes, and so on.
    Browse-only: you cannot order or pay."""
    full_uuid = _resolve_store_uuid(store_uuid)
    if not full_uuid:
        return (
            f"store_uuid {store_uuid!r} が見つかりません。"
            "先に uber_search で店を検索してください。"
        )
    full_item = _resolve_item_uuid(item_uuid)
    if not full_item:
        return (
            f"item_uuid {item_uuid!r} が見つかりません。"
            "先に uber_store でメニューを表示してください。"
        )
    ok, data = await _run(
        _client.item(full_uuid, full_item), f"uber_item({item_uuid!r})"
    )
    if not ok:
        return f"商品詳細を取得できませんでした: {data}"
    head = data["name"] + (f"  {data['price']}" if data.get("price") else "")
    if data.get("sold_out"):
        head += "  [品切れ]"
    lines = [f"【{head}】"]
    if data.get("tags"):
        lines.append("🏷 " + " · ".join(data["tags"]))
    if data.get("desc"):
        lines.append(data["desc"])
    if not data.get("customizations"):
        lines.append("\n（選択肢なし。そのまま注文する商品です）")
    for g in data.get("customizations", []):
        lines.append(f"\n▼ {g['title']}{_sel_label(g.get('min'), g.get('max'))}")
        for o in g["options"]:
            so = "[品切れ] " if o.get("sold_out") else ""
            extra = f"  {o['extra']}" if o.get("extra") else ""
            lines.append(f"  ・{so}{o['name']}{extra}")
            for ch in o.get("children", []):
                opts = "、".join(
                    c["name"] + (f"({c['extra']})" if c.get("extra") else "")
                    for c in ch.get("options", [])[:8]
                )
                if opts:
                    lines.append(f"      ＞ {ch['title']}: {opts}")
    return "\n".join(lines) + _RESULT_NOTE


def _sel_label(mn, mx) -> str:
    """Human label for a customization group's selection constraint."""
    if mn == mx == 1:
        return "（1つ選択）"
    if not mn:
        return f"（任意・最大{mx}）" if mx else "（任意）"
    return f"（{mn}〜{mx}個）" if mx else f"（{mn}個以上）"


if __name__ == "__main__":
    logger.info(f"Starting uber-eats MCP server (headless={_HEADLESS}).")
    mcp.run()  # stdio transport
