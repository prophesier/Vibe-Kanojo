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
import os
import pathlib
import sys

# Make sibling modules importable regardless of the launch cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from loguru import logger  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from uber_client import UberEatsClient, UberUnavailable  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
logger.remove()  # don't write to stdout — stdout is the MCP stdio channel!
logger.add(sys.stderr, level="INFO")
logger.add(str(HERE / "uber_mcp.log"), rotation="2 MB", retention=3, level="INFO")

_HEADLESS = os.environ.get("UBER_EATS_HEADLESS", "1") != "0"
_TOOL_TIMEOUT = 25  # seconds — under the MCP client's 30s read timeout

# Appended to every result. Tool results are NOT persisted to the chat context
# (they vanish next turn), so the character must relay anything worth keeping in
# its actual reply to the user.
_RESULT_NOTE = (
    "\n\n（メモ：この結果はあなたの参照用で会話履歴には残らない。"
    "気になった店・料理・価格などは、ユーザーへの返信の中で必ず自分の言葉で伝えること。"
    "この内容は次のターンには消えている。）"
)

mcp = FastMCP("uber-eats")
_client = UberEatsClient(headless=_HEADLESS)


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
        return False, "Uberの応答が遅すぎます（タイムアウト）。少し待って試してください。"
    except Exception:
        logger.exception(f"{what}: unexpected error")
        return False, "Uberの処理中に問題が発生しました。"


@mcp.tool()
async def uber_search(keyword: str) -> str:
    """Uber Eats（日本）で店舗を検索する。料理名・店名などのキーワードで、配達可能な
    店舗の一覧（店名・評価・store_uuid）を返す。これは閲覧専用で、注文や支払いはできない。
    気になる店があれば、その store_uuid を uber_store に渡すとメニューを見られる。
    返り値はあなたの参照用で会話には残らないので、伝えたいことは返信本文に書くこと。
    ［PR］が付く店はスポンサー（広告枠）の店。
    配送費に「Uber One」と付く店は会員特典価格。閲覧用アカウントはUber One非加入だが、
    ユーザー本人はUber One会員なので、その特典価格（=通常より安い/無料）がユーザーの
    実際の支払額になる。「通常¥…」は非会員の参考価格。"""
    ok, data = await _run(_client.search(keyword), f"uber_search({keyword!r})")
    if not ok:
        return f"検索できませんでした: {data}"
    if not data:
        return f"「{keyword}」に一致する配達可能な店舗は見つかりませんでした。"
    lines = [f"「{keyword}」の検索結果（{len(data)}件）:"]
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
        meta = "  " + " · ".join(bits) if bits else ""
        line = f"- {pr}{s['name']}{meta}"
        if s.get("promos"):
            line += "\n    🎁 " + " / ".join(s["promos"][:2])
        line += f"\n    store_uuid: {s['store_uuid']}"
        lines.append(line)
    lines.append("\n※ メニューを見るには uber_store に store_uuid を渡してください。")
    return "\n".join(lines) + _RESULT_NOTE


@mcp.tool()
async def uber_store(store_uuid: str) -> str:
    """指定した store_uuid の店舗のメニューを取得する。store_uuid は uber_search の
    結果から渡す。店名・評価・配達目安と、メニュー（カテゴリ・料理名・価格）を返す。
    閲覧専用で、注文や支払いはできない。
    返り値はあなたの参照用で会話には残らないので、伝えたいことは返信本文に書くこと。"""
    ok, data = await _run(_client.store(store_uuid), f"uber_store({store_uuid!r})")
    if not ok:
        return f"メニューを取得できませんでした: {data}"
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
    if not data.get("menu"):
        lines.append("（メニュー項目を取得できませんでした）")
    for sec in data["menu"]:
        if sec.get("section"):
            lines.append(f"\n■ {sec['section']}")
        for it in sec["items"]:
            so = "[品切れ] " if it.get("sold_out") else ""
            price = f"  {it['price']}" if it.get("price") else ""
            line = f"- {so}{it['name']}{price}"
            if it.get("desc"):
                line += f"\n    {it['desc']}"
            # only customizable items are worth drilling into; expose their id
            if it.get("customizable") and it.get("item_uuid"):
                line += f"\n    ⚙ トッピング/セット選択あり → uber_item(item_uuid: {it['item_uuid']})"
            lines.append(line)
    if data.get("truncated"):
        lines.append("\n…（メニューは一部のみ表示）")
    return "\n".join(lines) + _RESULT_NOTE


@mcp.tool()
async def uber_item(store_uuid: str, item_uuid: str) -> str:
    """商品の詳細を取得する。store_uuid と item_uuid（uber_store の結果で「⚙」が付いた
    商品に表示される）を渡す。トッピング・セット内容・サイズなどの選択肢（と追加料金）を返す。
    閲覧専用で、注文や支払いはできない。
    返り値はあなたの参照用で会話には残らないので、伝えたいことは返信本文に書くこと。"""
    ok, data = await _run(_client.item(store_uuid, item_uuid), f"uber_item({item_uuid!r})")
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
