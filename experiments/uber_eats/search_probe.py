"""Probe: capture the FULL search-results request so uber_client can return many
results instead of the ~few autocomplete suggestions.

getSearchSuggestionsV1 only returns a handful of matches. The full results page
(/jp/search) fetches a much bigger list via getFeedV1/getSearchV1 with extra
search params (searchSource/searchType/...). We can't guess that body, so this
drives a REAL in-app search (type → submit → land on the results page) and
captures whichever /_p/api request actually returned many stores, dumping its
exact body + headers to search_probe_out/ so we can replicate it.

Run (browser stays open; solve a captcha manually if one appears):
    python experiments/uber_eats/search_probe.py "ラーメン"
"""

import json
import pathlib
import sys
from urllib.parse import quote

from playwright.sync_api import sync_playwright

# Reuse the spike's (already battle-tested) navigation helpers.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from spike import (  # noqa: E402
    _dismiss_interstitial,
    _open_with_retry,
    _solve_captcha_if_present,
    _wait_for_feed,
)

QUERY = sys.argv[1] if len(sys.argv) > 1 else "ラーメン"
HERE = pathlib.Path(__file__).resolve().parent
PROFILE = HERE / "uber_profile"
OUT = HERE / "search_probe_out"
OUT.mkdir(exist_ok=True)

_CAP: list[dict] = []  # {url, method, post_data, headers, body, n_stores}


def _count_stores(body: str) -> int:
    try:
        t = body.lstrip()
        if t.startswith(")]}'"):
            t = t[t.index("\n") + 1:]
        d = (json.loads(t) or {}).get("data") or {}
        n = 0
        for fi in (d.get("feedItems") or []):
            if isinstance(fi, dict):
                if "carousel" in fi:
                    n += len((fi["carousel"] or {}).get("stores") or [])
                elif fi.get("store") or fi.get("storeUuid"):
                    n += 1
        return n
    except Exception:
        return 0


def _on_response(resp) -> None:
    try:
        url = resp.url
        if "/_p/api/" not in url:
            return
        if "application/json" not in (resp.headers or {}).get("content-type", ""):
            return
        req = resp.request
        try:
            post = req.post_data
        except Exception:
            post = None
        body = resp.text()
        _CAP.append({
            "url": url, "method": req.method, "post_data": post,
            "headers": dict(req.headers), "body": body, "n_stores": _count_stores(body),
        })
    except Exception:
        pass


def main() -> None:
    print(f"Query: {QUERY!r}")
    with sync_playwright() as p:
        opts = dict(
            user_data_dir=str(PROFILE), headless=False, locale="ja-JP",
            timezone_id="Asia/Tokyo", viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
        )
        try:
            ctx = p.chromium.launch_persistent_context(channel="chrome", **opts)
        except Exception:
            ctx = p.chromium.launch_persistent_context(**opts)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on("response", _on_response)
        ctx.on("page", lambda pg: pg.close() if pg is not page and "ubereats.com" not in (pg.url or "") else None)
        try:
            if not _open_with_retry(page, "https://www.ubereats.com/jp"):
                print("Uber 502; try again later.")
                return
            _solve_captcha_if_present(page)
            _wait_for_feed(page)
            _dismiss_interstitial(page)
            _CAP.clear()

            # Go straight to the full results page. This bypasses the search box +
            # floating ads entirely; the network capture grabs the search request
            # regardless of any overlay. A captcha may appear — solve it in the
            # window and it continues + captures the real search request.
            print(f"\nNavigating directly to the full search results for {QUERY!r}…")
            _open_with_retry(page, f"https://www.ubereats.com/jp/search?q={quote(QUERY)}")
            _solve_captcha_if_present(page)
            _dismiss_interstitial(page)
            page.wait_for_timeout(5000)

            print(f"\n  landed on: {page.url[:90]}")
            # Rank captured /_p/api requests by how many stores they returned.
            ranked = sorted(_CAP, key=lambda r: r["n_stores"], reverse=True)
            print(f"  captured {len(_CAP)} /_p/api responses; by store count:")
            for r in ranked[:6]:
                ep = r["url"].split("/_p/api/")[1].split("?")[0]
                print(f"    {r['n_stores']:>4} stores  {r['method']} {ep}")
            if ranked and ranked[0]["n_stores"] > 0:
                win = ranked[0]
                ep = win["url"].split("/_p/api/")[1].split("?")[0]
                (OUT / "search_request.json").write_text(json.dumps({
                    "endpoint": ep, "url": win["url"], "method": win["method"],
                    "post_data": win["post_data"], "headers": win["headers"],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                (OUT / "search_response.json").write_text(win["body"], encoding="utf-8")
                print(f"\n  ✓ winner: {ep} with {win['n_stores']} stores")
                print("    saved request body+headers → search_probe_out/search_request.json")
                # show the body params (so we can see userQuery/searchSource/etc.)
                try:
                    b = json.loads(win["post_data"]) if win["post_data"] else {}
                    print("    body params:", {k: (v if isinstance(v, (str, int, bool)) and len(str(v)) < 40 else f"<{type(v).__name__}>") for k, v in b.items()})
                except Exception:
                    pass
            else:
                print("\n  ✗ no request returned multiple stores — check search_probe_out/ "
                      "and the browser (did it reach the results page?).")
            print("\nPress Enter to close…")
            input()
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
