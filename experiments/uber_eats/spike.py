"""Stage-0 feasibility spike v2 for the Uber Eats *browse* feature (THROWAWAY).

v1 finding: a throwaway JP account logs in + reaches the feed fine, but the
SCRIPT navigating directly to a search URL trips Uber's reCAPTCHA ("自動セキュリ
ティチェック"). Manual, human-paced clicking did NOT. So v2:
  - real Chrome (channel="chrome") instead of bundled Chromium — less detectable
  - lightweight dependency-free stealth (hide the common automation tells)
  - HUMAN-LIKE navigation: type into the search box char-by-char, click result
    cards — never goto a search URL
  - if a captcha still appears, PAUSE and let you solve it in the (headed) window,
    then auto-continue
Goal: get ONE clean search + store view, and capture where the structured data
actually lives (embedded JSON vs a GraphQL/XHR response) to design Stage 1.

SETUP (conda env; spike only — don't add to pyproject yet):
    pip install playwright
    playwright install chromium      # real Chrome is used if installed; this is the fallback
RUN:
    python experiments/uber_eats/spike.py "ラーメン"
Browser stays open the whole time; just finish login/captcha there when asked.
"""

import json
import pathlib
import random
import re
import sys
import time

from playwright.sync_api import sync_playwright

QUERY = sys.argv[1] if len(sys.argv) > 1 else "ラーメン"
HERE = pathlib.Path(__file__).parent
PROFILE = HERE / "uber_profile"  # persistent login (gitignored — holds cookies!)
OUT = HERE / "out"  # inspection artifacts (gitignored)

# Dependency-free stealth: patch the properties reCAPTCHA/bot-scoring inspect.
_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = window.chrome || { runtime: {} };
Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP','ja','en-US','en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
const _q = navigator.permissions && navigator.permissions.query;
if (_q) { navigator.permissions.query = (p) => p && p.name === 'notifications'
    ? Promise.resolve({state: Notification.permission}) : _q(p); }
"""

# JSON API responses the page fetched itself (the real data source), WITH the
# request shape so Stage 1 can call these endpoints directly instead of scraping.
_CAPTURED: list[dict] = []
_NOISE = ("google.com", "gstatic.com", "recaptcha", "mixpanel", "heatpipe",
          "sentry", "facebook", "doubleclick", "/rgstr", "m.stripe")


def _dump(name: str, text: str) -> None:
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    p.write_text(text or "", encoding="utf-8")
    print(f"    saved {p.relative_to(HERE)} ({len(text or '')} chars)")


def _on_response(resp) -> None:
    try:
        url = resp.url
        if any(d in url.lower() for d in _NOISE):
            return
        if "application/json" not in (resp.headers or {}).get("content-type", ""):
            return
        interesting = any(
            k in url.lower()
            for k in ("api", "graphql", "feed", "search", "store", "catalog", "menu", "getstore")
        )
        body = resp.text()
        if body and (interesting or len(body) > 2000):
            req = resp.request
            try:
                post = req.post_data
            except Exception:
                post = None
            _CAPTURED.append({
                "url": url, "status": resp.status, "method": req.method,
                "req_headers": dict(req.headers), "post_data": post, "body": body,
            })
    except Exception:
        pass


def _dump_captured(phase: str) -> list:
    """Write each captured response body + a .req.json sidecar (method/url/
    headers/post_data) so the request can be replayed. Returns the records and
    clears the buffer."""
    if not _CAPTURED:
        print(f"  [{phase}] no JSON API responses captured")
        return []
    net = OUT / "network"
    net.mkdir(parents=True, exist_ok=True)
    print(f"  [{phase}] captured {len(_CAPTURED)} JSON API response(s):")
    for i, rec in enumerate(_CAPTURED):
        (net / f"{phase}_{i:02d}.json").write_text(rec["body"], encoding="utf-8")
        meta = {k: rec[k] for k in ("url", "status", "method", "post_data", "req_headers")}
        (net / f"{phase}_{i:02d}.req.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"      [{i:02d}] {rec['status']} {rec['method']} {len(rec['body'])}B  {rec['url'][:100]}")
    recs = list(_CAPTURED)
    _CAPTURED.clear()
    return recs


def _replay_direct(ctx, rec: dict) -> None:
    """Replay a captured /_p/api request via the context's APIRequestContext —
    NO page, NO DOM. If this returns the same JSON, Stage 1 can hit these
    endpoints directly (cookies are shared from the logged-in context)."""
    print(f"\n=== Direct-API replay (no navigation): {rec['method']} {rec['url'][:90]} ===")
    drop = {"host", "cookie", "content-length", "accept-encoding", "connection"}
    headers = {k: v for k, v in rec["req_headers"].items() if k.lower() not in drop}
    try:
        r = ctx.request.fetch(
            rec["url"], method=rec["method"], headers=headers, data=rec["post_data"]
        )
        body = r.text()
        (OUT / "network" / "REPLAY.json").write_text(body, encoding="utf-8")
        ok = r.status == 200 and len(body) > 500
        print(f"  -> status {r.status}, {len(body)}B  (saved out/network/REPLAY.json)")
        print("  ✓ DIRECT API WORKS — Stage 1 can skip the DOM entirely."
              if ok else "  ✗ direct call didn't return useful data (check status/REPLAY.json).")
    except Exception as e:
        print(f"  ✗ direct call failed: {e}")


def grab_embedded_json(page) -> dict:
    found: dict[str, str] = {}
    for key in ("__REDUX_STATE__", "__INITIAL_STATE__", "__PRELOADED_STATE__", "__APP_STATE__"):
        try:
            val = page.evaluate(f"() => window.{key} ? JSON.stringify(window.{key}) : null")
            if val:
                found[key] = val
        except Exception:
            pass
    try:
        blobs = page.eval_on_selector_all(
            'script[type="application/json"]', "els => els.map(e => e.textContent)"
        )
        for i, b in enumerate(blobs or []):
            if b and len(b) > 200:
                found[f"script_json_{i}"] = b
    except Exception:
        pass
    return found


def _pause(page, lo: int = 500, hi: int = 1300) -> None:
    page.wait_for_timeout(random.randint(lo, hi))


def _is_captcha(page) -> bool:
    """A VISIBLE reCAPTCHA challenge only. Two earlier false positives:
      - the invisible background reCAPTCHA (v3/Enterprise) iframe Uber loads on
        EVERY page for bot-scoring;
      - the challenge text living in Uber's JS i18n bundle on every page, so a
        raw ``"…" in page.content()`` always matched.
    So check for the challenge wording as a VISIBLE rendered text node (get_by_text
    excludes <script>; is_visible() excludes preloaded hidden templates)."""
    try:
        loc = page.get_by_text(re.compile("自動セキュリティチェック|まもなく完了です"))
        return loc.count() > 0 and loc.first.is_visible()
    except Exception:
        return False


def _dismiss_interstitial(page) -> None:
    """Close promo/interstitial modals (mod=messagingInterstitial) that overlay
    the feed and block the search box."""
    for sel in (
        'button[aria-label="閉じる"]',
        'button[aria-label="Close"]',
        '[aria-label="閉じる"]',
        '[aria-label="Close"]',
    ):
        try:
            loc = page.locator(sel)
            if loc.count() and loc.first.is_visible():
                loc.first.click(timeout=2000)
                page.wait_for_timeout(700)
                print("  (dismissed an interstitial/promo modal)")
                return
        except Exception:
            pass
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass


def _is_bad_gateway(page) -> bool:
    try:
        return "502" in (page.title() or "") or "Bad gateway" in page.content()[:3000]
    except Exception:
        return False


def _open_with_retry(page, url: str, attempts: int = 4) -> bool:
    for i in range(1, attempts + 1):
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            if _is_bad_gateway(page):
                print(f"  attempt {i}/{attempts}: Uber 502 (origin hiccup) — waiting 10s…")
                page.wait_for_timeout(10000)
                continue
            return True
        except Exception as e:
            print(f"  attempt {i}/{attempts}: navigation error: {e}")
            page.wait_for_timeout(4000)
    return False


def _solve_captcha_if_present(page, minutes: int = 5) -> None:
    if not _is_captcha(page):
        return
    print("\n" + "!" * 64)
    print("!! reCAPTCHA appeared. Solve it in the BROWSER WINDOW (click the box).")
    print("!! This auto-continues once it clears.")
    print("!" * 64)
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        page.wait_for_timeout(2000)
        if not _is_captcha(page):
            print("  ✓ captcha cleared — continuing.\n")
            return
    print("  (timed out waiting for captcha to clear)\n")


def _wait_for_feed(page, minutes: int = 6) -> bool:
    print("\n" + "=" * 64)
    print(">>> In the BROWSER: log in + set a Tokyo address if asked.")
    print(">>> Leave it open — this auto-continues once the feed loads.")
    print("=" * 64)
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            _solve_captcha_if_present(page, minutes=2)
            if "/feed" in page.url and page.locator('a[href*="/store/"]').count() > 0:
                print("  ✓ feed detected — continuing.\n")
                return True
        except Exception:
            pass
        page.wait_for_timeout(2000)
    print("  (timed out waiting for the feed; continuing anyway)\n")
    return False


def _human_search(page, query: str) -> bool:
    """Type the query into Uber's search box (never goto a URL). Dismisses
    overlays first and waits for the box to be actually VISIBLE, so a
    late-loading promo can't steal the click. Locator clicks auto-wait for the
    target to be stable + unobscured, so this no longer fires into an ad."""
    _dismiss_interstitial(page)
    box = None
    for make in (
        lambda: page.get_by_placeholder(re.compile("Uber Eats|検索|レストラン|料理|search", re.I)),
        lambda: page.get_by_role("searchbox"),
        lambda: page.get_by_role("combobox"),
    ):
        try:
            cand = make().first
            cand.wait_for(state="visible", timeout=6000)
            box = cand
            break
        except Exception:
            continue
    if box is None:
        print("  search box not found")
        return False
    try:
        _dismiss_interstitial(page)  # one may have popped up while we waited
        box.click(timeout=5000)
        _pause(page)
        box.fill(query)  # atomic set (no stray clicks), then submit
        _pause(page, 300, 700)
        page.keyboard.press("Enter")
        return True
    except Exception as e:
        print(f"  search interaction failed: {e}")
        return False


def _store_links(page) -> list[str]:
    try:
        hrefs = page.eval_on_selector_all(
            'a[href*="/store/"]', "els => Array.from(new Set(els.map(e => e.getAttribute('href'))))"
        )
        return [h for h in (hrefs or []) if h]
    except Exception:
        return []


def main() -> None:
    print(f"Query  : {QUERY!r}")
    print(f"Profile: {PROFILE}\n")
    with sync_playwright() as p:
        opts = dict(
            user_data_dir=str(PROFILE),
            headless=False,
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],  # drop the automation tell
        )
        try:
            ctx = p.chromium.launch_persistent_context(channel="chrome", **opts)
            print("Launched real Chrome (channel=chrome).")
        except Exception as e:
            print(f"Real Chrome unavailable ({e}); using bundled Chromium.")
            ctx = p.chromium.launch_persistent_context(**opts)
        ctx.add_init_script(_STEALTH_JS)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on("response", _on_response)

        # Popup guard: ads/promos that open a NEW TAB derailed us before. Close
        # any new tab that isn't an Uber Eats page so a stray interaction can't
        # leave us operating on the wrong (or a dead) page.
        def _on_new_page(pg) -> None:
            try:
                if pg is page:
                    return
                pg.wait_for_load_state("domcontentloaded", timeout=3000)
                if "ubereats.com" not in (pg.url or ""):
                    print(f"  (closed popup tab: {(pg.url or '')[:70]})")
                    pg.close()
            except Exception:
                pass

        ctx.on("page", _on_new_page)
        try:
            print("Opening Uber Eats JP…")
            if not _open_with_retry(page, "https://www.ubereats.com/jp"):
                print("\nUber kept 502-ing (their origin). Wait a few minutes and re-run.")
                return
            _solve_captcha_if_present(page)
            _wait_for_feed(page)
            _dismiss_interstitial(page)  # close promo modal overlaying the feed
            _CAPTURED.clear()

            # ---- human-like search (NO url goto) ----
            print(f"\nTyping {QUERY!r} into the search box like a human…")
            if not _human_search(page, QUERY):
                print("  couldn't find the search box; last-resort goto (may captcha)…")
                _open_with_retry(page, f"https://www.ubereats.com/jp/search?q={QUERY}")
            # Wait for results to actually render (store links appear) instead of
            # a blind sleep, so a slow load doesn't get screenshotted mid-render.
            try:
                page.wait_for_selector('a[href*="/store/"]', timeout=15000)
            except Exception:
                print("  (no store links within 15s — capturing whatever is there)")
            _solve_captcha_if_present(page)
            _dismiss_interstitial(page)
            page.wait_for_timeout(1200)

            page.screenshot(path=str(OUT / "search.png"), full_page=True)
            _dump("search.html", page.content())
            for k, v in grab_embedded_json(page).items():
                _dump(f"search_{k}.json", v)
            links = _store_links(page)
            print(f"  store links found on page: {len(links)}")
            for h in links[:5]:
                print(f"      {h}")
            search_recs = _dump_captured("search")

            # ---- Direct-API probe: replay a captured search/feed request with
            # NO page/DOM. If it works, Stage 1 hits these endpoints directly. ----
            target = next(
                (r for r in search_recs
                 if any(k in r["url"].lower() for k in ("getsearch", "getfeed"))),
                search_recs[0] if search_recs else None,
            )
            if target:
                _replay_direct(ctx, target)

            # ---- open a VISIBLE store to capture the menu endpoint ----
            # (the earlier timeout was clicking a hidden search-suggestion link)
            print("\nOpening a store to capture the menu endpoint…")
            try:
                page.keyboard.press("Escape")  # close the search-suggestion dropdown
                page.wait_for_timeout(500)
                _dismiss_interstitial(page)
                vis = page.locator('a[href*="/jp/store/"]:visible')
                if vis.count() == 0:
                    print("  no VISIBLE store card on the page — skipping store capture.")
                else:
                    link = vis.first
                    print(f"  clicking visible store: {(link.get_attribute('href') or '')[:80]}")
                    link.scroll_into_view_if_needed(timeout=3000)
                    _pause(page)
                    link.click(timeout=6000)
                    try:
                        page.wait_for_url(re.compile(r"/jp/store/"), timeout=12000)
                    except Exception:
                        page.wait_for_timeout(3000)
                    _solve_captcha_if_present(page)
                    _dismiss_interstitial(page)
                    page.wait_for_timeout(1500)
                    page.screenshot(path=str(OUT / "store.png"), full_page=True)
                    _dump("store.html", page.content())
                    for k, v in grab_embedded_json(page).items():
                        _dump(f"store_{k}.json", v)
                    _dump_captured("store")
            except Exception as e:
                print(f"  could not open store: {e}")

            print("\nDone. Inspect experiments/uber_eats/out/. Press Enter to close…")
            input()
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
