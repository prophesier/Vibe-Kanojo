"""One-time Uber Eats login + session-header capture for the MCP server.

Run on first setup, or whenever the server reports the session expired:
    python experiments/uber_eats/login.py

Opens a real Chrome on the persistent profile. Log in + set a Tokyo delivery
address. Once the restaurant feed loads it captures the request-header template
Uber's own site sends to ``/_p/api/*`` (csrf + ``x-uber-*`` + delivery location)
and saves it to ``uber_session.json``. The MCP server then replays those headers
together with the profile's cookies, so it can call the API directly without
ever rendering a page (no captcha surface).
"""

import json
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from uber_client import PROFILE, SESSION_FILE, filter_replay_headers  # noqa: E402

_captured: dict = {}


def _on_request(req) -> None:
    # Grab the headers off the first /_p/api/getFeed* request the site makes.
    if not _captured and "/_p/api/" in req.url and "getfeed" in req.url.lower():
        try:
            _captured.update(req.headers)
        except Exception:
            pass


def main() -> None:
    print("Opening Chrome for Uber Eats login…")
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
        page.on("request", _on_request)
        try:
            page.goto("https://www.ubereats.com/jp", wait_until="domcontentloaded")
            print("\n>>> Log in + set a Tokyo delivery address in the browser window.")
            print(">>> Waiting for the restaurant feed (capturing session headers)…")
            deadline = time.time() + 360
            while time.time() < deadline:
                if "/feed" in page.url and _captured:
                    break
                page.wait_for_timeout(1500)
            if not _captured:
                # Feed may have loaded from cache without an API call — force one.
                try:
                    page.goto("https://www.ubereats.com/jp", wait_until="domcontentloaded")
                    page.wait_for_timeout(3500)
                except Exception:
                    pass
            hdrs = filter_replay_headers(_captured)
            if not hdrs.get("x-csrf-token"):
                print(
                    "\n✗ Could not capture session headers. Make sure you reached the "
                    "restaurant feed (not the login/captcha page), then re-run."
                )
                return
            SESSION_FILE.write_text(
                json.dumps(
                    {"headers": hdrs, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\n✓ Saved {len(hdrs)} session headers → {SESSION_FILE.name}")
            print("  The MCP server can now browse Uber Eats. You can close this window.")
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
