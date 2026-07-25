"""Scan spike-era request captures for the original uev2.loc cookie value.

Prints ONLY the decoded location payload (address/coords/reference) — never
the session tokens that share the cookie header. Saves the raw cookie value to
_original_loc_cookie.txt for the repin probe.
"""

import json
import pathlib
import re
import sys
from urllib.parse import unquote

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "_original_loc_cookie.txt"

CANDIDATES = list((HERE / "out" / "network").glob("*.req.json")) + [
    HERE / "search_probe_out" / "search_request.json",
    HERE / "out" / "network" / "REPLAY.json",
]

pat = re.compile(r"uev2\.loc=([^;\"\\]+)")

for f in CANDIDATES:
    if not f.exists():
        continue
    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    m = pat.search(text)
    if not m:
        continue
    raw = m.group(1)
    try:
        loc = json.loads(unquote(raw))
    except Exception as e:
        print(f"[{f.name}] found but decode failed: {e}")
        continue
    print(f"[{f.name}] uev2.loc decoded — top keys: {sorted(loc.keys())}")
    addr = loc.get("address") or {}
    safe = {
        "address.title": addr.get("title"),
        "address.address1": addr.get("address1"),
        "address.subtitle": addr.get("subtitle"),
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
        "reference": (str(loc.get("reference"))[:50] if loc.get("reference") else None),
        "referenceType": loc.get("referenceType"),
        "type": loc.get("type"),
        "source": loc.get("source"),
        "addressId": loc.get("addressId") or loc.get("id"),
    }
    print(json.dumps({k: v for k, v in safe.items() if v}, ensure_ascii=False, indent=2))
    OUT.write_text(raw, encoding="utf-8")
    print(f"→ raw cookie value saved to {OUT.name}")
    sys.exit(0)

print("No uev2.loc found in any capture.")
