"""Scan the spike's REPLAY.json / page dumps for the full eater-location
payload (address + coordinates + place reference) around 東京駅.

Prints structure info and location payloads only — no tokens.
"""

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def walk(node, path=""):
    """Yield (path, dict) for every dict in the tree."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node[:200]):
            yield from walk(v, f"{path}[{i}]")


def scan_json_tree(data, label):
    hits = 0
    for path, d in walk(data):
        # A location payload: has coordinates and mentions 東京駅 nearby.
        if "latitude" in d and "longitude" in d:
            blob = json.dumps(d, ensure_ascii=False)
            if "東京駅" in blob and len(blob) < 4000:
                hits += 1
                print(f"\n[{label}] location-like dict at {path}:")
                print(blob[:1500])
                if hits >= 3:
                    return
    if not hits:
        print(f"[{label}] no 東京駅 location dicts found")


replay = HERE / "out" / "network" / "REPLAY.json"
if replay.exists():
    try:
        data = json.loads(replay.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, list):
            print(f"[REPLAY.json] list of {len(data)} entries; keys of first:",
                  sorted(data[0].keys()) if data else "-")
            urls = []
            for e in data:
                if isinstance(e, dict):
                    u = e.get("url") or ""
                    if "/_p/api/" in u:
                        urls.append(u.split("/_p/api/")[1].split("?")[0])
            print("[REPLAY.json] api endpoints seen:", sorted(set(urls)))
        else:
            print("[REPLAY.json] top-level type:", type(data).__name__,
                  "keys:", sorted(data.keys())[:15] if isinstance(data, dict) else "")
        scan_json_tree(data, "REPLAY.json")
    except Exception as e:
        print("[REPLAY.json] parse failed:", e)

# The SSR page dump: pull JSON <script> blobs that mention 東京駅.
html = (HERE / "out" / "search.html").read_text(encoding="utf-8", errors="replace")
for m in re.finditer(r"<script[^>]*>(.*?)</script>", html, re.S):
    s = m.group(1)
    if "東京駅" not in s:
        continue
    print(f"\n[search.html] script blob with 東京駅, {len(s)} chars; head:")
    idx = s.find("東京駅")
    print(s[max(0, idx - 600) : idx + 200].replace("\\u0022", '"')[:900])
    break
else:
    print("[search.html] no script blob contains 東京駅 directly")
sys.exit(0)
