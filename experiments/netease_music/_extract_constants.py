"""Re-derive the weapi crypto constants from NetEase's live web bundle and
diff them against what ``ncm_crypto.py`` hardcodes.

Run this if every weapi call suddenly starts failing — that is what a rotated
key looks like from the outside:

    uv run python experiments/netease_music/_extract_constants.py

How the constants are hidden (as of 2026-07): ``core_*.js`` defines a lookup
table ``nm.x.ek.emj`` mapping emoji names to 5-char hex fragments, plus three
name orders. ``md`` is the RSA modulus; the preset AES key and the exponent
are assembled from four and two more names at the ``window.asrsea(...)`` call
site. A second order named ``nmd`` is a decoy — it interleaves the preset-key
fragments into the modulus and yields a key the server rejects.

Two encoding traps, both real: the bundle mixes UTF-8 and GBK for the *same*
emoji names, so keys are matched on normalized bytes rather than decoded text;
and it contains at least one byte that is invalid in both codecs, so a strict
whole-file decode always fails.
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Dict, List

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ncm_crypto import _PRESET_KEY, _PUB_EXPONENT, _PUB_MODULUS  # noqa: E402

_HOME = "https://music.163.com/"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _canon(raw: bytes) -> str:
    """Encoding-agnostic key: the bundle stores the same names as UTF-8 in
    some places and GBK in others."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", "replace")


def _fetch_core_js() -> bytes:
    with httpx.Client(timeout=30, headers={"User-Agent": _UA}) as client:
        html = client.get(_HOME).text
        src = next(
            s for s in re.findall(r'src="([^"]+\.js[^"]*)"', html) if "/core_" in s
        )
        if src.startswith("//"):
            src = "https:" + src
        return client.get(src, headers={"Referer": _HOME}).content


def _table(js: bytes) -> Dict[str, str]:
    blob = re.search(rb"\.emj=\{(.*?)\};", js, re.S).group(1)
    return {
        _canon(k): v.decode()
        for k, v in re.findall(rb'"([^"]*)"\s*:\s*"([^"]*)"', blob)
    }


def _order(js: bytes, name: bytes) -> List[str]:
    blob = re.search(rb"\.%s=\[(.*?)\];" % name, js, re.S).group(1)
    return [_canon(k) for k in re.findall(rb'"([^"]*)"', blob)]


def main() -> int:
    js = _fetch_core_js()
    emj = _table(js)
    modulus = "".join(emj[k] for k in _order(js, b"md"))

    # The call site is `asrsea(json, <exponent>, <modulus>, <preset key>)`;
    # the exponent and preset key are spelled out as name lists there.
    call = re.search(rb"window\.asrsea\(.*?\)\);", js, re.S).group(0)
    lists = re.findall(rb"\[((?:\"[^\"]*\",?)+)\]", call)
    picked = [
        "".join(emj[_canon(n)] for n in re.findall(rb'"([^"]*)"', group))
        for group in lists
    ]
    exponent = next(p for p in picked if p == "010001")
    preset = next(p for p in picked if not re.fullmatch(r"[0-9a-f]+", p))

    ok = True
    for label, live, hardcoded in (
        ("modulus", modulus, f"{_PUB_MODULUS:x}".rjust(len(modulus), "0")),
        ("exponent", exponent, f"{_PUB_EXPONENT:06x}"),
        ("preset key", preset, _PRESET_KEY.decode()),
    ):
        same = live == hardcoded
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {label}: {live}")
        if not same:
            print(f"     hardcoded: {hardcoded}")
    print("\nAll constants match ncm_crypto.py." if ok else "\nUPDATE ncm_crypto.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
