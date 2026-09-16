"""Place-name → JMA area-code resolution.

The Japan Meteorological Agency publishes its whole forecast-area tree as
one JSON (``bosai/common/const/area.json``): centers → offices (都道府県,
58) → class10 (予報区, 142) → class15 (375) → class20 (市区町村, ~1800, with
kana). A forecast file is keyed by OFFICE and carries one column per
class10, so any municipality resolves to "which office file, which class10
column". Temperature points are AMeDAS stations whose codes/coordinates
come from ``amedas/const/amedastable.json``.

Matching ladder (09-16 design, あさひ: unknown place names must work
without pre-known codes): exact name → kana → suffix-stripped
(市/区/町/村/都/道/府/県) → prefix. Candidates collapse by class10 — the
level the forecast actually varies at — so 横浜市北部/南部 (two class20
rows, one column) is NOT ambiguous, while 府中市 (東京都 vs 広島県) is and
comes back as a candidate list for the character to ask about.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LEVELS = ("offices", "class10s", "class15s", "class20s")
_PARENT_OF = {
    "class20s": "class15s",
    "class15s": "class10s",
    "class10s": "offices",
    "offices": "centers",
}
_SUFFIX_RE = re.compile(r"(都|道|府|県|市|区|町|村|郡|地方)$")


def normalize(s: str) -> str:
    """NFKC (full-width digits → ASCII), no whitespace, ヶ/ヵ unified."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[\s　]", "", s)
    return s.replace("ヶ", "ケ").replace("ヵ", "カ")


def to_hiragana(s: str) -> str:
    return "".join(
        chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in normalize(s)
    )


@dataclass
class Resolution:
    office: str
    office_name: str
    class10: str
    class10_name: str
    class20: Optional[str] = None
    class20_name: Optional[str] = None
    level: str = ""  # which level the query matched

    @property
    def display(self) -> str:
        return self.class20_name or self.class10_name

    @property
    def key(self) -> Tuple[str, str]:
        return (self.office, self.class10)


@dataclass
class Ambiguous:
    query: str
    candidates: List[Resolution] = field(default_factory=list)

    def describe(self) -> str:
        return " / ".join(f"{c.display}（{c.office_name}）" for c in self.candidates)


class AreaIndex:
    def __init__(self, area: Dict[str, Any], amedas: Dict[str, Any]) -> None:
        self._area = area
        self._amedas = amedas or {}
        self._by_name: Dict[str, List[Tuple[str, str]]] = {}
        self._by_kana: Dict[str, List[Tuple[str, str]]] = {}
        self._by_base: Dict[str, List[Tuple[str, str]]] = {}
        for lvl in _LEVELS:
            for code, e in (area.get(lvl) or {}).items():
                name = normalize(e.get("name", ""))
                if not name:
                    continue
                self._by_name.setdefault(name, []).append((lvl, code))
                base = _SUFFIX_RE.sub("", name)
                if base and base != name:
                    self._by_base.setdefault(base, []).append((lvl, code))
                kana = to_hiragana(e.get("kana", ""))
                if kana:
                    self._by_kana.setdefault(kana, []).append((lvl, code))
        self._office_names = {
            normalize(e["name"]): code
            for code, e in (area.get("offices") or {}).items()
        }

    # -- tree walking ---------------------------------------------------------
    def _entry(self, lvl: str, code: str) -> Dict[str, Any]:
        return (self._area.get(lvl) or {}).get(code) or {}

    def _chain(self, lvl: str, code: str) -> Dict[str, Tuple[str, str]]:
        """{level: (code, name)} from the matched node up to the office."""
        out: Dict[str, Tuple[str, str]] = {}
        while lvl in _LEVELS:
            e = self._entry(lvl, code)
            if not e:
                break
            out[lvl] = (code, e.get("name", ""))
            if lvl == "offices":
                break
            nxt = _PARENT_OF[lvl]
            code = e.get("parent", "")
            lvl = nxt
        return out

    def _resolution(self, lvl: str, code: str) -> Optional[Resolution]:
        chain = self._chain(lvl, code)
        if "offices" not in chain:
            return None
        office, office_name = chain["offices"]
        if "class10s" in chain:
            class10, class10_name = chain["class10s"]
        else:  # an office-level match: its first forecast column
            children = self._entry("offices", office).get("children") or []
            if not children:
                return None
            class10 = children[0]
            class10_name = self._entry("class10s", class10).get("name", "")
        c20 = chain.get("class20s")
        return Resolution(
            office=office,
            office_name=office_name,
            class10=class10,
            class10_name=class10_name,
            class20=c20[0] if c20 else None,
            class20_name=c20[1] if c20 else None,
            level=lvl,
        )

    # -- public API -----------------------------------------------------------
    def office_name(self, code: str) -> str:
        return self._entry("offices", code).get("name", "") or code

    def office_by_name(self, name: str) -> Optional[str]:
        n = normalize(name)
        return self._office_names.get(n) or self._office_names.get(n + "県")

    def resolve(self, query: str, prefer_office: Optional[str] = None):
        """→ Resolution | Ambiguous | None.

        A leading prefecture name ("広島県府中市") narrows the search to that
        office; ``prefer_office`` (the home prefecture) breaks remaining
        ties silently."""
        q = normalize(query)
        if not q:
            return None
        office_filter = None
        for oname, ocode in self._office_names.items():
            if q.startswith(oname) and len(q) > len(oname):
                office_filter, q = ocode, q[len(oname) :]
                break
        tiers: List[List[Tuple[str, str]]] = [
            self._by_name.get(q, []),
            self._by_kana.get(to_hiragana(q), []),
            self._by_base.get(q, []),
        ]
        if len(q) >= 2:
            tiers.append(
                [
                    hit
                    for name, hits in self._by_name.items()
                    if name.startswith(q) and name != q
                    for hit in hits
                ]
            )
        for hits in tiers:
            found: Dict[Tuple[str, str], Resolution] = {}
            for lvl, code in hits:
                r = self._resolution(lvl, code)
                if r is None or (office_filter and r.office != office_filter):
                    continue
                found.setdefault(r.key, r)
            if not found:
                continue
            if len(found) == 1:
                return next(iter(found.values()))
            if prefer_office:
                home = [r for r in found.values() if r.office == prefer_office]
                if len(home) == 1:
                    return home[0]
            return Ambiguous(query=query, candidates=list(found.values()))
        return None

    # -- AMeDAS stations ------------------------------------------------------
    def station_name(self, code: str) -> str:
        return (self._amedas.get(str(code)) or {}).get("kjName", "") or str(code)

    def station_latlon(self, code: str) -> Optional[Tuple[float, float]]:
        e = self._amedas.get(str(code)) or {}
        lat, lon = e.get("lat"), e.get("lon")
        if not (isinstance(lat, list) and isinstance(lon, list)):
            return None
        try:
            return (lat[0] + lat[1] / 60.0, lon[0] + lon[1] / 60.0)
        except (TypeError, IndexError):
            return None

    def nearest_station(
        self, lat: float, lon: float, codes: Sequence[str]
    ) -> Optional[int]:
        """Index into ``codes`` of the station nearest to (lat, lon)."""
        best, best_d = None, math.inf
        for i, code in enumerate(codes):
            ll = self.station_latlon(code)
            if ll is None:
                continue
            d = math.hypot(ll[0] - lat, (ll[1] - lon) * math.cos(math.radians(lat)))
            if d < best_d:
                best, best_d = i, d
        return best
