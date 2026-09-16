"""Compact Japanese rendering of the weather data (~220 tokens for a full
JMA report). One shape, no parameters (あさひ 09-16: the weekly rides
after today/tomorrow instead of being a separate mode)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .areas import Resolution

_WD = "月火水木金土日"
_WCODE_HUNDREDS = {"1": "晴", "2": "曇", "3": "雨", "4": "雪"}
# JMA 警報・注意報 codes (bosai warning JSON); 32+ are 特別警報.
_WARN = {
    "02": "大雨警報",
    "03": "洪水警報",
    "04": "暴風警報",
    "05": "暴風雪警報",
    "06": "大雪警報",
    "07": "波浪警報",
    "08": "高潮警報",
    "10": "大雨注意報",
    "12": "大雪注意報",
    "13": "風雪注意報",
    "14": "雷注意報",
    "15": "強風注意報",
    "16": "波浪注意報",
    "17": "融雪注意報",
    "18": "洪水注意報",
    "19": "高潮注意報",
    "20": "濃霧注意報",
    "21": "乾燥注意報",
    "22": "なだれ注意報",
    "23": "低温注意報",
    "24": "霜注意報",
    "25": "着氷注意報",
    "26": "着雪注意報",
    "32": "暴風特別警報",
    "33": "大雨特別警報",
    "35": "暴風雪特別警報",
    "36": "大雪特別警報",
    "37": "波浪特別警報",
    "38": "高潮特別警報",
}
_NO_WARNING = ("解除", "発表警報・注意報はなし")
_DIR16 = "北 北北東 北東 東北東 東 東南東 南東 南南東 南 南南西 南西 西南西 西 西北西 北西 北北西".split()
# Open-Meteo / WMO weather codes → short Japanese.
_WMO = [
    (0, "快晴"),
    (1, "晴"),
    (2, "晴時々曇"),
    (3, "曇"),
    (45, "霧"),
    (48, "霧"),
    (51, "霧雨"),
    (53, "霧雨"),
    (55, "霧雨"),
    (56, "着氷性霧雨"),
    (57, "着氷性霧雨"),
    (61, "雨"),
    (63, "雨"),
    (65, "大雨"),
    (66, "着氷性の雨"),
    (67, "着氷性の雨"),
    (71, "雪"),
    (73, "雪"),
    (75, "大雪"),
    (77, "霧雪"),
    (80, "にわか雨"),
    (81, "にわか雨"),
    (82, "激しいにわか雨"),
    (85, "にわか雪"),
    (86, "にわか雪"),
    (95, "雷雨"),
    (96, "雷雨(雹)"),
    (99, "雷雨(雹)"),
]


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _day(s: str) -> str:
    t = _dt(s)
    return f"{t.day}日({_WD[t.weekday()]})"


def _squash(s: str) -> str:
    return re.sub(r"[\s　]+", "", s or "")


def _first_sentences(text: str, n: int = 3, cap: int = 220) -> str:
    parts = [p for p in _squash(text).split("。") if p]
    out = "。".join(parts[:n])
    if len(out) > cap:
        out = out[:cap].rstrip("、") + "…"
    return out + ("。" if out and not out.endswith("…") else "")


def weather_code_label(code: str) -> str:
    code = str(code or "")
    return _WCODE_HUNDREDS.get(code[:1], code) if code else "-"


def wmo_label(code: Any) -> str:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "-"
    label = "-"
    for k, v in _WMO:
        if c >= k:
            label = v
    return label


def active_warnings(
    warning: Dict[str, Any], class20: Optional[str], class10: str
) -> List[str]:
    """Names of warnings in force for the municipality (class20 row when
    present, else the class10 row)."""
    codes: List[str] = []
    wanted = {c for c in (class20, class10) if c}
    for at in warning.get("areaTypes") or []:
        for area in at.get("areas") or []:
            if area.get("code") not in wanted:
                continue
            for w in area.get("warnings") or []:
                code = str(w.get("code") or "")
                if code and w.get("status") not in _NO_WARNING and code not in codes:
                    codes.append(code)
            if class20 and area.get("code") == class20:
                break  # the municipality row is authoritative
    return [_WARN.get(c, f"コード{c}") for c in codes]


def _num(v: Any) -> Optional[float]:
    """AMeDAS values are [value, quality]; None when missing."""
    if isinstance(v, list):
        v = v[0] if v else None
    return float(v) if isinstance(v, (int, float)) else None


def _diff_note(value: Any, ref: Any, label: str) -> str:
    """「（前日比 -5）」 — empty when either side is unknown."""
    try:
        d = round(float(value) - float(ref))
    except (TypeError, ValueError):
        return ""
    return f"（{label} {'±0' if d == 0 else f'{d:+d}'}）"


_TS_WEATHER = {"晴れ": "晴", "くもり": "曇", "雨": "雨", "雪": "雪", "みぞれ": "みぞれ"}


def render_timeseries(ts: Optional[Dict[str, Any]]) -> List[str]:
    """3-hour rows as a small table with a header (あさひ 09-16: the
    one-liner read like precipitation chances):

        時系列:
         日時 天気 気温 風
         16日12時 雨 20℃ 北2m/s
    """
    if not ts:
        return []
    area = ts.get("areaTimeSeries") or {}
    point = ts.get("pointTimeSeries") or {}
    temps = {
        d.get("dateTime"): v
        for d, v in zip(point.get("timeDefines") or [], point.get("temperature") or [])
        if isinstance(d, dict)
    }
    winds = area.get("wind") or []
    rows: List[str] = []
    has_wind = False
    for k, (d, w) in enumerate(
        zip(area.get("timeDefines") or [], area.get("weather") or [])
    ):
        when = d.get("dateTime") if isinstance(d, dict) else None
        if not when:
            continue
        dt = _dt(when)
        temp = temps.get(when)
        cells = [
            f"{dt.day}日{dt.hour}時",
            _TS_WEATHER.get(w, w) or "-",
            f"{temp}℃" if temp not in (None, "") else "-",
        ]
        wind = winds[k] if k < len(winds) and isinstance(winds[k], dict) else None
        if wind and wind.get("direction") is not None:
            cells.append(f"{wind.get('direction', '')}{wind.get('speed', '')}m/s")
            has_wind = True
        rows.append(" " + " ".join(cells))
    if not rows:
        return []
    return ["時系列:", " 日時 天気 気温" + (" 風" if has_wind else "")] + rows


def render_observation(row: Optional[Dict[str, Any]], prev_temp: Any = None) -> str:
    if not row:
        return ""

    def val(k: str):
        v = row.get(k)
        return v[0] if isinstance(v, list) and v and v[0] is not None else None

    t = str(row.get("_time") or "")
    stamp = f"{t[8:10]}:{t[10:12]}" if len(t) >= 12 else ""
    bits = []
    if val("temp") is not None:
        note = ""
        if isinstance(prev_temp, (int, float)):
            note = f"（前日同時刻 {val('temp') - prev_temp:+.1f}）"
        bits.append(f"{val('temp')}℃{note}")
    if val("humidity") is not None:
        bits.append(f"湿度{val('humidity')}%")
    p = val("precipitation1h")
    if p:
        bits.append(f"1h雨量{p}mm")
    w = val("wind")
    if w is not None:
        d = val("windDirection")
        name = (
            _DIR16[int(d) - 1]
            if isinstance(d, (int, float)) and 1 <= int(d) <= 16
            else ""
        )
        bits.append(f"{name}{w}m/s")
    return f"実況 {stamp}: " + " ".join(bits) if bits else ""


def render_jma(
    res: Resolution,
    forecast: List[Dict[str, Any]],
    overview: Dict[str, Any],
    warning: Dict[str, Any],
    observation: Optional[Dict[str, Any]],
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """``extras``: prev_temp (yesterday, same time), y_max / y_min
    (yesterday's observed extremes) and timeseries (VPFD JSON) — all
    optional; each missing piece just drops its annotation."""
    extras = extras or {}
    b0 = forecast[0]
    ts_w, ts_p, ts_t = b0["timeSeries"][0], b0["timeSeries"][1], b0["timeSeries"][2]
    idx = next(
        (
            i
            for i, a in enumerate(ts_w["areas"])
            if a["area"].get("code") == res.class10
        ),
        0,
    )
    area_name = ts_w["areas"][idx]["area"].get("name", res.class10_name)
    report = _dt(b0["reportDatetime"])
    lines = [f"【天気】{res.display}（{area_name}・気象庁 {report:%H:%M}発表）"]
    if overview.get("text"):
        lines.append("概況: " + _first_sentences(overview["text"]))
    warns = active_warnings(warning, res.class20, res.class10)
    if warns:
        lines.append("注意報・警報: " + "・".join(warns))
    obs = render_observation(observation, extras.get("prev_temp"))
    if obs:
        lines.append(obs)

    weathers = ts_w["areas"][idx].get("weathers") or []
    pops = list(
        zip(
            ts_p["timeDefines"],
            (ts_p["areas"][idx] if idx < len(ts_p["areas"]) else ts_p["areas"][0]).get(
                "pops"
            )
            or [],
        )
    )
    tarea = ts_t["areas"][idx] if idx < len(ts_t["areas"]) else ts_t["areas"][0]
    temps = list(zip(ts_t["timeDefines"], tarea.get("temps") or []))
    days = ts_w["timeDefines"][:2]
    today_hi: Any = None
    for i, td in enumerate(days):
        d = _dt(td).date()
        label = "今日" if i == 0 else "明日"
        pop_txt = " ".join(
            f"{_dt(t).hour:02d}-{_dt(t).hour + 6:02d} {v}%"
            for t, v in pops
            if _dt(t).date() == d and v != ""
        )
        lo = [
            v for t, v in temps if _dt(t).date() == d and _dt(t).hour == 0 and v != ""
        ]
        hi = [
            v for t, v in temps if _dt(t).date() == d and _dt(t).hour == 9 and v != ""
        ]
        tparts = []
        if lo and (not hi or lo[0] != hi[0]):
            tparts.append(f"最低{lo[0]}℃")
        if hi:
            # Today against yesterday's OBSERVED max; tomorrow against
            # today's forecast (no observation to compare with yet).
            ref = extras.get("y_max") if i == 0 else today_hi
            diff_label = "前日比" if i == 0 else "今日比"
            tparts.append(f"最高{hi[0]}℃" + _diff_note(hi[0], ref, diff_label))
            if i == 0:
                today_hi = hi[0]
        seg = [
            f"{label} {_day(td)}: {_squash(weathers[i]) if i < len(weathers) else '-'}"
        ]
        if pop_txt:
            seg.append(f"降水 {pop_txt}")
        if tparts:
            seg.append(" ".join(tparts))
        lines.append(" | ".join(seg))

    lines.extend(render_timeseries(extras.get("timeseries")))

    if len(forecast) > 1 and forecast[1].get("timeSeries"):
        wts_w = forecast[1]["timeSeries"][0]
        wts_t = (
            forecast[1]["timeSeries"][1] if len(forecast[1]["timeSeries"]) > 1 else None
        )
        wa = wts_w["areas"][0]
        for a in wts_w["areas"]:
            if a["area"].get("code") in (res.class10, res.office):
                wa = a
                break
        wt = wts_t["areas"][0] if wts_t and wts_t.get("areas") else {}
        last_daily = _dt(days[-1]).date() if days else None
        week: List[str] = []
        for j, td in enumerate(wts_w["timeDefines"]):
            if last_daily and _dt(td).date() <= last_daily:
                continue
            code = (wa.get("weatherCodes") or [""] * 99)[j]
            pop = (wa.get("pops") or [""] * 99)[j]
            rel = (wa.get("reliabilities") or [""] * 99)[j]
            tmax = (wt.get("tempsMax") or [""] * 99)[j] or "-"
            tmin = (wt.get("tempsMin") or [""] * 99)[j] or "-"
            row = f" {_day(td)} {weather_code_label(code)} {tmax}/{tmin}℃"
            if pop != "":
                row += f" {pop}%"
            if rel in ("B", "C"):
                row += f" ({rel})"
            week.append(row)
        if week:
            lines.append("週間:")
            lines.append(" 日付 天気 最高/最低 降水確率 信頼度")
            lines.extend(week)
    return "\n".join(lines)


def render_open_meteo(place: str, data: Dict[str, Any]) -> str:
    lines = [f"【天気】{place}（Open-Meteo 予報モデル）"]
    cur = data.get("current") or {}
    if cur:
        bits = []
        if cur.get("temperature_2m") is not None:
            bits.append(f"{cur['temperature_2m']}℃")
        if cur.get("relative_humidity_2m") is not None:
            bits.append(f"湿度{cur['relative_humidity_2m']}%")
        if cur.get("weather_code") is not None:
            bits.append(wmo_label(cur["weather_code"]))
        if cur.get("wind_speed_10m") is not None:
            bits.append(f"風{cur['wind_speed_10m']}km/h")
        stamp = str(cur.get("time") or "")[11:16]
        lines.append(f"現在 {stamp}: " + " ".join(bits))
    d = data.get("daily") or {}
    times = d.get("time") or []
    for j, day in enumerate(times):
        label = "今日" if j == 0 else "明日" if j == 1 else ""
        tmax = (d.get("temperature_2m_max") or [None] * 99)[j]
        tmin = (d.get("temperature_2m_min") or [None] * 99)[j]
        pop = (d.get("precipitation_probability_max") or [None] * 99)[j]
        code = (d.get("weather_code") or [None] * 99)[j]
        row = f" {_day(day)} {wmo_label(code)} {tmax if tmax is not None else '-'}/{tmin if tmin is not None else '-'}℃"
        if pop is not None:
            row += f" {pop}%"
        lines.append((label + row) if label else row)
    return "\n".join(lines)
