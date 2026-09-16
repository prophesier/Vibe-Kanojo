"""WeatherService: one call → one compact Japanese report.

Resolution order for a place name: JMA area table (exact/kana/suffix/
prefix, ties broken by the home prefecture, true ambiguity returned as a
candidate list) → Open-Meteo geocoding (a Japanese hit is mapped back to
its prefecture's JMA file and the nearest forecast station picks the
column; a foreign hit gets Open-Meteo's own model forecast). Never raises:
every failure is a Japanese, actionable error dict."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from loguru import logger

from .areas import Ambiguous, AreaIndex, Resolution
from .client import JmaClient, OpenMeteoClient, WeatherUnavailable
from .render import _num, render_jma, render_open_meteo


class WeatherService:
    def __init__(
        self,
        home: str = "",
        cache_dir: str = "cache/weather",
        jma: Optional[JmaClient] = None,
        open_meteo: Optional[OpenMeteoClient] = None,
        areas: Optional[AreaIndex] = None,
        open_meteo_fallback: bool = True,
        home_hint: str = "",
    ) -> None:
        self._home = (home or "").strip()
        # How the tool schema names the default area ("home/school (Tokyo
        # area)"); read by the agent when it builds the schema.
        self.home_hint = (home_hint or "").strip()
        self._jma = jma or JmaClient(cache_dir)
        self._om = open_meteo or OpenMeteoClient()
        self._areas = areas
        self._fallback = open_meteo_fallback
        self._home_res: Optional[Resolution] = None
        self._index_lock = asyncio.Lock()

    # -- setup ------------------------------------------------------------------
    async def _ensure_index(self) -> AreaIndex:
        if self._areas is not None:
            return self._areas
        async with self._index_lock:
            if self._areas is None:
                area, amedas = await asyncio.gather(
                    self._jma.area_table(), self._jma.amedas_table()
                )
                self._areas = AreaIndex(area, amedas)
        return self._areas

    async def _home_resolution(self) -> Optional[Resolution]:
        if self._home_res is not None or not self._home:
            return self._home_res
        idx = await self._ensure_index()
        r = idx.resolve(self._home)
        if isinstance(r, Ambiguous):
            logger.warning(
                f"[weather] home {self._home!r} is ambiguous ({r.describe()}); "
                "prefix it with the prefecture in weather_config.home"
            )
            r = r.candidates[0]
        elif r is None:
            logger.warning(
                f"[weather] home {self._home!r} not found in the JMA area table"
            )
        self._home_res = r
        return r

    async def warmup(self) -> None:
        """Load the tables and resolve the home area ahead of the first call."""
        try:
            await self._ensure_index()
            r = await self._home_resolution()
            logger.info(
                "[weather] ready: home="
                + (
                    f"{r.display}/{r.office_name} ({r.office}:{r.class10})"
                    if r
                    else "(unset)"
                )
            )
        except Exception as e:
            logger.warning(f"[weather] warmup failed (will retry lazily): {e}")

    # -- the tool ---------------------------------------------------------------
    async def describe(self, area: Optional[str] = None) -> Dict[str, Any]:
        query = (area or "").strip()
        try:
            idx = await self._ensure_index()
            home = await self._home_resolution()
            if not query:
                if home is None:
                    return {
                        "status": "error",
                        "message": "自宅の地域が未設定のため、場所を指定してほしい（例: 東京都中央区）。",
                    }
                return await self._jma_report(idx, home)
            r = idx.resolve(query, prefer_office=home.office if home else None)
            if isinstance(r, Resolution):
                return await self._jma_report(idx, r)
            if isinstance(r, Ambiguous):
                return {
                    "status": "need_clarification",
                    "message": f"「{query}」は複数の地域に該当する: {r.describe()}。都道府県名を付けて指定してほしい。",
                    "candidates": [f"{c.office_name}{c.display}" for c in r.candidates],
                }
            return await self._geocode_report(idx, query)
        except WeatherUnavailable as e:
            logger.warning(f"[weather] source unavailable: {e}")
            return {
                "status": "error",
                "message": "天気データの取得に失敗した。少し待って再試行してほしい。",
            }
        except Exception as e:  # never break the tool loop
            logger.exception(f"[weather] unexpected failure for {query!r}: {e}")
            return {
                "status": "error",
                "message": "天気データの処理中に内部エラーが起きた。",
            }

    async def _jma_report(self, idx: AreaIndex, res: Resolution) -> Dict[str, Any]:
        forecast, overview, warning = await asyncio.gather(
            self._jma.forecast(res.office),
            self._jma.overview(res.office),
            self._jma.warning(res.office),
        )
        observation = None
        extras: Dict[str, Any] = {}
        station = _station_for(forecast, res.class10)
        if station:
            try:
                observation = await self._jma.amedas_latest(station)
                extras.update(await self._previous_day(station, observation))
            except WeatherUnavailable as e:
                logger.debug(f"[weather] amedas {station} unavailable: {e}")
        try:
            extras["timeseries"] = await self._jma.timeseries(res.class10)
        except WeatherUnavailable as e:
            logger.debug(f"[weather] timeseries {res.class10} unavailable: {e}")
        text = render_jma(res, forecast, overview, warning, observation, extras)
        return {"status": "ok", "area": res.display, "text": text}

    async def _previous_day(
        self, station: str, observation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Yesterday's observed max/min (from the last row of its 21:00
        block — the day's running extremes) and the temperature at the
        same clock time, for the 前日比 / 前日同時刻 annotations."""
        key = str((observation or {}).get("_time") or "")
        try:
            now = datetime.strptime(key, "%Y%m%d%H%M%S")
        except ValueError:
            return {}
        y = now - timedelta(days=1)
        out: Dict[str, Any] = {}
        last = await self._jma.amedas_block(station, y, 21)
        if last:
            row = last[max(last)]
            out["y_max"] = _num(row.get("maxTemp"))
            out["y_min"] = _num(row.get("minTemp"))
        rows = await self._jma.amedas_block(station, y, (y.hour // 3) * 3)
        if rows:
            ykey = f"{y:%Y%m%d}{key[8:]}"
            earlier = [k for k in rows if k <= ykey]
            if earlier:
                out["prev_temp"] = _num(rows[max(earlier)].get("temp"))
        return out

    async def _geocode_report(self, idx: AreaIndex, query: str) -> Dict[str, Any]:
        if not self._fallback:
            return _not_found(query)
        hits = await self._om.geocode(query)
        if not hits:
            return _not_found(query)
        hit = hits[0]
        label = hit["name"] + (f"（{hit['admin1']}）" if hit.get("admin1") else "")
        if hit.get("country") == "JP":
            office = idx.office_by_name(hit.get("admin1", ""))
            if office:
                forecast = await self._jma.forecast(office)
                res = _resolution_by_station(
                    idx, forecast, office, hit["lat"], hit["lon"], hit["name"]
                )
                if res is not None:
                    return await self._jma_report(idx, res)
        data = await self._om.forecast(hit["lat"], hit["lon"])
        return {"status": "ok", "area": label, "text": render_open_meteo(label, data)}


def _not_found(query: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "message": f"「{query}」の場所を特定できなかった。市区町村名か都道府県名で言い直してほしい。",
    }


def _station_for(forecast, class10: str) -> Optional[str]:
    """AMeDAS station of the class10 column (index-aligned with the temps
    areas of the 3-day block — verified on 4 offices, 09-16)."""
    try:
        ts = forecast[0]["timeSeries"]
        cols = [a["area"].get("code") for a in ts[0]["areas"]]
        i = cols.index(class10) if class10 in cols else 0
        temps = ts[2]["areas"]
        return (
            str(temps[i]["area"]["code"])
            if i < len(temps)
            else str(temps[0]["area"]["code"])
        )
    except (KeyError, IndexError, TypeError):
        return None


def _resolution_by_station(
    idx: AreaIndex, forecast, office: str, lat: float, lon: float, name: str
) -> Optional[Resolution]:
    try:
        ts = forecast[0]["timeSeries"]
        stations = [str(a["area"]["code"]) for a in ts[2]["areas"]]
        cols = ts[0]["areas"]
    except (KeyError, IndexError, TypeError):
        return None
    i = idx.nearest_station(lat, lon, stations)
    if i is None or i >= len(cols):
        return None
    col = cols[i]["area"]
    office_name = idx.office_name(office)
    return Resolution(
        office=office,
        office_name=office_name,
        class10=str(col.get("code")),
        class10_name=col.get("name", ""),
        class20=None,
        class20_name=name,
        level="geocode",
    )
