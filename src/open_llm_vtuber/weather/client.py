"""HTTP clients for the weather tool: JMA's keyless bosai JSON (primary)
and Open-Meteo (geocoding + global fallback). Both are read-only, need no
key, and answer in a few KB.

JMA endpoints (undocumented but what jma.go.jp itself renders from; stable
since 2021, verified 2026-09-16):
  forecast/data/forecast/{office}.json          3-day + weekly, per office
  forecast/data/overview_forecast/{office}.json  Japanese 概況 text
  warning/data/warning/{office}.json             warnings per class10/class20
  amedas/data/point/{station}/{YYYYMMDD}_{HH}.json  10-min observations,
      one file per 3-hour block (HH ∈ 00,03,…,21); the current block's file
      appears a few minutes into the block, so fall back to the previous one
  common/const/area.json, amedas/const/amedastable.json  static tables
      (262 KB / 188 KB) — cached on disk for a month.
Every call goes through a short in-memory TTL cache so a chatty turn never
hits the agency twice for the same thing."""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

JST = timezone(timedelta(hours=9))
_JMA = "https://www.jma.go.jp/bosai"
_UA = "VibeKanojo-Weather/0.1 (+https://github.com/prophesier/Vibe-Kanojo)"
_TIMEOUT = 15.0
_TABLE_TTL_S = 30 * 24 * 3600


class WeatherUnavailable(Exception):
    """Any failure to reach/parse a source. ``str(e)`` is short and safe."""


class _TTLCache:
    def __init__(self, ttl_s: float) -> None:
        self._ttl = ttl_s
        self._d: Dict[str, tuple] = {}

    def get(self, key: str) -> Any:
        hit = self._d.get(key)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        return None

    def put(self, key: str, value: Any) -> Any:
        self._d[key] = (time.monotonic(), value)
        return value


async def _get_json(url: str, timeout: float = _TIMEOUT) -> Any:
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _UA, "Accept": "application/json"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise WeatherUnavailable(f"HTTP {e.response.status_code}: {url}") from e
    except Exception as e:  # network, JSON, timeout
        raise WeatherUnavailable(f"{type(e).__name__}: {url}") from e


class JmaClient:
    def __init__(
        self, cache_dir: str | pathlib.Path = "cache/weather", ttl_s: float = 600
    ) -> None:
        self._cache_dir = pathlib.Path(cache_dir)
        self._cache = _TTLCache(ttl_s)
        self._lock = asyncio.Lock()

    async def _cached(self, path: str) -> Any:
        url = f"{_JMA}/{path}"
        hit = self._cache.get(url)
        if hit is not None:
            return hit
        return self._cache.put(url, await _get_json(url))

    async def forecast(self, office: str) -> List[Dict[str, Any]]:
        data = await self._cached(f"forecast/data/forecast/{office}.json")
        if not isinstance(data, list) or not data:
            raise WeatherUnavailable(f"forecast {office}: unexpected shape")
        return data

    async def overview(self, office: str) -> Dict[str, Any]:
        data = await self._cached(f"forecast/data/overview_forecast/{office}.json")
        return data if isinstance(data, dict) else {}

    async def warning(self, office: str) -> Dict[str, Any]:
        data = await self._cached(f"warning/data/warning/{office}.json")
        return data if isinstance(data, dict) else {}

    async def amedas_block(
        self, station: str, day: datetime, hh: int
    ) -> Optional[Dict[str, Any]]:
        """One 3-hour file of 10-minute rows ({timestamp: row}); None when
        the block has not been published yet (404)."""
        path = f"amedas/data/point/{station}/{day:%Y%m%d}_{hh:02d}.json"
        try:
            data = await self._cached(path)
        except WeatherUnavailable as e:
            if "HTTP 404" in str(e):
                return None
            raise
        return data if isinstance(data, dict) and data else None

    async def amedas_latest(
        self, station: str, now: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Newest 10-minute observation row for ``station`` (or None)."""
        now = now or datetime.now(JST)
        for t in (now, now - timedelta(hours=3)):
            data = await self.amedas_block(station, t, (t.hour // 3) * 3)
            if data:
                key = max(data)
                row = dict(data[key])
                row["_time"] = key
                return row
        return None

    async def timeseries(self, class10: str) -> Dict[str, Any]:
        """3-hourly weather/wind/temperature for the class10 column over the
        next ~36 h (the site's 時系列予報; ``jmatile/data/wdist/VPFD``)."""
        data = await self._cached(f"jmatile/data/wdist/VPFD/{class10}.json")
        return data if isinstance(data, dict) else {}

    # -- static tables (disk-cached) -----------------------------------------
    async def _table(self, name: str, path: str) -> Dict[str, Any]:
        f = self._cache_dir / name
        try:
            if f.exists() and time.time() - f.stat().st_mtime < _TABLE_TTL_S:
                return json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[weather] cached table {name} unreadable: {e}")
        async with self._lock:
            data = await _get_json(f"{_JMA}/{path}", timeout=30.0)
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[weather] could not cache {name}: {e}")
            return data

    async def area_table(self) -> Dict[str, Any]:
        return await self._table("area.json", "common/const/area.json")

    async def amedas_table(self) -> Dict[str, Any]:
        return await self._table("amedastable.json", "amedas/const/amedastable.json")


class OpenMeteoClient:
    """Geocoding (any language) + a global daily forecast. Keyless."""

    def __init__(self, ttl_s: float = 600) -> None:
        self._cache = _TTLCache(ttl_s)

    async def _cached(self, url: str) -> Any:
        hit = self._cache.get(url)
        if hit is not None:
            return hit
        return self._cache.put(url, await _get_json(url))

    async def geocode(self, name: str, count: int = 5) -> List[Dict[str, Any]]:
        q = httpx.QueryParams(
            {"name": name, "count": count, "language": "ja", "format": "json"}
        )
        data = await self._cached(f"https://geocoding-api.open-meteo.com/v1/search?{q}")
        out = []
        for r in (data or {}).get("results") or []:
            try:
                out.append(
                    {
                        "name": r.get("name", ""),
                        "admin1": r.get("admin1", ""),
                        "country": r.get("country_code", ""),
                        "lat": float(r["latitude"]),
                        "lon": float(r["longitude"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return out

    async def forecast(self, lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
        q = httpx.QueryParams(
            {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "timezone": "auto",
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "forecast_days": days,
            }
        )
        data = await self._cached(f"https://api.open-meteo.com/v1/forecast?{q}")
        if not isinstance(data, dict) or "daily" not in data:
            raise WeatherUnavailable("open-meteo: unexpected shape")
        return data
