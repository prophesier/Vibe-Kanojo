"""Read-only client for aistupidlevel.info's public API (no auth).

Tracks per-model benchmark scores so we can detect degradation the site's own
alert feed misses. Endpoint reality as of 2026-09-16 — the site moved
``/dashboard/scores`` and ``/dashboard/alerts`` behind a free API key at
``/api/v1``; the old paths answer 401 ``api_key_required``, which silently
killed the monitor from 09-04 until this rewrite:

- ``GET /dashboard/cached?period=<latest|7d>&sortBy=<axis>&analyticsPeriod=<same>``
  — still keyless (it is what the dashboard page itself renders from) and the
  ONE request we now make per axis. ``data.modelScores[]`` are the model cards
  with the same fields the retired ``/dashboard/scores`` returned:
  ``currentScore``, ``status`` ∈ {excellent, good, warning, critical},
  ``trend``, ``isStale``, and with ``period=7d`` the aggregates ``periodAvg``
  / ``stability`` / ``dataPoints`` (``trend`` becomes the 7-day trend).
  ``data.historyMap[model_id]`` is the per-point timeline the chart plots
  (newest first; each point carries a ``suite`` — ``hourly`` for
  combined/coding, ``deep`` for reasoning, ``tooling`` for tooling). The
  timeline is the same 7-day window whichever ``period`` is asked for; only
  the card blend differs (``latest`` shows a higher, differently blended
  score — never mix the two).
- ``GET /models/{id}/history?period=<7d|30d>`` → points under ``history``
  (NOT ``data``), score field ``displayScore``; over half of a 7d window is
  ``[SYNTHETIC]`` backfill. Still keyless; kept for ad-hoc use only.
- ``GET /dashboard/batch-status`` → ``data.{nextScheduledRun, ...}``.

Model ids move with versions (opus-4-8 was 268, opus-4-6 was 220, opus-5 is
281 today) — always resolve by ``name`` prefix, never hardcode an id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

_BASE = "https://aistupidlevel.info/api"
_UA = "VibeKanojo-ModelHealth/0.1 (+https://github.com/; degradation-monitor)"
_TIMEOUT = 20.0


class AiStupidLevelUnavailable(Exception):
    """Any failure to reach/parse the API. ``str(e)`` is short and safe."""


@dataclass
class ModelScore:
    """One model card (``data.modelScores[]`` of /dashboard/cached) for one axis."""

    id: str
    name: str
    provider: str
    current_score: Optional[float]
    status: str
    trend: str
    is_stale: bool
    stale_duration: Optional[int]
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    standard_error: Optional[float]
    last_updated: str
    # Only present when fetched with period != "latest" (e.g. "7d"): the real
    # N-day average currentScore (the model's typical level) and a stability
    # score. These give an immediate, real baseline — no need to track from
    # startup or lean on the (heavily synthetic) displayScore history.
    period_avg: Optional[float] = None
    stability: Optional[float] = None
    data_points: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ModelScore":
        def num(v: Any) -> Optional[float]:
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        score = d.get("currentScore")
        if score is None:
            score = d.get("score")
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            provider=str(d.get("provider", "")),
            current_score=num(score),
            status=str(d.get("status", "") or ""),
            trend=str(d.get("trend", "") or ""),
            is_stale=bool(d.get("isStale")),
            stale_duration=d.get("staleDuration"),
            confidence_lower=num(d.get("confidenceLower")),
            confidence_upper=num(d.get("confidenceUpper")),
            standard_error=num(d.get("standardError")),
            last_updated=str(d.get("lastUpdated", "") or ""),
            period_avg=num(d.get("periodAvg")),
            stability=num(d.get("stability")),
            data_points=d.get("dataPoints"),
            raw=d,
        )


@dataclass
class HistoryPoint:
    timestamp: str
    score: Optional[float]  # displayScore
    axes: Dict[str, Any] = field(default_factory=dict)
    # Over half of a 7d window is "[SYNTHETIC] Generated from …" backfill, not a
    # real benchmark run — the detector excludes these from the baseline.
    synthetic: bool = False


class AiStupidLevelClient:
    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def _get(self, path: str) -> Dict[str, Any]:
        url = f"{_BASE}/{path.lstrip('/')}"
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            raise AiStupidLevelUnavailable(
                f"aistupidlevel HTTP {e.response.status_code}"
            ) from e
        except Exception as e:  # network, JSON, timeout
            raise AiStupidLevelUnavailable(f"aistupidlevel unreachable: {e}") from e

    async def fetch_dashboard(
        self, sort_by: str = "combined", period: str = "7d"
    ) -> Tuple[List[ModelScore], Dict[str, List[float]]]:
        """One axis in one request: ``(model cards, {model_id: [scores]})``.

        Cards come from ``data.modelScores`` (``period="7d"`` adds
        ``periodAvg``/``stability``/``dataPoints`` and makes ``trend`` the
        7-day trend — the baseline the detector and the self-check report
        want); the series from ``data.historyMap`` (see
        _series_from_history_map). Both consumers call this once per axis."""
        data = await self._get(
            f"dashboard/cached?period={period}&sortBy={sort_by}"
            f"&analyticsPeriod={period}"
        )
        payload = (data or {}).get("data") or {}
        cards = [
            ModelScore.from_json(r)
            for r in (payload.get("modelScores") or [])
            if isinstance(r, dict)
        ]
        return cards, self._series_from_history_map(payload.get("historyMap") or {})

    async def fetch_scores(
        self, sort_by: str = "combined", period: str = "7d"
    ) -> List[ModelScore]:
        """Cards half of fetch_dashboard (same full request underneath — a
        caller that also wants the series should call fetch_dashboard)."""
        return (await self.fetch_dashboard(sort_by, period))[0]

    async def fetch_history(
        self, model_id: str, period: str = "7d"
    ) -> List[HistoryPoint]:
        """Raw per-run score series for one model. Points are under ``history``;
        the score is ``displayScore``. Sorted oldest→newest."""
        data = await self._get(f"models/{model_id}/history?period={period}")
        rows = (data or {}).get("history") or []
        out: List[HistoryPoint] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            s = r.get("displayScore")
            if s is None:
                s = r.get("stupidScore")
            try:
                score = float(s) if s is not None else None
            except (TypeError, ValueError):
                score = None
            out.append(
                HistoryPoint(
                    timestamp=str(r.get("timestamp", "")),
                    score=score,
                    axes=r.get("axes") or {},
                    synthetic=str(r.get("note", "")).startswith("[SYNTHETIC]"),
                )
            )
        out.sort(key=lambda p: p.timestamp)
        return out

    async def fetch_series(
        self, sort_by: str = "combined", period: str = "7d"
    ) -> Dict[str, List[float]]:
        """Series half of fetch_dashboard (same full request underneath)."""
        return (await self.fetch_dashboard(sort_by, period))[1]

    @staticmethod
    def _series_from_history_map(
        history_map: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        """``{model_id: [scores]}`` from ``data.historyMap`` — the real per-model
        TIMELINE the dashboard chart plots (the plain ``/history`` endpoint is
        >½ synthetic and diverges). Each point has a ``suite``: keep ``hourly``
        (what the COMBINED/CODING chart shows) when present, else all points
        (reasoning uses ``deep``, tooling uses ``tooling``)."""
        out: Dict[str, List[float]] = {}
        for mid, pts in history_map.items():
            if not isinstance(pts, list):
                continue

            def _scores(points, only_hourly):
                vals = []
                for p in points:
                    if not isinstance(p, dict):
                        continue
                    if only_hourly and p.get("suite") != "hourly":
                        continue
                    s = p.get("score")
                    if isinstance(s, (int, float)):
                        vals.append(float(s))
                return vals

            series = _scores(pts, True) or _scores(pts, False)
            if series:
                out[str(mid)] = series
        return out

    async def fetch_batch_status(self) -> Dict[str, Any]:
        data = await self._get("dashboard/batch-status")
        return (data or {}).get("data") or {}

    @staticmethod
    def find(scores: List[ModelScore], name_prefix: str) -> Optional[ModelScore]:
        """First card whose name starts with ``name_prefix`` (case-insensitive).
        Prefers a non-stale, exact-name match, then any match."""
        pref = (name_prefix or "").lower().strip()
        if not pref:
            return None
        matches = [s for s in scores if s.name.lower().startswith(pref)]
        if not matches:
            return None
        # exact name beats a longer-versioned sibling; fresh beats stale
        matches.sort(key=lambda s: (s.name.lower() != pref, s.is_stale, s.name))
        return matches[0]
