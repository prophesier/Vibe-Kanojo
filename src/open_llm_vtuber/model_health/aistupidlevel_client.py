"""Read-only client for aistupidlevel.info's public API (no auth).

Tracks per-model benchmark scores so we can detect degradation the site's own
alert feed misses. Field shapes confirmed live 2026-07-06:

- ``GET /dashboard/scores?period=latest&sortBy=<combined|coding|reasoning|tooling>``
  → ``data[]`` of model cards. Score is ``currentScore`` (a blended/uncertainty-
  weighted estimate; falls back to ``score``); ``status`` ∈ {good, warning,
  critical}; ``isStale`` marks a repeat of the previous batch (skip it).
- ``GET /models/{id}/history?period=<7d|30d>`` → points live under the
  ``history`` key (NOT ``data``); the score field is ``displayScore`` (== the
  raw per-run "stupidScore"), one ``suite`` ("hourly"). ``currentScore`` from
  /scores can diverge from the latest ``displayScore`` because it blends more
  than the hourly suite — so the detector keeps its OWN currentScore series and
  only bootstraps the baseline from history.
- ``GET /dashboard/batch-status`` → ``data.{nextScheduledRun, ...}``.

Model ids move with versions (opus-4-8 was 268, opus-4-6 was 220) — always
resolve by ``name`` prefix, never hardcode an id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

_BASE = "https://aistupidlevel.info/api"
_UA = "VibeKanojo-ModelHealth/0.1 (+https://github.com/; degradation-monitor)"
_TIMEOUT = 20.0


class AiStupidLevelUnavailable(Exception):
    """Any failure to reach/parse the API. ``str(e)`` is short and safe."""


@dataclass
class ModelScore:
    """One model card from ``/dashboard/scores`` for a single ``sortBy`` axis."""

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

    async def fetch_scores(
        self, sort_by: str = "combined", period: str = "latest"
    ) -> List[ModelScore]:
        """All model cards for one axis (combined/coding/reasoning/tooling).

        ``period="7d"`` additionally returns ``periodAvg`` (the 7-day average
        currentScore) and ``stability``, and ``trend`` becomes the 7-day trend
        (more meaningful than the latest-only trend for spotting degradation)."""
        data = await self._get(f"dashboard/scores?period={period}&sortBy={sort_by}")
        rows = (data or {}).get("data") or []
        return [ModelScore.from_json(r) for r in rows if isinstance(r, dict)]

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
