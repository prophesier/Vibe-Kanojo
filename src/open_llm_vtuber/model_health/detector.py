"""Degradation detector + state machine.

Fed one model's snapshot across all 4 axes (combined/reasoning/coding/tooling),
each with its CURRENT value + the scraped timeline series. We compute the
mean/std OURSELVES per axis from the series (the dashboard chart's real points
via /dashboard/cached), so the σ is honest and no from-startup accumulation is
needed. One warning line (``floor``). Signals (any → degraded):

- **A**: the site ``status`` is warning/critical, or ANY axis current < ``floor``.
- **P**: the combined current sits ``cur_drop`` below its own (self-computed) mean.
- **T**: the site 7-day trend is down AND combined is below its mean.

State per model: ``OK`` → ``DEGRADED`` (alert) → ``ESCALATED`` (alert again only
after a further ``escalate_delta`` drop) → ``RECOVERED`` (after ``recover_streak``
clean batches) → ``OK``. Deduped on the batch timestamp so re-polling never
re-fires. A separate OK↔DEGRADED machine tracks Anthropic's official status.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .anthropic_status_client import AnthropicStatus


@dataclass
class DetectorParams:
    floor: float = 60.0    # the single warning line (a score below this is flagged)
    cur_drop: float = 8.0  # P: this far below the axis's own mean = degraded
    recover_streak: int = 2
    escalate_delta: float = 6.0    # further combined-score drop to re-alert


@dataclass
class AxisStat:
    """One axis's current value + the mean/std WE compute from the scraped
    timeline series (the dashboard chart's points)."""

    current: Optional[float] = None
    status: str = ""
    trend: str = ""
    mean: Optional[float] = None
    std: Optional[float] = None
    n: int = 0  # points the mean/std used


@dataclass
class DegradationEvent:
    model: str
    provider: str
    event: str        # DEGRADED | ESCALATED | RECOVERED
    severity: str     # warning | critical
    current_score: Optional[float]  # combined
    status: str
    trend: str
    baseline: Optional[float]  # combined mean (computed from the scraped series)
    drop: Optional[float]      # baseline - current
    reasons: List[str]
    detected_at: str
    source_url: str
    floor: Optional[float] = None
    # axis name -> AxisStat (combined/reasoning/coding/tooling)
    axes: Dict[str, AxisStat] = field(default_factory=dict)


@dataclass
class OfficialStatusEvent:
    """A transition in Anthropic's OFFICIAL status (independent of the benchmark
    signals). ``degraded_performance`` is the priority case — subtler than a full
    outage and easy to miss."""

    event: str        # DEGRADED | ESCALATED | RECOVERED
    indicator: str    # none | minor | major | critical | maintenance
    claude_api_status: str
    degraded_components: List[str]
    incidents: List[str]  # unresolved incident names
    latest_body: str
    detected_at: str


def _median(xs: List[float]) -> Optional[float]:
    """Median of non-None values (used by the self-check report too)."""
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


class Detector:
    def __init__(
        self,
        state_path: pathlib.Path,
        params: Optional[DetectorParams] = None,
    ) -> None:
        self._path = pathlib.Path(state_path)
        self._params = params or DetectorParams()
        self._state: Dict[str, Dict[str, Any]] = {}
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        try:
            if self._path.exists():
                self._state = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[model_health] detector state unreadable ({e}); reset")
            self._state = {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[model_health] could not persist detector state: {e}")

    @staticmethod
    def axis_stats(current, status, trend, series: List[float]) -> AxisStat:
        """Build an AxisStat: current + mean/std computed from the scraped series."""
        vals = [s for s in (series or []) if isinstance(s, (int, float))]
        mean = statistics.fmean(vals) if vals else None
        std = statistics.pstdev(vals) if len(vals) >= 2 else (0.0 if vals else None)
        return AxisStat(
            current=current, status=status or "", trend=trend or "",
            mean=mean, std=std, n=len(vals),
        )

    # -- core ---------------------------------------------------------------
    def evaluate(
        self,
        model: Dict[str, Any],
        now_iso: str,
    ) -> Optional[DegradationEvent]:
        """Feed one watched model's snapshot across all 4 axes, each with its
        CURRENT value + the scraped timeline series:

            {"name","provider","axes": {"combined": {"current","status","trend",
             "period_avg","series":[...],"last_updated"}, "reasoning": {...}, ...}}

        We compute the mean/std OURSELVES per axis from the series (the dashboard
        chart's real points), so the σ is honest and no from-startup accumulation
        is needed. One warning line (``floor``). Returns a DegradationEvent on a
        state transition, else None.
        """
        name = model.get("name") or ""
        raw = model.get("axes") or {}
        combined = raw.get("combined")
        if not name or not combined:
            return None
        p = self._params
        st = self._state.setdefault(
            name, {"state": "OK", "last_batch": "", "last_event_current": None,
                   "recover_streak": 0},
        )
        # Dedup once per batch timestamp; the state machine prevents re-alerting.
        batch = combined.get("last_updated") or ""
        if batch and batch == st.get("last_batch"):
            return None
        st["last_batch"] = batch

        # Per-axis current + self-computed mean/std.
        axes: Dict[str, AxisStat] = {}
        for ax in ("combined", "reasoning", "coding", "tooling"):
            d = raw.get(ax)
            if not d:
                continue
            axes[ax] = self.axis_stats(
                d.get("current"), d.get("status"), d.get("trend"), d.get("series") or []
            )

        c = axes["combined"]
        cur, mean = c.current, c.mean
        trend = (c.trend or "").lower()
        reasons: List[str] = []

        # A — site status / warning line (combined + any sub-axis below the line)
        if (c.status or "").lower() in ("warning", "critical"):
            reasons.append(f"A:status={c.status}")
        for ax, s in axes.items():
            if s.current is not None and s.current < p.floor:
                reasons.append(f"A:{ax} {s.current:.0f}<{p.floor:.0f}")

        # P — combined below its own mean by cur_drop
        drop = (mean - cur) if (cur is not None and mean is not None) else None
        if drop is not None and drop >= p.cur_drop:
            reasons.append(f"P:{cur:.0f}≤mean {mean:.0f}−{p.cur_drop:.0f}")

        # T — 7-day trend down and below mean
        if trend == "down" and cur is not None and mean is not None and cur < mean:
            reasons.append(f"T:↓ {cur:.0f}<mean {mean:.0f}")

        degraded = bool(reasons)
        severity = "critical" if (c.status or "").lower() == "critical" else "warning"

        event = self._transition(st, degraded, cur, severity, p)
        self._save()
        if event is None:
            return None
        return DegradationEvent(
            model=name,
            provider=model.get("provider") or "",
            event=event,
            severity=severity,
            current_score=cur,
            status=c.status,
            trend=c.trend,
            baseline=mean,
            drop=drop,
            reasons=reasons,
            detected_at=now_iso,
            floor=p.floor,
            axes=axes,
            source_url=f"https://aistupidlevel.info/?model={name}",
        )

    def _transition(
        self,
        st: Dict[str, Any],
        degraded: bool,
        cur: Optional[float],
        severity: str,
        p: DetectorParams,
    ) -> Optional[str]:
        state = st.get("state", "OK")
        if state == "OK":
            if degraded:
                st["state"] = "DEGRADED"
                st["last_event_current"] = cur
                st["recover_streak"] = 0
                return "DEGRADED"
            return None
        # state == DEGRADED
        if degraded:
            st["recover_streak"] = 0
            prev = st.get("last_event_current")
            if cur is not None and prev is not None and (prev - cur) >= p.escalate_delta:
                st["last_event_current"] = cur
                return "ESCALATED"
            return None
        # not degraded while DEGRADED → count clean batches toward recovery
        st["recover_streak"] = st.get("recover_streak", 0) + 1
        if st["recover_streak"] >= p.recover_streak:
            st["state"] = "OK"
            st["last_event_current"] = None
            st["recover_streak"] = 0
            return "RECOVERED"
        return None

    def state_of(self, model_name: str) -> str:
        return self._state.get(model_name, {}).get("state", "OK")

    # -- official Anthropic status ------------------------------------------
    _OFFICIAL_KEY = "__anthropic_official__"

    def evaluate_official(
        self, status: AnthropicStatus, now_iso: str
    ) -> Optional[OfficialStatusEvent]:
        """Track Anthropic's official status with an OK↔DEGRADED state machine.
        Alerts once when it goes degraded, again if the incident set changes
        (ESCALATED), and once when it clears (RECOVERED). A fetch failure
        (``ok=False``) is treated as "unknown" — never alerts on it."""
        if not status.ok:
            return None
        st = self._state.setdefault(self._OFFICIAL_KEY, {"degraded": False, "sig": ""})
        deg = status.is_degraded
        incidents = [i.name for i in status.unresolved_incidents]
        sig = "|".join(
            [
                status.indicator,
                status.claude_api_status,
                ",".join(sorted(status.degraded_components)),
                ",".join(sorted(incidents)),
            ]
        )
        prev_deg, prev_sig = st.get("degraded", False), st.get("sig", "")
        event: Optional[str] = None
        if deg and not prev_deg:
            event = "DEGRADED"
        elif deg and prev_deg and sig != prev_sig:
            event = "ESCALATED"
        elif not deg and prev_deg:
            event = "RECOVERED"
        st["degraded"], st["sig"] = deg, sig
        self._save()
        if event is None:
            return None
        return OfficialStatusEvent(
            event=event,
            indicator=status.indicator,
            claude_api_status=status.claude_api_status,
            degraded_components=status.degraded_components,
            incidents=incidents,
            latest_body=(
                status.unresolved_incidents[0].latest_body
                if status.unresolved_incidents
                else ""
            ),
            detected_at=now_iso,
        )
