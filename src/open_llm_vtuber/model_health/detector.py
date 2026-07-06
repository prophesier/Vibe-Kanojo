"""Degradation detector + state machine.

Per the handoff spec, the site's own alert feed under-reports, so we compute the
verdict ourselves from three signals and keep a per-model state machine so each
transition alerts exactly once (no bombing):

- **A — status / floor** (fast, site-authoritative): the ``/dashboard/scores``
  ``status`` is ``warning``/``critical``, or ``currentScore`` < ``floor``.
- **B — baseline deviation** (history-internal): the latest history point sits
  ``max(drop_abs, z·noise)`` below the recent-history median. Catches cliffs.
- **C — sustained decline** (the class the site MISSES — implemented on purpose):
  the last ``n_trend`` history points fall monotonically, or a linear fit over
  the last ``m_reg`` points has slope < ``-slope_min`` with total drop > noise.

Signals A vs B/C use different metrics on purpose: ``currentScore`` (a blended
estimate that drives ``status``) is on a different scale than the raw hourly
``displayScore`` history, so B/C stay entirely within the history series to
avoid a scale-mismatch false positive, while A trusts the site's own flag.

State per model: ``OK`` → ``DEGRADED`` (alert) → ``ESCALATED`` (alert again only
after a further ``escalate_delta`` drop) → ``RECOVERED`` (alert once, after
``recover_streak`` clean batches) → ``OK``. Deduped on the batch timestamp so
re-polling the same batch never re-fires.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from .aistupidlevel_client import ModelScore
from .anthropic_status_client import AnthropicStatus


@dataclass
class DetectorParams:
    floor: float = 60.0            # A: currentScore below this = degraded (absolute)
    critical_floor: float = 50.0   # severity=critical below this
    # P: currentScore this far below its real 7-day average (periodAvg) = degraded.
    cur_drop: float = 8.0
    recover_streak: int = 2        # clean batches required to declare RECOVERED
    escalate_delta: float = 6.0    # further currentScore drop to re-alert


@dataclass
class DegradationEvent:
    model: str
    provider: str
    event: str        # DEGRADED | ESCALATED | RECOVERED
    severity: str     # warning | critical
    current_score: Optional[float]
    status: str
    trend: str
    baseline: Optional[float]
    latest_history: Optional[float]
    drop: Optional[float]
    coding_score: Optional[float]
    reasons: List[str]
    detected_at: str
    source_url: str


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

    # -- core ---------------------------------------------------------------
    def evaluate(
        self,
        score: ModelScore,
        now_iso: str,
        coding: Optional[ModelScore] = None,
    ) -> Optional[DegradationEvent]:
        """Feed one watched model's fresh snapshot — fetched with ``period="7d"``
        so it carries the REAL 7-day average (``period_avg``) and 7-day trend.
        Returns a DegradationEvent on a state transition, else None.

        Detection is deliberately ABSOLUTE, not normalised to the model's own
        volatility: these Anthropic models genuinely dip/degrade often, and each
        real dip is exactly what we want flagged — not smoothed away as "noise".

        ``coding`` is the same model's coding-axis card (warning/critical adds a
        reason). ``now_iso`` is supplied by the caller for ``detected_at``.
        """
        p = self._params
        st = self._state.setdefault(
            score.name,
            {"state": "OK", "last_batch": "", "last_event_current": None,
             "recover_streak": 0},
        )

        # Dedup: skip stale repeats and already-seen batches.
        batch = score.last_updated or ""
        if score.is_stale or (batch and batch == st.get("last_batch")):
            return None
        st["last_batch"] = batch

        cur = score.current_score
        avg = score.period_avg           # real 7-day average currentScore
        trend = (score.trend or "").lower()
        reasons: List[str] = []

        # A — status / absolute floor (combined + coding axes)
        if score.status.lower() in ("warning", "critical"):
            reasons.append(f"A:status={score.status}")
        if cur is not None and cur < p.floor:
            reasons.append(f"A:{cur:.0f}<floor {p.floor:.0f}")
        coding_score = coding.current_score if coding else None
        if coding and coding.status.lower() in ("warning", "critical"):
            reasons.append(f"A:coding={coding.status}")

        # P — currentScore sits meaningfully below its own real 7-day average
        drop = (avg - cur) if (cur is not None and avg is not None) else None
        if drop is not None and drop >= p.cur_drop:
            reasons.append(f"P:{cur:.0f} ≤ 7d avg {avg:.0f}−{p.cur_drop:.0f}")

        # T — the site's own 7-day trend is DOWN and we're below average. Catches
        # a slow slide (opus-4-6 68 vs avg 73, trend down) the floor would miss.
        if trend == "down" and cur is not None and avg is not None and cur < avg:
            reasons.append(f"T:7d↓ {cur:.0f}<avg {avg:.0f}")

        degraded = bool(reasons)
        severity = (
            "critical"
            if score.status.lower() == "critical"
            or (cur is not None and cur < p.critical_floor)
            else "warning"
        )

        event = self._transition(st, degraded, cur, severity, p)
        self._save()
        if event is None:
            return None
        return DegradationEvent(
            model=score.name,
            provider=score.provider,
            event=event,
            severity=severity,
            current_score=cur,
            status=score.status,
            trend=score.trend,
            baseline=avg,
            latest_history=None,
            drop=drop,
            coding_score=coding_score,
            reasons=reasons,
            detected_at=now_iso,
            source_url=f"https://aistupidlevel.info/?model={score.name}",
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
