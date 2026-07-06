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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .aistupidlevel_client import ModelScore
from .anthropic_status_client import AnthropicStatus


@dataclass
class DetectorParams:
    floor: float = 60.0            # warning floor (absolute)
    critical_floor: float = 50.0   # danger floor
    cur_drop: float = 8.0          # P: this far below our own mean = degraded
    # We compute the mean/std OURSELVES from a rolling window of recorded
    # currentScore samples — the site's standardError is unreliable and the
    # chart's series isn't fetchable, so we record it each batch.
    cur_window: int = 48           # samples for the self mean/std (~2 days hourly)
    cur_min_samples: int = 5       # trust the self-stats only past this many
    cur_series_cap: int = 240      # cap the persisted series length
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
    baseline: Optional[float]  # the mean we compare against (self mean, or periodAvg)
    latest_history: Optional[float]
    drop: Optional[float]      # baseline - current
    coding_score: Optional[float]
    reasons: List[str]
    detected_at: str
    source_url: str
    standard_error: Optional[float] = None  # site's spread (fallback only)
    floor: Optional[float] = None           # configured warning floor
    critical_floor: Optional[float] = None  # configured danger floor
    mean: Optional[float] = None      # self-computed mean of recorded currentScore
    std: Optional[float] = None       # self-computed population std
    n_samples: int = 0                # samples the self mean/std used
    # axis -> {"score": float|None, "status": str, "trend": str}
    axes: Dict[str, Any] = field(default_factory=dict)


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
        scores_by_axis: Dict[str, ModelScore],
        now_iso: str,
    ) -> Optional[DegradationEvent]:
        """Feed one watched model's fresh snapshot across ALL axes
        (``{"combined","reasoning","coding","tooling": ModelScore}``, fetched with
        period="7d"). Returns a DegradationEvent on a state transition, else None.

        We record the combined currentScore each batch and compute the mean/std
        OURSELVES from that rolling window (the site's standardError is unreliable
        and the chart's series isn't fetchable). Until the window warms up we fall
        back to the site's 7-day average. Detection stays ABSOLUTE (floor/status),
        plus a below-mean drop and a 7-day downtrend — dips aren't smoothed away.
        """
        combined = scores_by_axis.get("combined")
        if combined is None:
            return None
        p = self._params
        st = self._state.setdefault(
            combined.name,
            {"state": "OK", "last_batch": "", "last_event_current": None,
             "recover_streak": 0, "cur_series": []},
        )

        # Dedup on the batch timestamp: process a batch once (even if the site
        # flags it isStale — on startup we still haven't seen it, and a degraded
        # model must alert). Re-alerting is prevented by the state machine, not
        # by skipping. Same timestamp seen again → skip.
        batch = combined.last_updated or ""
        if batch and batch == st.get("last_batch"):
            return None
        st["last_batch"] = batch

        cur = combined.current_score
        trend = (combined.trend or "").lower()

        # Record this batch's combined score and compute our OWN mean/std from the
        # window (excluding the point we just added, so "mean" is the prior level).
        series = st.setdefault("cur_series", [])
        if cur is not None:
            series.append([batch, cur])
            del series[: -p.cur_series_cap]
        prior = [s for _, s in series[:-1] if s is not None][-p.cur_window:]
        if len(prior) >= p.cur_min_samples:
            my_mean = statistics.fmean(prior)
            my_std = statistics.pstdev(prior) if len(prior) >= 2 else 0.0
            n = len(prior)
        else:
            my_mean = combined.period_avg  # fallback until the window warms up
            my_std = None
            n = len(prior)
        baseline = my_mean

        reasons: List[str] = []
        # A — status / absolute floor
        if combined.status.lower() in ("warning", "critical"):
            reasons.append(f"A:status={combined.status}")
        if cur is not None and cur < p.floor:
            reasons.append(f"A:{cur:.0f}<floor {p.floor:.0f}")

        # Per-axis breakdown (all shown). A sub-axis only TRIGGERS when it hits
        # the danger floor — reasoning etc. sit inherently low (~35-62) on these
        # benchmarks, so firing on mere "warning" would flag every model always.
        axes: Dict[str, Any] = {}
        for ax, ms in scores_by_axis.items():
            if ms is None:
                continue
            axes[ax] = {
                "score": ms.current_score,
                "status": ms.status,
                "trend": ms.trend,
            }
            if (
                ax != "combined"
                and ms.current_score is not None
                and ms.current_score < p.critical_floor
            ):
                reasons.append(f"A:{ax}<{p.critical_floor:.0f}")

        # P — combined score sits meaningfully below our own recent mean
        drop = (baseline - cur) if (cur is not None and baseline is not None) else None
        if drop is not None and drop >= p.cur_drop:
            reasons.append(f"P:{cur:.0f}≤mean {baseline:.0f}−{p.cur_drop:.0f}")

        # T — site's 7-day trend is DOWN and we're below the mean
        if trend == "down" and cur is not None and baseline is not None and cur < baseline:
            reasons.append(f"T:7d↓ {cur:.0f}<mean {baseline:.0f}")

        degraded = bool(reasons)
        severity = (
            "critical"
            if combined.status.lower() == "critical"
            or (cur is not None and cur < p.critical_floor)
            else "warning"
        )

        event = self._transition(st, degraded, cur, severity, p)
        self._save()
        if event is None:
            return None
        return DegradationEvent(
            model=combined.name,
            provider=combined.provider,
            event=event,
            severity=severity,
            current_score=cur,
            status=combined.status,
            trend=combined.trend,
            baseline=baseline,
            latest_history=None,
            drop=drop,
            coding_score=axes.get("coding", {}).get("score"),
            reasons=reasons,
            detected_at=now_iso,
            standard_error=combined.standard_error,
            floor=p.floor,
            critical_floor=p.critical_floor,
            mean=my_mean,
            std=my_std,
            n_samples=n,
            axes=axes,
            source_url=f"https://aistupidlevel.info/?model={combined.name}",
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
