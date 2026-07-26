"""Read-only client for Anthropic's official Statuspage.

``status.anthropic.com`` 302-redirects to ``status.claude.com`` (Atlassian
Statuspage, page "Claude"); we hit status.claude.com directly and also follow
redirects as a belt-and-suspenders. One request to ``summary.json`` gives the
overall indicator, every component's status, and all unresolved incidents.

Shapes confirmed live 2026-07-06:
- ``status.indicator`` ∈ {none, minor, major, critical, maintenance}
  (``none`` == all good). Never trust the human ``status.description`` alone.
- ``components[]`` each ``{name, status}``; the API component is named exactly
  ``"Claude API (api.anthropic.com)"``. Component ``status`` ∈ {operational,
  degraded_performance, partial_outage, major_outage, under_maintenance}.
- There is NO "models" component — a model problem shows up as an INCIDENT (e.g.
  "Elevated errors on several models") that flips components to
  ``degraded_performance``. So we flag on BOTH unresolved incidents AND a
  non-operational Claude API component. ``degraded_performance`` is the priority
  signal: subtler than a full outage and easy to miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import httpx
from loguru import logger

_SUMMARY_URL = "https://status.claude.com/api/v2/summary.json"
_UA = "VibeKanojo-ModelHealth/0.1 (+degradation-monitor)"
_TIMEOUT = 15.0

_CLAUDE_API_COMPONENT = "Claude API (api.anthropic.com)"
_OK_COMPONENT = "operational"
_OK_INDICATOR = "none"


@dataclass
class Incident:
    name: str
    impact: str  # none | minor | major | critical
    status: str  # investigating | identified | monitoring | resolved | ...
    updated_at: str
    latest_body: str
    affected: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Incident":
        updates = d.get("incident_updates") or []
        latest = updates[0] if updates else {}
        affected = []
        for u in updates:
            for c in u.get("affected_components") or []:
                nm = c.get("name")
                if nm and nm not in affected:
                    affected.append(nm)
        return cls(
            name=str(d.get("name", "") or ""),
            impact=str(d.get("impact", "") or ""),
            status=str(d.get("status", "") or ""),
            updated_at=str(d.get("updated_at", "") or ""),
            latest_body=str((latest or {}).get("body", "") or ""),
            affected=affected,
        )


@dataclass
class AnthropicStatus:
    ok: bool  # did the fetch succeed
    indicator: str  # none | minor | major | critical | maintenance
    description: str
    claude_api_status: str  # status of the "Claude API" component
    components: Dict[str, str] = field(default_factory=dict)  # name -> status
    unresolved_incidents: List[Incident] = field(default_factory=list)
    error: str = ""

    @property
    def is_degraded(self) -> bool:
        """True if Anthropic is reporting anything less than all-clear that could
        touch the Claude API — including the subtle ``degraded_performance``."""
        if not self.ok:
            return False  # unknown ≠ degraded; caller sees ok=False
        if self.indicator and self.indicator != _OK_INDICATOR:
            return True
        if self.claude_api_status and self.claude_api_status != _OK_COMPONENT:
            return True
        if self.unresolved_incidents:
            return True
        # any component in a degraded/outage state (catch model-wide issues that
        # flipped a component but not the overall indicator)
        return any(
            v and v != _OK_COMPONENT and v != "under_maintenance"
            for v in self.components.values()
        )

    @property
    def degraded_components(self) -> List[str]:
        return [
            f"{n}={s}" for n, s in self.components.items() if s and s != _OK_COMPONENT
        ]


class AnthropicStatusClient:
    def __init__(self, timeout: float = _TIMEOUT) -> None:
        self._timeout = timeout

    async def fetch(self) -> AnthropicStatus:
        """Never raises — returns ``ok=False`` with an error string on failure."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(_SUMMARY_URL)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"[model_health] Anthropic status unavailable: {e}")
            return AnthropicStatus(
                ok=False,
                indicator="",
                description="",
                claude_api_status="",
                error=str(e),
            )

        status = data.get("status") or {}
        components = {}
        for c in data.get("components") or []:
            if isinstance(c, dict) and c.get("name"):
                components[str(c["name"])] = str(c.get("status", "") or "")
        incidents = [
            Incident.from_json(i)
            for i in (data.get("incidents") or [])
            if isinstance(i, dict)
        ]
        return AnthropicStatus(
            ok=True,
            indicator=str(status.get("indicator", "") or ""),
            description=str(status.get("description", "") or ""),
            claude_api_status=components.get(_CLAUDE_API_COMPONENT, ""),
            components=components,
            unresolved_incidents=incidents,
        )
