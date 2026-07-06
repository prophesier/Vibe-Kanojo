"""Degradation monitor: poll AIStupidLevel on a timer, run each watched model
through the :class:`Detector`, and hand transitions to a caller-supplied alert
sink. Discord-free on purpose — the bot supplies the sink and builds the embed.
"""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from loguru import logger

from .aistupidlevel_client import AiStupidLevelClient, AiStupidLevelUnavailable
from .anthropic_status_client import AnthropicStatusClient
from .detector import (
    DegradationEvent,
    Detector,
    DetectorParams,
    OfficialStatusEvent,
)


@dataclass
class MonitorConfig:
    enabled: bool = False
    watch: List[str] = field(
        default_factory=lambda: ["claude-opus-4-8", "claude-fable-5", "claude-opus-4-6"]
    )
    # subset of `watch` for which the coding axis is also checked/surfaced
    coding_models: List[str] = field(
        default_factory=lambda: ["claude-opus-4-8", "claude-fable-5"]
    )
    poll_seconds: int = 1200  # 20 min; site data is hourly, faster is pointless
    state_path: Optional[pathlib.Path] = None


class DegradationMonitor:
    def __init__(
        self,
        config: MonitorConfig,
        asl_client: Optional[AiStupidLevelClient] = None,
        params: Optional[DetectorParams] = None,
    ) -> None:
        self._cfg = config
        self._asl = asl_client or AiStupidLevelClient()
        self._status = AnthropicStatusClient()
        state = config.state_path or pathlib.Path("cache/model_health_state.json")
        self._detector = Detector(state, params)

    async def check_official(self, now_iso: str) -> Optional[OfficialStatusEvent]:
        """Fetch Anthropic's official status and return a transition event
        (degraded/recovered) or None. Never raises."""
        status = await self._status.fetch()  # ok=False on failure, never raises
        if status.is_degraded:
            logger.info(
                f"[model_health] Anthropic official degraded: "
                f"indicator={status.indicator} api={status.claude_api_status} "
                f"incidents={[i.name for i in status.unresolved_incidents]}"
            )
        return self._detector.evaluate_official(status, now_iso)

    async def poll_once(self, now_iso: str) -> List[DegradationEvent]:
        """One polling pass. Returns state-transition events (usually empty).
        Never raises — a source failure logs and yields no events this pass."""
        try:
            # period="7d" carries the real 7-day average + 7-day trend.
            combined = await self._asl.fetch_scores("combined", period="7d")
        except AiStupidLevelUnavailable as e:
            logger.warning(f"[model_health] poll skipped (scores): {e}")
            return []
        coding_cards = []
        if self._cfg.coding_models:
            try:
                coding_cards = await self._asl.fetch_scores("coding", period="7d")
            except AiStupidLevelUnavailable as e:
                logger.warning(f"[model_health] coding axis unavailable: {e}")

        events: List[DegradationEvent] = []
        for name in self._cfg.watch:
            m = AiStupidLevelClient.find(combined, name)
            if not m:
                logger.debug(f"[model_health] watched model not listed: {name}")
                continue
            coding = (
                AiStupidLevelClient.find(coding_cards, name)
                if name in self._cfg.coding_models
                else None
            )
            ev = self._detector.evaluate(m, now_iso, coding=coding)
            if ev:
                logger.info(
                    f"[model_health] {ev.model} -> {ev.event} "
                    f"(sev={ev.severity}, score={ev.current_score}, reasons={ev.reasons})"
                )
                events.append(ev)
        return events

    async def run(
        self,
        send_alert: Callable[[dict], Awaitable[None]],
        now_fn: Callable[[], str],
    ) -> None:
        """Poll forever. ``send_alert`` receives a render-ready alert view (dict);
        the bot turns it into an embed. Covers BOTH the per-model benchmark
        signals AND Anthropic's official status (degraded_performance included).
        ``now_fn`` supplies an ISO-8601 UTC timestamp per pass. Each iteration is
        isolated so one failure never kills the loop."""
        logger.info(
            f"[model_health] monitor started: watch={self._cfg.watch} "
            f"+ Anthropic official status, every {self._cfg.poll_seconds}s"
        )
        while True:
            try:
                now = now_fn()
                views: List[dict] = [format_alert_zh(ev) for ev in await self.poll_once(now)]
                oev = await self.check_official(now)
                if oev is not None:
                    views.append(format_official_alert_zh(oev))
                for view in views:
                    try:
                        await send_alert(view)
                    except Exception as e:
                        logger.error(f"[model_health] alert delivery failed: {e}")
            except Exception as e:
                logger.exception(f"[model_health] monitor pass errored: {e}")
            await asyncio.sleep(self._cfg.poll_seconds)


# -- Chinese alert view (user-facing) ---------------------------------------
_EVENT_ZH = {"DEGRADED": "降智", "ESCALATED": "进一步降智", "RECOVERED": "已恢复"}
_EVENT_EMOJI = {"DEGRADED": "⚠️", "ESCALATED": "🔴", "RECOVERED": "✅"}
_COLOR = {"DEGRADED": 0xE67E22, "ESCALATED": 0xE74C3C, "RECOVERED": 0x2ECC71}


def format_alert_zh(e: DegradationEvent) -> dict:
    """Structured Chinese alert for the user. The bot turns this into an embed."""
    emoji = _EVENT_EMOJI.get(e.event, "•")
    color = _COLOR["ESCALATED"] if e.severity == "critical" else _COLOR.get(e.event, 0x95A5A6)
    title = f"{emoji} 模型{_EVENT_ZH.get(e.event, e.event)}: {e.model}"

    def _f(v, suffix=""):
        return f"{v:.0f}{suffix}" if isinstance(v, (int, float)) else "?"

    fields = [("综合分", f"{_f(e.current_score)}  (状态 {e.status or '?'}, 7天趋势 {e.trend or '?'})")]
    if e.baseline is not None:
        # currentScore vs its real 7-day average. Only label a positive gap as low.
        low = f"，低 {e.drop:.0f}" if isinstance(e.drop, (int, float)) and e.drop > 0 else ""
        fields.append(("7天均值", f"~{_f(e.baseline)}，当前 {_f(e.current_score)}{low}"))
    if e.coding_score is not None:
        fields.append(("编程分", _f(e.coding_score)))
    fields.append(("触发信号", "、".join(e.reasons) or "—"))
    if e.event == "RECOVERED":
        desc = "已回到正常范围，可以考虑切回。"
    elif e.severity == "critical":
        desc = "跌破危险水位，建议尽快切换模型。"
    else:
        desc = "基准/官方检测到下降，注意观察，必要时切换模型。"
    return {
        "title": title,
        "description": desc,
        "color": color,
        "fields": fields,
        "url": e.source_url,
        "footer": f"severity={e.severity} · detected {e.detected_at}",
    }


def format_official_alert_zh(e: OfficialStatusEvent) -> dict:
    """Structured Chinese alert for an Anthropic OFFICIAL-status transition."""
    if e.event == "RECOVERED":
        return {
            "title": "✅ Anthropic 官方状态已恢复",
            "description": "Anthropic 官方状态恢复正常。",
            "color": 0x2ECC71,
            "fields": [("状态", "全部正常")],
            "url": "https://status.claude.com",
            "footer": f"detected {e.detected_at}",
        }
    title = "🔴 Anthropic 官方报告异常"
    fields = [
        ("总体", f"indicator={e.indicator or '?'}"),
        ("Claude API", e.claude_api_status or "?"),
    ]
    if e.degraded_components:
        fields.append(("受影响组件", "、".join(e.degraded_components)))
    if e.incidents:
        fields.append(("事故", "、".join(f"「{n}」" for n in e.incidents)))
    if e.latest_body:
        fields.append(("最新", e.latest_body[:300]))
    # degraded_performance is the priority case あさひ flagged — call it out.
    dp = any("degraded_performance" in c for c in e.degraded_components)
    desc = (
        "Anthropic 官方报告性能下降（degraded_performance）——比整站挂更隐蔽，重点注意。"
        if dp
        else "Anthropic 官方报告故障/异常。"
    )
    return {
        "title": title,
        "description": desc,
        "color": 0xE74C3C,
        "fields": fields,
        "url": "https://status.claude.com",
        "footer": f"official · detected {e.detected_at}",
    }
