"""Degradation monitor: poll AIStupidLevel on a timer, run each watched model
through the :class:`Detector`, and hand transitions to a caller-supplied alert
sink. Discord-free on purpose — the bot supplies the sink and builds the embed.
"""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from loguru import logger

from .aistupidlevel_client import AiStupidLevelClient, AiStupidLevelUnavailable
from .anthropic_status_client import AnthropicStatus, AnthropicStatusClient
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


    AXES = ("combined", "reasoning", "coding", "tooling")

    async def _fetch_all(self):
        """Fetch, once per poll and shared across models, the current cards AND
        the scraped timeline series for all four axes. Returns (cards, series) or
        (None, None) if the combined axis is unreachable."""
        cards: Dict[str, list] = {}
        series: Dict[str, dict] = {}
        for ax in self.AXES:
            try:
                cards[ax] = await self._asl.fetch_scores(ax, period="7d")
            except AiStupidLevelUnavailable as e:
                if ax == "combined":
                    logger.warning(f"[model_health] poll skipped (scores): {e}")
                    return None, None
                logger.warning(f"[model_health] {ax} scores unavailable: {e}")
                cards[ax] = []
            try:
                series[ax] = await self._asl.fetch_series(ax)
            except AiStupidLevelUnavailable as e:
                logger.warning(f"[model_health] {ax} series unavailable: {e}")
                series[ax] = {}
        return cards, series

    def _assemble(self, name, cards, series) -> Optional[dict]:
        """Build the per-model dict (name/provider + per-axis current+series)."""
        cc = AiStupidLevelClient.find(cards.get("combined") or [], name)
        if cc is None:
            return None
        axes_data = {}
        for ax in self.AXES:
            card = AiStupidLevelClient.find(cards.get(ax) or [], name)
            if card is None:
                continue
            axes_data[ax] = {
                "current": card.current_score,
                "status": card.status,
                "trend": card.trend,
                "period_avg": card.period_avg,
                "series": (series.get(ax) or {}).get(card.id, []),
                "last_updated": card.last_updated,
            }
        return {"name": name, "provider": cc.provider, "axes": axes_data}

    async def poll_once(self, now_iso: str) -> List[DegradationEvent]:
        """One polling pass. Returns state-transition events (usually empty).
        Never raises — a source failure logs and yields no events this pass."""
        cards, series = await self._fetch_all()
        if cards is None:
            return []
        events: List[DegradationEvent] = []
        for name in self._cfg.watch:
            model = self._assemble(name, cards, series)
            if model is None:
                logger.debug(f"[model_health] watched model not listed: {name}")
                continue
            ev = self._detector.evaluate(model, now_iso)
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
                model_events = await self.poll_once(now)
                # Fetch the official status ONCE and reuse it: for the per-model
                # panel's "C" line AND for its own state-machine alert.
                official = await self._status.fetch()
                if official.is_degraded:
                    logger.info(
                        f"[model_health] Anthropic official degraded: "
                        f"indicator={official.indicator} api={official.claude_api_status}"
                    )
                views: List[dict] = [
                    format_alert_zh(ev, official) for ev in model_events
                ]
                oev = self._detector.evaluate_official(official, now)
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
# Site fields, translated for humans. status = the site's rating band for the
# current score; trend = the site's 7-day SCORE trend (about the benchmark
# score, NOT server uptime).
STATUS_ZH = {"good": "良好", "warning": "警告", "critical": "危险", "": "未知"}
TREND_ZH = {"up": "上升 ↑", "down": "下降 ↓", "stable": "平稳", "": "未知"}


_AXIS_ZH = {"combined": "综合", "reasoning": "逻辑推理", "coding": "代码", "tooling": "工具使用"}


def _num(v, dash: str = "?") -> str:
    return f"{v:.0f}" if isinstance(v, (int, float)) else dash


def build_axis_lines(axes: dict, floor: float) -> list:
    """One line per axis: score（均值 M · σ S · 近7天走势）with a ⚠️/✅ mark (by the
    single warning line). Shared by the alert and the /model-status embed."""
    out = []
    for ax in ("combined", "reasoning", "coding", "tooling"):
        s = axes.get(ax)
        if not s:
            continue
        cur = getattr(s, "current", None) if not isinstance(s, dict) else s.get("current")
        mean = getattr(s, "mean", None) if not isinstance(s, dict) else s.get("mean")
        std = getattr(s, "std", None) if not isinstance(s, dict) else s.get("std")
        trend = getattr(s, "trend", "") if not isinstance(s, dict) else s.get("trend", "")
        mark = "⚠️" if isinstance(cur, (int, float)) and cur < floor else (
            "✅" if isinstance(cur, (int, float)) else "❔")
        parts = [f"{_num(cur)}/100"]
        if isinstance(mean, (int, float)):
            parts.append(f"均值{_num(mean)}")
        if isinstance(std, (int, float)):
            parts.append(f"σ{std:.1f}")
        tr = TREND_ZH.get((trend or "").lower(), "")
        if ax == "combined" and tr:
            parts.append(f"近7天{tr}")
        out.append((f"{_AXIS_ZH[ax]} {mark}", " · ".join(parts)))
    return out


def _humanize_reasons(reasons: list) -> str:
    m = {"A:status": "站点评级为警告/危险", "A:combined": "综合分低于警戒线",
         "A:reasoning": "逻辑推理低于警戒线", "A:coding": "代码低于警戒线",
         "A:tooling": "工具使用低于警戒线", "P:": "综合分明显低于自身均值",
         "T:": "近7天走低且低于均值"}
    seen, out = set(), []
    for r in reasons:
        txt = next((v for k, v in m.items() if r.startswith(k)), r)
        if txt not in seen:
            seen.add(txt)
            out.append(txt)
    return "；".join(out) or "—"


def format_alert_zh(e: DegradationEvent, official: Optional[AnthropicStatus] = None) -> dict:
    """Alert panel: one line per axis (score · self-computed 均值/σ · trend · mark),
    then why-it-fired and the Anthropic official status. ``official`` is the shared
    snapshot fetched once per poll (it has its own alert too)."""
    color = _COLOR["ESCALATED"] if e.severity == "critical" else _COLOR.get(e.event, 0x95A5A6)
    floor = e.floor if e.floor is not None else 60.0
    if e.event == "RECOVERED":
        return {
            "title": f"✅ {e.model}: 跑分已恢复",
            "description": "综合分回到正常范围，可以考虑切回。",
            "color": _COLOR["RECOVERED"], "fields": [], "url": e.source_url,
            "footer": f"数据源 aistupidlevel · {e.detected_at}",
        }
    emoji = "🔴" if e.event == "ESCALATED" or e.severity == "critical" else "⚠️"
    fields = build_axis_lines(e.axes, floor)
    fields.append(("为什么报警", _humanize_reasons(e.reasons)))

    if official is None or not official.ok:
        c_line = "❔ 未知"
    elif official.is_degraded:
        inc = official.unresolved_incidents[0].name if official.unresolved_incidents else ""
        c_line = (
            f"⚠️ {official.claude_api_status or official.indicator or '异常'}"
            + (f"（{inc}）" if inc else "")
        )
    else:
        c_line = "✅ 正常"
    fields.append(("Claude 官方状态", c_line))

    return {
        "title": f"{emoji} {e.model}: 模型有降智风险",
        "description": f"分数越高越聪明（满分100，低于 {floor:.0f} 为警戒线）。"
        "均值/σ 由我从时间线数据自算；「走势」指跑分趋势，不是服务器状态。",
        "color": color,
        "fields": fields,
        "url": e.source_url,
        "footer": f"数据源 aistupidlevel + status.claude.com · {e.detected_at}",
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
