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

    async def poll_once(self, now_iso: str) -> List[DegradationEvent]:
        """One polling pass. Returns state-transition events (usually empty).
        Never raises — a source failure logs and yields no events this pass.
        Fetches all four axes (combined/reasoning/coding/tooling) once, shared
        across every watched model."""
        cards: Dict[str, list] = {}
        for ax in self.AXES:
            try:
                cards[ax] = await self._asl.fetch_scores(ax, period="7d")
            except AiStupidLevelUnavailable as e:
                if ax == "combined":
                    logger.warning(f"[model_health] poll skipped (scores): {e}")
                    return []
                logger.warning(f"[model_health] {ax} axis unavailable: {e}")
                cards[ax] = []

        events: List[DegradationEvent] = []
        for name in self._cfg.watch:
            by_axis = {
                ax: AiStupidLevelClient.find(cards.get(ax) or [], name)
                for ax in self.AXES
            }
            if by_axis.get("combined") is None:
                logger.debug(f"[model_health] watched model not listed: {name}")
                continue
            ev = self._detector.evaluate(by_axis, now_iso)
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


def _axis_mark(score, floor: float, crit: float) -> str:
    """Per-axis mark by the same floor standard: <crit=danger, <floor=warning."""
    if not isinstance(score, (int, float)):
        return "❔"
    if score < crit:
        return "🔴"
    if score < floor:
        return "⚠️"
    return "✅"


def format_alert_zh(e: DegradationEvent, official: Optional[AnthropicStatus] = None) -> dict:
    """Alert panel: header (score / self-computed mean / σ), a per-axis breakdown
    (综合/逻辑推理/代码/工具, each marked by the floor standard), then A/B/C/D
    reference signals. ``official`` is the shared status snapshot fetched once per
    poll (the C line; official degradation has its own alert)."""
    color = _COLOR["ESCALATED"] if e.severity == "critical" else _COLOR.get(e.event, 0x95A5A6)
    if e.event == "RECOVERED":
        return {
            "title": f"✅ {e.model}: 跑分已恢复",
            "description": "综合分回到正常范围，可以考虑切回。",
            "color": _COLOR["RECOVERED"],
            "fields": [],
            "url": e.source_url,
            "footer": f"数据源 aistupidlevel · {e.detected_at}",
        }
    emoji = "🔴" if e.event == "ESCALATED" or e.severity == "critical" else "⚠️"
    title = f"{emoji} {e.model}: 模型有降智风险"
    floor = e.floor if e.floor is not None else 60.0
    crit = e.critical_floor if e.critical_floor is not None else 50.0

    cur = e.current_score
    header = [f"当前综合分: {_num(cur)} / 100"]
    if e.std is not None and e.mean is not None:  # self-computed stats ready
        header.append(f"自算均值 {_num(e.mean)}（{e.n_samples}样本）")
        header.append(f"波动 σ≈{e.std:.1f}")
    elif e.mean is not None:  # warming up — fall back to the site's 7-day average
        header.append(f"近7天均值 {_num(e.mean)}")
        header.append(f"σ 待样本积累({e.n_samples})")

    fields = []
    # Per-axis breakdown, each judged by the floor standard.
    for ax in ("combined", "reasoning", "coding", "tooling"):
        info = (e.axes or {}).get(ax)
        if not info:
            continue
        s = info.get("score")
        fields.append((f"{_AXIS_ZH[ax]}分", f"{_num(s)} / 100  {_axis_mark(s, floor, crit)}"))

    # A — score deviation vs our own mean / 7-day trend
    a_hit = any(r.startswith(("P:", "T:")) for r in e.reasons)
    gap = e.drop if isinstance(e.drop, (int, float)) and e.drop > 0 else None
    spread = e.std if e.std else e.standard_error
    sigma = f"，{gap / spread:.1f}σ" if (gap and spread) else ""
    trend_zh = TREND_ZH.get((e.trend or "").lower(), e.trend or "?")
    tail = f"，低于均值 {gap:.0f} 分{sigma}" if gap else ""
    a_line = (f"⚠️ 近7天{trend_zh}{tail}" if a_hit else f"✅ 近7天{trend_zh}{tail}")

    # B — aistupidlevel combined rating
    b_hit = any(r.startswith("A:status") for r in e.reasons)
    b_line = f"{'⚠️' if b_hit else '✅'} stupidmeter 评级: {STATUS_ZH.get((e.status or '').lower(), e.status or '?')}"

    # C — Anthropic official status (context; has its own alert)
    if official is None or not official.ok:
        c_line = "❔ Claude 官方状态: 未知"
    elif official.is_degraded:
        inc = official.unresolved_incidents[0].name if official.unresolved_incidents else ""
        c_line = (
            f"⚠️ Claude 官方状态: {official.claude_api_status or official.indicator or '异常'}"
            + (f"（{inc}）" if inc else "")
        )
    else:
        c_line = "✅ Claude 官方状态: 正常"

    # D — warning floor (configured)
    d_hit = cur is not None and cur < floor
    d_line = (
        f"⚠️ 跌破警戒线 {floor:.0f} 分（当前 {_num(cur)}）"
        if d_hit
        else f"✅ 未跌破警戒线 {floor:.0f} 分（当前 {_num(cur)}）"
    )

    fields += [
        ("A 跑分偏离", a_line),
        ("B 站点评级", b_line),
        ("C 官方状态", c_line),
        ("D 警戒线", d_line),
    ]
    return {
        "title": title,
        "description": "，".join(header),
        "color": color,
        "fields": fields,
        "url": e.source_url,
        "footer": f"各维度 <{floor:.0f}=⚠️ <{crit:.0f}=🔴 · 自算σ · {e.detected_at}",
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
