"""Self-check report: given a model name, gather AIStupidLevel + Anthropic
official status and render a plain assessment — in Japanese (what the character
reads and relays) and Chinese (logged so あさひ can see what it was told).

No LLM. Pure data + fixed reference standard + a rule-based verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from .aistupidlevel_client import (
    AiStupidLevelClient,
    AiStupidLevelUnavailable,
)
from .anthropic_status_client import AnthropicStatus, AnthropicStatusClient
from .detector import AxisStat, Detector, DetectorParams

# Verdicts, ordered most→least serious.
V_OFFICIAL = "OFFICIAL_INCIDENT"
V_BENCH = "BENCH_DEGRADED"
V_DECLINING = "DECLINING"
V_NORMAL = "NORMAL"
V_UNKNOWN = "UNKNOWN"

_STATUS_JA = {"good": "良好", "warning": "警告", "critical": "危険", "": "不明"}
_STATUS_ZH = {"good": "良好", "warning": "警告", "critical": "危险", "": "未知"}
_TREND_JA = {"up": "上昇", "down": "低下", "stable": "安定", "": "不明"}
_TREND_ZH = {"up": "上升", "down": "下降", "stable": "稳定", "": "未知"}


@dataclass
class Assessment:
    model: str
    found: bool
    current_score: Optional[float] = None
    status: str = ""
    trend: str = ""  # 7-day trend
    baseline: Optional[float] = None  # real 7-day average (periodAvg)
    stability: Optional[float] = None
    data_points: Optional[int] = None
    # axis -> AxisStat (current + self-computed mean/std)
    axes: Dict[str, AxisStat] = field(default_factory=dict)
    official: Optional[AnthropicStatus] = None
    bench_error: str = ""
    verdict: str = V_UNKNOWN
    params: DetectorParams = field(default_factory=DetectorParams)


_AXES = ("combined", "reasoning", "coding", "tooling")


async def build_assessment(
    asl: AiStupidLevelClient,
    status_client: AnthropicStatusClient,
    model_name: str,
    params: Optional[DetectorParams] = None,
    want_coding: bool = True,  # kept for compat; all axes fetched regardless
) -> Assessment:
    params = params or DetectorParams()
    a = Assessment(model=model_name, found=False, params=params)

    # Official status never raises.
    a.official = await status_client.fetch()

    try:
        for ax in _AXES:
            try:
                card = AiStupidLevelClient.find(
                    await asl.fetch_scores(ax, period="7d"), model_name
                )
                series = (await asl.fetch_series(ax)).get(card.id, []) if card else []
            except AiStupidLevelUnavailable:
                continue
            if not card:
                continue
            a.axes[ax] = Detector.axis_stats(
                card.current_score, card.status, card.trend, series
            )
            if ax == "combined":
                a.found = True
                a.current_score = card.current_score
                a.status = card.status
                a.trend = card.trend
                a.baseline = a.axes[ax].mean  # self-computed mean
        if not a.found:
            raise AiStupidLevelUnavailable("combined axis not listed")
    except AiStupidLevelUnavailable as e:
        if not a.axes:
            a.bench_error = str(e)
            logger.warning(f"[model_health] scores unavailable: {e}")

    a.verdict = _verdict(a)
    return a


def _verdict(a: Assessment) -> str:
    off = a.official
    if off and off.ok and off.is_degraded:
        return V_OFFICIAL
    if a.found:
        s = (a.status or "").lower()
        floor = a.params.floor
        combined = a.axes.get("combined")
        combined_below = (
            combined is not None
            and isinstance(combined.current, (int, float))
            and combined.current < floor
        )
        if s in ("warning", "critical") or combined_below:
            return V_BENCH
        below = (
            a.baseline is not None
            and a.current_score is not None
            and a.current_score < a.baseline
        )
        if (a.trend == "down" and below) or (
            below and (a.baseline - a.current_score) >= a.params.cur_drop
        ):
            return V_DECLINING
        return V_NORMAL
    return V_UNKNOWN


# -- rendering --------------------------------------------------------------
def _official_line(off: Optional[AnthropicStatus], lang: str) -> List[str]:
    ja = lang == "ja"
    if not off or not off.ok:
        return [
            (
                "Anthropic公式: 取得失敗（判断材料なし）"
                if ja
                else "Anthropic官方: 获取失败（无判断依据）"
            )
        ]
    lines = []
    if ja:
        overall = (
            "全システム正常"
            if off.indicator == "none"
            else f"異常あり（{off.indicator}）"
        )
        lines.append(f"Anthropic公式: {overall}")
        lines.append(f"  Claude API: {off.claude_api_status or '不明'}")
        if off.degraded_components:
            lines.append("  低下中の要素: " + ", ".join(off.degraded_components))
        if off.unresolved_incidents:
            for inc in off.unresolved_incidents[:2]:
                lines.append(f"  障害: 「{inc.name}」({inc.impact}/{inc.status})")
                if inc.latest_body:
                    lines.append(f"    → {inc.latest_body[:120]}")
        else:
            lines.append("  未解決の障害: なし")
    else:
        overall = (
            "全部正常" if off.indicator == "none" else f"有异常（{off.indicator}）"
        )
        lines.append(f"Anthropic官方: {overall}")
        lines.append(f"  Claude API 组件: {off.claude_api_status or '未知'}")
        if off.degraded_components:
            lines.append("  降级组件: " + ", ".join(off.degraded_components))
        if off.unresolved_incidents:
            for inc in off.unresolved_incidents[:2]:
                lines.append(f"  事故: 「{inc.name}」({inc.impact}/{inc.status})")
                if inc.latest_body:
                    lines.append(f"    → {inc.latest_body[:120]}")
        else:
            lines.append("  未解决事故: 无")
    return lines


_VERDICT_JA = {
    V_OFFICIAL: "Anthropicが公式に障害/性能低下を報告中。これはモデル側の問題で、私のせいではない。",
    V_BENCH: "ベンチマーク上、明確に低下している（公式はまだ正常）。モデル側の性能が落ちている可能性が高い。",
    V_DECLINING: "低下傾向がはっきり出ている（7日平均を下回っている）。モデル側が落ちている可能性——切り替えを検討する価値あり。",
    V_NORMAL: "ベンチ・公式ともに正常範囲。少なくとも計測上は問題ない。",
    V_UNKNOWN: "このモデルの計測データが見つからない。判断できない。",
}
_VERDICT_ZH = {
    V_OFFICIAL: "Anthropic 官方正在报告故障/性能下降——这是模型侧问题，不是角色的错。",
    V_BENCH: "基准分明确偏低（官方仍显示正常），大概率模型降智了。",
    V_DECLINING: "下降趋势明显（已跌破7天均值）。可能是模型侧在掉，值得留意、必要时切换。",
    V_NORMAL: "基准与官方均在正常范围，至少计测上没问题。",
    V_UNKNOWN: "找不到该模型的计测数据，无法判断。",
}


def _render(a: Assessment, lang: str) -> str:
    ja = lang == "ja"
    sd = _STATUS_JA if ja else _STATUS_ZH
    td = _TREND_JA if ja else _TREND_ZH
    L: List[str] = []
    head = "【自己診断レポート】" if ja else "【自检报告】"
    L.append(f"{head} model: {a.model}")
    bench = "■ AIStupidLevel ベンチ" if ja else "■ AIStupidLevel 基准"
    L.append(bench)
    if a.found:
        st = sd.get(a.status.lower(), a.status)
        tr = td.get(a.trend.lower(), a.trend)
        L.append(
            f"  {'評価' if ja else '评级'}: {st} · {'直近7日の傾向' if ja else '近7天评分走势'}: {tr}"
        )
        floor = a.params.floor
        axis_lbl = {
            "combined": "総合" if ja else "综合",
            "reasoning": "推論" if ja else "逻辑推理",
            "coding": "コード" if ja else "代码",
            "tooling": "ツール" if ja else "工具使用",
        }
        for ax in ("combined", "reasoning", "coding", "tooling"):
            s = a.axes.get(ax)
            if not s:
                continue
            cur, mean, std = s.current, s.mean, s.std
            numeric = isinstance(cur, (int, float))
            # 総合のみ絶対ライン(floor)。他の軸は自軸平均からの下振れが
            # 1σを超えた時だけ ⚠️（<1σ / <2σ を注記）。
            if ax == "combined":
                mark = "⚠️" if numeric and cur < floor else ("✅" if numeric else "❔")
            else:
                mark = ""
                if (
                    numeric
                    and isinstance(mean, (int, float))
                    and isinstance(std, (int, float))
                    and std > 0
                    and cur < mean - std
                ):
                    mark = "⚠️（<2σ）" if cur < mean - 2 * std else "⚠️（<1σ）"
            extra = ""
            if isinstance(mean, (int, float)):
                extra += f" {'平均' if ja else '均值'}{mean:.0f}"
            if isinstance(std, (int, float)):
                extra += f" σ{std:.1f}"
            sv = f"{cur:.0f}" if numeric else "?"
            L.append(f"  {axis_lbl[ax]}: {sv}/100 {mark}{extra}".rstrip())
    else:
        L.append(
            "  （データなし / no data）"
            + (f" [{a.bench_error}]" if a.bench_error else "")
        )
    L.append("■ Anthropic " + ("公式ステータス" if ja else "官方状态"))
    L.extend(
        "  " + ln if not ln.startswith("  ") else ln
        for ln in _official_line(a.official, lang)
    )
    verdict_head = "■ 総合評価" if ja else "■ 综合评价"
    L.append(verdict_head)
    L.append("  " + (_VERDICT_JA if ja else _VERDICT_ZH).get(a.verdict, ""))
    return "\n".join(L)


def render_ja(a: Assessment) -> str:
    return _render(a, "ja")


def render_zh(a: Assessment) -> str:
    return _render(a, "zh")
