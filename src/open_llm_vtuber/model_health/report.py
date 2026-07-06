"""Self-check report: given a model name, gather AIStupidLevel + Anthropic
official status and render a plain assessment — in Japanese (what the character
reads and relays) and Chinese (logged so あさひ can see what it was told).

No LLM. Pure data + fixed reference standard + a rule-based verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from .aistupidlevel_client import (
    AiStupidLevelClient,
    AiStupidLevelUnavailable,
)
from .anthropic_status_client import AnthropicStatus, AnthropicStatusClient
from .detector import DetectorParams

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
    coding_score: Optional[float] = None
    coding_status: str = ""
    official: Optional[AnthropicStatus] = None
    bench_error: str = ""
    verdict: str = V_UNKNOWN
    params: DetectorParams = field(default_factory=DetectorParams)


async def build_assessment(
    asl: AiStupidLevelClient,
    status_client: AnthropicStatusClient,
    model_name: str,
    params: Optional[DetectorParams] = None,
    want_coding: bool = False,
) -> Assessment:
    params = params or DetectorParams()
    a = Assessment(model=model_name, found=False, params=params)

    # Official status never raises.
    a.official = await status_client.fetch()

    try:
        # period="7d" → real 7-day average + stability + 7-day trend (all real;
        # the displayScore history is skipped — it's >half synthetic backfill).
        m = AiStupidLevelClient.find(
            await asl.fetch_scores("combined", period="7d"), model_name
        )
        if m:
            a.found = True
            a.current_score = m.current_score
            a.status = m.status
            a.trend = m.trend
            a.baseline = m.period_avg
            a.stability = m.stability
            a.data_points = m.data_points
            if want_coding:
                try:
                    cm = AiStupidLevelClient.find(
                        await asl.fetch_scores("coding", period="7d"), model_name
                    )
                    if cm:
                        a.coding_score = cm.current_score
                        a.coding_status = cm.status
                except AiStupidLevelUnavailable:
                    pass
    except AiStupidLevelUnavailable as e:
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
        cs = (a.coding_status or "").lower()
        if (
            s in ("warning", "critical")
            or cs in ("warning", "critical")
            or (a.current_score is not None and a.current_score < a.params.floor)
        ):
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
            ("Anthropic公式: 取得失敗（判断材料なし）" if ja else "Anthropic官方: 获取失败（无判断依据）")
        ]
    lines = []
    if ja:
        overall = "全システム正常" if off.indicator == "none" else f"異常あり（{off.indicator}）"
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
        overall = "全部正常" if off.indicator == "none" else f"有异常（{off.indicator}）"
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
    V_BENCH: "ベンチマーク上、明確に低下している（公式はまだ正常）。モデルが降智している可能性が高い。",
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
        sc = f"{a.current_score:.0f}" if a.current_score is not None else "?"
        if ja:
            L.append(f"  総合スコア: {sc}/100 (状態: {sd.get(a.status.lower(),a.status)}, 傾向: {td.get(a.trend.lower(),a.trend)})")
        else:
            L.append(f"  综合分: {sc}/100 (状态: {sd.get(a.status.lower(),a.status)}, 趋势: {td.get(a.trend.lower(),a.trend)})")
        if a.baseline is not None:
            base_lbl = "7日平均" if ja else "7天均值"
            if a.stability is not None:
                stab = f" · {'安定性' if ja else '稳定性'}{a.stability:.0f}"
            else:
                stab = ""
            L.append(f"  {base_lbl}: ~{a.baseline:.0f}{stab}")
        if a.coding_score is not None:
            clbl = "コーディング" if ja else "编程"
            L.append(f"  {clbl}: {a.coding_score:.0f} ({sd.get(a.coding_status.lower(),a.coding_status)})")
        ref = (
            f"  ※基準: {a.params.floor:.0f}未満=警告, {a.params.critical_floor:.0f}未満=危険"
            if ja
            else f"  ※参照标准: 低于{a.params.floor:.0f}=警告, 低于{a.params.critical_floor:.0f}=危险"
        )
        L.append(ref)
    else:
        L.append("  （データなし / no data）" + (f" [{a.bench_error}]" if a.bench_error else ""))
    L.append("■ Anthropic " + ("公式ステータス" if ja else "官方状态"))
    L.extend("  " + ln if not ln.startswith("  ") else ln for ln in _official_line(a.official, lang))
    verdict_head = "■ 総合評価" if ja else "■ 综合评价"
    L.append(verdict_head)
    L.append("  " + (_VERDICT_JA if ja else _VERDICT_ZH).get(a.verdict, ""))
    L.append(
        "  出典: aistupidlevel.info / status.anthropic.com"
        if ja
        else "  数据源: aistupidlevel.info / status.anthropic.com"
    )
    return "\n".join(L)


def render_ja(a: Assessment) -> str:
    return _render(a, "ja")


def render_zh(a: Assessment) -> str:
    return _render(a, "zh")
