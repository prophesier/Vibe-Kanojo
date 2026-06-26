"""Builds the Japanese prompt injected when an alarm fires.

The fired prompt becomes the "user turn" for a proactive conversation: it
reminds the character of the note it left itself and hands it the opening to
speak. Wording is intentionally unambiguous about *when* (a reminder set
earlier whose time has come) and, when overdue, explains that the scheduled
time already passed and the client simply wasn't running then.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

from .store import _parse_iso

# An alarm at most this many seconds late counts as "on time" (the scheduler
# just woke a hair late); beyond it, the prompt frames the alarm as overdue.
ON_TIME_GRACE_SECONDS = 60


def _fmt(dt: datetime.datetime) -> str:
    """Local 'YYYY-MM-DD HH:MM' for display."""
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def format_elapsed(delta: datetime.timedelta) -> str:
    """Human-readable lateness in Japanese, e.g. 約45分 / 約3時間 / 約2日."""
    secs = int(max(delta.total_seconds(), 0))
    if secs < 3600:
        return f"約{max(secs // 60, 1)}分"
    if secs < 86400:
        return f"約{secs // 3600}時間"
    return f"約{secs // 86400}日"


def build_fire_prompt(
    due: List[Dict[str, Any]], now_utc: datetime.datetime | None = None
) -> str:
    """Build the proactive-turn text for one or more fired alarms."""
    now = now_utc or datetime.datetime.now(datetime.timezone.utc)

    def parts(a: Dict[str, Any]):
        fire = _parse_iso(a.get("fire_at_utc", "")) or now
        late = now - fire
        overdue = late.total_seconds() > ON_TIME_GRACE_SECONDS
        return fire, late, overdue

    if len(due) == 1:
        a = due[0]
        note = a.get("note", "")
        fire, late, overdue = parts(a)
        if not overdue:
            return (
                "【アラーム】前にセットしておいたリマインダーの予定時刻になった。\n"
                f"自分へのメモ:「{note}」\n"
                "このメモの内容を思い出して、今、自分から自然にユーザーに話しかけて。"
            )
        return (
            "【アラーム】前にセットしておいたリマインダーの予定時刻は、すでに過ぎている。\n"
            f"自分へのメモ:「{note}」\n"
            f"予定時刻は {_fmt(fire)} だったが、その時はクライアントが起動しておらず、"
            f"今（{_fmt(now)}、{format_elapsed(late)} 遅れ）になって通知が届いた。\n"
            "もう過ぎてしまったことを踏まえて、今から自然にユーザーに話しかけて。"
        )

    # Multiple alarms came due together (e.g. the client reconnected after the
    # server had been down): fold them into one turn.
    lines = []
    for a in sorted(due, key=lambda x: x.get("fire_at_utc", "")):
        fire, late, overdue = parts(a)
        tail = f" ※{format_elapsed(late)} 遅れ" if overdue else ""
        lines.append(f"- 予定 {_fmt(fire)}（メモ:{a.get('note', '')}）{tail}")
    body = "\n".join(lines)
    return (
        "【アラーム】前にセットしておいたリマインダーが複数、まとめて届いた"
        "（クライアント未起動だった分も含む）。"
        f"今は {_fmt(now)}。\n"
        f"{body}\n"
        "それぞれのメモを思い出して、まとめて自然にユーザーに話しかけて。"
    )
