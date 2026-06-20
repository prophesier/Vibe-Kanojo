"""LLM relevance judge for memory RAG (shared by diary and facts).

Hybrid (vector + lexical) scoring is good at pulling candidates but can't tell an
entry that *is* what the user is talking about from one that merely shares the
words while discussing something else (e.g. an entry about debugging the RAG test
itself). A cheap LLM judges the shortlist and returns only the genuinely relevant
ones, ordered.

Two improvements over a bare reranker:
- It is given the **recent conversation**, not just the isolated last message, so
  it knows what is actually being discussed. A lone sentence has too little signal
  and anything keyword-adjacent looks relevant; with context the judge can tell.
- Listwise / ordered selection (à la RankGPT) — no numeric scores, since LLMs
  rank reliably but calibrate absolute scores poorly. An empty result is the
  natural "nothing relevant" exit, so no similarity threshold has to be tuned.

The same class serves both subsystems via ``item_label`` ("日記" / "事実"); diary
and facts each construct their own instance over their own data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

# System instruction for the judge. Plain tool prompt — no roleplay. Japanese to
# match the memory/query language. ``{item}`` is the entry kind ("日記"/"事実").
_RERANK_SYSTEM = (
    "あなたは記憶検索の関連性判定ツールです。これは会話ではありません。"
    "ロールプレイやキャラクターとしての応答はせず、判定結果のみを出力してください。\n\n"
    "AIキャラクターとユーザーの「最近の会話」と、自動検索された「{item}の候補」の"
    "リストが与えられます。あなたの仕事は、キャラクターが**会話の最後のユーザー発言に"
    "返答するにあたって**、各候補が実際に役立つ・参照する必要があるかを判定することです。\n\n"
    "判定基準:\n"
    "- 関連 = その候補の内容が、今まさに話している事柄・最後の発言が求めていることに"
    "直接関係し、返答に**具体的に役立つ**。\n"
    "- **同じ大まかな話題に属するというだけでは選ばない**（例:「どちらも食べ物の話」程度では"
    "不十分）。その候補が、今の具体的なやり取りに対して具体的で有用な中身を持つ場合のみ選ぶ。\n"
    "- **会話の流れを踏まえて**判断する。最後の一文だけを見て語句が一致するからといって"
    "関連とは限らない。今の話題と無関係なら入れない。\n"
    "- 単にキーワードが一致するだけ、または別の文脈(デバッグ・テスト・検索の失敗・"
    "うまく思い出せなかった等)でその語に触れているだけのものは**関連ではない**。"
    "出来事そのものではなく、それを「思い出そうとした/検索した」というメタな言及は除外する。\n"
    "- 迷うときは**入れない**。「これが無くても自然に返答できる」なら不要。"
    "関連が確実なものだけに絞る。\n\n"
    "出力は関連する候補だけを、**関連度の高い順**に並べ、各要素に短い理由を一言添える。"
    "関連するものが一つも無ければ空配列を返す。"
)


def _schema(item: str) -> Dict[str, Any]:
    """Structured-output schema: ordered relevant items (1-based index + reason)."""
    return {
        "name": "memory_relevance",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "array",
                    "description": f"Relevant {item} excerpts, most relevant first. Empty if none.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer", "description": "1-based index of the excerpt"},
                            "reason": {"type": "string", "description": "short reason (a few words)"},
                        },
                        "required": ["index", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["relevant"],
            "additionalProperties": False,
        },
    }


class MemoryReranker:
    """Cheap, context-aware LLM judge that filters/orders shortlisted candidates."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        model: str = "gpt-4o-mini",
        item_label: str = "日記",
        timeout: float = 20.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._item = item_label
        self._timeout = timeout

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        context: str = "",
    ) -> Optional[List[Dict[str, Any]]]:
        """Judge ``candidates`` against the conversation, not just ``query``.

        ``candidates`` is ``[{"id", "date", "content"}, ...]`` (already shortlisted
        by hybrid score). ``context`` is the recent conversation (most recent last);
        ``query`` is the latest user message. Returns the relevant subset as
        ``[{"id", "date", "content", "reason"}, ...]`` in descending relevance, or
        ``[]`` when none are relevant. Returns ``None`` on any failure so the caller
        can fall back to score-based selection. Never raises.
        """
        if not query or not candidates:
            return []
        numbered = "\n".join(
            f"{i + 1}. [{c.get('date', '')}] {(c.get('content') or '').strip()}"
            for i, c in enumerate(candidates)
        )
        parts = []
        if context.strip():
            parts.append(f"最近の会話:\n{context.strip()}")
        parts.append(f"最後のユーザー発言:\n{query}")
        parts.append(f"{self._item}の候補:\n{numbered}")
        user = "\n\n".join(parts)
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM.format(item=self._item)},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_schema", "json_schema": _schema(self._item)},
                temperature=0,
                timeout=self._timeout,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            logger.warning(f"[memory_rag] rerank failed ({self._model}, {self._item}): {e}")
            return None

        out: List[Dict[str, Any]] = []
        seen: set = set()
        for item in data.get("relevant", []):
            try:
                idx = int(item.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                picked = dict(candidates[idx])
                picked["reason"] = str(item.get("reason", ""))[:60]
                out.append(picked)
        return out
