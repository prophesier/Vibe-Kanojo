"""LLM relevance judge for diary RAG.

Hybrid (vector + lexical) scoring is good at pulling candidate diaries but can't
tell a diary that *is* the event the user asks about from one that merely
mentions the words while discussing something else (e.g. a diary about debugging
the RAG test itself). A cheap LLM judges the shortlisted candidates against the
user's actual message and returns only the genuinely relevant ones, ordered.

Listwise / ordered selection (à la RankGPT) — no numeric scores, since LLMs rank
reliably but calibrate absolute scores poorly. An empty result is the natural
"nothing relevant" exit, so no similarity threshold has to be tuned.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

# System instruction for the judge. Plain tool prompt — no roleplay. Japanese to
# match the diary/query language.
_RERANK_SYSTEM = (
    "あなたは記憶検索の関連性判定ツールです。これは会話ではありません。"
    "ロールプレイやキャラクターとしての応答はせず、判定結果のみを出力してください。\n\n"
    "ユーザーの「今の発言」と、過去のセッションから自動検索された「日記の抜粋」のリストが"
    "与えられます。各抜粋について、それが今の発言に**実際に関連し、答える助けになるか**を"
    "判定してください。\n\n"
    "判定基準:\n"
    "- 関連 = その抜粋の内容が、ユーザーが尋ねている事柄そのものに触れている。\n"
    "- 単にキーワードが一致するだけ、または別の話題(デバッグ・テスト・検索の失敗・"
    "うまく思い出せなかった等)の中でその語に触れているだけのものは**関連ではない**。"
    "出来事そのものではなく、それを「思い出そうとした/検索した」というメタな言及は除外する。\n"
    "- 迷うときは、その抜粋が今の発言に本当に役立つかで判断する。役立たないなら入れない。\n\n"
    "出力は関連する抜粋だけを、**関連度の高い順**に並べること。各要素に短い理由を一言添える。"
    "関連するものが一つも無ければ空配列を返す。"
)

# Structured-output schema: ordered relevant items (1-based index + short reason).
_RERANK_SCHEMA = {
    "name": "diary_relevance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "array",
                "description": "Relevant excerpts, most relevant first. Empty if none.",
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


class DiaryReranker:
    """Cheap LLM judge that filters/orders shortlisted diary candidates."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        model: str = "gpt-4o-mini",
        timeout: float = 20.0,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._timeout = timeout

    async def rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Judge ``candidates`` against ``query``.

        ``candidates`` is ``[{"id", "date", "content"}, ...]`` (already shortlisted
        by hybrid score). Returns the relevant subset as
        ``[{"id", "date", "content", "reason"}, ...]`` in descending relevance,
        or ``[]`` when none are relevant. Returns ``None`` on any failure so the
        caller can fall back to score-based selection. Never raises.
        """
        if not query or not candidates:
            return []
        numbered = "\n".join(
            f"{i + 1}. [{c.get('date', '')}] {(c.get('content') or '').strip()}"
            for i, c in enumerate(candidates)
        )
        user = f"今の発言:\n{query}\n\n過去の日記の抜粋:\n{numbered}"
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_schema", "json_schema": _RERANK_SCHEMA},
                temperature=0,
                timeout=self._timeout,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            logger.warning(f"[diary_rag] rerank failed ({self._model}): {e}")
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
