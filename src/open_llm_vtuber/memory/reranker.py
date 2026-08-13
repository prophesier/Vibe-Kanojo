"""LLM relevance judge for memory RAG (shared by diary and facts).

Hybrid (vector + lexical) scoring is good at pulling candidates but can't tell an
entry that *is* what the user is talking about from one that merely shares the
words while discussing something else (e.g. an entry about debugging the RAG test
itself). A cheap LLM judges the shortlist and returns only the genuinely relevant
ones, ordered.

Two improvements over a bare reranker:
- It is given the **recent conversation** — but (since the 07-24 tightening)
  strictly for reference resolution: pronouns and ellipses in the latest message
  are resolved against it, while relevance itself must be judged against the
  latest user message ALONE. Judging against the whole window made the judge
  re-inject candidates matching earlier topics turn after turn.
- Listwise / ordered selection (à la RankGPT) — no numeric scores, since LLMs
  rank reliably but calibrate absolute scores poorly. An empty result is the
  natural "nothing relevant" exit (framed in the prompt as the NORMAL outcome),
  so no similarity threshold has to be tuned.

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
# 2026-07-24 tightening (あさひ): the judge was still letting too much through,
# and kept re-injecting candidates that matched EARLIER topics turn after turn.
# The context is therefore demoted to reference-resolution ONLY — relevance must
# come from the latest user message alone — and the default is now framed as
# "an empty array is the normal outcome".
_RERANK_SYSTEM = (
    "あなたは記憶検索の関連性判定ツールです。これは会話ではありません。"
    "ロールプレイやキャラクターとしての応答はせず、判定結果のみを出力してください。\n\n"
    "AIキャラクターへのユーザーの「最後のユーザー発言」と、自動検索された"
    "「{item}の候補」のリストが与えられます。あなたの仕事は、キャラクターが"
    "その発言に返答するにあたって、**候補の具体的な中身を参照しなければ返答の質が"
    "落ちる・事実を誤る**——という厳格な基準で候補を絞り込むことです。\n\n"
    "大原則:\n"
    "- **判定対象は「最後のユーザー発言」ただ一つ**。添付される「最近の会話」は、"
    "最後の発言に含まれる代名詞・指示語・省略を解決するための**参照解決専用**であり、"
    "**関連性の根拠として使ってはならない**。\n"
    "- **前の会話の話題と一致するだけの候補は選ばない**。最後の発言自身がその話題を"
    "持ち出していない限り、それは過ぎた話題である。過去の話題に紐づく候補を"
    "ターンごとに繰り返し追加し続けることが、まさに防ぐべき失敗パターンである。\n"
    "- 関連 = 最後の発言に返答するとき、その候補の中身を参照しないと"
    "**返答が不正確になる・嘘をつく・的外れになる**もの。"
    "「あれば会話が少し豊かになる」程度は関連ではない。\n"
    "- **空配列が通常の結果である**。挨拶・相槌・スキンシップ・短い感情表現・"
    "その場限りの雑談には、原則として何も要らない。選ぶのは例外的な場合だけで、"
    "多くても1〜2件、確信があるものに限る。\n"
    "- 同じ大まかな話題に属するだけでは不十分（「どちらも食べ物の話」程度は無関連）。\n"
    "- キーワードが一致するだけのもの、別の文脈(デバッグ・テスト・検索の失敗等)で"
    "その語に触れただけのもの、出来事そのものではなく「思い出そうとした/検索した」"
    "というメタな言及は**無関連**。\n"
    "- 迷うときは**落とす**。「これが無くても自然に返答できる」なら不要。\n\n"
    "出力は関連する候補だけを、**関連度の高い順**に並べ、各要素に短い理由を一言添える。"
    "関連するものが無ければ空配列を返す（それが通常の結果である）。"
)


# Sentence-mode variant (diary RAG only; facts keep the whole-entry judge).
# Same 07-24 strictness core, but the judge now picks SENTENCES inside each
# relevant diary: ordering exists only at diary granularity (あさひ 08-13 —
# intra-diary sentence order is chronology/semantics and must not be ranked),
# sentences already in context are marked 注入済み and must not be re-picked,
# and picking nothing from a masked diary is an explicitly normal outcome.
_RERANK_SENTENCE_SYSTEM = (
    "あなたは記憶検索の関連性判定ツールです。これは会話ではありません。"
    "ロールプレイやキャラクターとしての応答はせず、判定結果のみを出力してください。\n\n"
    "AIキャラクターへのユーザーの「最後のユーザー発言」と、自動検索された"
    "「日記の候補」のリストが与えられます。各候補の本文は s1, s2, … と"
    "文番号付きで示されます。あなたの仕事は、キャラクターがその発言に返答する"
    "にあたって、**その文の具体的な中身を参照しなければ返答の質が落ちる・"
    "事実を誤る**——という厳格な基準で、関連する日記と、その中の必要な文だけを"
    "選ぶことです。\n\n"
    "大原則:\n"
    "- **判定対象は「最後のユーザー発言」ただ一つ**。添付される「最近の会話」は、"
    "最後の発言に含まれる代名詞・指示語・省略を解決するための**参照解決専用**であり、"
    "**関連性の根拠として使ってはならない**。\n"
    "- **前の会話の話題と一致するだけの候補は選ばない**。最後の発言自身がその話題を"
    "持ち出していない限り、それは過ぎた話題である。\n"
    "- 関連 = 最後の発言に返答するとき、その文の中身を参照しないと"
    "**返答が不正確になる・嘘をつく・的外れになる**もの。"
    "「あれば会話が少し豊かになる」程度は関連ではない。\n"
    "- **空配列が通常の結果である**。挨拶・相槌・スキンシップ・短い感情表現・"
    "その場限りの雑談には、原則として何も要らない。\n"
    "- 迷うときは**落とす**。\n\n"
    "文の選び方:\n"
    "- 日記単位では**関連度の高い順**に並べる。文単位の順序付けはしない"
    "（文は日記内の番号で指定するだけでよい。提示順は原文順で復元される）。\n"
    "- 関連する日記からは、核心の文に加えて、**その文が誤解なく読める分の"
    "前後文も一緒に**選ぶ。1文だけの抜粋は場面・主語・経緯を失いやすい——"
    "場面が伝わる最小のまとまり（目安2〜4文）で切り出すこと。\n"
    "- 「注入済み」と表示された文は**既にキャラクターの文脈内にある**。"
    "再選択してはならない。注入済みの部分だけで返答に足りるなら、"
    "その日記からは何も選ばない（それも正常な結果である）。\n"
    "- 関連する日記が複数あれば複数選んでよい。選択の合計はおおむね"
    "{budget}文を目安にする（厳密な上限ではない）。\n\n"
    "出力は関連する日記だけを関連度の高い順に並べ、各要素に候補番号・"
    "選んだ文番号の配列・短い理由を一言添える。"
    "関連するものが無ければ空配列を返す（それが通常の結果である）。"
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
                            "index": {
                                "type": "integer",
                                "description": "1-based index of the excerpt",
                            },
                            "reason": {
                                "type": "string",
                                "description": "short reason (a few words)",
                            },
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


def _sentence_schema() -> Dict[str, Any]:
    """Sentence-mode schema: ordered diaries, each with picked sentence numbers."""
    return {
        "name": "diary_sentence_relevance",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "array",
                    "description": (
                        "Relevant diaries, most relevant first. Empty if none."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "1-based index of the diary candidate",
                            },
                            "sentences": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "1-based sentence numbers (the sN labels) "
                                    "picked from this diary"
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "short reason (a few words)",
                            },
                        },
                        "required": ["index", "sentences", "reason"],
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
            parts.append(
                "最近の会話（参照解決専用 — 関連性の根拠には使わないこと）:\n"
                f"{context.strip()}"
            )
        parts.append(f"最後のユーザー発言（判定対象はこれのみ）:\n{query}")
        parts.append(f"{self._item}の候補:\n{numbered}")
        user = "\n\n".join(parts)
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": _RERANK_SYSTEM.format(item=self._item),
                    },
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": _schema(self._item),
                },
                temperature=0,
                timeout=self._timeout,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            logger.warning(
                f"[memory_rag] rerank failed ({self._model}, {self._item}): {e}"
            )
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

    async def rerank_sentences(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        context: str = "",
        budget: int = 8,
    ) -> Optional[List[Dict[str, Any]]]:
        """Sentence-mode judge for diary RAG (あさひ 08-13 redesign).

        ``candidates`` is ``[{"id", "date", "sents": [str, ...],
        "injected": [int, ...]}, ...]`` — pre-split sentences (1-based sN
        numbering matches the diary chunk index) plus the sentence numbers
        already injected into context this session (the 既出 mask). Returns
        ``[{"id", "date", "sents", "sentences": [int, ...], "reason"}, ...]``
        in descending diary relevance, with ``sentences`` validated, deduped
        and sorted ascending (intra-diary order is always original order).
        ``[]`` means nothing relevant (the normal outcome). Returns ``None``
        on any API/parse failure — the caller SKIPS injection this round
        (宁缺勿整篇回退; no whole-diary fallback in sentence mode). Never
        raises.
        """
        if not query or not candidates:
            return []
        blocks: List[str] = []
        for i, c in enumerate(candidates):
            mask = sorted({int(n) for n in (c.get("injected") or [])})
            head = f"{i + 1}. [{c.get('date', '')}]"
            if mask:
                head += f"（注入済み: {_compact_sent_ranges(mask)}）"
            body = "\n".join(
                f"s{j + 1}: {s}" + ("　←注入済み" if (j + 1) in mask else "")
                for j, s in enumerate(c.get("sents") or [])
            )
            blocks.append(f"{head}\n{body}")
        parts = []
        if context.strip():
            parts.append(
                "最近の会話（参照解決専用 — 関連性の根拠には使わないこと）:\n"
                f"{context.strip()}"
            )
        parts.append(f"最後のユーザー発言（判定対象はこれのみ）:\n{query}")
        parts.append("日記の候補:\n" + "\n\n".join(blocks))
        user = "\n\n".join(parts)
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": _RERANK_SENTENCE_SYSTEM.format(budget=budget),
                    },
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": _sentence_schema(),
                },
                temperature=0,
                timeout=self._timeout,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            relevant = data.get("relevant")
            if not isinstance(relevant, list):
                raise ValueError(f"malformed judge output: {data!r}")
        except Exception as e:
            logger.warning(f"[memory_rag] sentence rerank failed ({self._model}): {e}")
            return None

        out: List[Dict[str, Any]] = []
        seen: set = set()
        for item in relevant:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(candidates)) or idx in seen:
                continue
            seen.add(idx)
            n_sents = len(candidates[idx].get("sents") or [])
            nums: set = set()
            for raw in item.get("sentences") or []:
                try:
                    n = int(raw)
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= n_sents:
                    nums.add(n)
            if not nums:
                continue  # empty pick = the diary was not actually selected
            picked = dict(candidates[idx])
            picked["sentences"] = sorted(nums)
            picked["reason"] = str(item.get("reason", ""))[:60]
            out.append(picked)
        return out


def _compact_sent_ranges(nums: List[int]) -> str:
    """``[1,2,3,5]`` → ``"s1-3, s5"`` — compact 既出 mask for prompts/labels."""
    if not nums:
        return ""
    parts: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"s{start}" if start == prev else f"s{start}-{prev}")
        start = prev = n
    parts.append(f"s{start}" if start == prev else f"s{start}-{prev}")
    return ", ".join(parts)
