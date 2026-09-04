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
# match the memory/query language. ``{item}`` is the entry kind ("日記"/"事実";
# facts are the only live caller since diaries moved to the paragraph judge).
# Relevance bar recalibrated 08-26 (あさひ) — same treatment as the diary
# variant got on 08-23: the 07-24 strictness ("must-reference or drop", "empty
# is THE normal outcome", "1〜2 picks max") was written to compensate a leaky
# gpt-4o-mini judge; swapping in the compliant 5.6-luna collapsed the accept
# rate 87.5% → 25% on an unchanged pre-judge score distribution (day-one log
# stats, top-hyb median 0.666 vs 0.679). Bar is now "topic actually raised →
# include"; the anti-repeat core (latest-message-only judging,
# reference-resolution-only context) and the anti-patterns stay.
_RERANK_SYSTEM = (
    "あなたは記憶検索の関連性判定ツールです。これは会話ではありません。"
    "ロールプレイやキャラクターとしての応答はせず、判定結果のみを出力してください。\n\n"
    "AIキャラクターへのユーザーの「最後のユーザー発言」と、自動検索された"
    "「{item}の候補」のリストが与えられます。あなたの仕事は、キャラクターが"
    "その発言に返答するにあたって、**中身に触れると返答がより具体的で正確になる**"
    "候補を選ぶことです。\n\n"
    "大原則:\n"
    "- **判定対象は「最後のユーザー発言」ただ一つ**。添付される「最近の会話」は、"
    "最後の発言に含まれる代名詞・指示語・省略を解決するための**参照解決専用**であり、"
    "**関連性の根拠として使ってはならない**。\n"
    "- **前の会話の話題と一致するだけの候補は選ばない**。最後の発言自身がその話題を"
    "持ち出していない限り、それは過ぎた話題である。過去の話題に紐づく候補を"
    "ターンごとに繰り返し追加し続けるのは、防ぐべき失敗パターンである。\n"
    "- 関連 = 最後の発言の話題（人・物・出来事・場所）について、候補に"
    "**具体的な記録**——好み・習慣・経緯・約束——があり、それに触れると返答が"
    "具体的で正確になるもの。事実を誤らないために必須のものはもちろん、"
    "**話題に直接つながる{item}も関連に含める**。\n"
    "- ただし、同じ大分類に属するだけの緩いつながり（「どちらも食べ物の話」程度）、"
    "キーワードが重なるだけのもの、別の文脈（デバッグ・テスト・検索の失敗等）で"
    "その語に触れただけのもの、出来事そのものではなく「思い出そうとした/検索した」"
    "というメタな言及は**無関連**。\n"
    "- 挨拶・相槌・スキンシップ・短い感情表現だけの発言には、通常何も要らない。\n"
    "- 迷うときは:最後の発言がその話題を**実際に持ち出しているなら入れる**、"
    "キーワードが重なるだけなら落とす。\n\n"
    "出力は関連する候補だけを、**関連度の高い順**に並べ、各要素に短い理由を一言添える。"
    "関連するものが無ければ空配列を返す。"
)


# Paragraph-mode variant (diary RAG only; facts keep the whole-entry judge).
# 08-13 introduced in-diary picking at sentence granularity; 08-23 re-grained
# it to PARAGRAPHS. Relevance bar recalibrated same day (あさひ): the 07-24
# strictness ("must-reference or drop", "empty is THE normal outcome") was
# written to compensate a leaky gpt-4o-mini judge — the compliant 5.6-luna
# executed it into near-zero injection on day one, exactly as あさひ
# predicted. Bar is now "topic actually raised → include"; the anti-repeat
# core (latest-message-only judging, reference-resolution-only context) and
# the anti-patterns stay. Ordering still exists only at diary granularity,
# paragraphs already in context are marked 注入済み and must not be re-picked.
_RERANK_SENTENCE_SYSTEM = (
    "あなたは記憶検索の関連性判定ツールです。これは会話ではありません。"
    "ロールプレイやキャラクターとしての応答はせず、判定結果のみを出力してください。\n\n"
    "AIキャラクターへのユーザーの「最後のユーザー発言」と、自動検索された"
    "「日記の候補」のリストが与えられます。各候補の本文は p1, p2, … と"
    "段落番号付きで示されます。あなたの仕事は、キャラクターがその発言に返答する"
    "にあたって、**中身に触れると返答がより具体的で正確になる**段落を、"
    "関連する日記から選ぶことです。\n\n"
    "大原則:\n"
    "- **判定対象は「最後のユーザー発言」ただ一つ**。添付される「最近の会話」は、"
    "最後の発言に含まれる代名詞・指示語・省略を解決するための**参照解決専用**であり、"
    "**関連性の根拠として使ってはならない**。\n"
    "- **前の会話の話題と一致するだけの候補は選ばない**。最後の発言自身がその話題を"
    "持ち出していない限り、それは過ぎた話題である。過去の話題に紐づく候補を"
    "ターンごとに繰り返し追加し続けるのは、防ぐべき失敗パターンである。\n"
    "- 関連 = 最後の発言の話題（人・物・出来事・場所）について、日記に"
    "**具体的な記録**——経緯・評価・約束・感想——があり、それに触れると返答が"
    "具体的で正確になるもの。事実を誤らないために必須のものはもちろん、"
    "**話題に直接つながる思い出も関連に含める**。\n"
    "- ただし、同じ大分類に属するだけの緩いつながり（「どちらも食べ物の話」程度）、"
    "キーワードが重なるだけのもの、別の文脈（デバッグ・テスト・検索の失敗等）で"
    "その語に触れただけのもの、「思い出そうとした/検索した」というメタな言及は"
    "**無関連**。\n"
    "- 挨拶・相槌・スキンシップ・短い感情表現だけの発言には、通常何も要らない。\n"
    "- 迷うときは:最後の発言がその話題を**実際に持ち出しているなら入れる**、"
    "キーワードが重なるだけなら落とす。\n\n"
    "段落の選び方:\n"
    "- 日記単位では**関連度の高い順**に並べる。段落単位の順序付けはしない"
    "（段落は日記内の番号で指定するだけでよい。提示順は原文順で復元される）。\n"
    "- 段落は場面のまとまりとして書かれている。核心の段落だけで意味が通るなら"
    "1段落でよい。前後がないと誤解される場合のみ、隣接する段落を足す。\n"
    "- 「注入済み」と表示された段落は**既にキャラクターの文脈内にある**。"
    "再選択してはならない。注入済みの部分だけで返答に足りるなら、"
    "その日記からは何も選ばない。\n"
    "- 関連する日記が複数あれば複数選んでよい。選択の合計はおおむね"
    "{budget}段落を目安にする（厳密な上限ではない）。\n\n"
    "出力は関連する日記だけを関連度の高い順に並べ、各要素に候補番号・"
    "選んだ段落番号の配列・短い理由を一言添える。"
    "関連するものが無ければ空配列を返す。"
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
    """Paragraph-mode schema: ordered diaries, each with picked paragraph
    numbers (field/function names keep the sentence-era spelling)."""
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
                                    "1-based paragraph numbers (the pN labels) "
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
        # max_retries=0: the per-request timeout below is only a real bound
        # without the library's default 2 retries (20s × 3 + backoff ≈ 60s+ —
        # observed as 56s/88s RAG stalls when a flaky local network wedged
        # individual connections, 09-02). A failed judge degrades to the
        # score-based fallback for that turn, which is the cheaper failure.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url or None, max_retries=0
        )
        self._model = model
        self._item = item_label
        self._timeout = timeout
        # gpt-4o-era judges run temperature=0 for determinism; gpt-5.x
        # reasoning models 400-reject an explicit temperature. Omit the param
        # for anything that isn't a known temperature-taking family (their
        # defaults are deterministic enough for a structured judge).
        self._temperature_kwargs: Dict[str, Any] = (
            {"temperature": 0} if model.lower().startswith("gpt-4") else {}
        )

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
                timeout=self._timeout,
                **self._temperature_kwargs,
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
        budget: int = 4,
    ) -> Optional[List[Dict[str, Any]]]:
        """Paragraph-mode judge for diary RAG (08-13 sentence redesign,
        re-grained to paragraphs 08-23 — names keep the sentence spelling).

        ``candidates`` is ``[{"id", "date", "sents": [str, ...],
        "injected": [int, ...]}, ...]`` — pre-split paragraphs (1-based pN
        numbering matches the diary chunk index) plus the paragraph numbers
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
                f"p{j + 1}: {s}" + ("　←注入済み" if (j + 1) in mask else "")
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
                timeout=self._timeout,
                **self._temperature_kwargs,
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
    """``[1,2,3,5]`` → ``"p1-3, p5"`` — compact 既出 mask for prompts/labels."""
    if not nums:
        return ""
    parts: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append(f"p{start}" if start == prev else f"p{start}-{prev}")
        start = prev = n
    parts.append(f"p{start}" if start == prev else f"p{start}-{prev}")
    return ", ".join(parts)
