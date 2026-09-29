"""Three-layer persistent memory for Open-LLM-VTuber.

Layer 1 – sliding window: loaded by BasicMemoryAgent at session start.
Layer 2 – structured facts: key assertions about the user, stored in facts.json.
Layer 3 – session diaries: per-session mood summaries, stored in diaries/.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import unicodedata
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set
from loguru import logger

from ..chat_history_manager import SESSION_UID_RE, strip_context_excluded
from .vector_index import VectorIndex, extract_keywords


def _split_paragraphs(text: str) -> List[str]:
    """Split diary content into paragraph chunks on blank lines.

    Paragraph redesign (あさひ 08-23, was pysbd sentence splitting): diaries
    are written in tidy natural paragraphs now, and a paragraph is the scene-
    sized unit a recall actually needs — sentence excerpts kept losing
    subject/setting. A legacy diary with no blank-line breaks comes back as
    ONE whole-text chunk (short early diaries inject whole, by design).
    Single newlines inside a paragraph are preserved verbatim.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# Matches the timestamp tags _to_text_prompt / history load prepend to user
# messages: "[HH:MM]" since 09-29, plus the older "[YYYY-MM-DD HH:MM:SS 曜]"
# shape still held in memory by a session that predates the change (disk
# records carry no tag; it is re-rendered at load).
_TIMESTAMP_RE = re.compile(
    r"^\[(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2}(?::\d{2})?(?: \w+)?\]\s*",
    re.MULTILINE,
)

# A session uid: "YYYY-MM-DD_HH-MM-SS_<hex>". Diary files and session files
# are named by it; directory scans filter on this so a stray file dropped
# into diaries/ or chat_history/ never gets injected or embedded
# (あさひ 08-20).
# Single source in chat_history_manager (get_history_list allowlists on the
# same pattern since 09-02); the old local name stays for the callers here.
_SESSION_UID_RE = SESSION_UID_RE


_FACT_EXTRACT_SYSTEM = (
    "あなたはメモリ抽出ツールです。これは会話ではありません。"
    "ロールプレイ、キャラクターとしての応答、感情表現タグ（[neutral]、[smirk]等）、"
    "前置き、コメント、Markdown装飾、コードフェンス（```）は一切禁止です。\n"
    "出力は**生のJSON配列のみ**。それ以外のテキストを1文字でも含めると失敗とみなされます。\n\n"
    "タスク：会話からユーザーに関する**長期的に価値のある**事実を抽出する。\n\n"
    "【抽出すべき情報】\n"
    "- 個人情報：出身地、学歴（学部・専攻など）、職業、年齢層\n"
    "- 価値観、信念、性格特性\n"
    "- 長期的な好み・趣味・習慣（その場限りでなく繰り返し見られる傾向、"
    "あるいはユーザーが明示的に表明したもの）\n"
    "- 人間関係\n"
    "- 進行中の長期プロジェクト、使用ツール・技術\n"
    "- 重要な約束・合意事項\n"
    "- ユーザーの目標・課題・悩み\n"
    "- 代表性のある経験：初めての出来事、転機となる事象、"
    "ユーザーが「これは大事」と示したこと\n\n"
    "【統合の原則】\n"
    "今回抽出する新しい事実の中に、互いに密接に関連するものが2つ以上あれば、"
    "別々の項目として出力せず、1つに統合した事実として書く。\n"
    "※ このバッチ内での統合のみを指す。既存の事実リストとの整合・統合は"
    "ここでは行わない。\n"
    "例：\n"
    "× 「ユーザーはAアニメを視聴」「ユーザーはBアニメを視聴」「ユーザーはCアニメを視聴」\n"
    "○ 「ユーザーはA、B、Cのアニメを視聴している」\n"
    "× 「ユーザーは物理学部出身」「ユーザーは物理学を専攻」\n"
    "○ 「ユーザーは物理学部出身で物理学を専攻していた」\n\n"
    "【抽出しない情報】\n"
    "日々の些細な行動・状態は通常は抽出しない。会話履歴と日記に既に残るため、"
    "facts に書くと冗長で、本質的な情報がノイズに埋もれる：\n"
    "- 今日食べたもの、今日のゲーム進捗、今日の体調、今日着た服\n"
    "- その場限りの感情・状況\n"
    "- 一度きりの細かい話題内容\n\n"
    "ただし、長期的価値を持つ場合は例外的に抽出する：\n"
    "- 初めての経験（「初めて〇〇を食べた」など）\n"
    "- ユーザーが明示的に表明した好み・嫌い（「〇〇が好きだ」と発言）\n"
    "- 繰り返し見られる習慣・パターン\n\n"
    "【特定の飲食店・メニューへの評価は抽出禁止】\n"
    "特定の店や注文メニューへの評価・感想（美味しい/まずい・量・値段感・"
    "リピート意向など）は、明示的に表明されていても**抽出しない**。"
    "上の例外規定よりこの禁止が優先される。"
    "店の評価はキャラクター自身が注文の場で別経路の記憶として記録しており、"
    "ここで抽出すると古い評価との重複が積もるだけである。\n\n"
    "判断基準：「1ヶ月後にもこの情報を参照する価値があるか？」を問う。"
    "No なら抽出しない。\n\n"
    "【ガイドライン】\n"
    "- 中国語・日本語・英語が混在していても、すべての言語の発言を対象にする\n"
    "- 既存の事実リストが提供される場合、それらをそのまま繰り返さない\n"
    "- 既存の事実には先頭に `[importance]`（現在の重要度）が付いている。"
    "新しい事実の重要度を判定する際の一貫性の参考にしてよい。\n"
    "- `[archive]` の付いた既存事実は「古い・期限切れだが残してある」もの。"
    "同じ内容を再抽出しない（重複扱い）。会話中でその内容が現在の事実として"
    "明確に再確立された場合のみ、新しい事実として抽出してよい。\n\n"
    "【重要度の判定（importance）】\n"
    '抽出する各事実に importance を付ける。値は "high" か "low" のどちらか。\n'
    '（"user" は使わない——それは人間が手動で指定する専用の値で、あなたが付けてはいけない。）\n'
    '- "high" = ユーザー像を定義する中核的な事実で、常にキャラクターの念頭にあるべきもの:\n'
    "  学歴・専攻・資格、職業・専門スキル、出身地、年齢層、家族・重要な人間関係、\n"
    "  価値観・信念・性格特性、長期的な目標や進行中の重要プロジェクト、\n"
    "  重要な約束・合意、人生の節目・転機・トラウマ。\n"
    '- "low" = 覚えておく価値はあるが、関連する話題のときに思い出せれば十分な事実:\n'
    "  具体的な好みの細部、個別のエピソード、特定の物事（食べた物・買った物・観た作品など）、\n"
    "  中核とまでは言えない習慣や傾向。\n"
    '迷ったら "low"。"high" は本当に常時参照する価値があるものだけに厳選する。\n\n'
    "【タグ（tags）】\n"
    "各事実に tags（0〜3個、短い名詞）を付けてよい。"
    "プロンプトに「既存のタグ一覧」がある場合は、そこにある名前を優先して再利用し、"
    "似た意味の新しい名前を作らない。該当する名前が無ければ空配列でよい。\n\n"
    "**出力形式（厳守）**：\n"
    '[{"fact": "ユーザーは物理学部出身で物理学を専攻していた", "importance": "high", "tags": ["学歴"]}, '
    '{"fact": "ユーザーは白黒のポテトチップスが好き", "importance": "low", "tags": ["食べ物", "好み"]}]\n'
    "本当に新しい事実が1件もない場合のみ、空の配列のみを出力する: []\n"
    '繰り返す：JSON配列のみ。各要素は必ず "fact" と "importance" を持つ（"tags" は任意）。'
    '"importance" は "high" か "low"。`[`で始まり`]`で終わる。他のテキスト・記号は一切含めない。'
)

_DIARY_SYSTEM = (
    "あなたは記憶アシスタントです。AIキャラクターの一人称視点から、"
    "この会話セッションを日記としてまとめてください。\n\n"
    "【この日記の用途（重要）】\n"
    "この日記は後で2通りに使われる：(1) 最近の数件がシステムプロンプトに常時注入される、"
    "(2) それより古いものは、後の会話でユーザーの発言に関連した時だけ自動で検索されて参照される。"
    "そのため、後から思い出したり検索したりする価値のある**具体的な出来事・物事は、"
    "固有の言葉のまま具体的に書き残す**こと（例:「大きい飲み物」ではなく「グランドサイズのコーラ」、"
    "「お菓子」ではなく「白黒のポテトチップス」のように、特徴的な固有名・数量・状況を残す）。"
    "抽象化して要約しすぎると、後で検索に引っかからず、思い出せなくなる。\n\n"
    "【長さの制約】\n"
    "全体で**400〜700字程度**を目安に。最大でも900字以内。"
    "具体性は保ちつつ、逐語的な再現・引用は避けて簡潔にまとめる。\n\n"
    "【記録する内容】\n"
    "以下のうち、このセッションで**実際に発生したもの**だけを記録する。"
    "該当しないカテゴリは省略する（穴埋め式に全項目を埋める必要はない）：\n"
    "- 具体的な出来事・エピソード（何があったか。固有名・固有の物事をそのまま残す）\n"
    "- 未解決の約束・タスク・宿題\n"
    "- ユーザーが示した判断パターン・選好・価値観\n"
    "- 感情の節目（嬉しさ・落ち込み・葛藤・転機）\n"
    "- AI（あなた自身）の誤り・謝罪、ユーザーに訂正された事柄\n\n"
    "【触れた話題の書き方】\n"
    "セッションで触れた話題は本文の中に自然な文章として織り込む。"
    "「話した話題：」のような見出しや箇条書きにはしない。\n"
    "- 軽く触れただけの話題は一言で済ませる（「〇〇の話題にも触れた」程度）。\n"
    "- 深く議論した話題は、**論点・結論・ユーザーの主な意見**に加えて、"
    "**後で参照されそうな具体的な事項（固有名・物・出来事）も一言添える**。"
    "ただし会話の逐語的な再現や、AI自身の応答の引き写しは不要。\n\n"
    "【省いてよい内容】\n"
    "繰り返しの数値（ゲームの細かい進捗など）や、後で話題に上らないことが明らかな"
    "純粋な雑事は省いてよい。ただし「何を食べた・買った・見た」のような具体は、"
    "後で話題に上りうるので、特徴的なら固有の物事として一言残す。\n\n"
    "【時刻表現】\n"
    "「今日」「本日」のような日付レベルの曖昧表現は避ける。"
    "ただし「19時42分から21時5分の会話で」のような分単位の精密表現も避ける——"
    "日記冒頭の日付・セッション時刻と二重になるため。\n"
    "代わりに「夕方頃」「深夜に」「午前中の」「昼過ぎから」のような"
    "時間帯レベルの言葉を使う。\n"
    "※ 一日に複数のセッションがある場合があるため、「今日」では他のセッションと"
    "区別がつかない。時間帯レベルなら区別できる。\n\n"
    "【文体】\n"
    "人格設定が提供されている場合、その口調・性格・思考パターンを反映する。"
    "自然な文章で。[neutral]などの表現タグは含めない。\n\n"
    "出力は日記本文のみ。見出し・装飾・前置きは一切含めない。"
)

# Framing for the always-on facts block appended to the diary system prompt.
# Without it the model tends to either copy facts verbatim into the diary or
# treat them as things that happened this session. Placed right before the
# conversation so it reads as background context, not new events.
_DIARY_FACTS_NOTE = (
    "【参考：ユーザーに関する既知の長期記憶】\n"
    "以下は、すでに長期記憶として保存されているユーザーに関する事実。"
    "日記を書く際、ユーザーの人物像を踏まえた自然な記述をするための"
    "背景知識として参照してよい。\n"
    "ただし厳守すること：\n"
    "- これらは過去に確定済みの情報であり、今回のセッションで起きた出来事ではない。\n"
    "- 日記本文にそのまま列挙・コピーしてはならない。"
    "今回の会話で新しく分かったこと・起きたことだけを日記に書く。"
    "既知の事実は、今回それに関連する話題が出たときの文脈理解にのみ使う。\n"
    "- 各事実の冒頭の `[YYYY-MM-DD]` は記録日であり、出来事が起きた日ではない。"
)

_FACT_PRUNE_SYSTEM = (
    "あなたはメモリ整理ツールです。これは会話ではありません。"
    "ロールプレイ、キャラクターとしての応答、感情表現タグ（[neutral]、[smirk]等）、"
    "前置き、コメント、Markdown装飾、コードフェンス（```）は一切禁止です。\n"
    "出力は**生のJSON配列のみ**。それ以外のテキストを1文字でも含めると失敗とみなされます。\n\n"
    "タスク：ユーザーに関する事実リストが保存上限を超えたため、"
    "最も価値の低い項目を選んで削除する。\n\n"
    "各事実には記録日が付いている（形式: [YYYY-MM-DD]）。"
    "この日付は事実が記録された日であり、出来事が起きた日ではない点に注意。\n\n"
    "【絶対に削除してはならない（最高優先度で保持）】\n"
    "記録日に関わらず、以下に該当する情報はユーザーの本質を定義する：\n"
    "- 学歴・資格・試験合格（JLPT合格、IT資格、卒業学部・専攻など）\n"
    "- 職業・キャリア上の達成、専門スキル\n"
    "- 出身地、年齢層、家族構成、重要な人間関係\n"
    "- 価値観、信念、性格特性、長期的な趣味\n"
    "- 過去の重要な経験・トラウマ・転機となった出来事\n"
    "- 健康状態・宗教・政治信条など個人を定義する基本属性\n"
    "これらは「古いから」「最近触れていないから」「日付が古いから」"
    "という理由で削除してはならない。\n\n"
    "【優先的に削除】\n"
    "- 新しい事実によって上書き・無効化された古い情報\n"
    "  （例: 古い「Aプロジェクト取り組み中」と新しい「Bプロジェクトに移行」が両方ある場合、古い方）\n"
    "- 時間の経過により時効・陳腐化した一時的情報\n"
    "  （例: 数週間前の「明日締切のタスク」、過去の一日限りの予定）\n"
    "- 同じ内容の重複（古い方）\n"
    "- 今この瞬間の状況・行動のうち、ユーザー像を理解する上で重要でないもの\n"
    "  （例: 「今プレイ中のゲーム名」「今夜食べたメニュー」など）\n\n"
    "【重要原則】\n"
    "「新しい」ことそれ自体は重要性の指標ではない。"
    "**新しいが些細な情報より、古いが本質的な情報の方が常に価値が高い**。\n"
    "例：「[2026-03-01] JLPT N1合格」のような達成事項は、"
    "「[2026-05-30] 今プレイ中のゲーム名」のような一時情報より、"
    "たとえ前者が古くても優先的に保持する。\n\n"
    "【記録日の使い方】\n"
    "削除候補の重要性が完全に同じレベルで甲乙つけがたい場合に限り、"
    "「より新しい記録日のものを残す」をタイブレーカーとして使ってよい。"
    "それ以外で日付を主要な判断基準にしてはならない。\n\n"
    "【複合事実の扱い（重要）】\n"
    "1つの事実に複数の独立した情報がまとまっている場合、"
    "**その一部だけが古くなっていても削除しない**。"
    "全体を削除すると、まだ有効な情報まで失うため。\n"
    "例：「ユーザーはAプロジェクトに取り組み中で、Bツールを使用している」のうち、"
    "Aだけが古い情報になっていても、Bツールの情報は現在も有効。"
    "このような複合事実は削除候補から除外する。\n"
    "削除してよいのは**事実全体が陳腐化・無効化されている**ケースのみ。\n\n"
    "**出力形式（厳守）**：\n"
    "削除するインデックス（数字）のみをJSON配列で出力する: [3, 7, 12]\n"
    "繰り返す：JSON配列のみ。他のテキスト・記号は一切含めない。"
)


def model_short_name(model_id: str) -> str:
    """Compact display form of a model id — the annotation style あさひ chose
    (統一モデル番号): claude-opus-4-6 → opus4.6, claude-opus-5 → opus5,
    gpt-5.1 → gpt5.1, gpt-5.6-sol → gpt5.6-sol. Number segments join with
    dots (the first attaches bare), non-number tails keep a hyphen. Unknown
    ids fall through recognizably instead of erroring."""
    m = (model_id or "").strip().lower()
    if not m:
        return "?"
    if m.startswith("claude-"):
        m = m[len("claude-") :]
    parts = m.split("-")
    out = parts[0]
    seen_num = False
    for p in parts[1:]:
        if re.fullmatch(r"\d+(\.\d+)*", p):
            out += p if not seen_num else "." + p
            seen_num = True
        else:
            out += "-" + p
    return out


class PersistentMemoryManager:
    """Manages facts.json and per-session diaries for one character (conf_uid)."""

    # Process-wide set tracking which conf_uids have a backfill currently
    # running. Prevents concurrent connections from kicking off duplicate
    # backfills against the same character.
    _backfill_in_progress: ClassVar[Set[str]] = set()

    # Class-level defaults so partially-constructed instances (tests build
    # via __new__) degrade gracefully: no chat model recorded, no map
    # snapshot loaded. Real instances overwrite both in __init__/boot.
    _chat_model: str = ""
    _model_map_snapshot: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        conf_uid: str,
        *,
        max_facts: int = 50,
        diary_count: int = 5,
        recent_sessions: int = 3,
        diary_rag_config: Any = None,
        facts_rag_config: Any = None,
        embed_api_key: str = "",
        embed_base_url: str = "",
        long_fact_chars: int = 500,
    ) -> None:
        self._conf_uid = conf_uid
        self._max_facts = max_facts
        # Long-fact rule (あさひ 09-29): a fact longer than this many chars
        # appears in every LIST path (auto recall, memory_search hits, the
        # Uber related-facts section) as its title + id only; memory_read
        # gives the full text. 0 disables the collapse. The resident header
        # never collapses (resident = meant to be read in full).
        self._long_fact_chars = max(0, int(long_fact_chars or 0))
        self._diary_count = diary_count
        self._recent_sessions = recent_sessions
        self._base_dir = os.path.join("chat_history", conf_uid)
        self._facts_path = os.path.join(self._base_dir, "facts.json")
        self._diaries_dir = os.path.join(self._base_dir, "diaries")
        # Optional dedicated LLM for memory tasks (diary/fact). When
        # set, _call_llm uses it instead of the chat model — keeps big uncached
        # one-shot memory calls off the (pricier) chat model. set via setter.
        self._memory_llm: Any = None
        # reasoning_effort sent on memory-task calls. Empty = don't send (use the
        # model's default). gpt-5.1 defaults to "none" (no reasoning → lazy "[]"
        # for extraction), so it needs an explicit "low"; gpt-5.5 defaults to
        # "medium" so empty is fine there. Set from config via the setter.
        self._memory_reasoning_effort: str = ""
        # The CHAT model's id (set by the agent at wiring) — the "experiencer"
        # recorded on fact history events and used for the in-progress session
        # in the model map snapshot. Distinct from _memory_llm (the writer of
        # diaries/extractions).
        self._chat_model: str = ""
        # Session→model attribution snapshot (chat_history/_model_map.json,
        # refreshed at boot by refresh_model_map). Keys are session uids.
        self._model_map_snapshot = {}

        # Diary RAG (long-tail recall). Built only when enabled and an embedding
        # key resolves; otherwise stays None and every RAG call is a no-op so a
        # missing key degrades gracefully instead of breaking memory.
        self._rag_cfg = diary_rag_config
        self._diary_index: Optional[VectorIndex] = None
        if diary_rag_config is not None and getattr(diary_rag_config, "enabled", False):
            if embed_api_key:
                self._diary_index = VectorIndex(
                    os.path.join(self._base_dir, "diaries.embeddings.json"),
                    api_key=embed_api_key,
                    base_url=embed_base_url,
                    model=getattr(
                        diary_rag_config, "embedding_model", "text-embedding-3-small"
                    ),
                )
                # Echo auto_inject so a typo'd conf key (silently dropped by
                # pydantic) is visible at startup.
                logger.info(
                    "[memory] Diary RAG enabled (auto_inject={}).".format(
                        getattr(diary_rag_config, "auto_inject", True)
                    )
                )
            else:
                logger.warning(
                    "[memory] diary_rag enabled but no embedding API key resolved "
                    "(set diary_rag.openai_api_key or configure the openai_llm provider); "
                    "RAG disabled."
                )

        # Optional LLM relevance judge over the hybrid shortlist (reuses the
        # embedding key/endpoint). None → fall back to pure score-based selection.
        self._diary_reranker = None
        if (
            self._diary_index is not None
            and getattr(diary_rag_config, "rerank_enabled", False)
            and embed_api_key
        ):
            from .reranker import MemoryReranker

            model = getattr(diary_rag_config, "rerank_model", "gpt-4o-mini")
            self._diary_reranker = MemoryReranker(
                api_key=embed_api_key,
                base_url=embed_base_url,
                model=model,
                item_label="日記",
            )
            logger.info(f"[memory] Diary RAG reranker enabled ({model}).")

        # Facts RAG — a separate, independent subsystem (own index, own config,
        # own reranker). Index ALL facts; tier filtering happens only at
        # injection (user/llm-tier facts stay in the header, `low` go to RAG).
        self._facts_rag_cfg = facts_rag_config
        self._facts_index: Optional[VectorIndex] = None
        self._facts_reranker = None
        if facts_rag_config is not None and getattr(facts_rag_config, "enabled", False):
            if embed_api_key:
                self._facts_index = VectorIndex(
                    os.path.join(self._base_dir, "facts.embeddings.json"),
                    api_key=embed_api_key,
                    base_url=embed_base_url,
                    model=getattr(
                        diary_rag_config, "embedding_model", "text-embedding-3-small"
                    ),
                )
                logger.info(
                    "[memory] Facts RAG enabled (auto_inject={}).".format(
                        getattr(facts_rag_config, "auto_inject", True)
                    )
                )
                if getattr(facts_rag_config, "rerank_enabled", False):
                    from .reranker import MemoryReranker

                    fmodel = getattr(facts_rag_config, "rerank_model", "gpt-4o-mini")
                    self._facts_reranker = MemoryReranker(
                        api_key=embed_api_key,
                        base_url=embed_base_url,
                        model=fmodel,
                        item_label="事実",
                    )
                    logger.info(f"[memory] Facts RAG reranker enabled ({fmodel}).")
            else:
                logger.warning(
                    "[memory] facts_rag enabled but no embedding API key resolved; "
                    "facts RAG disabled."
                )
        # Session UIDs currently loaded in the agent's sliding window — their
        # diaries are excluded from the injected memory block to avoid
        # duplicating content the agent already has verbatim.
        self._active_session_uids: set = set()
        # The session that is currently being written to (i.e. in progress).
        # Backfill skips this UID so it doesn't summarise an unfinished session.
        self._current_session_uid: str = ""
        # Frozen header-facts snapshot (captured on first prompt build, reset
        # only when backfill settles). The system-prompt facts block must stay
        # byte-stable within a session or every facts.json write (extraction,
        # the character's own memory_* edits, hand edits) busts the prompt
        # cache. Disk stays the live truth for writes and RAG retrieval.
        self._header_snapshot: Optional[List[Dict[str, Any]]] = None
        # Frozen diaries-block snapshot, same discipline and lifecycle as the
        # facts snapshot above (rendered string — no downstream consumers).
        self._diaries_snapshot: Optional[str] = None
        # Frozen tag-vocabulary block (あさひ 09-29), same lifecycle: the
        # list of every tag in use rides block 2 so the character reuses
        # names instead of coining near-duplicates; it catches up at the
        # next boot like the facts header.
        self._tags_snapshot: Optional[str] = None
        # The frozen header is also persisted to disk at freeze time, and a
        # --resume boot RELOADS it instead of re-reading live facts.json
        # (あさひ 09-02): mid-session fact writes changed the rebuilt header
        # on resume and busted the whole prefix cache — resume means "same
        # session", so it must mean "same bytes". Fresh boots overwrite the
        # file; a missing/corrupt file degrades to a live freeze (one-time
        # rewrite, logged). Lives OUTSIDE the conf dir — the excerpts.json /
        # _model_map.json precedent — on top of get_history_list's uid
        # allowlist: session scanners must never even see it.
        self._resume_boot = os.environ.get("OLV_RESUME") == "1"
        self._header_snapshot_path = os.path.join(
            "chat_history", f"_prompt_header_snapshot_{conf_uid}.json"
        )
        # Read once and cached: the facts half freezes (and re-persists the
        # file) before the diaries half is even asked for, so the diaries
        # restore must NOT re-read the just-overwritten file.
        self._resume_snap_loaded = False
        self._resume_snap: Optional[Dict[str, Any]] = None
        # Third tenant of the frozen block: the Steam library digest shares
        # the facts+diaries cache block, and playtime moves while a session
        # runs — a resume that rebuilt it from the live snapshot missed the
        # whole block and everything behind it (09-12: read 12k, write 115k,
        # digest 933→954 chars). Frozen through freeze_steam_digest().
        self._steam_digest_snapshot: Optional[str] = None
        os.makedirs(self._diaries_dir, exist_ok=True)
        # Run the importance migrations unconditionally (not only when facts
        # RAG is on) so the llm→high rename reaches every setup.
        self._migrate_facts_importance()
        self._migrate_fact_ids()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_memory_llm(self, llm: Any) -> None:
        """Route memory tasks (diary/fact) through this LLM instead
        of the chat model. Pass None to fall back to the caller-supplied LLM."""
        self._memory_llm = llm

    def set_memory_reasoning_effort(self, effort: str) -> None:
        """Set reasoning_effort for memory-task LLM calls (none/low/medium/high).
        Empty string = don't send it, i.e. use the model's own default."""
        self._memory_reasoning_effort = (effort or "").strip()

    def set_chat_model(self, model: str) -> None:
        """Record the CHAT model's id (the experiencer) — stamped onto fact
        history events and the in-progress session's model-map entry."""
        self._chat_model = (model or "").strip()

    def set_active_sessions(self, uids) -> None:
        """Mark these session UIDs as already loaded in the sliding window."""
        self._active_session_uids = set(uids or [])

    def set_current_session(self, uid: str) -> None:
        """Register the session that is currently in progress.

        Backfill will skip this UID so an unfinished session is never
        summarised into a diary or used for premature fact extraction.
        """
        self._current_session_uid = uid or ""

    def get_facts_prompt(self) -> str:
        """Return the facts block for the system prompt (empty string if no facts).

        When facts RAG is active only the header-tier facts (user/llm) go here;
        ``low`` facts are recalled on demand. With RAG off, all facts go here.

        Uses the FROZEN snapshot — the block is byte-stable for the whole
        session (cache safety); facts.json changes land here on next restart.
        """
        facts = self._header_facts_frozen()
        if not facts:
            return ""
        lines = []
        for f in facts:
            updated = str(f.get("updated", ""))
            date = updated[:10] if len(updated) >= 10 else "不明"
            tag = f"{date} {self._fact_ref(f)[:8]}"
            sid = str(f.get("store_id", "") or "")[:8]
            if sid:
                tag += f" {sid}"
            # Resident facts are never collapsed; a title (long facts carry
            # one, あさひ 09-29) simply leads the text.
            title = self._clean_title(f.get("title", ""))
            text = f"【{title}】 {f['fact']}" if title else f["fact"]
            lines.append(f"- [{tag}] {text}")
        body = "\n".join(lines)
        header = (
            "## ユーザーに関する長期記憶（事実）\n"
            "各事実の冒頭は `[記録日 id]` または `[記録日 id store_id]`。"
            "記録日はその事実が**このリストに記録された日**であり、"
            "出来事が実際に起きた日ではない"
            "（事実抽出は次のセッション開始時にまとめて行われるため、"
            "実際の出来事はその数時間〜数日前の可能性がある）。"
            "id は事実そのものの短縮IDで、memory_read / memory_update / memory_delete に"
            "そのまま渡せる。store_id は Uber の店舗ID——uber_store に渡せば"
            "その店のメニューが開き、memory_add / memory_update の store_id "
            "引数と同じもの。"
        )
        return f"{header}\n\n{body}"

    def get_diaries_prompt(self) -> str:
        """Return the diary block for the system prompt (empty string if no diaries).

        Uses a FROZEN snapshot like the facts header — a diary written by
        startup backfill AFTER the first turn (overdue alarm firing at boot)
        must not change the system prompt mid-session (cache discipline,
        08-07 turn-2 hit 13%). Zero information loss: a backfilled diary
        summarizes a sliding-window session whose full text is already in
        context, and diary RAG excludes in-window sessions anyway.
        """
        if self._diaries_snapshot is None:
            if self._resume_boot:
                snap = self._resume_header_snapshot()
                restored = snap.get("diaries_prompt") if snap else None
                if isinstance(restored, str):
                    self._diaries_snapshot = restored
                    logger.info(
                        "[memory] diaries header restored from resume "
                        f"snapshot ({len(restored)} chars)."
                    )
            if self._diaries_snapshot is None:
                self._diaries_snapshot = self._render_diaries_prompt()
                logger.info(
                    "[memory] diaries header frozen for this session "
                    f"({len(self._diaries_snapshot)} chars)."
                )
            self._persist_header_snapshot()
        return self._diaries_snapshot

    def tag_vocabulary(self) -> List[tuple]:
        """``(tag, count)`` over every fact (all tiers), most used first."""
        counts: Dict[str, int] = {}
        for f in self._load_facts():
            for t in self._normalize_tags(f.get("tags")):
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def _render_tags_prompt(self) -> str:
        vocab = self.tag_vocabulary()
        if not vocab:
            return ""
        # The resident lines carry no tags (あさひ 09-29: +881 tok for no new
        # information); say so here so she does not read the resident list
        # as untagged.
        return (
            "## タグ一覧（起動時点）\n"
            + " ".join(f"#{t}({n})" for t, n in vocab)
            + "\n常駐の事実リストの各行にはタグを表示していないが、常駐の事実にも"
            "タグは付いている（memory_search の tags 絞り込みと memory_read で見える）。"
        )

    def get_tags_prompt(self) -> str:
        """Tag-vocabulary block for the system prompt (あさひ 09-29) — the
        names in use with their counts, so the character reuses popular
        tags instead of coining near-duplicates. FROZEN per session like the
        facts header (same resume/persist rules): a tag coined mid-session
        shows up here at the next boot; until then the write result names
        it (``new_tags`` / ``similar_existing``)."""
        if getattr(self, "_tags_snapshot", None) is None:
            self._tags_snapshot = None
            if getattr(self, "_resume_boot", False):
                snap = self._resume_header_snapshot()
                restored = snap.get("tags_prompt") if snap else None
                if isinstance(restored, str):
                    self._tags_snapshot = restored
                    logger.info(
                        "[memory] tag vocabulary restored from resume snapshot "
                        f"({len(restored)} chars)."
                    )
            if self._tags_snapshot is None:
                self._tags_snapshot = self._render_tags_prompt()
                logger.info(
                    "[memory] tag vocabulary frozen for this session "
                    f"({len(self._tags_snapshot)} chars)."
                )
            self._persist_header_snapshot()
        return self._tags_snapshot

    @staticmethod
    def _similar_tags(tag: str, known: List[str]) -> List[str]:
        """Known tags a new one probably duplicates: substring either way, or
        character-bigram overlap ≥ 0.5 of the shorter tag."""
        t = tag.lower()
        tb = {t[i : i + 2] for i in range(len(t) - 1)} if len(t) > 1 else {t}
        out: List[str] = []
        for k in known:
            kl = k.lower()
            if kl == t:
                continue
            if t in kl or kl in t:
                out.append(k)
                continue
            kb = {kl[i : i + 2] for i in range(len(kl) - 1)} if len(kl) > 1 else {kl}
            if len(tb & kb) / min(len(tb), len(kb)) >= 0.5:
                out.append(k)
        return out

    def tag_feedback(
        self, tags: List[str], before: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Result extras for a write that used tags: ``new_tags`` (absent
        from the vocabulary before the write) and ``similar_existing``
        (new tag → known tags it probably duplicates). Advisory only — the
        write goes through either way (あさひ 09-29: hint, never refuse)."""
        known = sorted({t for f in before for t in self._normalize_tags(f.get("tags"))})
        new = [t for t in tags if t not in known]
        if not new:
            return {}
        out: Dict[str, Any] = {"new_tags": new}
        similar = {t: s for t in new if (s := self._similar_tags(t, known))}
        if similar:
            out["similar_existing"] = similar
        return out

    def _render_diaries_prompt(self) -> str:
        diaries = self._load_recent_diaries()
        if not diaries:
            return ""

        def _head(d: Dict[str, Any]) -> str:
            tag = self.diary_display_tag(d.get("history_uid", ""), entry=d)
            return f"[{d['date']} {tag}]" if tag else f"[{d['date']}]"

        entries = "\n\n".join(f"{_head(d)}\n{d['content']}" for d in diaries)
        return (
            "## 過去セッションの日記\n"
            "後続の会話履歴より前に行われたセッションの要約。"
            "各エントリ冒頭の日付がそのセッションの実時間。"
            "日付の横のモデル名は、その記録を実際に体験した当時の会話モデル。"
            "「本人執筆」付きは当時の自分が書いた日記で、"
            "無印の日記は記録係（メモリ用の別モデル）が代筆したもの。\n"
            "※ 日記中の「未解決」「これから」「明日」など、当時の予定や保留事項を"
            "表す記述は、その日記が書かれた時点の状態を反映している。"
            "その後すでに解決・完了している可能性があるため、現状を断定せず、"
            "必要に応じてユーザーに確認すること。\n\n"
            f"{entries}"
        )

    def get_memory_prompt(self) -> str:
        """Return the combined memory block (facts + diaries) for non-Claude LLMs."""
        parts = [p for p in (self.get_facts_prompt(), self.get_diaries_prompt()) if p]
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Diary RAG (long-tail recall)
    # ------------------------------------------------------------------

    @property
    def diary_rag_active(self) -> bool:
        """True when the diary vector index is built and ready to query."""
        return self._diary_index is not None

    @property
    def diary_rag_config(self) -> Any:
        """The DiaryRagConfig (ttl_turns / max_in_context / ... ) or None."""
        return self._rag_cfg

    def injected_diary_uids(self) -> Set[str]:
        """UIDs of the diaries currently injected in the system prompt block.

        The agent unions these into the retrieval exclude set so RAG never
        surfaces a diary the model already has verbatim in its prompt.
        """
        return {d.get("history_uid", "") for d in self._load_recent_diaries()} - {""}

    # ------------------------------------------------------------------
    # Facts RAG (low-importance fact recall) — sibling of diary RAG
    # ------------------------------------------------------------------

    @property
    def facts_rag_active(self) -> bool:
        """True when the fact vector index is built and ready to query."""
        return self._facts_index is not None

    @property
    def facts_rag_config(self) -> Any:
        """The FactsRagConfig or None."""
        return self._facts_rag_cfg

    @staticmethod
    def _fact_id(fact_text: str) -> str:
        """Stable content fingerprint used as the vector-index id for a fact.

        The id *is* the content hash, so an edited/merged fact gets a new id —
        ensure_indexed then re-embeds it and prunes the stale vector
        automatically, with no separate content-change check (facts, unlike
        immutable diaries, get rewritten by consolidation/pruning).
        """
        norm = " ".join((fact_text or "").split())
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _id_matches(full_id: str, given: str) -> bool:
        """Exact 16-hex id, or a >=8-hex prefix — the displayed short id
        (08-20: every injection path shows `[date id8]`, so the tools must
        accept what the model actually sees)."""
        return full_id == given or (len(given) >= 8 and full_id.startswith(given))

    @staticmethod
    def _row_store_id(f: Dict[str, Any]) -> Dict[str, str]:
        """``{"store_id": ...}`` when the fact carries the uber linkage,
        else empty — splatted into recall/search rows so every path can
        render `[記録日 id store_id]` without special-casing."""
        sid = str(f.get("store_id", "") or "")[:8]
        return {"store_id": sid} if sid else {}

    # ------------------------------------------------------------------
    # Titles, tags and the long-fact rule (あさひ 09-29)
    # ------------------------------------------------------------------
    # A fact may carry ``title`` (≤60 chars; only long facts need one) and
    # ``tags`` (≤5, each ≤20 chars, NFKC + ASCII-lowercased). Neither is
    # embedded — the vector index keys on the text alone — they are display
    # and filter metadata. Long facts (over ``_long_fact_chars``) collapse to
    # `【title】（本文 N字 → memory_read）` in every list path; a collapsed row
    # still counts as injected for the session, so it is shown once and read
    # by id when actually needed.
    _TITLE_MAX = 60
    _TAG_MAX_COUNT = 5
    _TAG_MAX_CHARS = 20

    @staticmethod
    def _clean_title(title: Any) -> str:
        """One-line title, inner whitespace collapsed, capped at _TITLE_MAX."""
        t = " ".join(str(title or "").split())
        return t[: PersistentMemoryManager._TITLE_MAX]

    @staticmethod
    def _normalize_tags(tags: Any) -> List[str]:
        """Canonical tag list from a list or a loosely separated string:
        NFKC, trimmed, leading '#' dropped, ASCII lowercased, deduped in
        order, each cut to _TAG_MAX_CHARS, at most _TAG_MAX_COUNT."""
        if tags is None:
            return []
        if isinstance(tags, str):
            raw = re.split(r"[,、，\s#]+", tags)
        elif isinstance(tags, (list, tuple, set)):
            raw = [str(t) for t in tags if t is not None]
        else:
            return []
        out: List[str] = []
        for t in raw:
            t = unicodedata.normalize("NFKC", str(t or "")).strip().strip("#").strip()
            if not t:
                continue
            t = "".join(ch.lower() if ch.isascii() else ch for ch in t)
            t = t[: PersistentMemoryManager._TAG_MAX_CHARS]
            if t and t not in out:
                out.append(t)
            if len(out) >= PersistentMemoryManager._TAG_MAX_COUNT:
                break
        return out

    def is_long_fact(self, f: Dict[str, Any]) -> bool:
        limit = int(getattr(self, "_long_fact_chars", 500) or 0)
        return bool(limit) and len(str(f.get("fact", "") or "")) > limit

    def _display_fields(self, f: Dict[str, Any]) -> Dict[str, Any]:
        """Model-facing extras for a fact row: ``title`` / ``tags`` when
        present, plus ``collapsed``/``chars`` for a long fact. Renderers and
        tool results decide what to do with ``collapsed``; the row's ``fact``
        (when a caller keeps it) stays the full text for the judge/logs."""
        out: Dict[str, Any] = {}
        title = self._clean_title(f.get("title", ""))
        if title:
            out["title"] = title
        tags = self._normalize_tags(f.get("tags"))
        if tags:
            out["tags"] = tags
        if self.is_long_fact(f):
            out["collapsed"] = True
            out["chars"] = len(str(f.get("fact", "") or ""))
        return out

    @staticmethod
    def fact_row_body(row: Dict[str, Any]) -> str:
        """Line body after the `[記録日 id]` tag for the text renderers (auto
        recall block, Uber section): the collapsed stub for a long fact,
        otherwise the text led by its title when it has one."""
        title = str(row.get("title") or "").strip()
        if row.get("collapsed"):
            head = f"【{title}】" if title else "【無題】"
            return f"{head}（本文 {int(row.get('chars') or 0)}字 → memory_read）"
        text = str(row.get("fact") or "").strip()
        return f"【{title}】 {text}" if title else text

    # Persistent handle (あさひ 09-22). The content hash used to do three jobs:
    # vector-index key, "this exact text is already in context" dedup key, and
    # the id the character addresses a fact by. The first two MUST follow the
    # content (an edited fact is new content: re-embed it, allow re-injection;
    # it is also what makes external edits sync for free). The third must NOT:
    # her long "shelf" facts are edited daily, and an id that changes on every
    # edit dies in the frozen header block after the first edit of a session.
    # So a fact carries its own ``id`` — the content hash AT CREATION, 16 hex,
    # same shape as before — and only the model-facing paths (display rows,
    # tool arguments, tool results) use it. Existing facts are stamped with
    # their current hash, so no displayed id changed when this shipped.

    @staticmethod
    def _fact_ref(f: Dict[str, Any]) -> str:
        """A fact's persistent handle; a not-yet-stamped fact answers with its
        content hash — exactly the value stamping will give it."""
        return str(f.get("id") or "") or PersistentMemoryManager._fact_id(
            f.get("fact", "")
        )

    @staticmethod
    def _ensure_fact_ids(facts: List[Dict[str, Any]]) -> int:
        """Stamp ``id`` on every fact that lacks one (in place); returns how
        many were stamped. Mutation paths call this BEFORE touching a text —
        stamping afterwards would hash the new text and the handle would move
        on a fact's first edit. Handles are unique: a hash already taken (a
        fact edited away from that text, or a duplicate) gets a deterministic
        salt. Static so the memory viewer stamps by the very same rule."""
        fid = PersistentMemoryManager._fact_id
        taken = {str(f.get("id")) for f in facts if f.get("id")}
        stamped = 0
        for f in facts:
            if f.get("id") or not f.get("fact"):
                continue
            ref = fid(f["fact"])
            n = 0
            while ref in taken:
                n += 1
                ref = fid(f"{f['fact']}\0{f.get('updated', '')}\0{n}")
            f["id"] = ref
            taken.add(ref)
            stamped += 1
        return stamped

    def _migrate_fact_ids(self) -> None:
        """Stamp the persistent handle on every fact that lacks one, at boot
        (09-22) — the whole existing pool on the first boot with this code,
        and afterwards any straggler a hand edit added. Each gets its CURRENT
        content hash, i.e. exactly the id it was already showing. Uniquely
        named backup first, like the importance migration. An unreadable
        facts.json loads as [] and is left alone. Idempotent; never raises."""
        try:
            facts = self._load_facts()
            if not any(f.get("fact") and not f.get("id") for f in facts):
                return
            backup = self._facts_path + ".pre-fact-id.bak"
            if os.path.exists(self._facts_path) and not os.path.exists(backup):
                shutil.copy2(self._facts_path, backup)
                logger.info(
                    f"[memory] Backed up facts.json → {backup} before id stamping."
                )
            stamped = self._ensure_fact_ids(facts)
            self._save_facts(facts)
            logger.info(
                f"[memory] persistent fact ids stamped: {stamped} of {len(facts)} "
                "(each = its current content hash; no displayed id changed)."
            )
        except Exception as e:
            logger.warning(f"[memory] fact id stamping failed: {e}")

    def _facts_by_given_id(self, facts, fact_id: str):
        """All (index, fact) whose handle matches ``fact_id`` (prefix aware).
        More than one hit = ambiguous prefix; callers must refuse. Content
        hashes are still honoured when no handle matches (ids shown by
        anything that predates the handle, e.g. the memory viewer)."""
        out = [
            (i, f)
            for i, f in enumerate(facts)
            if f.get("fact") and self._id_matches(self._fact_ref(f), fact_id)
        ]
        if out:
            return out
        return [
            (i, f)
            for i, f in enumerate(facts)
            if f.get("fact") and self._id_matches(self._fact_id(f["fact"]), fact_id)
        ]

    def _ambiguous_id_error(self, fact_id: str, matches) -> Dict[str, Any]:
        """Refusal for an ambiguous short id. Every display path shows only
        the 8-hex form, so the model has nowhere else to get a longer id —
        the error itself must carry the full ids (08-20 あさひ)."""
        listing = " / ".join(
            f"{self._fact_ref(f)}＝{f['fact'][:30]}" for _, f in matches
        )
        return {
            "status": "error",
            "message": (
                f"id {fact_id} が複数の記憶に一致して曖昧: {listing}。"
                "完全なidで再指定を。"
            ),
        }

    def _header_facts(self) -> List[Dict[str, Any]]:
        """Facts that belong in the system-prompt header.

        With facts RAG active, only ``user``/``llm``-tier facts; the rest
        (``low``, the default) are recalled on demand. With RAG off, all facts
        (preserves the original always-inject-everything behaviour).
        """
        facts = self._load_facts()
        if not self.facts_rag_active:
            return facts
        return [
            f
            for f in facts
            # "llm" tolerated as legacy spelling of "high" (renamed 2026-07-09).
            if (f.get("importance") or "low") in ("user", "high", "llm")
        ]

    def _persist_header_snapshot(self) -> None:
        """Best-effort atomic write of the frozen header (facts + diaries).

        Called whenever either half freezes; the file always describes the
        session currently on the wire, which is exactly what a --resume boot
        needs back. Never raises."""
        try:
            data = {
                "frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "facts": self._header_snapshot,
                "diaries_prompt": self._diaries_snapshot,
                "tags_prompt": getattr(self, "_tags_snapshot", None),
                "steam_digest": getattr(self, "_steam_digest_snapshot", None),
            }
            tmp = self._header_snapshot_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._header_snapshot_path)
        except Exception as e:
            logger.warning(f"[memory] header snapshot persist failed: {e}")

    def _load_header_snapshot(self) -> Optional[Dict[str, Any]]:
        """The persisted frozen header, or None (missing/corrupt — logged)."""
        try:
            with open(self._header_snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"[memory] header snapshot unreadable: {e}")
            return None

    def _resume_header_snapshot(self) -> Optional[Dict[str, Any]]:
        """The pre-restart frozen header on a --resume boot, loaded once."""
        if not self._resume_snap_loaded:
            self._resume_snap_loaded = True
            self._resume_snap = self._load_header_snapshot()
        return self._resume_snap

    def freeze_steam_digest(self, digest: str) -> str:
        """Freeze the Steam library digest for this session and return the
        text the agent must actually mount.

        The digest sits in the same cached system block as the facts and
        diaries headers, so it obeys the same rule: one value per session,
        byte-identical across a --resume. A fresh boot freezes the live
        digest; a resume boot returns the one persisted with the header
        snapshot (playtime that accrued mid-session is picked up by the next
        FRESH boot, exactly like facts.json edits). A resume whose snapshot
        predates this field degrades to the live digest with a warning —
        one rewrite, then the file carries it. Persists immediately when the
        header halves already froze (Steam wiring normally finishes before
        the first turn, but an overdue alarm can fire first)."""
        live = (digest or "").strip()
        chosen = live
        if self._resume_boot:
            snap = self._resume_header_snapshot()
            restored = snap.get("steam_digest") if snap else None
            if isinstance(restored, str) and restored.strip():
                chosen = restored
                logger.info(
                    "[steam] digest restored from resume snapshot "
                    f"({len(chosen)} chars; live build was {len(live)} chars)."
                )
            else:
                logger.warning(
                    "[steam] --resume but the header snapshot carries no "
                    "digest; freezing the live one (one-time cache rewrite)."
                )
        self._steam_digest_snapshot = chosen
        if self._header_snapshot is not None or self._diaries_snapshot is not None:
            self._persist_header_snapshot()
        return chosen

    def _header_facts_frozen(self) -> List[Dict[str, Any]]:
        """Header facts, captured once and reused for the whole session.

        Frozen so the system-prompt facts block never changes mid-session —
        previously facts were re-read from disk every turn, so ANY facts.json
        write (fact extraction, hand edit, and now the character's own
        memory_* tools) rewrote the prompt and busted the prefix cache
        (observed 90%→17%). New/edited facts remain immediately reachable
        through facts RAG and memory_search; the header catches up on the
        next restart. Startup-backfill facts make it in only when backfill
        settles before the first build — once a turn has shipped the header,
        it stays frozen even through backfill (an overdue alarm can fire
        before backfill settles; resetting then cost turn-2 87% of its
        cache, 08-07).
        """
        if self._header_snapshot is None:
            if self._resume_boot:
                snap = self._resume_header_snapshot()
                if snap and isinstance(snap.get("facts"), list):
                    self._header_snapshot = [dict(f) for f in snap["facts"]]
                    logger.info(
                        "[memory] facts header restored from resume snapshot "
                        f"({len(self._header_snapshot)} fact(s), frozen_at="
                        f"{snap.get('frozen_at', '?')}) — byte-stable resume."
                    )
                else:
                    logger.warning(
                        "[memory] --resume but no usable header snapshot; "
                        "freezing from live facts (one-time cache rewrite)."
                    )
            if self._header_snapshot is None:
                self._header_snapshot = [dict(f) for f in self._header_facts()]
                logger.info(
                    f"[memory] facts header frozen for this session "
                    f"({len(self._header_snapshot)} fact(s))."
                )
            self._persist_header_snapshot()
        return self._header_snapshot

    def injected_fact_ids(self) -> Set[str]:
        """Fingerprints of the facts already in the header (user/llm tier).

        The agent unions these into the fact-retrieval exclude set so RAG never
        surfaces a fact the model already has verbatim in its prompt. Uses the
        same frozen snapshot as the header block so the two stay consistent
        (a fact added mid-session is absent from both → RAG may surface it).
        """
        return {
            self._fact_id(f["fact"])
            for f in self._header_facts_frozen()
            if f.get("fact")
        }

    def _facts_items_for_index(self) -> List[Dict[str, Any]]:
        """Every fact as ``{id, text, meta}`` for the fact vector index.

        Indexes ALL tiers — tier filtering is applied only at injection, so a
        manual tier change takes effect without re-indexing.
        """
        items: List[Dict[str, Any]] = []
        for f in self._load_facts():
            text = f.get("fact", "")
            if not text:
                continue
            items.append(
                {
                    "id": self._fact_id(text),
                    "text": text,
                    "meta": {"date": str(f.get("updated", ""))[:10]},
                }
            )
        return items

    def _migrate_facts_importance(self) -> None:
        """One-time migrations of the ``importance`` field.

        (a) tag facts lacking ``importance`` with the default ``low``;
        (b) rename the legacy ``llm`` tier to ``high`` (2026-07-09, semantic
        consistency with user/high/low). Backs up to a uniquely-named file
        first (distinct from the rolling ``facts.json.bak`` that
        ``_save_facts`` overwrites each save). Idempotent.
        """
        facts = self._load_facts()
        needs_default = any("importance" not in f for f in facts)
        needs_rename = any(f.get("importance") == "llm" for f in facts)
        if not facts or not (needs_default or needs_rename):
            return
        backup = self._facts_path + ".pre-importance.bak"
        try:
            if os.path.exists(self._facts_path) and not os.path.exists(backup):
                shutil.copy2(self._facts_path, backup)
                logger.info(
                    f"[memory] Backed up facts.json → {backup} before importance migration."
                )
        except Exception as e:
            logger.warning(f"[memory] facts importance backup failed: {e}")
        renamed = 0
        for f in facts:
            f.setdefault("importance", "low")
            if f["importance"] == "llm":
                f["importance"] = "high"
                renamed += 1
        self._save_facts(facts)
        logger.info(
            f"[memory] importance migration: defaults filled={needs_default}, "
            f"llm→high renamed={renamed}."
        )

    async def retrieve_facts_context(
        self, query: str, exclude_ids: Set[str], context: str = ""
    ) -> tuple:
        """Return low-importance facts relevant to the conversation.

        Mirrors :meth:`retrieve_diary_context` but over the fact index (facts are
        single sentences, so no parent grouping). ``context`` is the recent
        conversation handed to the judge. Returns ``(hits, candidates, keywords)``
        where hits is ``[{"id", "fact", "score", "reason"}, ...]``.
        """
        if self._facts_index is None or not query or not query.strip():
            return [], [], []
        cfg = self._facts_rag_cfg
        max_n = getattr(cfg, "max_retrievals_per_turn", 3)
        lex_w = getattr(cfg, "lexical_weight", 0.5)
        keywords = extract_keywords(query)
        embed_q = " ".join(keywords) if keywords else query
        by_id = {
            self._fact_id(f["fact"]): f for f in self._load_facts() if f.get("fact")
        }
        # The archive tier (below low, あさひ 08-31) is reachable ONLY through
        # deliberate memory_search — it stays in the index but never rides any
        # automatic channel. Excluding here keeps even the floor gate blind to
        # archived facts (an archived top candidate must not open the judge).
        exclude_ids = set(exclude_ids) | {
            fid
            for fid, f in by_id.items()
            if (f.get("importance") or "low") == "archive"
        }

        if self._facts_reranker is None:
            hits, candidates = await self._facts_index.retrieve(
                embed_q,
                exclude_ids=exclude_ids,
                similarity_threshold=getattr(cfg, "similarity_threshold", 0.55),
                topn_threshold=getattr(cfg, "topn_threshold", 0.70),
                max_retrievals=max_n,
                lexical_weight=lex_w,
                keywords=keywords,
            )
            # Rows: "id" stays the CONTENT hash (the agent's in-context dedup
            # key); "ref" is the persistent handle the row displays.
            out = [
                {
                    "id": h["id"],
                    "ref": self._fact_ref(by_id[h["id"]]),
                    "fact": by_id[h["id"]]["fact"],
                    "date": str(by_id[h["id"]].get("updated", ""))[:10],
                    "score": h["score"],
                    "reason": "",
                    **self._row_store_id(by_id[h["id"]]),
                    **self._display_fields(by_id[h["id"]]),
                }
                for h in hits
                if h["id"] in by_id
            ]
            return out, candidates, keywords

        top_k = getattr(cfg, "rerank_candidates", 12)
        _, candidates = await self._facts_index.retrieve(
            embed_q,
            exclude_ids=exclude_ids,
            similarity_threshold=-1.0,
            topn_threshold=-1.0,
            max_retrievals=top_k,
            debug_k=top_k,
            lexical_weight=lex_w,
            keywords=keywords,
        )
        floor = getattr(cfg, "prefilter_floor", 0.3)
        if not candidates or candidates[0][2] < floor:
            return [], candidates, keywords
        shortlist = [
            {"id": cid, "date": date, "content": by_id[cid]["fact"]}
            for cid, date, _h, _v, _lx in candidates
            if cid in by_id
        ]
        judged = await self._facts_reranker.rerank(query, shortlist, context=context)
        if judged is None:
            judged = [dict(s, reason="(rerank-fallback)") for s in shortlist[:max_n]]
        out = [
            {
                "id": j["id"],
                "ref": self._fact_ref(by_id[j["id"]]) if j["id"] in by_id else j["id"],
                "fact": j["content"],
                "date": j.get("date", "")
                or str(by_id.get(j["id"], {}).get("updated", ""))[:10],
                "score": 0.0,
                "reason": j.get("reason", ""),
                **self._row_store_id(by_id.get(j["id"], {})),
                **self._display_fields(by_id.get(j["id"], {})),
            }
            for j in judged[:max_n]
        ]
        return out, candidates, keywords

    async def uber_related_facts(
        self,
        query: str,
        store_ids: List[str],
        exclude_ids: Set[str],
        cap: int,
    ) -> List[Dict[str, Any]]:
        """Two-wave facts recall for Uber search results.

        Wave B (primary, 08-15 redesign): facts carry an optional
        ``store_id`` — the store_uuid's 8-char short form — and match by
        EQUALITY against the ids of the stores in the search results.
        This replaced the 08-13 title-string matcher: name fuzz (prefix
        overreach, same-name different-store, comparison mentions) is gone
        by construction, and a rotated uuid fails silently to a miss, never
        to a wrong injection. Keyless facts are invisible to this wave.
        Ordering follows the stores' positions in the search results.
        Wave A: semantic hits on the search keyword itself (no judge;
        floor = facts_rag.uber_topic_floor, falling back to
        similarity_threshold) — catches store-independent preference facts
        (「日式中華は口に合わない」). Merge is B-first, but when B alone
        fills ``cap`` the top A hit keeps one guaranteed slot. Zero API for
        B; one embedding for A. Never raises; returns ``[]`` on failure.
        """
        try:
            facts = self._load_facts()
        except Exception:
            return []
        by_id = {
            self._fact_id(f["fact"]): f
            for f in facts
            if f.get("fact")
            # archive tier: invisible to every automatic channel, both waves.
            and (f.get("importance") or "low") != "archive"
        }
        cap = max(1, int(cap))

        # ---- Wave B: store-id equality ----
        shorts = [s[:8] for s in (store_ids or []) if s]
        rank = {s: i for i, s in enumerate(shorts)}
        scored: Dict[str, int] = {}
        for fid, f in by_id.items():
            if fid in exclude_ids:
                continue
            key = str(f.get("store_id", "") or "")[:8]
            if key and key in rank:
                scored[fid] = rank[key]

        # ---- Wave A: semantic hits on the search keyword ----
        a_order: List[str] = []
        if self._facts_index is not None and query.strip():
            cfg = self._facts_rag_cfg
            floor = getattr(cfg, "uber_topic_floor", 0.0) or getattr(
                cfg, "similarity_threshold", 0.6
            )
            keywords = extract_keywords(query)
            embed_q = " ".join(keywords) if keywords else query
            try:
                hits, _ = await self._facts_index.retrieve(
                    embed_q,
                    exclude_ids=set(exclude_ids),
                    similarity_threshold=floor,
                    topn_threshold=floor,
                    max_retrievals=cap,
                    lexical_weight=getattr(cfg, "lexical_weight", 0.5),
                    keywords=keywords,
                )
                a_order = [
                    h["id"] for h in hits if h["id"] in by_id and h["id"] not in scored
                ]
            except Exception as e:
                logger.warning(f"[uber_facts] semantic wave failed: {e}")

        # ---- Merge: B-first (result order); A's best keeps a slot when full.
        # A contributes at most 2 either way (あさひ 08-15: unlimited topical
        # backfill just stuffed every free slot with noise). ----
        b_order = sorted(scored, key=lambda i: scored[i])
        if len(b_order) >= cap and a_order:
            final = b_order[: cap - 1] + a_order[:1]
        else:
            final = (b_order + a_order[:2])[:cap]
        return [
            {
                "id": fid,
                "ref": self._fact_ref(by_id[fid]),
                "fact": by_id[fid]["fact"],
                "date": str(by_id[fid].get("updated", ""))[:10],
                "via": "store" if fid in scored else "topic",
                **self._row_store_id(by_id[fid]),
                **self._display_fields(by_id[fid]),
            }
            for fid in final
        ]

    # ------------------------------------------------------------------
    # Character self-service memory (memory_* in-process tools)
    # ------------------------------------------------------------------
    # CRUD is facts-only; search covers facts AND diaries. All writes go
    # through _save_facts (disk = live truth) + an immediate index sync, so
    # changes are RAG-searchable at once; the frozen header block catches up
    # on the next restart. The ``user`` importance tier is manual-only: the
    # character may neither create (clamped) nor modify nor delete it.

    def find_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Fact dict by content-fingerprint id (short-prefix aware), or None.
        An ambiguous prefix also returns None — the mutation paths surface
        the distinction; this read path stays conservative."""
        matches = self._facts_by_given_id(self._load_facts(), fact_id)
        return matches[0][1] if len(matches) == 1 else None

    @staticmethod
    def _diary_short_id(uid: str) -> str:
        """8-hex display id of a diary uid (the random tail of
        ``date_time_hex32``) — mirrors the agent's excerpt-block id."""
        tail = (uid or "").rsplit("_", 1)[-1]
        if len(tail) >= 8 and all(c in "0123456789abcdef" for c in tail.lower()):
            return tail[:8]
        return uid

    def read_memories_by_ids(self, ids) -> Dict[str, Any]:
        """Memories in full by id (memory_read; facts 09-28, diaries merged
        in 09-29): fact handles and diary ids mixed, 1-10 per call, rows in
        request order. Facts come back whole — never collapsed — with their
        title/tags; diaries as ``{id, diary_uid, date, content, model?,
        written_by?}``. An id that resolves to nothing, to several facts, or
        to BOTH a fact and a diary is reported per id instead of failing the
        call; something asked for twice comes back once. Fact rows carry
        ``_hash`` (the content hash) for the agent's in-context dedup set —
        the agent strips it before the model sees the result. Diary rows are
        NOT cap-checked here: the agent applies the per-turn diary cap and
        the sentence ledger."""
        wanted = [str(x or "").strip() for x in (ids or []) if str(x or "").strip()]
        if not wanted or len(wanted) > 10:
            return {"status": "error", "message": "ids は1〜10件。"}
        facts = self._load_facts()
        rows: List[Dict[str, Any]] = []
        diaries: List[Dict[str, Any]] = []
        problems: List[str] = []
        seen_f: Set[str] = set()
        seen_d: Set[str] = set()
        for fid in wanted:
            fmatches = self._facts_by_given_id(facts, fid)
            duid, dmatches = self.resolve_diary_uid(fid)
            if fmatches and (duid or dmatches):
                problems.append(
                    f"id {fid} は事実と日記の両方に一致して曖昧: 事実 "
                    + " / ".join(self._fact_ref(f) for _, f in fmatches)
                    + " / 日記 "
                    + ", ".join(sorted(dmatches)[:5])
                    + "。完全なidで再指定を。"
                )
                continue
            if len(fmatches) > 1:
                problems.append(self._ambiguous_id_error(fid, fmatches)["message"])
                continue
            if fmatches:
                f = fmatches[0][1]
                ref = self._fact_ref(f)
                if ref in seen_f:
                    continue
                seen_f.add(ref)
                meta = {
                    k: v
                    for k, v in self._display_fields(f).items()
                    if k in ("title", "tags")
                }
                rows.append(
                    {
                        "id": ref[:8],
                        "date": str(f.get("updated", ""))[:10],
                        "importance": f.get("importance") or "low",
                        **meta,
                        "fact": f.get("fact", ""),
                        **self._row_store_id(f),
                        "_hash": self._fact_id(f.get("fact", "")),
                    }
                )
                continue
            if duid:
                if duid in seen_d:
                    continue
                seen_d.add(duid)
                entry = self.read_diary_full(duid)
                if not entry:
                    problems.append(f"日記 {fid} が見つからない。")
                    continue
                diaries.append(
                    {"id": self._diary_short_id(duid), "diary_uid": duid, **entry}
                )
                continue
            if dmatches:
                problems.append(
                    f"id {fid} が曖昧（日記{len(dmatches)}件一致）。候補: "
                    + ", ".join(sorted(dmatches)[:5])
                )
                continue
            problems.append(f"id {fid} の記憶が見つからない。")
        if not rows and not diaries:
            return {"status": "error", "message": " ".join(problems)}
        out: Dict[str, Any] = {
            "status": "ok",
            "note": "この全文はこのまま会話の文脈に残る（再読は不要）。",
        }
        if rows:
            out["facts"] = rows
        if diaries:
            out["diaries"] = diaries
        if problems:
            out["problems"] = problems
        return out

    async def _sync_facts_index(self) -> None:
        """Re-sync the fact vector index after a manual mutation (no-op when
        facts RAG is off; never raises)."""
        if self._facts_index is None:
            return
        try:
            await self._facts_index.ensure_indexed(self._facts_items_for_index())
        except Exception as e:
            logger.warning(f"[memory_tool] facts index sync failed: {e}")

    @staticmethod
    def _history_event(op: str, model: str, now: str = "") -> Dict[str, str]:
        """One provenance event for a fact's ``history`` list (あさひ 08-20:
        pure record — original creator, later editors — no consumer yet).
        ``m`` is the full model id; unknown stays '' rather than a guess."""
        return {
            "t": now or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "m": (model or "").strip(),
            "op": op,
        }

    async def add_fact_manual(
        self,
        text: str,
        importance: str = "low",
        store_id: str = "",
        title: str = "",
        tags: Any = None,
    ) -> Dict[str, Any]:
        """Append a fact on the character's behalf (memory_add).

        ``title`` / ``tags`` (あさひ 09-29) are optional metadata (see the
        title/tag helpers). A long fact without a title is still created —
        the result carries ``needs_title`` and asks for one via
        memory_update; refusing would make her re-emit the whole text.

        ``importance`` is clamped to high/low/archive — ``user`` is manual-only and an
        LLM must never assign it. Duplicate content (same fingerprint) is
        rejected instead of silently re-added. ``store_id`` (optional, Uber
        facts only) links the fact to a store for the search-time recall;
        stored as the 8-char short form. It is NOT part of the fingerprint,
        so adding/fixing it later never churns the embedding index.
        """
        text = self._clean_fact_text(text)
        if not text:
            return {"status": "error", "message": "fact本文が空。"}
        if importance == "llm":  # legacy spelling of "high"
            importance = "high"
        importance = importance if importance in ("high", "low", "archive") else "low"
        facts = self._load_facts()
        fid = self._fact_id(text)
        dup = next(
            (f for f in facts if f.get("fact") and self._fact_id(f["fact"]) == fid),
            None,
        )
        if dup is not None:
            return {
                "status": "error",
                "message": "同内容の記憶が既にある。",
                "id": self._fact_ref(dup)[:8],
            }
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "fact": text,
            "updated": now,
            "importance": importance,
            # Provenance trail (あさひ 08-20): pure record, no consumer yet.
            "history": [self._history_event("create", self._chat_model, now)],
        }
        store_id = (store_id or "").strip()
        if store_id:
            entry["store_id"] = store_id[:8]
        title = self._clean_title(title)
        if title:
            entry["title"] = title
        tag_list = self._normalize_tags(tags)
        if tag_list:
            entry["tags"] = tag_list
        # Vocabulary feedback is judged against the pool BEFORE this fact
        # joins it (its own tags must not count as "known").
        feedback = self.tag_feedback(tag_list, facts) if tag_list else {}
        facts.append(entry)
        self._save_facts(facts)  # stamps the handle (entry["id"])
        await self._sync_facts_index()
        ref = self._fact_ref(entry)
        # Full text on purpose — the log is the recovery trail for accidental
        # memory operations (facts.json.bak is only one save deep).
        logger.info(f"[memory_tool] fact ADDED ({importance}, {ref}): {text}")
        out: Dict[str, Any] = {
            "status": "ok",
            "id": ref[:8],
            "importance": importance,
            "note": "保存した。検索には即時反映、常駐の事実リストへは次回起動から。",
            **feedback,
        }
        if self.is_long_fact(entry) and not title:
            out["needs_title"] = True
            out["note"] += (
                f" {self._long_fact_chars}字を超える長い事実——"
                "memory_update で title を付けること。"
            )
        return out

    async def update_fact_manual(
        self,
        fact_id: str,
        new_text: str = "",
        importance: Optional[str] = None,
        store_id: Optional[str] = None,
        old_string: str = "",
        new_string: str = "",
        append: str = "",
        title: Optional[str] = None,
        tags: Any = None,
    ) -> Dict[str, Any]:
        """Change one fact (memory_update): rewrite the text whole
        (``new_text``), edit it in place (``old_string``→``new_string``
        and/or ``append``; the two combine, replace first — the former
        memory_edit, merged 09-29), and/or change importance, Uber store
        linkage, title or tags. ``new_text`` and an in-place edit are
        mutually exclusive; that call changes nothing and returns an error.

        Text edits are allowed on ALL tiers, including ``user`` (あさひ
        2026-07-09: content is editable; what stays forbidden is CREATING
        user-tier facts and DELETING them). The TIER of a user-tier fact is
        equally the user's own — the character may not promote or demote it
        (あさひ 2026-08-09: importance change added for high/low —
        "archive" joined 08-31 as the search-only shelf tier; ``user``
        stays manual-only in both directions). ``store_id`` (08-15) /
        ``title`` / ``tags``: None = untouched, empty = clear, else set.
        Atomic: every check runs before anything is written — an error
        leaves the fact exactly as it was."""
        new_text = self._clean_fact_text(new_text)
        old_string = str(old_string or "")
        new_string = str(new_string or "")
        append = str(append or "")
        importance = (importance or "").strip().lower() or None
        if importance == "llm":  # legacy spelling of "high"
            importance = "high"
        if importance is not None and importance not in ("high", "low", "archive"):
            return {
                "status": "error",
                "message": "importance は high / low / archive のみ（user は本人管理で指定不可）。",
            }
        if store_id is not None:
            store_id = store_id.strip()
        if title is not None:
            title = self._clean_title(title)
        tag_list = self._normalize_tags(tags) if tags is not None else None
        in_place = bool(old_string or append)
        if new_text and in_place:
            return {
                "status": "error",
                "message": "new_fact と old_string/append は同時に指定できない（何も変更していない）。",
            }
        if (
            not new_text
            and not in_place
            and importance is None
            and store_id is None
            and title is None
            and tag_list is None
        ):
            return {
                "status": "error",
                "message": (
                    "変更内容が無い（new_fact / old_string+new_string / append / "
                    "importance / store_id / title / tags のどれかを指定）。"
                ),
            }
        facts = self._load_facts()
        self._ensure_fact_ids(facts)  # before any text moves — see there
        matches = self._facts_by_given_id(facts, fact_id)
        if len(matches) > 1:
            return self._ambiguous_id_error(fact_id, matches)
        if not matches:
            return {
                "status": "error",
                "message": f"id {fact_id} の記憶が見つからない。memory_searchで確認を。",
            }
        _, f = matches[0]
        old = f["fact"]
        old_tier = f.get("importance") or "low"
        if importance is not None and old_tier == "user":
            return {
                "status": "error",
                "message": (
                    "userレベルの記憶の優先度は本人管理のため変更"
                    "できない（本文の修正は可）。"
                ),
            }
        edited = old
        if new_text:
            edited = new_text
        elif in_place:
            edited, err = self._apply_partial_edit(old, old_string, new_string, append)
            if err:
                return {"status": "error", "message": err}
            edited = self._clean_fact_text(edited)
            if not edited:
                return {
                    "status": "error",
                    "message": "編集後の本文が空になる（削除は memory_delete で）。",
                }
        text_changed = edited != old
        if text_changed:
            new_hash = self._fact_id(edited)
            if new_hash != self._fact_id(old) and any(
                g is not f and g.get("fact") and self._fact_id(g["fact"]) == new_hash
                for g in facts
            ):
                return {
                    "status": "error",
                    "message": "編集後と同内容の記憶が既にある。",
                }
        feedback = self.tag_feedback(tag_list, facts) if tag_list else {}
        # ---- all checks passed: mutate ----
        if text_changed:
            f["fact"] = edited
        if importance is not None:
            f["importance"] = importance
        if store_id is not None:
            if store_id:
                f["store_id"] = store_id[:8]
            else:
                f.pop("store_id", None)
        if title is not None:
            if title:
                f["title"] = title
            else:
                f.pop("title", None)
        if tag_list is not None:
            if tag_list:
                f["tags"] = tag_list
            else:
                f.pop("tags", None)
        f["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Provenance: append the edit; a legacy fact without a history
        # list starts one here (its creator stays unrecorded — never
        # fabricated).
        f.setdefault("history", []).append(
            self._history_event("edit", self._chat_model, f["updated"])
        )
        self._save_facts(facts)
        await self._sync_facts_index()
        ref = self._fact_ref(f)
        # Full old/new text on purpose — recovery trail for accidental
        # edits (restore by hand from the log if needed).
        if text_changed:
            logger.info(
                f"[memory_tool] fact {'UPDATED' if new_text else 'EDITED'} {ref} "
                f"(tier {old_tier}→{f.get('importance') or 'low'}, "
                f"{len(old)}→{len(edited)} chars):\n"
                f"  OLD: {old}\n  NEW: {edited}"
            )
        else:
            logger.info(
                f"[memory_tool] fact UPDATED {ref} (fields only; tier "
                f"{old_tier}→{f.get('importance') or 'low'}): {old}"
            )
        notes = []
        if new_text:
            notes.append("本文を更新した（idはそのまま）。")
        elif in_place:
            notes.append("本文を編集した（idはそのまま）。")
        else:
            notes.append("本文は変更なし。")
        if importance is not None and importance != old_tier:
            notes.append(f"importance を {old_tier}→{importance} に変更。")
        if store_id is not None:
            notes.append(
                f"store_id を {store_id[:8]} に設定。"
                if store_id
                else "store_id を削除。"
            )
        if title is not None:
            notes.append(f"title を「{title}」に設定。" if title else "title を削除。")
        if tag_list is not None:
            notes.append(
                "tags を " + " ".join(f"#{t}" for t in tag_list) + " に設定。"
                if tag_list
                else "tags を削除。"
            )
        notes.append("常駐リストへの反映は次回起動から。")
        # "fact" rides along for the chat audit marker: field-only calls
        # carry no new_fact, so the marker needs the target's text from
        # here to stay legible.
        out: Dict[str, Any] = {
            "status": "ok",
            "id": ref[:8],
            "fact": f["fact"],
            "chars": len(f["fact"]),
            "note": " ".join(notes),
            **feedback,
        }
        if self.is_long_fact(f) and not self._clean_title(f.get("title", "")):
            out["needs_title"] = True
            out["note"] += (
                f" {self._long_fact_chars}字を超える長い事実——title を付けること。"
            )
        return out

    @staticmethod
    def _clean_fact_text(text: str) -> str:
        """Stored form of a fact text: line endings unified, outer whitespace
        trimmed — nothing else. The old collapse-to-one-line rule is gone
        (あさひ 09-22): long shelf facts were an unreadable slab in the memory
        viewer, and line breaks are hers to place. ``_fact_id`` still hashes
        whitespace-insensitively, so re-wrapping a text never changes its
        content hash (no re-embed, duplicates still caught)."""
        return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _apply_partial_edit(
        text: str, old_string: str, new_string: str, append: str
    ) -> tuple:
        """One in-place edit of ``text`` → ``(new_text, error_message)``.

        Two forms that combine (あさひ 09-29; were exclusive until then):
        ``old_string``→``new_string`` on a passage that occurs EXACTLY once
        (the str_replace contract coding agents use: zero or several
        matches change nothing), then ``append`` added verbatim at the end.
        Matching is literal — no whitespace or width folding (あさひ 09-22:
        a model copies its own tokens faithfully; that class of slip is a
        human one). Shared by memory_update and memory_write_diary."""
        if not old_string and not append:
            return None, "old_string（置き換える箇所）か append のどちらかが必要。"
        out = text
        if old_string:
            n = text.count(old_string)
            if n == 0:
                return None, "old_string が本文に見つからない（何も変更していない）。"
            if n > 1:
                return None, (
                    f"old_string が{n}箇所に一致した（何も変更していない）。"
                    "前後を含めて一意になるように指定を。"
                )
            if old_string == new_string:
                return None, "old_string と new_string が同じ。"
            out = text.replace(old_string, new_string, 1)
        if append:
            out = out + append
        return out, ""

    async def delete_fact_manual(self, fact_id: str) -> Dict[str, Any]:
        """Remove one fact (memory_delete). The caller (agent) is responsible
        for having completed the user-approval flow BEFORE calling this.
        ``user``-tier facts cannot be deleted by the character at all."""
        facts = self._load_facts()
        matches = self._facts_by_given_id(facts, fact_id)
        if len(matches) > 1:
            return self._ambiguous_id_error(fact_id, matches)
        for i, f in matches:
            if (f.get("importance") or "low") == "user":
                return {
                    "status": "error",
                    "message": "userレベルの記憶は本人管理のため削除不可。",
                }
            removed = facts.pop(i)
            self._save_facts(facts)
            await self._sync_facts_index()
            # Full text on purpose — this line is what lets あさひ restore
            # an accidentally deleted fact by hand from the log.
            logger.info(
                f"[memory_tool] fact DELETED ({fact_id}, "
                f"importance={removed.get('importance', 'low')}): "
                f"{removed.get('fact', '')}"
            )
            return {"status": "ok", "deleted": removed.get("fact", "")}
        return {"status": "error", "message": f"id {fact_id} の記憶が見つからない。"}

    def _search_fact_row(
        self, f: Dict[str, Any], score: Optional[float] = None
    ) -> Dict[str, Any]:
        """memory_search / tag-listing row. A long fact collapses to its
        title + id (あさひ 09-29): the text is NOT in the row — memory_read
        gives it — so a search can never drop a whole shelf into context."""
        row: Dict[str, Any] = {
            "id": self._fact_ref(f)[:8],
            "date": str(f.get("updated", ""))[:10],
            "importance": f.get("importance", "low"),
        }
        disp = self._display_fields(f)
        if not disp.get("collapsed"):
            row["fact"] = f.get("fact", "")
        row.update(disp)
        if score is not None:
            row["score"] = round(float(score), 3)
        row.update(self._row_store_id(f))
        return row

    @staticmethod
    def _in_date_range(day: str, date_range: Optional[tuple]) -> bool:
        if date_range is None:
            return True
        if not day:
            return False
        return not (
            (date_range[0] and day < date_range[0])
            or (date_range[1] and day > date_range[1])
        )

    async def search_memory_tool(
        self,
        query: str,
        target: str = "both",
        n: int = 5,
        date_from: str = "",
        date_to: str = "",
        tags: Any = None,
    ) -> Dict[str, Any]:
        """Explicit memory search for the memory_search tool.

        Unlike the auto-injection RAG path this skips the LLM judge and the
        tuned thresholds (thresholds -1 → always return the top-n): an
        explicit lookup wants recall, and the character judges relevance
        itself. Fact hits carry the id used by memory_update/memory_delete.

        ``date_from``/``date_to`` ("YYYY-MM-DD") prefilter candidates by date
        metadata (diary date / fact updated-date) BEFORE ranking, so the model
        can scope a period without polluting the semantic query with dates.

        ``tags`` (あさひ 09-29): only facts carrying ALL the given tags are
        candidates (AND — two calls for OR). With tags and no query the call
        LISTS those facts newest first instead of ranking; diaries carry no
        tags, so the listing mode is facts-only.
        """
        query = (query or "").strip()
        tag_list = self._normalize_tags(tags)
        if not query and not tag_list:
            return {"status": "error", "message": "queryが空。"}
        try:
            n = max(1, min(int(n or 5), 10))
        except (TypeError, ValueError):
            n = 5
        date_from = (date_from or "").strip()
        date_to = (date_to or "").strip()
        for d, label in ((date_from, "date_from"), (date_to, "date_to")):
            if d and not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                return {
                    "status": "error",
                    "message": f"{label} は YYYY-MM-DD 形式で指定すること: {d!r}",
                }
        date_range = (date_from, date_to) if (date_from or date_to) else None
        keywords = extract_keywords(query) if query else []
        embed_q = " ".join(keywords) if keywords else query
        out: Dict[str, Any] = {"status": "ok"}
        if query:
            out["query"] = query
        if tag_list:
            out["tags"] = tag_list
        if date_range:
            out["date_filter"] = f"{date_from or '...'} 〜 {date_to or '...'}"

        if target in ("facts", "both"):
            by_id = {
                self._fact_id(f["fact"]): f for f in self._load_facts() if f.get("fact")
            }
            tagged_ids: Optional[Set[str]] = None
            if tag_list:
                want = set(tag_list)
                tagged_ids = {
                    fid
                    for fid, f in by_id.items()
                    if want <= set(self._normalize_tags(f.get("tags")))
                }
            if not query:
                # Tag listing mode: no ranking, newest first.
                listed = [
                    f
                    for fid, f in by_id.items()
                    if fid in (tagged_ids or set())
                    and self._in_date_range(str(f.get("updated", ""))[:10], date_range)
                ]
                listed.sort(key=lambda f: str(f.get("updated", "")), reverse=True)
                out["facts"] = [self._search_fact_row(f) for f in listed[:n]]
                out["facts_total"] = len(listed)
            elif self._facts_index is None:
                out["facts"] = []
                out["facts_note"] = "facts RAGが無効のため検索不可。"
            else:
                exclude = (
                    set()
                    if tagged_ids is None
                    else {fid for fid in by_id if fid not in tagged_ids}
                )
                hits, _ = await self._facts_index.retrieve(
                    embed_q,
                    exclude_ids=exclude,
                    similarity_threshold=-1.0,
                    topn_threshold=-1.0,
                    max_retrievals=n,
                    debug_k=n,
                    lexical_weight=getattr(self._facts_rag_cfg, "lexical_weight", 0.5),
                    keywords=keywords,
                    date_range=date_range,
                )
                out["facts"] = [
                    self._search_fact_row(by_id[h["id"]], h.get("score", 0.0))
                    for h in hits
                    if h["id"] in by_id
                ]
            if tag_list and not out.get("facts"):
                out["facts_note"] = "指定タグを全部持つ事実は無い。"

        if target in ("diaries", "both") and not query:
            out["diaries_note"] = "タグだけの検索は事実のみが対象（日記にタグは無い）。"
        elif target in ("diaries", "both"):
            if self._diary_index is None:
                out["diaries"] = []
                out["diaries_note"] = "diary RAGが無効のため検索不可。"
            else:
                chunk_map = {c["id"]: c for c in self._all_diary_chunks_for_index()}
                hits, _ = await self._diary_index.retrieve(
                    embed_q,
                    exclude_ids=set(),
                    similarity_threshold=-1.0,
                    topn_threshold=-1.0,
                    max_retrievals=n,
                    debug_k=n,
                    lexical_weight=getattr(self._rag_cfg, "lexical_weight", 0.5),
                    keywords=keywords,
                    date_range=date_range,
                )
                diaries = []
                for h in hits:
                    c = chunk_map.get(h["id"])
                    if not c:
                        continue
                    meta = c.get("meta") or {}
                    parent = meta.get("parent", "")
                    row = {
                        "date": meta.get("date", ""),
                        # the containing diary — usable with
                        # memory_read_diary
                        "diary_uid": parent,
                        "text": c.get("text", ""),
                        "score": round(float(h.get("score", 0.0)), 3),
                    }
                    tag = self.diary_display_tag(parent)
                    if tag.endswith(" 本人執筆"):
                        row["model"] = tag[: -len(" 本人執筆")]
                        row["written_by"] = "本人"
                    elif tag:
                        row["model"] = tag
                    diaries.append(row)
                out["diaries"] = diaries
        return out

    def read_diary_full(self, diary_uid: str) -> Optional[Dict[str, Any]]:
        """Full diary entry ``{date, content, model?, written_by?}`` by uid
        (memory_read_diary), or None when missing. ``model`` is the session's
        experiencer (short form); ``written_by``: 本人 for self-written
        diaries, the pen model's short name for annotated auto diaries
        (post-08-20), absent for legacy unmarked ones (= the era's memory
        model, gpt5.1, per the prompt-side note)."""
        uid = (diary_uid or "").strip()
        entry = self._read_diary(uid)
        if not entry:
            return None
        out = {"date": entry.get("date", ""), "content": entry.get("content", "")}
        label = self.model_label_for_session(uid)
        if label:
            out["model"] = label
        w = str(entry.get("writer", "") or "")
        if w == "self":
            out["written_by"] = "本人"
        elif w:
            # Annotated pen model (post-08-20 auto diaries) — short form,
            # e.g. "gpt5.6-luna" (あさひ 08-23: without this, a luna-penned
            # diary reads identically to a legacy unmarked gpt5.1 one).
            out["written_by"] = model_short_name(w)
        return out

    def self_written_diary(self, diary_uid: str) -> str:
        """Text of the diary the character wrote HERSELF for this session
        (``writer == "self"``), or "" — for the past-transcript inline
        (あさひ 09-19). Auto-generated diaries never qualify: they only
        summarise a transcript that is already in context, and startup
        backfill may land them AFTER the transcript froze, so a resume
        would re-render different bytes. A self-written diary is on disk
        before the next boot and immutable from then on."""
        entry = self._read_diary((diary_uid or "").strip())
        if not entry or entry.get("writer") != "self":
            return ""
        return str(entry.get("content") or "").strip()

    def write_session_diary(self, history_uid: str, content: str) -> Dict[str, Any]:
        """Save the CURRENT session's diary written by the character herself
        (memory_write_diary, あさひ 08-14).

        Same file shape as the boot-time generation, so backfill's
        exists→skip check makes them perfect complements: written → the
        gpt pass never runs; unwritten → generated as before. No
        ``facts_extracted`` flag, so the fact-extraction pass still
        processes the session either way. Embedding is deliberately left
        to the next boot's ensure_indexed — a live session never needs to
        retrieve its own diary, and the text-drift check re-embeds
        overwritten drafts. Never raises."""
        uid = (history_uid or "").strip()
        content = (content or "").strip()
        if not uid or not content:
            return {"status": "error", "message": "セッションか内容が空。"}
        try:
            os.makedirs(self._diaries_dir, exist_ok=True)
            path = os.path.join(self._diaries_dir, f"{uid}.json")
            overwrote = os.path.exists(path)
            entry = {
                "date": self._session_date_from_uid(uid),
                "history_uid": uid,
                "content": content,
                # Self-written via memory_write_diary — drives the 本人執筆
                # annotation; auto-generated diaries record their pen model.
                "writer": "self",
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            return {
                "status": "ok",
                "date": entry["date"],
                "overwrote": overwrote,
                "note": "保存した。"
                + (
                    "以前の下書きを上書きした。"
                    if overwrote
                    else "次回起動時の自動生成はスキップされる。"
                ),
            }
        except Exception as e:
            logger.warning(f"[memory] write_session_diary failed: {e}")
            return {"status": "error", "message": "日記の保存に失敗した。"}

    def edit_session_diary(
        self,
        history_uid: str,
        old_string: str = "",
        new_string: str = "",
        append: str = "",
    ) -> Dict[str, Any]:
        """Partial edit of the CURRENT session's self-written diary
        (memory_edit_diary, 09-22) — same two forms as memory_edit. A session
        that continues past its first goodnight used to cost a full re-emit
        of the diary to add one paragraph. No writer check (あさひ 09-22):
        backfill only ever ghost-writes PAST sessions, so the current
        session's diary is hers by construction — and the one odd path
        (resuming a session a fresh boot had already backfilled) shows her
        "日記: 未記録", so she rewrites rather than edits. Past diaries stay
        immutable (the caller only ever passes the current uid). All other
        fields of the file, ``writer`` included, are preserved. Never raises."""
        uid = (history_uid or "").strip()
        try:
            entry = self._read_diary(uid) if uid else None
            if not entry or not entry.get("content"):
                return {
                    "status": "error",
                    "message": "このセッションの日記はまだ無い。memory_write_diary で書くこと。",
                }
            old = str(entry["content"]).replace("\r\n", "\n")
            edited, err = self._apply_partial_edit(
                old,
                str(old_string or "").replace("\r\n", "\n"),
                str(new_string or "").replace("\r\n", "\n"),
                str(append or "").replace("\r\n", "\n"),
            )
            if err:
                return {"status": "error", "message": err}
            edited = edited.strip()
            if len(edited) < 100:
                return {
                    "status": "error",
                    "message": "編集後の日記が短すぎる（100字以上）。",
                }
            entry = dict(entry, content=edited)
            entry.setdefault("history_uid", uid)
            with open(
                os.path.join(self._diaries_dir, f"{uid}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            return {
                "status": "ok",
                "date": entry.get("date", ""),
                "chars": len(edited),
                "note": "日記を編集した。",
            }
        except Exception as e:
            logger.warning(f"[memory] edit_session_diary failed: {e}")
            return {"status": "error", "message": "日記の編集に失敗した。"}

    def resolve_diary_uid(self, fragment: str) -> tuple:
        """Full diary uid from a fragment — the 8-hex short id shown in RAG
        excerpt blocks, or any unique substring, or the full uid itself.

        Returns ``(uid, matches)``: ``uid`` only on a unique resolution;
        ambiguity leaves it None with the candidates in ``matches`` so the
        caller can name them."""
        frag = (fragment or "").strip()
        if not frag:
            return None, []
        if self._read_diary(frag):  # full uid — the common fast path
            return frag, [frag]
        if not os.path.isdir(self._diaries_dir):
            return None, []
        stems = [f[:-5] for f in os.listdir(self._diaries_dir) if f.endswith(".json")]
        matches = [s for s in stems if frag in s]
        if len(matches) == 1:
            return matches[0], matches
        return None, matches

    def diary_sentences(self, diary_uid: str) -> List[str]:
        """A diary's PARAGRAPHS in order (same splitter as the chunk index, so
        1-based paragraph numbers = chunk ``#i`` + 1 and stay stable across
        turns — diaries are immutable). Name kept from the sentence era
        (08-23 paragraph redesign): every consumer treats the unit opaquely.
        ``[]`` when the diary is missing."""
        entry = self._read_diary((diary_uid or "").strip())
        if not entry or not entry.get("content"):
            return []
        return _split_paragraphs(entry["content"])

    async def retrieve_diary_context(
        self,
        query: str,
        exclude_uids: Set[str],
        context: str = "",
        injected_sents: Optional[Dict[str, Set[int]]] = None,
    ) -> tuple:
        """Return diary excerpts relevant to ``query`` (long-tail recall).

        Pipeline: denoise the query to content keywords (drop framing words) →
        hybrid candidate generation (keywords drive embedding + lexical, grouped
        back to whole diaries — retrieval granularity is unchanged) → LLM judge
        picks PARAGRAPHS inside each relevant diary (08-13 sentence redesign,
        re-grained to paragraphs 08-23; "sentence" names kept, unit is opaque).
        Reads content fresh from disk so it's never stale.

        ``injected_sents`` maps uid → 1-based paragraph numbers already injected
        into context this session; shortlisted diaries carry that mask into the
        judge prompt (the caller still subtracts it when packing).

        Returns ``(hits, candidates, keywords)`` where hits is
        ``[{"uid", "date", "sents", "sentences", "reason"}, ...]`` in judge
        relevance order — ``sents`` is the full ordered sentence list, and
        ``sentences`` the picked 1-based numbers (ascending; original order).
        The old diary-count cap is retired: the caller packs against the
        sentence budget. A judge API/parse failure returns no hits (the round
        skips injection — no whole-diary fallback). candidates stays the scored
        shortlist ``(uid, date, hybrid, vec, lex)`` for log tuning.
        """
        if self._diary_index is None or not query or not query.strip():
            return [], [], []
        cfg = self._rag_cfg
        max_n = getattr(cfg, "max_retrievals_per_turn", 2)
        lex_w = getattr(cfg, "lexical_weight", 0.5)

        # Denoise the query: retrieve on the content keywords (embed the stitched
        # keywords; lexical matches each keyword), falling back to the raw query.
        keywords = extract_keywords(query)
        embed_q = " ".join(keywords) if keywords else query

        # No reranker → original score-based threshold + topN whole-diary
        # selection (sentence picking needs the judge). Uniform hit shape:
        # every sentence counts as picked, so the caller's ledger stays exact.
        if self._diary_reranker is None:
            hits, candidates = await self._diary_index.retrieve(
                embed_q,
                exclude_ids=exclude_uids,
                similarity_threshold=getattr(cfg, "similarity_threshold", 0.55),
                topn_threshold=getattr(cfg, "topn_threshold", 0.70),
                max_retrievals=max_n,
                group_by="parent",
                lexical_weight=lex_w,
                keywords=keywords,
            )
            out: List[Dict[str, Any]] = []
            for h in hits:
                sents = self.diary_sentences(h["id"])
                if not sents:
                    continue
                entry = self._read_diary(h["id"])
                out.append(
                    {
                        "uid": h["id"],
                        "date": (entry or {}).get("date", h["meta"].get("date", "")),
                        "sents": sents,
                        "sentences": list(range(1, len(sents) + 1)),
                        "reason": "",
                    }
                )
            return out, candidates, keywords

        # Reranker path: pull a generous shortlist (no strict gate — the judge is
        # the real relevance filter), then let the LLM pick relevant sentences.
        top_k = getattr(cfg, "rerank_candidates", 12)
        _, candidates = await self._diary_index.retrieve(
            embed_q,
            exclude_ids=exclude_uids,
            similarity_threshold=-1.0,
            topn_threshold=-1.0,
            max_retrievals=top_k,
            debug_k=top_k,
            group_by="parent",
            lexical_weight=lex_w,
            keywords=keywords,
        )
        floor = getattr(cfg, "prefilter_floor", 0.3)
        if not candidates or candidates[0][2] < floor:
            return [], candidates, keywords  # nothing plausibly relevant; skip judge

        injected_sents = injected_sents or {}
        shortlist: List[Dict[str, Any]] = []
        for uid, date, _h, _v, _lx in candidates:
            entry = self._read_diary(uid)
            if not entry or not entry.get("content"):
                continue
            shortlist.append(
                {
                    "id": uid,
                    "date": entry.get("date", date),
                    "sents": _split_paragraphs(entry["content"]),
                    "injected": sorted(injected_sents.get(uid, set())),
                }
            )
        judged = await self._diary_reranker.rerank_sentences(
            query,
            shortlist,
            context=context,
            budget=getattr(cfg, "sentence_budget", 4),
        )
        if judged is None:
            # Judge API/parse failure → skip this round entirely (宁缺勿整篇
            # 回退 — a whole-diary fallback would break the sentence ledger).
            logger.info("[diary_rag] judge failed — no injection this round.")
            return [], candidates, keywords

        out = [
            {
                "uid": j["id"],
                "date": j["date"],
                "sents": j["sents"],
                "sentences": j["sentences"],
                "reason": j.get("reason", ""),
            }
            for j in judged
        ]
        return out, candidates, keywords

    def _read_diary(self, uid: str) -> Optional[Dict[str, Any]]:
        """Load a single diary entry by uid, or None if missing/unreadable."""
        path = os.path.join(self._diaries_dir, f"{uid}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            if isinstance(entry, dict) and "content" in entry:
                entry.setdefault("date", self._session_date_from_uid(uid))
                return entry
        except Exception:
            pass
        return None

    @staticmethod
    def _diary_chunks(uid: str, content: str, date: str) -> List[Dict[str, Any]]:
        """Split a diary into paragraph-level chunk items for the vector index.

        Each chunk id is ``<diary_uid>#<n>``; ``meta.parent`` points back at the
        diary so retrieval can group chunks and recall the whole diary. The
        first boot after the 08-23 paragraph redesign re-embeds every diary
        (same id form, drifted text) and prunes the stale sentence chunks —
        ensure_indexed handles both, no migration pass needed.
        """
        return [
            {"id": f"{uid}#{i}", "text": para, "meta": {"parent": uid, "date": date}}
            for i, para in enumerate(_split_paragraphs(content))
        ]

    def _all_diary_chunks_for_index(self) -> List[Dict[str, Any]]:
        """Every diary's chunks as ``{id, text, meta}`` for ensure_indexed."""
        items: List[Dict[str, Any]] = []
        if not os.path.isdir(self._diaries_dir):
            return items
        for fname in os.listdir(self._diaries_dir):
            if not fname.endswith(".json") or not _SESSION_UID_RE.match(fname[:-5]):
                continue
            uid = fname[:-5]
            entry = self._read_diary(uid)
            if entry and entry.get("content"):
                items.extend(
                    self._diary_chunks(uid, entry["content"], entry.get("date", uid))
                )
        return items

    @staticmethod
    def resolve_embed_credentials(diary_rag_config: Any, agent_config: Any) -> tuple:
        """Resolve (api_key, base_url) for embeddings.

        Falls back to the ``openai_llm`` provider's credentials when the
        ``diary_rag`` block leaves them blank, so a user already on OpenAI needs
        no extra config. The framework's ``"default_api_key"`` placeholder is
        treated as absent.
        """
        key = (
            (getattr(diary_rag_config, "openai_api_key", "") or "")
            if diary_rag_config
            else ""
        )
        base = (
            (getattr(diary_rag_config, "base_url", "") or "")
            if diary_rag_config
            else ""
        )
        if not key and agent_config is not None:
            openai_cfg = getattr(
                getattr(agent_config, "llm_configs", None), "openai_llm", None
            )
            if openai_cfg is not None:
                key = getattr(openai_cfg, "llm_api_key", "") or ""
                base = base or (getattr(openai_cfg, "base_url", "") or "")
        if key == "default_api_key":
            key = ""
        return key, base

    @staticmethod
    def _norm_ts(ts: Any) -> str:
        """Comparable ``YYYY-MM-DD HH:MM:SS`` form. History records stamp
        ISO-T (isoformat) while facts' ``updated`` uses a space; comparing
        the two lexicographically breaks on same-day values (' ' < 'T') —
        which silently emptied the ENTIRE low listing on 08-23 (77/354
        shown = zero low facts, duplicate re-extraction observed live).
        Every timestamp entering a window comparison goes through here."""
        return str(ts or "").strip().replace("T", " ")

    @classmethod
    def _earliest_timestamp(cls, messages: List[Dict[str, Any]]) -> str:
        """Earliest ``timestamp`` among history records ('' when none carry
        one), normalized via _norm_ts so min() = chronological."""
        stamps = [
            cls._norm_ts(m.get("timestamp", ""))
            for m in messages
            if isinstance(m, dict)
        ]
        stamps = [s for s in stamps if s]
        return min(stamps) if stamps else ""

    @staticmethod
    def _facts_for_extraction_list(
        facts: List[Dict[str, Any]], window_start: str
    ) -> List[Dict[str, Any]]:
        """Existing-facts listing for the extraction prompt (あさひ 08-21).

        user/high always; ``low`` only when its ``updated`` falls inside the
        extraction input's own time window — a low fact older than every
        message/diary being analysed can't be re-derived from them, so
        listing it buys no dedup and only bloats the prompt (low was 73% of
        the listing's characters when this shipped). Empty ``window_start``
        keeps the historical pass-everything behaviour. The saved pool is
        NOT affected — this trims the prompt listing only.
        """
        if not window_start:
            return list(facts)
        window_start = PersistentMemoryManager._norm_ts(window_start)
        return [
            f
            for f in facts
            # user/high always shown; low AND archive are window-scoped (an
            # archived fact outside the window can't be re-summarised anyway).
            if (f.get("importance") or "low") not in ("low", "archive")
            or PersistentMemoryManager._norm_ts(f.get("updated", "")) >= window_start
        ]

    async def extract_facts_async(
        self,
        recent_messages: List[Dict[str, Any]],
        llm: Any,
        diary_context: str = "",
        persona: str = "",
        window_start: str = "",
    ) -> None:
        """Extract new facts from recent messages and append to facts.json.

        Runs as a fire-and-forget background task. ``diary_context`` is an
        optional summary of older sessions (used during backfill) so the LLM
        has context beyond the sliding window without burning tokens on full
        message history. ``persona`` is the character's system prompt; when
        provided it is prepended so fact selection and pruning reflect what
        the character would consider memorable. ``window_start`` bounds the
        low-tier part of the existing-facts listing (see
        _facts_for_extraction_list); callers pass the earliest date of the
        input they're feeding in.
        """
        try:
            # Excluded turns (api_error / thinking_only) never reached the
            # character's context — they are not extraction material.
            recent_messages = strip_context_excluded(recent_messages)
            existing = self._load_facts()
            shown = self._facts_for_extraction_list(existing, window_start)
            if len(shown) < len(existing):
                logger.info(
                    f"[memory] Existing-facts list trimmed for extraction: "
                    f"{len(shown)}/{len(existing)} shown (low updated before "
                    f"{window_start} omitted)"
                )
            # Show each existing fact's current importance so the LLM tags new
            # facts consistently with the established tiering (it must still
            # never output "user" — see _FACT_EXTRACT_SYSTEM).
            existing_text = (
                "\n".join(
                    f"- [{f.get('importance', 'low')}] {f['fact']}" for f in shown
                )
                if shown
                else "(まだありません)"
            )
            conv_text = self._format_messages(recent_messages)
            if not conv_text.strip() and not diary_context.strip():
                logger.info(
                    f"[memory] Fact extraction skipped: empty input ({len(recent_messages)} raw msgs)"
                )
                return

            prompt_parts = [f"既存の事実リスト（繰り返さないこと）:\n{existing_text}"]
            # Tag vocabulary (あさひ 09-29): names in use, most used first,
            # so the extractor reuses them instead of coining near-duplicates.
            vocab = self.tag_vocabulary()
            if vocab:
                prompt_parts.append(
                    "既存のタグ一覧（再利用を優先）: "
                    + " ".join(f"#{t}({c})" for t, c in vocab[:80])
                )
            if diary_context.strip():
                prompt_parts.append(
                    f"以前のセッションのまとめ（参考）:\n{diary_context}"
                )
            if conv_text.strip():
                prompt_parts.append(f"分析する会話:\n{conv_text}")
            prompt = "\n\n".join(prompt_parts)
            logger.info(
                f"[memory] Extracting facts from {len(recent_messages)} messages "
                f"({len(conv_text)} chars conversation, {len(diary_context)} chars diary context)"
            )
            logger.debug(
                f"[memory] Fact extraction conversation preview: {conv_text[:400]!r}"
            )
            # NOTE: fact extraction deliberately does NOT prepend persona.
            # Persona context was tried but conflicts directly with the
            # "no roleplay / no [tag] markers / raw JSON only" instructions
            # (the persona tells the model to be the character with tags),
            # causing it to defensively output []. Fact extraction wants an
            # objective, neutral lens on the user, not a character lens.
            raw = await self._call_llm(llm, _FACT_EXTRACT_SYSTEM, prompt)
            # Full raw output (not truncated): we want to see exactly what
            # the LLM returned, including any preamble that fooled the parser.
            logger.info(f"[memory] Fact-extraction LLM raw output:\n{raw}")
            new_facts = self._parse_json_list(raw)
            if not new_facts:
                logger.info("[memory] No new facts extracted.")
                return

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # The LLM tags each fact "high" (keep in header) or "low"
            # (RAG-recalled). "user" is manual-only: if the LLM assigns it,
            # demote to "high" rather than rework — but log it so the slip is
            # visible. Legacy "llm" (pre-rename) maps to "high"; anything else
            # missing/invalid → low.
            tagged: List[Dict[str, Any]] = []
            for f in new_facts:
                if "fact" not in f:
                    continue
                imp = f.get("importance")
                if imp == "user":
                    logger.warning(
                        f"[memory] Extraction LLM tagged a fact 'user' (manual-only); "
                        f"demoted to 'high': {f['fact']!r}"
                    )
                    imp = "high"
                elif imp == "llm":
                    imp = "high"
                elif imp not in ("high", "low"):
                    imp = "low"
                tag_list = self._normalize_tags(f.get("tags"))
                tagged.append(
                    {
                        "fact": f["fact"],
                        "updated": now,
                        "importance": imp,
                        # Creator = the pen model of this extraction run.
                        "history": [
                            self._history_event(
                                "create",
                                str(
                                    getattr(self._memory_llm or llm, "model", "") or ""
                                ),
                                now,
                            )
                        ],
                    }
                )
                if tag_list:
                    tagged[-1]["tags"] = tag_list
            merged = existing + tagged
            # Smart trim: ask the LLM to drop least-important entries when
            # over the cap. The whole merged pool (old + new) is the
            # candidate set — newly extracted facts are NOT privileged.
            if len(merged) > self._max_facts:
                merged = await self._prune_facts_with_llm(
                    merged, self._max_facts, llm, persona=persona
                )
            self._save_facts(merged)

            # Detailed multi-line summary: distinguish which newly-extracted
            # facts survived, which were dropped right after extraction, and
            # which existing facts were displaced.
            final_text = {m["fact"] for m in merged}
            new_kept = [t for t in tagged if t["fact"] in final_text]
            new_dropped = [t for t in tagged if t["fact"] not in final_text]
            existing_dropped = [e for e in existing if e["fact"] not in final_text]
            self._log_fact_update(
                added=new_kept,
                discarded_new=new_dropped,
                dropped_existing=existing_dropped,
                total=len(merged),
            )
        except Exception as e:
            logger.warning(f"[memory] Fact extraction failed: {e}", exc_info=True)

    async def create_diary_async(
        self,
        history_messages: List[Dict[str, Any]],
        history_uid: str,
        llm: Any,
        persona: str = "",
    ) -> None:
        """Generate and save a diary entry for the finished session.

        ``persona`` is the character's system prompt; when provided the diary
        is written in the character's voice rather than a generic narrator.
        """
        try:
            # Excluded turns (api_error / thinking_only) never reached the
            # character's context — the diary must not "remember" them either.
            history_messages = strip_context_excluded(history_messages)
            if not history_messages:
                return
            conv_text = self._format_messages(history_messages)
            if not conv_text.strip():
                return

            # Build time-range header so LLM uses specific times instead of "今日".
            session_date = self._session_date_from_uid(history_uid)
            end_hm = self._session_end_hm_from_messages(history_messages)
            start_hm = session_date[11:16] if len(session_date) > 10 else ""
            time_range = (
                f"{start_hm}〜{end_hm}" if start_hm and end_hm else start_hm or end_hm
            )
            nth = (
                self._count_same_day_diaries(session_date[:10], exclude_uid=history_uid)
                + 1
            )
            header_parts = []
            if time_range:
                header_parts.append(f"セッション時間: {time_range}")
            if nth > 1:
                header_parts.append(f"この日の{nth}回目の会話セッション")
            if header_parts:
                conv_text = (
                    "[セッション情報]\n" + "\n".join(header_parts) + "\n\n" + conv_text
                )

            # Inject the always-on facts (user/llm tier when facts RAG is on,
            # else all facts — exactly what the chat model keeps in its header)
            # as background, placed AFTER the persona but BEFORE the diary-writing
            # rules. This keeps _DIARY_SYSTEM adjacent to the conversation so the
            # task instruction holds the high-attention slot (mirrors how the
            # chat path puts _HISTORY_NOTE last); the facts are reference data
            # and sit fine in the labelled middle block.
            base = _DIARY_SYSTEM
            header_facts = self._header_facts()
            if header_facts:
                fact_lines = "\n".join(
                    f"- [{str(f.get('updated', ''))[:10] or '不明'}] {f['fact']}"
                    for f in header_facts
                )
                base = f"{_DIARY_FACTS_NOTE}\n\n{fact_lines}\n\n---\n\n{_DIARY_SYSTEM}"
            content = await self._call_llm(
                llm, self._with_persona(base, persona), conv_text
            )
            content = content.strip()
            if not content:
                return

            # Use the session's start time (encoded in history_uid) as the date,
            # so backfilled diaries sort correctly with newly-created ones.
            session_date = self._session_date_from_uid(history_uid)
            pen = self._memory_llm or llm
            diary_entry = {
                "date": session_date,
                "history_uid": history_uid,
                "content": content,
                # Pen model for the record (display treats non-self diaries
                # as gpt5.1-written per the prompt-side convention).
                "writer": str(getattr(pen, "model", "") or ""),
            }
            filename = f"{history_uid}.json"
            path = os.path.join(self._diaries_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(diary_entry, f, ensure_ascii=False, indent=2)
            logger.debug(f"[memory] Saved diary for session {history_uid}")
            if self._diary_index is not None:
                await self._diary_index.add_many(
                    self._diary_chunks(history_uid, content, session_date)
                )
        except Exception as e:
            logger.warning(f"[memory] Diary creation failed: {e}")

    async def end_of_session_async(
        self,
        history_messages: List[Dict[str, Any]],
        history_uid: str,
        llm: Any,
        persona: str = "",
    ) -> None:
        """Run diary generation and fact extraction concurrently at session end.

        Both tasks receive the full session history so neither is starved of
        context. ``persona`` is forwarded so the diary is in-character and
        fact selection / pruning reflect the character's perspective.
        Runs as a fire-and-forget background task.
        """
        # Filter here too (both workers also self-filter) so window_start is
        # computed over the messages that will actually be processed.
        history_messages = strip_context_excluded(history_messages)
        await asyncio.gather(
            self.create_diary_async(
                history_messages, history_uid, llm, persona=persona
            ),
            self.extract_facts_async(
                history_messages,
                llm,
                persona=persona,
                window_start=self._earliest_timestamp(history_messages),
            ),
            return_exceptions=True,
        )
        # Mark diary so backfill knows this session's facts were already extracted.
        self._mark_diary_facts_extracted(history_uid)

    async def backfill_async(self, conf_uid: str, llm: Any, persona: str = "") -> bool:
        """Generate diaries and facts for sessions that don't have them yet.

        Diary backfill: creates a diary for each session that has messages but
        no diary file yet.
        Fact backfill: processes any diary that doesn't carry a
        ``"facts_extracted": true`` marker, using the sliding-window approach
        (recent N sessions in full + older diary summaries) so the prompt stays
        bounded regardless of how many sessions exist.
        Both passes are idempotent. Guarded by a process-wide lock so concurrent
        connections don't kick off duplicate backfills for the same character.

        Returns ``True`` if this call actually ran the backfill (so the caller
        can signal "system prompt settled"), or ``False`` if it early-returned
        because another connection's backfill is already in progress — that
        concurrent caller must NOT signal settled, since the real run is still
        going. A run that completes with no work to do still returns ``True``.
        """
        if conf_uid in PersistentMemoryManager._backfill_in_progress:
            return False
        PersistentMemoryManager._backfill_in_progress.add(conf_uid)
        try:
            from ..chat_history_manager import get_history_list, get_history

            history_list = get_history_list(conf_uid)

            # --- Diary backfill ---
            # Skip the currently-active (in-progress) session: its diary should
            # only be written by end_of_session_async once the session finishes.
            skip_uid = self._current_session_uid
            missing_diaries = []
            for entry in history_list:
                uid = entry["uid"]
                if uid == skip_uid:
                    continue
                diary_path = os.path.join(self._diaries_dir, f"{uid}.json")
                if not os.path.exists(diary_path):
                    missing_diaries.append(uid)

            if missing_diaries:
                logger.info(
                    f"[memory] Backfilling {len(missing_diaries)} session diary entries…"
                )
                for uid in missing_diaries:
                    messages = get_history(conf_uid, uid)
                    if messages:
                        await self.create_diary_async(
                            messages, uid, llm, persona=persona
                        )
                logger.info("[memory] Diary backfill complete.")

            # --- Fact backfill: sessions whose diary lacks facts_extracted=True ---
            # This handles the server-restart case where end_of_session_async
            # never ran for the last active session.
            unprocessed_uids: List[str] = []
            if os.path.isdir(self._diaries_dir):
                for fname in sorted(os.listdir(self._diaries_dir)):
                    if not fname.endswith(".json"):
                        continue
                    uid = fname[:-5]
                    if uid == skip_uid:
                        continue
                    path = os.path.join(self._diaries_dir, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        if not d.get("facts_extracted"):
                            unprocessed_uids.append(uid)
                    except Exception:
                        continue

            # Fact extraction only runs when there are unprocessed sessions.
            # Note: we do NOT early-return here — the fact-limit enforcement
            # below must run on every startup regardless.
            if unprocessed_uids:
                logger.info(
                    f"[memory] {len(unprocessed_uids)} session(s) pending fact extraction."
                )

                # Use the most recent N unprocessed sessions in full; the rest
                # as diary summaries to keep token cost bounded.
                unprocessed_uids.sort()  # lexicographic = chronological
                recent_uids = set(unprocessed_uids[-self._recent_sessions :])
                recent_messages: List[Dict[str, Any]] = []
                for uid in unprocessed_uids[-self._recent_sessions :]:
                    # Excluded turns are filtered here so the extraction
                    # window (min timestamp below) matches the filtered input;
                    # extract_facts_async filters again defensively.
                    msgs = strip_context_excluded(get_history(conf_uid, uid))
                    if msgs:
                        recent_messages.extend(msgs)

                older_parts: List[str] = []
                older_dates: List[str] = []
                for uid in unprocessed_uids:
                    if uid in recent_uids:
                        continue
                    path = os.path.join(self._diaries_dir, f"{uid}.json")
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        if "content" in d:
                            older_parts.append(
                                f"[{d.get('date', uid)}]\n{d['content']}"
                            )
                            if d.get("date"):
                                older_dates.append(self._norm_ts(d["date"]))
                    except Exception:
                        continue
                diary_context = "\n\n".join(older_parts)
                # Earliest date across everything the prompt will contain —
                # bounds the low-tier existing-facts listing.
                window_candidates = [
                    c
                    for c in [self._earliest_timestamp(recent_messages)] + older_dates
                    if c
                ]
                window_start = min(window_candidates) if window_candidates else ""

                if recent_messages or diary_context:
                    logger.info(
                        f"[memory] Running fact extraction backfill "
                        f"({len(recent_uids)} recent session(s) full, "
                        f"{len(older_parts)} older diary summary/summaries)…"
                    )
                    await self.extract_facts_async(
                        recent_messages,
                        llm,
                        diary_context=diary_context,
                        persona=persona,
                        window_start=window_start,
                    )
                    # Mark all processed diaries so this doesn't repeat next startup.
                    for uid in unprocessed_uids:
                        self._mark_diary_facts_extracted(uid)
                    logger.info("[memory] Fact backfill complete.")

            # Enforce the fact cap unconditionally — covers the case where the
            # user lowered max_facts in config but no new facts were extracted
            # this run (in-place pruning otherwise only triggers when a fact is
            # added, so an oversized file would keep injecting every entry).
            await self._enforce_fact_limit_async(llm, persona=persona)

            # Embed any diaries that don't yet have a vector (first run embeds
            # them all; later runs only the freshly backfilled ones). Prunes
            # vectors whose diary was deleted.
            if self._diary_index is not None:
                await self._diary_index.ensure_indexed(
                    self._all_diary_chunks_for_index()
                )
            # Same for facts: embed new/edited facts, prune vectors of facts that
            # were consolidated or pruned away (id = content fingerprint, so an
            # edited fact is a new id + an orphaned old one).
            if self._facts_index is not None:
                await self._facts_index.ensure_indexed(self._facts_items_for_index())
        except Exception as e:
            logger.warning(f"[memory] Backfill failed: {e}", exc_info=True)
        finally:
            PersistentMemoryManager._backfill_in_progress.discard(conf_uid)
            # If the header hasn't been built yet (the usual case — the user
            # talks minutes after boot), the first build simply reads the
            # settled disk state. But when a turn already shipped this
            # session's header — an OVERDUE ALARM can fire within seconds of
            # boot, beating backfill — resetting here would change the system
            # prompt mid-session and bust the prefix cache (08-07: turn-2 hit
            # 13%). The settled facts stay RAG-reachable and enter the header
            # on the next boot, same as any mid-session memory_* write.
            if self._header_snapshot is not None:
                logger.info(
                    "[memory] backfill settled after the first turn — header "
                    "keeps the boot snapshot (new facts reach RAG only)."
                )
        # Reached only by the call that actually ran (the early-return above
        # exits first). True even if the work errored — facts are in their
        # final state for this startup either way, so the prompt is settled.
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_facts(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._facts_path):
            return []
        try:
            with open(self._facts_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    @staticmethod
    def _sort_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a copy of facts sorted by `updated` ascending (oldest first).

        Stored timestamps are ISO `YYYY-MM-DD HH:MM:SS`, so lexicographic
        sort = chronological sort. Entries with missing/empty `updated`
        sort first (treated as "earliest known").
        """
        return sorted(facts, key=lambda f: str(f.get("updated", "")))

    def _save_facts(self, facts: List[Dict[str, Any]]) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        # Backup current file before overwriting so accidental pruning can be
        # manually rolled back by renaming facts.json.bak → facts.json.
        if os.path.exists(self._facts_path):
            bak = self._facts_path + ".bak"
            try:
                shutil.copy2(self._facts_path, bak)
            except Exception as e:
                logger.warning(f"[memory] Failed to backup facts.json: {e}")
        # Every fact leaves here with its persistent handle (see _fact_ref):
        # new ones from any writer — tools, extraction, the viewer's handle-
        # less additions — get stamped on the first production save.
        self._ensure_fact_ids(facts)
        # Always persist in chronological order so the file is predictable
        # both for the LLM (oldest-first reading) and human review.
        ordered = self._sort_facts(facts)
        with open(self._facts_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False, indent=2)

    def _mark_diary_facts_extracted(self, history_uid: str) -> None:
        """Set facts_extracted=True on the diary file for history_uid (no-op if missing)."""
        diary_path = os.path.join(self._diaries_dir, f"{history_uid}.json")
        if not os.path.exists(diary_path):
            return
        try:
            with open(diary_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            if not entry.get("facts_extracted"):
                entry["facts_extracted"] = True
                with open(diary_path, "w", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_recent_diaries(self) -> List[Dict[str, Any]]:
        if not os.path.isdir(self._diaries_dir):
            return []
        try:
            entries = []
            for fname in os.listdir(self._diaries_dir):
                if not fname.endswith(".json") or not _SESSION_UID_RE.match(fname[:-5]):
                    continue
                history_uid = fname[:-5]
                # Skip diaries for sessions already in the agent's sliding
                # window — those messages are present verbatim, so the diary
                # would just duplicate them.
                if history_uid in self._active_session_uids:
                    continue
                path = os.path.join(self._diaries_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                    if isinstance(entry, dict) and "content" in entry:
                        entry.setdefault("history_uid", history_uid)
                        entries.append(entry)
                except Exception:
                    continue
            # Sort by history_uid: it begins with the session's start timestamp
            # (YYYY-MM-DD_HH-MM-SS_<hex>) so lexicographic order = chronological.
            entries.sort(key=lambda e: e.get("history_uid", ""))
            return entries[-self._diary_count :]
        except Exception:
            return []

    @staticmethod
    def _session_date_from_uid(history_uid: str) -> str:
        """Parse the human-readable session start time out of a history_uid."""
        parts = history_uid.split("_")
        if len(parts) >= 2 and len(parts[0]) == 10 and len(parts[1]) == 8:
            time_part = parts[1].replace("-", ":")
            return f"{parts[0]} {time_part}"
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Session→model attribution (chat_history/_model_map.json)
    # ------------------------------------------------------------------
    # The pre-08-18 era was hand-adjudicated (console logs + cost CSV +
    # あさひ rulings — see the map's own comment field); finished sessions
    # the map does not cover yet are attributed here at boot from their
    # thinking_seed.model fields (served truth). The in-progress session
    # rides only the in-memory snapshot — once it ends, the next boot
    # attributes it from seeds like any other.

    @property
    def _model_map_path(self) -> str:
        return os.path.join("chat_history", "_model_map.json")

    def refresh_model_map(self, current_uid: str = "") -> None:
        """Attribute unmapped finished sessions from their seeds, persist the
        map, and freeze an in-memory snapshot (the model_history tool and the
        diary annotations read only the snapshot). Never raises."""
        try:
            data: Dict[str, Any] = {}
            if os.path.exists(self._model_map_path):
                with open(self._model_map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            sessions = data.setdefault("sessions", {})
            added = 0
            if os.path.isdir(self._base_dir):
                for fname in os.listdir(self._base_dir):
                    if not fname.endswith(".json") or not _SESSION_UID_RE.match(
                        fname[:-5]
                    ):
                        continue
                    uid = fname[:-5]
                    if uid in sessions or uid == current_uid:
                        continue
                    entry = self._attribute_session_from_seeds(
                        os.path.join(self._base_dir, fname)
                    )
                    if entry is not None:
                        sessions[uid] = entry
                        added += 1
            if added:
                data["generated_at"] = datetime.now().isoformat(timespec="seconds")
                tmp = self._model_map_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self._model_map_path)
                logger.info(
                    f"[model_map] attributed {added} new session(s) from seeds."
                )
            snapshot = dict(sessions)
            if current_uid and self._chat_model:
                snapshot[current_uid] = {
                    "start": self._session_date_from_uid(current_uid).replace(" ", "T"),
                    "end": None,
                    "verdict": "single",
                    "ongoing": True,
                    "models": {
                        self._chat_model: {"messages": 0, "sources": {"current": 1}}
                    },
                }
            self._model_map_snapshot = snapshot
        except Exception as e:
            logger.warning(f"[model_map] refresh failed: {e}")

    @staticmethod
    def _attribute_session_from_seeds(path: str) -> Optional[Dict[str, Any]]:
        """Map entry for one finished session file, from its thinking seeds.
        Sessions with no seed at all get verdict 'uncertain' and no models —
        the display side degrades to no annotation, never a guess."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                msgs = json.load(f)
            if not isinstance(msgs, list) or not msgs:
                return None
            models: Dict[str, int] = {}
            first_ts = last_ts = ""
            for m in msgs:
                ts = str(m.get("timestamp") or "")
                if ts:
                    first_ts = first_ts or ts
                    last_ts = ts
                seed = m.get("thinking_seed")
                if m.get("role") == "ai" and isinstance(seed, dict):
                    mid = str(seed.get("model") or "")
                    if mid:
                        models[mid] = models.get(mid, 0) + 1
            verdict = (
                "single" if len(models) == 1 else ("mixed" if models else "uncertain")
            )
            return {
                "start": first_ts,
                "end": last_ts,
                "verdict": verdict,
                "models": {
                    k: {"messages": v, "sources": {"seed": v}}
                    for k, v in models.items()
                },
            }
        except Exception:
            return None

    def model_label_for_session(self, uid: str) -> str:
        """Short experiencer label for a session uid ('opus5'; a mixed
        session joins its models with '/'; unknown/unmapped → '')."""
        return "/".join(model_short_name(m) for m in self.session_model_ids(uid))

    def session_model_ids(self, uid: str) -> List[str]:
        """Full model ids attributed to a session, in order of first
        appearance ([] when unknown/unmapped) — the past-session banner shows
        them in the same form the current-session banner shows the running
        model. Reads the snapshot only, like every other annotation."""
        s = (self._model_map_snapshot or {}).get(uid)
        if not isinstance(s, dict):
            return []
        return [str(m) for m in (s.get("models") or {}).keys() if m]

    def diary_display_tag(
        self, uid: str, entry: Optional[Dict[str, Any]] = None
    ) -> str:
        """Model annotation for one diary: 'opus5', 'opus5 本人執筆', or ''.

        The experiencer comes from the session map; 本人執筆 marks diaries
        the character wrote herself via memory_write_diary (writer='self' on
        the diary file). Unmarked diaries were written by the memory model
        (gpt-5.1) — stated once in the prompt-side note, not per entry."""
        label = self.model_label_for_session(uid)
        if not label:
            return ""
        if entry is None:
            entry = self._read_diary(uid) or {}
        if entry.get("writer") == "self":
            label += " 本人執筆"
        return label

    def sessions_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Sessions whose [start, end] day-interval touches the given JST
        date (YYYY-MM-DD) — the model_history tool. Sorted by start."""
        rows: List[Dict[str, Any]] = []
        for uid, s in (self._model_map_snapshot or {}).items():
            if not isinstance(s, dict):
                continue
            start = str(s.get("start") or "")[:10]
            end = str(s.get("end") or "")[:10] or start
            if not start or not (start <= date_str <= end):
                continue
            rows.append(
                {
                    "start": str(s.get("start") or "").replace("T", " "),
                    "end": (
                        "進行中"
                        if s.get("ongoing")
                        else str(s.get("end") or "").replace("T", " ")
                    ),
                    "model": self.model_label_for_session(uid) or "不明",
                }
            )
        rows.sort(key=lambda r: r["start"])
        return rows

    @staticmethod
    def _session_end_hm_from_messages(messages: List[Dict[str, Any]]) -> str:
        """Return HH:MM of the last message's timestamp, or empty string."""
        for m in reversed(messages):
            ts = m.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return dt.strftime("%H:%M")
                except (ValueError, TypeError):
                    pass
        return ""

    def _count_same_day_diaries(self, date_str: str, exclude_uid: str = "") -> int:
        """Count diary files whose uid starts with date_str (YYYY-MM-DD)."""
        if not os.path.isdir(self._diaries_dir):
            return 0
        count = 0
        for fname in os.listdir(self._diaries_dir):
            if not fname.endswith(".json"):
                continue
            uid = fname[:-5]
            if uid == exclude_uid:
                continue
            if uid.startswith(date_str):
                count += 1
        return count

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            # Strip timestamp tags that _to_text_prompt prepends to user messages
            # so the fact-extraction LLM focuses on the actual speech content.
            content = _TIMESTAMP_RE.sub("", content).strip()
            if content:
                label = "ユーザー" if role in ("user", "human") else "AI"
                lines.append(f"{label}: {content}")
        return "\n".join(lines)

    async def _call_llm(
        self, llm: Any, system: str, prompt: str, max_tokens: int = 4096
    ) -> str:
        # Route every memory task (fact extraction, diary, prune)
        # through the dedicated memory model when one is configured — these are
        # big, uncached one-shot calls that don't need the chat model.
        llm = self._memory_llm or llm
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = ""
        # Memory tasks (fact extraction, diary summary, fact pruning,
        # consolidation) are one-shot tool-style calls with no cache and no
        # need for the chat agent's web tools. Pass:
        #   max_tokens=4096 — default 1024 has truncated long fact arrays
        #     mid-entry; these calls need more headroom.
        #   disable_server_tools=True — keeps the web_search / web_fetch
        #     tool definitions out of the request, saving ~200-300 tokens
        #     per memory call when those tools are enabled for chat.
        #   reasoning_effort=self._memory_reasoning_effort — gpt-5.1 defaults
        #     reasoning_effort to "none", so without this it does ZERO reasoning
        #     and lazily returns "[]" for extraction (observed: a 148-message
        #     session yielded 0 facts with reasoning=0). An explicit effort makes
        #     these judgment-heavy tasks actually think. Empty string → not sent
        #     (model default); also ignored by models/endpoints that don't
        #     support it (graceful API fallback).
        # All three kwargs fall back gracefully on LLM impls that don't accept
        # them (TypeError → retry with positional-only args).
        try:
            stream = llm.chat_completion(
                messages,
                system,
                max_tokens=max_tokens,
                disable_server_tools=True,
                reasoning_effort=self._memory_reasoning_effort,
            )
        except TypeError:
            try:
                stream = llm.chat_completion(messages, system, max_tokens=max_tokens)
            except TypeError:
                # Older LLM impls without max_tokens or disable_server_tools.
                stream = llm.chat_completion(messages, system)
        async for event in stream:
            if isinstance(event, str):
                result += event
            elif isinstance(event, dict) and event.get("type") == "text_delta":
                result += event.get("text", "")
        return result

    @staticmethod
    def _with_persona(base_system: str, persona: str) -> str:
        """Prepend the character persona block to a memory-task system prompt."""
        if not persona or not persona.strip():
            return base_system
        return f"あなたの人格設定:\n{persona.strip()}\n\n---\n\n{base_system}"

    @staticmethod
    def _parse_int_list(text: str) -> List[int]:
        """Extract a JSON array of integers from LLM output."""
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(text[start : end + 1])
            return [int(x) for x in data if isinstance(x, (int, float))]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def _log_fact_update(
        self,
        *,
        added: List[Dict[str, Any]],
        discarded_new: List[Dict[str, Any]],
        dropped_existing: List[Dict[str, Any]],
        total: int,
    ) -> None:
        """Multi-line summary of an extraction/pruning round.

        Each fact gets its own line so long updates stay readable in the
        log. Three buckets are reported separately:
          - added: newly extracted facts that survived any concurrent pruning
          - discarded_new: just-extracted facts that the prune step dropped
          - dropped_existing: pre-existing facts that the prune step dropped
        """
        lines = [f"[memory] Fact update → {self._facts_path} (total: {total})"]
        if added:
            lines.append(f"  Added {len(added)} new fact(s):")
            for f in added:
                lines.append(f"    + [{f.get('importance', 'low')}] {f['fact']}")
        if discarded_new:
            lines.append(
                f"  Discarded {len(discarded_new)} newly-extracted fact(s) "
                "(judged less valuable than alternatives):"
            )
            for f in discarded_new:
                lines.append(f"    - {f['fact']}")
        if dropped_existing:
            lines.append(f"  Dropped {len(dropped_existing)} existing fact(s):")
            for f in dropped_existing:
                date = str(f.get("updated", ""))[:10] or "不明"
                lines.append(f"    - [{date}] {f['fact']}")
        if not (added or discarded_new or dropped_existing):
            lines.append("  (no changes)")
        logger.info("\n".join(lines))

    async def _enforce_fact_limit_async(self, llm: Any, persona: str = "") -> None:
        """Trim facts.json down to max_facts if it currently exceeds the cap.

        In-place pruning otherwise only runs when a new fact is added, so a
        file that became oversized (e.g. the user lowered max_facts in config)
        would keep injecting every entry into the prompt until the next
        extraction. This is called once per startup from backfill_async.
        """
        facts = self._load_facts()
        if len(facts) <= self._max_facts:
            return
        logger.info(
            f"[memory] facts.json has {len(facts)} entries, over the "
            f"max_facts={self._max_facts} cap; pruning down."
        )
        pruned = await self._prune_facts_with_llm(
            facts, self._max_facts, llm, persona=persona
        )
        self._save_facts(pruned)
        pruned_text = {p["fact"] for p in pruned}
        dropped = [f for f in facts if f["fact"] not in pruned_text]
        self._log_fact_update(
            added=[],
            discarded_new=[],
            dropped_existing=dropped,
            total=len(pruned),
        )

    async def _prune_facts_with_llm(
        self,
        facts: List[Dict[str, Any]],
        target_count: int,
        llm: Any,
        persona: str = "",
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to drop the N least-important facts (N = excess).

        Falls back to FIFO trimming (drop oldest) if the LLM output is
        malformed or returns the wrong number of indices.
        """
        excess = len(facts) - target_count
        if excess <= 0:
            return facts
        # Include timestamp so the LLM can judge staleness / supersession.
        numbered = "\n".join(
            f"{i} [{f.get('updated', '不明')}]: {f['fact']}"
            for i, f in enumerate(facts)
        )
        prompt = (
            f"現在{len(facts)}個の事実があり、上限は{target_count}個です。\n"
            f"最も価値の低い{excess}個を選んで削除してください。\n\n"
            f"事実リスト（形式: インデックス [更新日時]: 内容）:\n{numbered}\n\n"
            f"削除する{excess}個のインデックスをJSON配列で出力: [n, n, ...]"
        )
        try:
            # Same rationale as extract_facts_async: skip persona prefix to
            # avoid the "be in character / output only raw JSON" contradiction.
            raw = await self._call_llm(llm, _FACT_PRUNE_SYSTEM, prompt)
            indices = sorted(
                {i for i in self._parse_int_list(raw) if 0 <= i < len(facts)}
            )
            if len(indices) != excess:
                logger.warning(
                    f"[memory] Fact-prune LLM returned {len(indices)} indices, "
                    f"expected {excess}; falling back to FIFO trimming."
                )
                return facts[-target_count:]
            # Verbose per-fact reporting is done by the caller via
            # _log_fact_update; keep only a debug breadcrumb here.
            logger.debug(
                f"[memory] LLM-prune picked indices {sorted(indices)} "
                f"of {len(facts)} fact(s) for removal."
            )
            return [f for i, f in enumerate(facts) if i not in set(indices)]
        except Exception as e:
            logger.warning(
                f"[memory] Fact pruning failed ({e}); falling back to FIFO trimming."
            )
            return facts[-target_count:]

    @staticmethod
    def _parse_json_list(text: str) -> List[Dict[str, Any]]:
        """Extract a JSON array of {"fact": ...} objects from LLM output.

        Robust to two failure modes seen in the wild:
        1) The LLM prefaces its response with in-character text containing
           [neutral] / [smirk] / etc. — naive "first [ to last ]" would span
           the whole thing and fail to parse.
        2) The LLM wraps the array in a ```json fenced block.
        3) The LLM is cut off mid-array by max_tokens, so the final entry
           is partial — recover everything up to the last complete object.
        """
        text = text.strip()
        if not text:
            return []

        # Prefer a ```json ... ``` fence if present.
        fence_start = text.find("```json")
        if fence_start != -1:
            after = text[fence_start + len("```json") :]
            fence_end = after.find("```")
            candidate = after[:fence_end] if fence_end != -1 else after
            parsed = PersistentMemoryManager._try_parse_fact_array(candidate)
            if parsed is not None:
                return parsed

        # Otherwise look for "[" followed (after any whitespace) by "{" —
        # the start of a JSON array-of-objects. Allow whitespace/newlines
        # between bracket and brace so cleanly-formatted multi-line output
        # parses too, while still skipping leading [tag] in-character
        # markers that have non-whitespace immediately after "[".
        import re

        m = re.search(r"\[\s*\{", text)
        if m is None:
            # Maybe the LLM correctly produced an empty array "[]".
            if re.search(r"\[\s*\]", text):
                return []
            return []
        candidate = text[m.start() :]
        parsed = PersistentMemoryManager._try_parse_fact_array(candidate)
        return parsed if parsed is not None else []

    @staticmethod
    def _try_parse_fact_array(candidate: str) -> Optional[List[Dict[str, Any]]]:
        """Try strict JSON first; on failure, recover entries object-by-object.

        Returns None if nothing useful could be parsed.
        """
        candidate = candidate.strip()
        if not candidate:
            return None
        # Strict parse: works when the LLM closed the array cleanly.
        end = candidate.rfind("]")
        if end != -1:
            try:
                data = json.loads(candidate[: end + 1])
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
            except json.JSONDecodeError:
                pass
        # Lenient parse: walk the string and extract balanced {...} objects.
        # Handles max_tokens truncation that left the array unclosed.
        results: List[Dict[str, Any]] = []
        i = 0
        n = len(candidate)
        while i < n:
            if candidate[i] != "{":
                i += 1
                continue
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = candidate[j]
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(candidate[i : j + 1])
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            break
                j += 1
            else:
                # Reached end without closing — truncated final object, drop it.
                break
            i = j + 1
        return results if results else None
