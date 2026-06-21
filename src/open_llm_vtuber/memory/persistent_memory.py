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
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set
from loguru import logger

from .vector_index import VectorIndex, extract_keywords

# Lazily-built Japanese sentence segmenter (pysbd) for diary chunking. Built on
# first use so importing this module stays cheap when RAG is disabled.
_JA_SEGMENTER = None


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences for chunk-level embedding (Japanese-aware)."""
    text = (text or "").strip()
    if not text:
        return []
    global _JA_SEGMENTER
    try:
        if _JA_SEGMENTER is None:
            import pysbd

            _JA_SEGMENTER = pysbd.Segmenter(language="ja", clean=False)
        sents = _JA_SEGMENTER.segment(text)
    except Exception:
        sents = [text]
    return [s.strip() for s in sents if s and s.strip()]

# Matches timestamp tags injected by _to_text_prompt: "[YYYY-MM-DD HH:MM:SS Weekday]"
_TIMESTAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+\]\s*", re.MULTILINE)


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
    "ここでは行わない。それは後から /facts-consolidate が担当する。\n"
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
    "判断基準：「1ヶ月後にもこの情報を参照する価値があるか？」を問う。"
    "No なら抽出しない。\n\n"
    "【ガイドライン】\n"
    "- 中国語・日本語・英語が混在していても、すべての言語の発言を対象にする\n"
    "- 既存の事実リストが提供される場合、それらをそのまま繰り返さない\n"
    "- 既存の事実には先頭に `[importance]`（現在の重要度）が付いている。"
    "新しい事実の重要度を判定する際の一貫性の参考にしてよい。\n\n"
    "【重要度の判定（importance）】\n"
    "抽出する各事実に importance を付ける。値は \"llm\" か \"low\" のどちらか。\n"
    "（\"user\" は使わない——それは人間が手動で指定する専用の値で、あなたが付けてはいけない。）\n"
    "- \"llm\" = ユーザー像を定義する中核的な事実で、常にキャラクターの念頭にあるべきもの:\n"
    "  学歴・専攻・資格、職業・専門スキル、出身地、年齢層、家族・重要な人間関係、\n"
    "  価値観・信念・性格特性、長期的な目標や進行中の重要プロジェクト、\n"
    "  重要な約束・合意、人生の節目・転機・トラウマ。\n"
    "- \"low\" = 覚えておく価値はあるが、関連する話題のときに思い出せれば十分な事実:\n"
    "  具体的な好みの細部、個別のエピソード、特定の物事（食べた物・買った物・観た作品など）、\n"
    "  中核とまでは言えない習慣や傾向。\n"
    "迷ったら \"low\"。\"llm\" は本当に常時参照する価値があるものだけに厳選する。\n\n"
    "**出力形式（厳守）**：\n"
    '[{"fact": "ユーザーは物理学部出身で物理学を専攻していた", "importance": "llm"}, '
    '{"fact": "ユーザーは白黒のポテトチップスが好き", "importance": "low"}]\n'
    "本当に新しい事実が1件もない場合のみ、空の配列のみを出力する: []\n"
    "繰り返す：JSON配列のみ。各要素は必ず \"fact\" と \"importance\" を持つ。"
    "\"importance\" は \"llm\" か \"low\"。`[`で始まり`]`で終わる。他のテキスト・記号は一切含めない。"
)

_CONSOLIDATE_SYSTEM = (
    "あなたはメモリ整理ツールです。これは会話ではありません。"
    "ロールプレイ、キャラクターとしての応答、感情表現タグ（[neutral]等）、"
    "前置き、コメント、Markdown装飾、コードフェンス（```）は一切禁止です。\n"
    "出力は**生のJSON配列のみ**。それ以外のテキストを1文字でも含めると失敗とみなされます。\n\n"
    "タスク：ユーザーに関する事実リストを整理し、関連性の高い項目を1つに統合する。"
    "総数を減らしつつ情報量は保つことが目的。\n\n"
    "【統合の対象例】\n"
    "- 同じカテゴリの列挙系：\n"
    "  例：複数の「視聴したアニメ」→「ユーザーは A, B, C を視聴した」\n"
    "  例：複数の「使用ツール」→「ユーザーは X, Y, Z を使用している」\n"
    "- 同じテーマの周辺事実を要約：\n"
    "  例：「物理学部出身」+「物理学を専攻」→「ユーザーは物理学部出身で物理学を専攻していた」\n"
    "- 微妙な重複・言い換え（古い方を新しい表現に統合）\n"
    "- **矛盾・状態更新**：新しい事実が古い事実を無効化している場合、"
    "新しい状態を反映するように統合する。どう統合するかは内容で判断：\n"
    "  ・**単純な進捗・状態の上書き** → 新しい状態のみ残し、古いものは捨てる\n"
    "    例：「ゼロエスケープの2章をプレイ中」+「3章をプレイ中」→「3章をプレイ中」\n"
    "    例：「Aタスクに取り組み中」+「Aタスクを完了した」→「Aタスクを完了した」\n"
    "  ・**経過自体に意味がある変化** → 「以前X、現在Y」の形で経過を残す\n"
    "    例：「就職活動中」+「会社Aに内定」→「就職活動を経て会社Aに内定した」\n"
    "    例：「雨が好き」+「雨が嫌いになった」→「以前は雨が好きだったが、今は嫌い」\n"
    "    例：「東京在住」+「大阪に引っ越した」→「以前は東京、現在は大阪在住」\n"
    "  ・判断基準：古い状態自体が**今後も参照する価値がある履歴か**。"
    "達成、価値観の変化、人生の節目、好みの変化、住所・職業の変遷などは履歴として残す。"
    "単なる進捗や状況の更新は上書きで構わない。\n"
    "- **複合事実の部分更新（外科的修正）**：1つの事実に複数の独立した情報が含まれていて、"
    "その**一部だけ**が新しい事実で無効化されている場合、その部分だけを更新する。"
    "事実全体を捨てない。\n"
    "  例：古い「ユーザーはAプロジェクト中で、Bツールを使用している」+ 新しい「Bプロジェクトに移行」\n"
    "  → 「ユーザーはBプロジェクト中で、Bツールを使用している」（Bツール部分は保持、A部分のみ更新）\n\n"
    "【統合の厳格なルール】\n"
    "- 統合元のうち**まだ有効な情報**は全て保持すること。"
    "新しい事実で明示的に無効化された古い情報は捨ててよいが、"
    "それ以外の意図的な省略・歪曲・過度な要約は禁止。\n"
    "- 推測で情報を追加してはならない。元の事実に明記されていない内容は書かない。\n"
    "- 単独で意味を持つ重要な事実は無理に統合しない（矛盾・更新がある場合は別）：\n"
    "  - 個人情報（出身、学歴、資格、職業、年齢、家族）\n"
    "  - 約束・合意事項\n"
    "  - 人生の節目・重要な出来事・トラウマ\n"
    "  - 価値観・信念\n"
    "- 統合すると元の情報の特異性が失われる場合は統合しない。\n"
    "- 各統合グループには**少なくとも2個**のインデックスを含めること（1個だけは統合ではない）。\n"
    "- 同じインデックスを複数のグループに含めてはならない。\n"
    "- 統合候補が無ければ空の配列 `[]` を出力する。\n\n"
    "**出力形式（厳守）**：\n"
    "[\n"
    '  {"merge": [元のインデックス1, インデックス2, ...], "into": "統合後の事実文"},\n'
    '  {"merge": [...], "into": "..."},\n'
    "  ...\n"
    "]\n\n"
    "例：\n"
    "[\n"
    '  {"merge": [3, 7, 12], "into": "ユーザーは『ひぐらしのなく頃に』『サマータイムレンダ』『xxx』を視聴した"},\n'
    '  {"merge": [5, 9], "into": "ユーザーは物理学部出身で物理学を専攻していた"}\n'
    "]\n\n"
    "繰り返す：JSON配列のみ。前置き・コメント・Markdownは禁止。"
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
    "このような複合事実は削除候補から除外し、後から /facts-consolidate で"
    "外科的に部分更新するのが正しい処理。\n"
    "削除してよいのは**事実全体が陳腐化・無効化されている**ケースのみ。\n\n"
    "**出力形式（厳守）**：\n"
    "削除するインデックス（数字）のみをJSON配列で出力する: [3, 7, 12]\n"
    "繰り返す：JSON配列のみ。他のテキスト・記号は一切含めない。"
)


class PersistentMemoryManager:
    """Manages facts.json and per-session diaries for one character (conf_uid)."""

    # Process-wide set tracking which conf_uids have a backfill currently
    # running. Prevents concurrent connections from kicking off duplicate
    # backfills against the same character.
    _backfill_in_progress: ClassVar[Set[str]] = set()

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
    ) -> None:
        self._conf_uid = conf_uid
        self._max_facts = max_facts
        self._diary_count = diary_count
        self._recent_sessions = recent_sessions
        self._base_dir = os.path.join("chat_history", conf_uid)
        self._facts_path = os.path.join(self._base_dir, "facts.json")
        self._diaries_dir = os.path.join(self._base_dir, "diaries")
        # Optional dedicated LLM for memory tasks (diary/fact/consolidate). When
        # set, _call_llm uses it instead of the chat model — keeps big uncached
        # one-shot memory calls off the (pricier) chat model. set via setter.
        self._memory_llm: Any = None

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
                    model=getattr(diary_rag_config, "embedding_model", "text-embedding-3-small"),
                )
                logger.info("[memory] Diary RAG enabled.")
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
                api_key=embed_api_key, base_url=embed_base_url, model=model,
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
                    model=getattr(diary_rag_config, "embedding_model", "text-embedding-3-small"),
                )
                logger.info("[memory] Facts RAG enabled.")
                # One-time: tag existing facts so they can be re-tiered by hand.
                self._migrate_facts_importance()
                if getattr(facts_rag_config, "rerank_enabled", False):
                    from .reranker import MemoryReranker

                    fmodel = getattr(facts_rag_config, "rerank_model", "gpt-4o-mini")
                    self._facts_reranker = MemoryReranker(
                        api_key=embed_api_key, base_url=embed_base_url, model=fmodel,
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
        os.makedirs(self._diaries_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_memory_llm(self, llm: Any) -> None:
        """Route memory tasks (diary/fact/consolidate) through this LLM instead
        of the chat model. Pass None to fall back to the caller-supplied LLM."""
        self._memory_llm = llm

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
        """
        facts = self._header_facts()
        if not facts:
            return ""
        lines = []
        for f in facts:
            updated = str(f.get("updated", ""))
            date = updated[:10] if len(updated) >= 10 else "不明"
            lines.append(f"- [{date}] {f['fact']}")
        body = "\n".join(lines)
        header = (
            "## ユーザーに関する長期記憶（事実）\n"
            "各事実の冒頭の `[YYYY-MM-DD]` は、その事実が**このリストに記録された日**で"
            "あり、出来事が実際に起きた日ではない。"
            "事実抽出は次のセッション開始時にまとめて行われるため、"
            "実際の出来事はその数時間〜数日前に起きている可能性がある点に注意。"
        )
        return f"{header}\n\n{body}"

    def get_diaries_prompt(self) -> str:
        """Return the diary block for the system prompt (empty string if no diaries)."""
        diaries = self._load_recent_diaries()
        if not diaries:
            return ""
        entries = "\n\n".join(f"[{d['date']}]\n{d['content']}" for d in diaries)
        return (
            "## 過去セッションの日記\n"
            "後続の会話履歴より前に行われたセッションの要約。"
            "各エントリ冒頭の日付がそのセッションの実時間。\n"
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
        return {
            d.get("history_uid", "") for d in self._load_recent_diaries()
        } - {""}

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

    def _header_facts(self) -> List[Dict[str, Any]]:
        """Facts that belong in the system-prompt header.

        With facts RAG active, only ``user``/``llm``-tier facts; the rest
        (``low``, the default) are recalled on demand. With RAG off, all facts
        (preserves the original always-inject-everything behaviour).
        """
        facts = self._load_facts()
        if not self.facts_rag_active:
            return facts
        return [f for f in facts if (f.get("importance") or "low") in ("user", "llm")]

    def injected_fact_ids(self) -> Set[str]:
        """Fingerprints of the facts already in the header (user/llm tier).

        The agent unions these into the fact-retrieval exclude set so RAG never
        surfaces a fact the model already has verbatim in its prompt.
        """
        return {self._fact_id(f["fact"]) for f in self._header_facts() if f.get("fact")}

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
        """One-time: tag facts lacking ``importance`` with the default ``low``.

        Lets the user re-tier by hand before relying on RAG. Backs up to a
        uniquely-named file first (distinct from the rolling ``facts.json.bak``
        that ``_save_facts`` overwrites each save). Idempotent — a no-op once
        every fact carries an ``importance``.
        """
        facts = self._load_facts()
        if not facts or all("importance" in f for f in facts):
            return
        backup = self._facts_path + ".pre-importance.bak"
        try:
            if os.path.exists(self._facts_path) and not os.path.exists(backup):
                shutil.copy2(self._facts_path, backup)
                logger.info(f"[memory] Backed up facts.json → {backup} before importance migration.")
        except Exception as e:
            logger.warning(f"[memory] facts importance backup failed: {e}")
        for f in facts:
            f.setdefault("importance", "low")
        self._save_facts(facts)
        logger.info(
            f"[memory] Tagged {len(facts)} fact(s) with default importance=low "
            "(RAG-gated until you re-tier them)."
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
            out = [
                {
                    "id": h["id"],
                    "fact": by_id[h["id"]]["fact"],
                    "date": str(by_id[h["id"]].get("updated", ""))[:10],
                    "score": h["score"],
                    "reason": "",
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
                "fact": j["content"],
                "date": j.get("date", "") or str(by_id.get(j["id"], {}).get("updated", ""))[:10],
                "score": 0.0,
                "reason": j.get("reason", ""),
            }
            for j in judged[:max_n]
        ]
        return out, candidates, keywords

    async def retrieve_diary_context(
        self, query: str, exclude_uids: Set[str], context: str = ""
    ) -> List[Dict[str, Any]]:
        """Return diaries relevant to ``query`` (long-tail recall).

        Pipeline: denoise the query to content keywords (drop framing words) →
        hybrid candidate generation (keywords drive embedding + lexical, grouped
        back to whole diaries) → optional LLM relevance judge over the shortlist
        → recall the full diaries. Reads content fresh from disk so it's never
        stale.

        Returns ``(hits, candidates)`` where hits is
        ``[{"uid", "date", "content", "score", "reason"}, ...]`` and candidates is
        the scored shortlist ``(uid, date, hybrid, vec, lex)`` for log tuning.
        Returns ``([], [])`` when RAG is off / query empty / embedding fails.
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

        # No reranker → original score-based threshold + topN selection.
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
            out, cands = self._resolve_diaries(hits, candidates)
            return out, cands, keywords

        # Reranker path: pull a generous shortlist (no strict gate — the judge is
        # the real relevance filter), then let the LLM pick/order the relevant.
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

        shortlist: List[Dict[str, Any]] = []
        for uid, date, _h, _v, _lx in candidates:
            entry = self._read_diary(uid)
            if entry and entry.get("content"):
                shortlist.append(
                    {"id": uid, "date": entry.get("date", date), "content": entry["content"]}
                )
        judged = await self._diary_reranker.rerank(query, shortlist, context=context)
        if judged is None:  # judge errored → fall back to top-N by hybrid
            judged = [dict(s, reason="(rerank-fallback)") for s in shortlist[:max_n]]

        out = [
            {
                "uid": j["id"],
                "date": j["date"],
                "content": j["content"],
                "score": 0.0,
                "reason": j.get("reason", ""),
            }
            for j in judged[:max_n]
        ]
        return out, candidates, keywords

    def _resolve_diaries(self, hits: List[Dict[str, Any]], candidates) -> tuple:
        """Map VectorIndex hits to full diaries read fresh from disk."""
        out: List[Dict[str, Any]] = []
        for h in hits:
            entry = self._read_diary(h["id"])
            if entry and entry.get("content"):
                out.append(
                    {
                        "uid": h["id"],
                        "date": entry.get("date", h["meta"].get("date", "")),
                        "content": entry["content"],
                        "score": h["score"],
                        "reason": "",
                    }
                )
        return out, candidates

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
        """Split a diary into sentence-level chunk items for the vector index.

        Each chunk id is ``<diary_uid>#<n>``; ``meta.parent`` points back at the
        diary so retrieval can group chunks and recall the whole diary.
        """
        return [
            {"id": f"{uid}#{i}", "text": sent, "meta": {"parent": uid, "date": date}}
            for i, sent in enumerate(_split_sentences(content))
        ]

    def _all_diary_chunks_for_index(self) -> List[Dict[str, Any]]:
        """Every diary's chunks as ``{id, text, meta}`` for ensure_indexed."""
        items: List[Dict[str, Any]] = []
        if not os.path.isdir(self._diaries_dir):
            return items
        for fname in os.listdir(self._diaries_dir):
            if not fname.endswith(".json"):
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
        key = (getattr(diary_rag_config, "openai_api_key", "") or "") if diary_rag_config else ""
        base = (getattr(diary_rag_config, "base_url", "") or "") if diary_rag_config else ""
        if not key and agent_config is not None:
            openai_cfg = getattr(getattr(agent_config, "llm_configs", None), "openai_llm", None)
            if openai_cfg is not None:
                key = getattr(openai_cfg, "llm_api_key", "") or ""
                base = base or (getattr(openai_cfg, "base_url", "") or "")
        if key == "default_api_key":
            key = ""
        return key, base

    async def extract_facts_async(
        self,
        recent_messages: List[Dict[str, Any]],
        llm: Any,
        diary_context: str = "",
        persona: str = "",
    ) -> None:
        """Extract new facts from recent messages and append to facts.json.

        Runs as a fire-and-forget background task. ``diary_context`` is an
        optional summary of older sessions (used during backfill) so the LLM
        has context beyond the sliding window without burning tokens on full
        message history. ``persona`` is the character's system prompt; when
        provided it is prepended so fact selection and pruning reflect what
        the character would consider memorable.
        """
        try:
            existing = self._load_facts()
            # Show each existing fact's current importance so the LLM tags new
            # facts consistently with the established tiering (it must still
            # never output "user" — see _FACT_EXTRACT_SYSTEM).
            existing_text = (
                "\n".join(f"- [{f.get('importance', 'low')}] {f['fact']}" for f in existing)
                if existing
                else "(まだありません)"
            )
            conv_text = self._format_messages(recent_messages)
            if not conv_text.strip() and not diary_context.strip():
                logger.info(f"[memory] Fact extraction skipped: empty input ({len(recent_messages)} raw msgs)")
                return

            prompt_parts = [f"既存の事実リスト（繰り返さないこと）:\n{existing_text}"]
            if diary_context.strip():
                prompt_parts.append(f"以前のセッションのまとめ（参考）:\n{diary_context}")
            if conv_text.strip():
                prompt_parts.append(f"分析する会話:\n{conv_text}")
            prompt = "\n\n".join(prompt_parts)
            logger.info(
                f"[memory] Extracting facts from {len(recent_messages)} messages "
                f"({len(conv_text)} chars conversation, {len(diary_context)} chars diary context)"
            )
            logger.debug(f"[memory] Fact extraction conversation preview: {conv_text[:400]!r}")
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
            # The LLM tags each fact "llm" (keep in header) or "low" (RAG-recalled).
            # "user" is manual-only: if the LLM assigns it, demote to "llm" rather
            # than rework — but log it so the slip is visible. Missing/invalid → low.
            tagged: List[Dict[str, Any]] = []
            for f in new_facts:
                if "fact" not in f:
                    continue
                imp = f.get("importance")
                if imp == "user":
                    logger.warning(
                        f"[memory] Extraction LLM tagged a fact 'user' (manual-only); "
                        f"demoted to 'llm': {f['fact']!r}"
                    )
                    imp = "llm"
                elif imp not in ("llm", "low"):
                    imp = "low"
                tagged.append({"fact": f["fact"], "updated": now, "importance": imp})
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
            existing_dropped = [
                e for e in existing if e["fact"] not in final_text
            ]
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
            if not history_messages:
                return
            conv_text = self._format_messages(history_messages)
            if not conv_text.strip():
                return

            # Build time-range header so LLM uses specific times instead of "今日".
            session_date = self._session_date_from_uid(history_uid)
            end_hm = self._session_end_hm_from_messages(history_messages)
            start_hm = session_date[11:16] if len(session_date) > 10 else ""
            time_range = f"{start_hm}〜{end_hm}" if start_hm and end_hm else start_hm or end_hm
            nth = self._count_same_day_diaries(session_date[:10], exclude_uid=history_uid) + 1
            header_parts = []
            if time_range:
                header_parts.append(f"セッション時間: {time_range}")
            if nth > 1:
                header_parts.append(f"この日の{nth}回目の会話セッション")
            if header_parts:
                conv_text = "[セッション情報]\n" + "\n".join(header_parts) + "\n\n" + conv_text

            content = await self._call_llm(
                llm, self._with_persona(_DIARY_SYSTEM, persona), conv_text
            )
            content = content.strip()
            if not content:
                return

            # Use the session's start time (encoded in history_uid) as the date,
            # so backfilled diaries sort correctly with newly-created ones.
            session_date = self._session_date_from_uid(history_uid)
            diary_entry = {
                "date": session_date,
                "history_uid": history_uid,
                "content": content,
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
        await asyncio.gather(
            self.create_diary_async(history_messages, history_uid, llm, persona=persona),
            self.extract_facts_async(history_messages, llm, persona=persona),
            return_exceptions=True,
        )
        # Mark diary so backfill knows this session's facts were already extracted.
        self._mark_diary_facts_extracted(history_uid)

    async def backfill_async(self, conf_uid: str, llm: Any, persona: str = "") -> None:
        """Generate diaries and facts for sessions that don't have them yet.

        Diary backfill: creates a diary for each session that has messages but
        no diary file yet.
        Fact backfill: processes any diary that doesn't carry a
        ``"facts_extracted": true`` marker, using the sliding-window approach
        (recent N sessions in full + older diary summaries) so the prompt stays
        bounded regardless of how many sessions exist.
        Both passes are idempotent. Guarded by a process-wide lock so concurrent
        connections don't kick off duplicate backfills for the same character.
        """
        if conf_uid in PersistentMemoryManager._backfill_in_progress:
            return
        PersistentMemoryManager._backfill_in_progress.add(conf_uid)
        try:
            from ..chat_history_manager import get_history_list, get_history

            # Promote any pending consolidation result BEFORE the rest of
            # backfill, so subsequent extraction/pruning operates on the
            # consolidated baseline rather than the pre-consolidation one.
            self._promote_staged_facts_if_present()

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
                        await self.create_diary_async(messages, uid, llm, persona=persona)
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
                    msgs = get_history(conf_uid, uid)
                    if msgs:
                        recent_messages.extend(msgs)

                older_parts: List[str] = []
                for uid in unprocessed_uids:
                    if uid in recent_uids:
                        continue
                    path = os.path.join(self._diaries_dir, f"{uid}.json")
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        if "content" in d:
                            older_parts.append(f"[{d.get('date', uid)}]\n{d['content']}")
                    except Exception:
                        continue
                diary_context = "\n\n".join(older_parts)

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
                await self._diary_index.ensure_indexed(self._all_diary_chunks_for_index())
            # Same for facts: embed new/edited facts, prune vectors of facts that
            # were consolidated or pruned away (id = content fingerprint, so an
            # edited fact is a new id + an orphaned old one).
            if self._facts_index is not None:
                await self._facts_index.ensure_indexed(self._facts_items_for_index())
        except Exception as e:
            logger.warning(f"[memory] Backfill failed: {e}", exc_info=True)
        finally:
            PersistentMemoryManager._backfill_in_progress.discard(conf_uid)

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
                if not fname.endswith(".json"):
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
        # Route every memory task (fact extraction, diary, prune, consolidate)
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
        #     per memory call when those tools are enabled for chat. Both
        #     kwargs fall back gracefully on LLM impls that don't accept
        #     them (TypeError → retry with positional-only args).
        try:
            stream = llm.chat_completion(
                messages, system, max_tokens=max_tokens, disable_server_tools=True
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

    @property
    def _staged_facts_path(self) -> str:
        """Path to the pending consolidated facts file."""
        return os.path.join(self._base_dir, "facts.consolidated.json")

    async def consolidate_facts_to_staged(self, llm: Any) -> Dict[str, Any]:
        """Run LLM-based fact consolidation and write the result to a staged
        file alongside facts.json.

        The current facts.json is **not modified** — the active session
        keeps using it. On the next OLV startup, backfill_async detects the
        staged file and promotes it to facts.json before running the
        normal extraction/pruning passes.

        Returns a dict with stats and per-merge breakdown for the caller
        (Discord bot) to surface to the user.
        """
        facts = self._load_facts()
        result: Dict[str, Any] = {
            "ok": False,
            "before": len(facts),
            "after": len(facts),
            "merges": [],
            "message": "",
        }
        if len(facts) < 2:
            result["message"] = "Need at least 2 facts to consolidate."
            return result

        numbered = "\n".join(
            f"{i} [{str(f.get('updated', ''))[:10] or '不明'}]: {f['fact']}"
            for i, f in enumerate(facts)
        )
        prompt = (
            f"現在 {len(facts)} 個の事実がある。\n\n"
            f"事実リスト（形式: インデックス [記録日]: 内容）:\n{numbered}\n\n"
            "統合できる項目があれば指定形式で出力。なければ `[]`。"
        )

        try:
            raw = await self._call_llm(llm, _CONSOLIDATE_SYSTEM, prompt)
            logger.info(f"[memory] Fact-consolidation LLM raw output:\n{raw}")
            if not raw.strip():
                result["message"] = "Consolidation LLM returned empty output."
                logger.warning(f"[memory] {result['message']}")
                return result
            proposals = self._parse_json_list(raw)
        except Exception as e:
            logger.warning(f"[memory] Consolidation LLM call failed: {e}", exc_info=True)
            result["message"] = f"LLM call failed: {e}"
            return result

        # Validate proposals: each merge needs ≥2 valid indices, into must
        # be a non-empty string, no index reused across groups.
        used_indices: set = set()
        valid: List[Dict[str, Any]] = []
        for m in proposals:
            into = m.get("into", "")
            if not isinstance(into, str) or not into.strip():
                continue
            raw_indices = m.get("merge", [])
            if not isinstance(raw_indices, list):
                continue
            indices = [
                i for i in raw_indices
                if isinstance(i, int) and 0 <= i < len(facts) and i not in used_indices
            ]
            if len(indices) < 2:
                continue
            used_indices.update(indices)
            valid.append({"merge": indices, "into": into.strip()})

        if not valid:
            logger.info("[memory] No valid consolidations proposed.")
            result["ok"] = True
            result["message"] = "No consolidation opportunities found."
            return result

        # Build the new fact list: keep unmerged entries; append one entry
        # per valid merge with `updated` set to the newest date among the
        # source facts (no new fact was created, just reorganised).
        merged_text_set = used_indices
        survivors = [
            f for i, f in enumerate(facts) if i not in merged_text_set
        ]
        _tier_rank = {"user": 2, "llm": 1, "low": 0}
        new_merged: List[Dict[str, Any]] = []
        for m in valid:
            source_dates = [
                str(facts[i].get("updated", "")) for i in m["merge"]
            ]
            source_dates = [d for d in source_dates if d]
            newest = max(source_dates) if source_dates else (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            # Keep the highest tier among the merged sources so consolidation
            # never silently demotes a hand-set `user` (or `llm`) fact.
            best_tier = max(
                (facts[i].get("importance", "low") for i in m["merge"]),
                key=lambda t: _tier_rank.get(t, 0),
            )
            new_merged.append(
                {"fact": m["into"], "updated": newest, "importance": best_tier}
            )

        new_facts = survivors + new_merged

        # Write to staged file (NOT facts.json). Backfill will promote it
        # atomically at the next OLV startup. Same chronological sort as
        # the main _save_facts path so the staged + promoted file is
        # immediately ordered.
        try:
            os.makedirs(self._base_dir, exist_ok=True)
            ordered = self._sort_facts(new_facts)
            with open(self._staged_facts_path, "w", encoding="utf-8") as f:
                json.dump(ordered, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[memory] Failed to write staged facts file: {e}")
            result["message"] = f"Failed to write staged file: {e}"
            return result

        # Build per-merge log + return payload.
        log_lines = [
            f"[memory] Consolidation staged → {self._staged_facts_path} "
            f"({len(facts)} → {len(new_facts)}, {len(valid)} merge group(s))"
        ]
        merges_summary: List[Dict[str, Any]] = []
        for m in valid:
            sources = [
                {
                    "date": str(facts[i].get("updated", ""))[:10] or "不明",
                    "fact": facts[i]["fact"],
                }
                for i in m["merge"]
            ]
            merges_summary.append({"into": m["into"], "sources": sources})
            log_lines.append(f"  Merged {len(m['merge'])} fact(s) into:")
            log_lines.append(f"    + {m['into']}")
            for s in sources:
                log_lines.append(f"    ← [{s['date']}] {s['fact']}")
        logger.info("\n".join(log_lines))

        result["ok"] = True
        result["after"] = len(new_facts)
        result["merges"] = merges_summary
        result["message"] = (
            f"Consolidated {len(facts)} → {len(new_facts)} fact(s) in "
            f"{len(valid)} merge group(s). Will take effect on next OLV restart."
        )
        return result

    def _promote_staged_facts_if_present(self) -> bool:
        """If a staged consolidated facts file exists, atomically replace
        facts.json with it (after backing up the current file).

        Called at the very start of backfill_async so the consolidated file
        becomes the base for any subsequent extraction/pruning in the same
        backfill run. Returns True if a promotion happened.

        The pre-consolidation snapshot is saved with a timestamped name so
        each manual consolidation's "before" state is preserved permanently
        (consolidations are manual + rare, so the backup files won't
        accumulate excessively — and the user explicitly wants the ability
        to review or roll back later).
        """
        staged = self._staged_facts_path
        if not os.path.exists(staged):
            return False
        try:
            if os.path.exists(self._facts_path):
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                bak = f"{self._facts_path}.bak.pre-consolidation-{ts}"
                try:
                    shutil.copy2(self._facts_path, bak)
                    logger.info(f"[memory] Pre-consolidation backup → {bak}")
                except Exception as e:
                    logger.warning(
                        f"[memory] Could not back up before consolidation: {e}"
                    )
            os.replace(staged, self._facts_path)
            logger.info(
                f"[memory] Promoted staged consolidated facts → {self._facts_path}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"[memory] Failed to promote staged consolidated facts: {e}"
            )
            return False

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
        candidate = text[m.start():]
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
