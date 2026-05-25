"""Three-layer persistent memory for Open-LLM-VTuber.

Layer 1 – sliding window: loaded by BasicMemoryAgent at session start.
Layer 2 – structured facts: key assertions about the user, stored in facts.json.
Layer 3 – session diaries: per-session mood summaries, stored in diaries/.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Set
from loguru import logger

# Matches timestamp tags injected by _to_text_prompt: "[YYYY-MM-DD HH:MM:SS Weekday]"
_TIMESTAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+\]\s*", re.MULTILINE)


_FACT_EXTRACT_SYSTEM = (
    "あなたは記憶アシスタントです。会話からユーザーに関する持続的な事実を抽出してください。\n"
    "抽出すべき情報（これに限らない）：\n"
    "- 個人情報：出身地、学歴（学部・専攻など）、職業、年齢層\n"
    "- 好み・趣味・習慣\n"
    "- 人間関係\n"
    "- 進行中のプロジェクト、使用ツール・技術\n"
    "- 約束・合意事項\n"
    "- ユーザーの目標・課題・悩み\n\n"
    "重要なガイドライン：\n"
    "- 会話の大半が技術的な内容でも、その中に1回だけ出てきたユーザー自身の情報も必ず抽出する\n"
    "- 中国語・日本語・英語が混在していても、すべての言語の発言を対象にする\n"
    "- 判断に迷うなら抽出する（省略するより多めに拾う方が良い）\n"
    "- 真に一時的・文脈依存で今後役に立たない情報だけをスキップする\n\n"
    "既存の事実リストが提供される場合、それらを繰り返さないこと。新しい情報のみ抽出してください。\n"
    "出力はJSONの配列のみ（日本語で記述）: "
    '[{"fact": "ユーザーは物理学部出身"}, {"fact": "ユーザーはWindowsを使用している"}]\n'
    "本当に新しい事実が1件もない場合のみ、空の配列を出力してください: []"
)

_DIARY_SYSTEM = (
    "あなたは記憶アシスタントです。AIキャラクターの一人称視点から、"
    "この会話セッションを簡潔な日記として2〜4文でまとめてください。"
    "含めるべき内容：主なトピック、ユーザーの感情状態、約束や合意事項、全体的な雰囲気。"
    "セッション情報に開始・終了時刻が含まれる場合、「今日」「本日」という曖昧な表現を避け、"
    "「〇〇時頃」「〇〇時から〇〇時の会話で」のように具体的な時刻を使って書いてください。"
    "人格設定が提供されている場合、その口調・性格・思考パターンを反映した文体で書いてください。"
    "自然な文体で書いてください。[neutral]などの表現タグは含めないでください。"
    "日記の本文のみを出力し、他は何も出力しないでください。"
)

_FACT_PRUNE_SYSTEM = (
    "あなたは記憶アシスタントです。ユーザーに関する事実リストが保存上限を超えました。"
    "AIキャラクターの視点から、最も価値の低い項目を選んで削除する必要があります。\n\n"
    "各事実には更新日時が付いています。以下の優先順位で削除対象を選んでください：\n\n"
    "【優先的に削除】\n"
    "- 新しい事実によって上書き・無効化された古い情報\n"
    "  （例: 古い「プロジェクトA取り組み中」と新しい「プロジェクトBに移行」が両方ある場合、古い方）\n"
    "- 時間の経過により時効・陳腐化した情報（古い日時のその場限りのタスク・状況など）\n"
    "- 一時的・状況依存で今後参照する可能性が低い情報\n"
    "- 同じ内容の重複（古い方）\n"
    "- 人格設定の視点から、ユーザーとの関係に影響が薄い些細な情報\n\n"
    "【残すべき】\n"
    "- 出身、学歴、職業、人間関係など長期的に変わらない個人情報\n"
    "- 価値観・性格・趣味・習慣など\n"
    "- 新しい日時の情報（古い情報より優先）\n\n"
    "削除するインデックス（数字）のみをJSON配列で出力してください: [3, 7, 12]\n"
    "他のテキストは一切出力しないこと。"
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
    ) -> None:
        self._conf_uid = conf_uid
        self._max_facts = max_facts
        self._diary_count = diary_count
        self._recent_sessions = recent_sessions
        self._base_dir = os.path.join("chat_history", conf_uid)
        self._facts_path = os.path.join(self._base_dir, "facts.json")
        self._diaries_dir = os.path.join(self._base_dir, "diaries")
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
        """Return the facts block for the system prompt (empty string if no facts)."""
        facts = self._load_facts()
        if not facts:
            return ""
        lines = "\n".join(f"- {f['fact']}" for f in facts)
        return f"## Long-term memory: facts about the user\n{lines}"

    def get_diaries_prompt(self) -> str:
        """Return the diary block for the system prompt (empty string if no diaries)."""
        diaries = self._load_recent_diaries()
        if not diaries:
            return ""
        entries = "\n\n".join(f"[{d['date']}]\n{d['content']}" for d in diaries)
        return f"## Recent session memories\n{entries}"

    def get_memory_prompt(self) -> str:
        """Return the combined memory block (facts + diaries) for non-Claude LLMs."""
        parts = [p for p in (self.get_facts_prompt(), self.get_diaries_prompt()) if p]
        return "\n\n".join(parts)

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
            existing_text = (
                "\n".join(f"- {f['fact']}" for f in existing) if existing else "(まだありません)"
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
            raw = await self._call_llm(
                llm, self._with_persona(_FACT_EXTRACT_SYSTEM, persona), prompt
            )
            logger.info(f"[memory] Fact-extraction LLM raw output: {raw[:500]!r}")
            new_facts = self._parse_json_list(raw)
            if not new_facts:
                logger.info("[memory] No new facts extracted.")
                return

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tagged = [{"fact": f["fact"], "updated": now} for f in new_facts if "fact" in f]
            merged = existing + tagged
            # Smart trim: ask the LLM (in-character) to drop least-important
            # entries when over the cap, instead of blindly dropping by age.
            if len(merged) > self._max_facts:
                merged = await self._prune_facts_with_llm(
                    merged, self._max_facts, llm, persona=persona
                )
            self._save_facts(merged)
            logger.info(
                f"[memory] Added {len(tagged)} new fact(s) → {self._facts_path} "
                f"(total: {len(merged)})"
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

            if not unprocessed_uids:
                return

            logger.info(
                f"[memory] {len(unprocessed_uids)} session(s) pending fact extraction."
            )

            # Use the most recent N unprocessed sessions in full; the rest as
            # diary summaries to keep token cost bounded.
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
                    recent_messages, llm, diary_context=diary_context, persona=persona
                )
                # Mark all processed diaries so this doesn't repeat next startup.
                for uid in unprocessed_uids:
                    self._mark_diary_facts_extracted(uid)
                logger.info("[memory] Fact backfill complete.")
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
        with open(self._facts_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)

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

    @staticmethod
    async def _call_llm(llm: Any, system: str, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        result = ""
        async for event in llm.chat_completion(messages, system):
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
            raw = await self._call_llm(
                llm, self._with_persona(_FACT_PRUNE_SYSTEM, persona), prompt
            )
            indices = sorted(
                {i for i in self._parse_int_list(raw) if 0 <= i < len(facts)}
            )
            if len(indices) != excess:
                logger.warning(
                    f"[memory] Fact-prune LLM returned {len(indices)} indices, "
                    f"expected {excess}; falling back to FIFO trimming."
                )
                return facts[-target_count:]
            dropped = [facts[i]["fact"] for i in indices]
            logger.info(f"[memory] LLM-pruned {excess} fact(s): {dropped}")
            return [f for i, f in enumerate(facts) if i not in set(indices)]
        except Exception as e:
            logger.warning(
                f"[memory] Fact pruning failed ({e}); falling back to FIFO trimming."
            )
            return facts[-target_count:]

    @staticmethod
    def _parse_json_list(text: str) -> List[Dict[str, Any]]:
        text = text.strip()
        # Find the first '[' and last ']'
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
