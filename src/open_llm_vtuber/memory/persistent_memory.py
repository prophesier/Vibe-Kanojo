"""Three-layer persistent memory for Open-LLM-VTuber.

Layer 1 – sliding window: loaded by BasicMemoryAgent at session start.
Layer 2 – structured facts: key assertions about the user, stored in facts.json.
Layer 3 – session diaries: per-session mood summaries, stored in diaries/.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Set
from loguru import logger


_FACT_EXTRACT_SYSTEM = (
    "あなたは記憶アシスタントです。会話からユーザーに関する重要で持続的な事実を抽出してください。"
    "注目すべき点：個人情報、好み、人間関係、進行中の状況、約束事。"
    "一時的な雑談はスキップしてください。\n"
    "既存の事実リストが提供される場合、それらを繰り返さないでください。新しい情報のみ抽出してください。\n"
    "出力はJSONの配列のみ: "
    '[{"fact": "ユーザーはミミという猫を飼っている"}, {"fact": "ユーザーはソフトウェアエンジニアである"}]\n'
    "保存する価値のある新しい事実がない場合は、空の配列を出力してください: []"
)

_DIARY_SYSTEM = (
    "あなたは記憶アシスタントです。AIキャラクターの一人称視点から、"
    "この会話セッションを簡潔な日記として2〜4文でまとめてください。"
    "含めるべき内容：主なトピック、ユーザーの感情状態、約束や合意事項、全体的な雰囲気。"
    "自然な文体で書いてください。[neutral]などの表現タグは含めないでください。"
    "日記の本文のみを出力し、他は何も出力しないでください。"
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
        os.makedirs(self._diaries_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_sessions(self, uids) -> None:
        """Mark these session UIDs as already loaded in the sliding window."""
        self._active_session_uids = set(uids or [])

    def get_memory_prompt(self) -> str:
        """Return the memory block to prepend to the system prompt."""
        parts: List[str] = []

        facts = self._load_facts()
        if facts:
            lines = "\n".join(f"- {f['fact']}" for f in facts)
            parts.append(f"## Long-term memory: facts about the user\n{lines}")

        diaries = self._load_recent_diaries()
        if diaries:
            entries = "\n\n".join(
                f"[{d['date']}]\n{d['content']}" for d in diaries
            )
            parts.append(f"## Recent session memories\n{entries}")

        return "\n\n".join(parts)

    async def extract_facts_async(
        self,
        recent_messages: List[Dict[str, Any]],
        llm: Any,
        diary_context: str = "",
    ) -> None:
        """Extract new facts from recent messages and append to facts.json.

        Runs as a fire-and-forget background task. ``diary_context`` is an
        optional summary of older sessions (used during backfill) so the LLM
        has context beyond the sliding window without burning tokens on full
        message history.
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
            raw = await self._call_llm(llm, _FACT_EXTRACT_SYSTEM, prompt)
            logger.info(f"[memory] Fact-extraction LLM raw output: {raw[:500]!r}")
            new_facts = self._parse_json_list(raw)
            if not new_facts:
                logger.info("[memory] No new facts extracted.")
                return

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tagged = [{"fact": f["fact"], "updated": now} for f in new_facts if "fact" in f]
            merged = existing + tagged
            # Trim to max_facts keeping newest
            if len(merged) > self._max_facts:
                merged = merged[-self._max_facts :]
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
    ) -> None:
        """Generate and save a diary entry for the finished session."""
        try:
            if not history_messages:
                return
            conv_text = self._format_messages(history_messages)
            if not conv_text.strip():
                return

            content = await self._call_llm(llm, _DIARY_SYSTEM, conv_text)
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
    ) -> None:
        """Run diary generation and fact extraction concurrently at session end.

        Both tasks receive the full session history so neither is starved of
        context. Runs as a fire-and-forget background task.
        """
        await asyncio.gather(
            self.create_diary_async(history_messages, history_uid, llm),
            self.extract_facts_async(history_messages, llm),
            return_exceptions=True,
        )

    async def backfill_async(self, conf_uid: str, llm: Any) -> None:
        """Generate diaries and facts for sessions that don't have them yet.

        Diary backfill: scans chat_history/{conf_uid}/diaries/ and creates a
        diary for each session that has messages but no diary file.
        Fact backfill: if facts.json doesn't exist yet, runs a one-time
        extraction across all historical sessions.
        Both are idempotent — running again is a no-op once everything exists.
        Guarded by a process-wide lock so concurrent connections don't double up.
        """
        if conf_uid in PersistentMemoryManager._backfill_in_progress:
            return
        PersistentMemoryManager._backfill_in_progress.add(conf_uid)
        try:
            from ..chat_history_manager import get_history_list, get_history

            history_list = get_history_list(conf_uid)

            # --- Diary backfill ---
            missing_diaries = []
            for entry in history_list:
                uid = entry["uid"]
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
                        await self.create_diary_async(messages, uid, llm)
                logger.info("[memory] Diary backfill complete.")

            # --- Fact backfill (one-time, only when facts.json doesn't exist) ---
            # Uses the sliding-window range of recent sessions in full, plus
            # diary summaries for older sessions. This keeps the prompt
            # bounded by config (recent_sessions + diary_count) rather than
            # ballooning if facts.json gets deleted with lots of history.
            if not os.path.exists(self._facts_path):
                # history_list is newest-first; take the N most recent sessions
                recent_entries = history_list[: self._recent_sessions]
                recent_uids = {e["uid"] for e in recent_entries}
                recent_messages: List[Dict[str, Any]] = []
                # Reverse so messages flow oldest→newest within the window
                for entry in reversed(recent_entries):
                    msgs = get_history(conf_uid, entry["uid"])
                    if msgs:
                        recent_messages.extend(msgs)

                # Pull diary summaries for older sessions (excluding the ones
                # already covered by the full-message window above).
                older_diary_entries = []
                if os.path.isdir(self._diaries_dir):
                    for fname in os.listdir(self._diaries_dir):
                        if not fname.endswith(".json"):
                            continue
                        diary_uid = fname[:-5]
                        if diary_uid in recent_uids:
                            continue
                        try:
                            with open(
                                os.path.join(self._diaries_dir, fname),
                                "r",
                                encoding="utf-8",
                            ) as f:
                                entry = json.load(f)
                            if isinstance(entry, dict) and "content" in entry:
                                entry.setdefault("history_uid", diary_uid)
                                older_diary_entries.append(entry)
                        except Exception:
                            continue
                older_diary_entries.sort(key=lambda e: e.get("history_uid", ""))

                diary_context = "\n\n".join(
                    f"[{d.get('date', '')}]\n{d['content']}"
                    for d in older_diary_entries
                )

                if recent_messages or diary_context:
                    logger.info(
                        f"[memory] facts.json not found — running one-time fact "
                        f"extraction (recent sessions: {len(recent_entries)}, "
                        f"older diaries: {len(older_diary_entries)})…"
                    )
                    await self.extract_facts_async(
                        recent_messages, llm, diary_context=diary_context
                    )
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
        with open(self._facts_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, ensure_ascii=False, indent=2)

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
    def _format_messages(messages: List[Dict[str, Any]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
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
