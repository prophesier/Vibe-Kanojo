"""Three-layer persistent memory for Open-LLM-VTuber.

Layer 1 – sliding window: loaded by BasicMemoryAgent at session start.
Layer 2 – structured facts: key assertions about the user, stored in facts.json.
Layer 3 – session diaries: per-session mood summaries, stored in diaries/.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List
from loguru import logger


_FACT_EXTRACT_SYSTEM = (
    "You are a memory assistant. Extract important, durable facts about the user "
    "from the conversation. Focus on: personal info, preferences, relationships, "
    "ongoing situations, commitments. Skip ephemeral chit-chat.\n"
    "Output ONLY a JSON array like: "
    '[{"fact": "User has a cat named Mimi"}, {"fact": "User is a software engineer"}]\n'
    "If there are no new facts worth saving, output an empty array: []"
)

_DIARY_SYSTEM = (
    "You are a memory assistant. Write a brief diary entry (2-4 sentences) "
    "summarising the conversation session from the AI character's first-person perspective. "
    "Capture: main topics, user's emotional state, anything that was agreed or promised, "
    "and the general vibe. Write naturally — do NOT include expression tags like [neutral]. "
    "Output only the diary text, nothing else."
)


class PersistentMemoryManager:
    """Manages facts.json and per-session diaries for one character (conf_uid)."""

    def __init__(self, conf_uid: str, *, max_facts: int = 50, diary_count: int = 5) -> None:
        self._conf_uid = conf_uid
        self._max_facts = max_facts
        self._diary_count = diary_count
        self._base_dir = os.path.join("chat_history", conf_uid)
        self._facts_path = os.path.join(self._base_dir, "facts.json")
        self._diaries_dir = os.path.join(self._base_dir, "diaries")
        os.makedirs(self._diaries_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
    ) -> None:
        """Extract new facts from recent messages and append to facts.json.

        Runs as a fire-and-forget background task.
        """
        try:
            existing = self._load_facts()
            existing_text = (
                "\n".join(f"- {f['fact']}" for f in existing) if existing else "(none yet)"
            )
            conv_text = self._format_messages(recent_messages)
            if not conv_text.strip():
                return

            prompt = (
                f"Existing facts (do NOT repeat these):\n{existing_text}\n\n"
                f"Conversation to analyse:\n{conv_text}"
            )
            raw = await self._call_llm(llm, _FACT_EXTRACT_SYSTEM, prompt)
            new_facts = self._parse_json_list(raw)
            if not new_facts:
                return

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            tagged = [{"fact": f["fact"], "updated": now} for f in new_facts if "fact" in f]
            merged = existing + tagged
            # Trim to max_facts keeping newest
            if len(merged) > self._max_facts:
                merged = merged[-self._max_facts :]
            self._save_facts(merged)
            logger.debug(f"[memory] Added {len(tagged)} new fact(s). Total: {len(merged)}")
        except Exception as e:
            logger.warning(f"[memory] Fact extraction failed: {e}")

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

            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            diary_entry = {
                "date": date_str,
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
                path = os.path.join(self._diaries_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                    if isinstance(entry, dict) and "date" in entry and "content" in entry:
                        entries.append(entry)
                except Exception:
                    continue
            # Sort by date string (ISO format sorts lexicographically)
            entries.sort(key=lambda e: e["date"])
            return entries[-self._diary_count :]
        except Exception:
            return []

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
                label = "User" if role == "user" else "AI"
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
