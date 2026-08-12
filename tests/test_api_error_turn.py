"""API-error turn handling (あさひ 08-05, post-hallucination incident).

A failed turn must leave the LLM context entirely — the surviving user input
made the context show two consecutive user turns, the exact shape that seeded
hallucinated inputs. Both sides stay on disk tagged ``context_excluded`` for
human review; the assembler skips tagged records.
"""

import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from src.open_llm_vtuber import chat_history_manager as chm
from src.open_llm_vtuber.agent.agents import basic_memory_agent as agent_module
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)


class _FakeClaudeLLM(ClaudeAsyncLLM):
    def __init__(self, events):
        self.model = "claude-opus-5"
        self._rounds = [list(events)]

    async def chat_completion(self, messages, system=None, tools=None):
        for event in self._rounds.pop(0):
            yield deepcopy(event)


def _bare_agent(llm) -> BasicMemoryAgent:
    agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
    agent._llm = llm
    agent._memory = [{"role": "user", "content": "[tag] 届かなかった入力"}]
    agent._memory_manager = None
    agent._build_system_for_llm = lambda: "system"
    agent._check_stateful_claims = lambda text: None
    return agent


class ApiErrorTurnTests(unittest.IsolatedAsyncioTestCase):
    async def _run_error_turn(self, events):
        agent = _bare_agent(_FakeClaudeLLM(events))
        chunks = []
        async for chunk in agent._claude_tool_interaction_loop(list(agent._memory), []):
            if isinstance(chunk, str):
                chunks.append(chunk)
        return agent, "".join(chunks)

    async def test_error_pops_input_and_yields_notice(self):
        agent, text = await self._run_error_turn(
            [{"type": "error", "message": "Claude API error: Overloaded"}]
        )
        self.assertIn("⚠️", text)
        self.assertIn("もう一度送ってほしい", text)
        self.assertEqual(agent._memory, [], "user input must leave the context")
        self.assertEqual(agent.pop_context_excluded(), "api_error")
        self.assertIsNone(agent.pop_context_excluded(), "pop must be one-shot")

    async def test_partial_reply_is_not_committed(self):
        agent, text = await self._run_error_turn(
            [
                {"type": "text_delta", "text": "言いかけた言葉"},
                {"type": "error", "message": "Overloaded"},
            ]
        )
        self.assertEqual(
            agent._memory, [], "neither input nor partial reply may remain"
        )

    async def test_error_notice_truncates_long_messages(self):
        agent, text = await self._run_error_turn(
            [{"type": "error", "message": "x" * 500}]
        )
        self.assertIn("…", text)
        self.assertLess(len(text), 400)

    async def test_normal_turn_leaves_no_excluded_tag(self):
        agent = _bare_agent(
            _FakeClaudeLLM(
                [
                    {"type": "text_delta", "text": "ok"},
                    {"type": "message_stop"},
                ]
            )
        )
        async for _ in agent._claude_tool_interaction_loop(list(agent._memory), []):
            pass
        self.assertIsNone(agent.pop_context_excluded())


class DiskTagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "conf").mkdir(parents=True, exist_ok=True)
        self.patcher = patch.object(
            chm,
            "_get_safe_history_path",
            lambda conf, uid: str(self.tmp / conf / f"{uid}.json"),
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, uid, records):
        (self.tmp / "conf" / f"{uid}.json").write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )

    def _read(self, uid):
        fp = self.tmp / "conf" / f"{uid}.json"
        return json.loads(fp.read_text(encoding="utf-8"))

    def test_store_message_persists_exclusion_tag(self):
        self._write("uid1", [])
        chm.store_message(
            "conf", "uid1", "ai", "⚠️ notice", context_excluded="api_error"
        )
        records = self._read("uid1")
        self.assertEqual(records[-1]["context_excluded"], "api_error")

    def test_normal_store_carries_no_tag(self):
        self._write("uid3", [])
        chm.store_message("conf", "uid3", "ai", "ordinary reply")
        self.assertNotIn("context_excluded", self._read("uid3")[-1])

    def test_mark_last_message_excluded_tags_matching_role_only(self):
        self._write(
            "uid2",
            [{"role": "human", "content": "input", "timestamp": "2026-08-05T16:38:45"}],
        )
        self.assertFalse(
            chm.mark_last_message_excluded("conf", "uid2", "ai", "api_error")
        )
        self.assertTrue(
            chm.mark_last_message_excluded("conf", "uid2", "human", "api_error")
        )
        records = self._read("uid2")
        self.assertEqual(records[-1]["context_excluded"], "api_error")


class AssemblySkipTests(unittest.TestCase):
    def test_msg_from_history_record_skips_tagged(self):
        agent = _bare_agent(llm=object())
        agent._format_timestamp = lambda ts: "[tag]"
        excluded = {
            "role": "human",
            "content": "届かなかった入力",
            "timestamp": "2026-08-05T16:38:45",
            "context_excluded": "api_error",
        }
        normal = {
            "role": "human",
            "content": "普通の入力",
            "timestamp": "2026-08-05T16:43:19",
        }
        self.assertIsNone(agent._msg_from_history_record(excluded))
        self.assertIsNotNone(agent._msg_from_history_record(normal))

    def test_window_load_drops_tagged_records(self):
        agent = _bare_agent(llm=_FakeClaudeLLM([]))
        past = [
            {"role": "human", "content": "A", "timestamp": "2026-08-05T16:38:45"},
            {
                "role": "ai",
                "content": "[Error from LLM: ...]",
                "timestamp": "2026-08-05T16:38:57",
                "context_excluded": "api_error",
            },
            {"role": "human", "content": "B", "timestamp": "2026-08-05T16:43:19"},
        ]
        with (
            patch.object(
                agent_module, "get_recent_histories", return_value=[("old", past)]
            ),
            patch.object(agent_module, "get_history", return_value=[]),
        ):
            agent.set_memory_from_recent_histories("conf", 3, current_uid="cur")

        contents = [m["content"] for m in agent._memory]
        self.assertEqual(len(agent._memory), 2)
        self.assertFalse(any("Error from LLM" in c for c in contents))


if __name__ == "__main__":
    unittest.main()
