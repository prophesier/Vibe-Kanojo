"""System-transcript restructure + runtime thinking toggle (あさひ 08-02).

Past-session history moves out of `messages` into a frozen system block
(persona | facts+diaries | past transcript | HISTORY_NOTE), so
thinking-parameter changes only invalidate the current-session message tail.
The /thinking Discord command toggles forced/adaptive at runtime without
touching conf.yaml.
"""

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.agents import basic_memory_agent as agent_module
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)
from src.open_llm_vtuber.discord_bot.bot import DiscordVTuberBot
from src.open_llm_vtuber.discord_bot.bridge import OLVBridge
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class _FakeClaudeLLM(ClaudeAsyncLLM):
    def __init__(self, model="claude-opus-4-6"):
        self.model = model
        self._thinking_force = False
        self._thinking_budget = 16000
        self._thinking_effort = "max"


def _bare_agent(llm) -> BasicMemoryAgent:
    agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
    agent._llm = llm
    agent._memory = []
    agent._memory_manager = None
    agent._past_history_cut = 0
    agent._past_transcript = ""
    return agent


class RenderPastTranscriptTests(unittest.TestCase):
    def test_renders_labelled_transcript_with_header(self):
        agent = _bare_agent(_FakeClaudeLLM())
        agent._memory = [
            {"role": "user", "content": "【セッション開始: X】\n[tag] こんにちは"},
            {"role": "assistant", "content": "やあ"},
            {"role": "user", "content": "[tag] 現在セッションの発言"},
        ]
        agent._past_history_cut = 2

        text = agent._render_past_transcript()

        self.assertIn("【過去セッションの転記】", text)
        self.assertIn("ユーザー: 【セッション開始: X】\n[tag] こんにちは", text)
        self.assertIn("アシスタント: やあ", text)
        self.assertNotIn("現在セッションの発言", text)

    def test_zero_cut_renders_empty(self):
        agent = _bare_agent(_FakeClaudeLLM())
        agent._memory = [{"role": "user", "content": "hi"}]
        self.assertEqual(agent._render_past_transcript(), "")


class BuildSystemTests(unittest.TestCase):
    def _agent(self, past_transcript=""):
        agent = _bare_agent(_FakeClaudeLLM())
        agent._system = "PERSONA"
        agent._steam_digest = ""
        agent._tool_capability_notes = lambda: []
        agent._memory_manager = SimpleNamespace(
            get_facts_prompt=lambda: "FACTS",
            get_diaries_prompt=lambda: "DIARIES",
        )
        agent._past_transcript = past_transcript
        return agent

    def test_layout_with_past_transcript(self):
        agent = self._agent(past_transcript="【過去セッションの転記】\n...")

        blocks = agent._build_system_for_llm()

        self.assertEqual(len(blocks), 4)
        self.assertIn("PERSONA", blocks[0]["text"])
        self.assertIn("cache_control", blocks[0])
        # facts + diaries merged into ONE cached block
        self.assertIn("FACTS", blocks[1]["text"])
        self.assertIn("DIARIES", blocks[1]["text"])
        self.assertIn("cache_control", blocks[1])
        self.assertIn("過去セッションの転記", blocks[2]["text"])
        self.assertIn("cache_control", blocks[2])
        # trailing HISTORY_NOTE stays uncached and positional
        self.assertIn("【以下の会話履歴について】", blocks[3]["text"])
        self.assertNotIn("cache_control", blocks[3])

    def test_layout_without_past_transcript(self):
        blocks = self._agent(past_transcript="")._build_system_for_llm()

        self.assertEqual(len(blocks), 3)
        self.assertIn("【以下の会話履歴について】", blocks[2]["text"])

    def test_steam_digest_lives_in_block2_not_block1(self):
        # The digest is session-scoped (playtime moves between sessions);
        # in block 1 it silently busted the persona cache on quick restarts
        # (あさひ 08-12).
        agent = self._agent(past_transcript="")
        agent._steam_digest = "STEAM-DIGEST"
        blocks = agent._build_system_for_llm()
        self.assertNotIn("STEAM-DIGEST", blocks[0]["text"])
        self.assertIn("STEAM-DIGEST", blocks[1]["text"])
        # After facts/diaries, sharing their breakpoint.
        self.assertIn("cache_control", blocks[1])
        self.assertTrue(blocks[1]["text"].startswith("FACTS"))

    def test_history_note_describes_new_structure(self):
        note = self._agent()._history_note()
        self.assertIn("現在進行中のセッション", note)
        self.assertIn("過去セッションの転記", note)


class ToMessagesSplitTests(unittest.TestCase):
    def _agent_with_window(self, llm):
        agent = _bare_agent(llm)
        agent._memory = [
            {"role": "user", "content": "past-1"},
            {"role": "assistant", "content": "past-2"},
            {"role": "user", "content": "current-1"},
        ]
        agent._past_history_cut = 2
        agent._to_text_prompt = lambda input_data: "new turn"
        agent._time_event_banner = lambda: ""
        agent._steam_pending_blocks = []
        agent._memory_pending_blocks = []
        agent._pending_rag_block = ""
        agent._pending_facts_block = ""
        # Persisting the fresh input to _memory is _add_message's job and not
        # under test here — the assertions only inspect the history prefix.
        agent._add_message = lambda *args, **kwargs: None
        return agent

    def test_claude_messages_start_at_current_session(self):
        agent = self._agent_with_window(_FakeClaudeLLM())
        messages = agent._to_messages(SimpleNamespace(images=None, metadata=None))
        # The cache breakpoint may rewrite an entry's content into block-array
        # form, so assert on the serialized payload rather than raw strings.
        payload = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("past-1", payload)
        self.assertNotIn("past-2", payload)
        self.assertIn("current-1", payload)

    def test_non_claude_keeps_full_window(self):
        agent = self._agent_with_window(llm=object())
        agent._openai_explicit_cache = lambda: False
        messages = agent._to_messages(SimpleNamespace(images=None, metadata=None))
        payload = json.dumps(messages, ensure_ascii=False)
        self.assertIn("past-1", payload)
        self.assertIn("current-1", payload)


class SetMemoryCutTests(unittest.TestCase):
    def test_cut_and_transcript_frozen_between_past_and_current(self):
        agent = _bare_agent(_FakeClaudeLLM())
        past = [
            {"role": "human", "content": "旧発言", "timestamp": "2026-08-01T10:00:00"},
            {"role": "ai", "content": "旧返事", "timestamp": "2026-08-01T10:00:05"},
        ]
        current = [
            {
                "role": "human",
                "content": "今の発言",
                "timestamp": "2026-08-02T01:00:00",
            },
        ]
        with (
            patch.object(
                agent_module, "get_recent_histories", return_value=[("old-uid", past)]
            ),
            patch.object(agent_module, "get_history", return_value=current),
        ):
            agent.set_memory_from_recent_histories("conf", 3, current_uid="cur-uid")

        self.assertEqual(agent._past_history_cut, 2)
        self.assertEqual(len(agent._memory), 3)
        self.assertIn("旧発言", agent._past_transcript)
        self.assertIn("旧返事", agent._past_transcript)
        self.assertNotIn("今の発言", agent._past_transcript)


class SetThinkingModeTests(unittest.TestCase):
    def test_forced_on_46_is_effective(self):
        agent = _bare_agent(_FakeClaudeLLM("claude-opus-4-6"))
        result = agent.set_thinking_mode("forced")
        self.assertEqual(
            result,
            {
                "ok": True,
                "requested": "forced",
                "effective": "forced",
                "model": "claude-opus-4-6",
            },
        )
        self.assertTrue(agent._llm._thinking_force)

    def test_forced_on_opus5_reports_adaptive_fallback(self):
        agent = _bare_agent(_FakeClaudeLLM("claude-opus-5"))
        result = agent.set_thinking_mode("forced")
        self.assertEqual(result["effective"], "adaptive")
        self.assertEqual(result["requested"], "forced")

    def test_adaptive_clears_force(self):
        llm = _FakeClaudeLLM()
        llm._thinking_force = True
        agent = _bare_agent(llm)
        result = agent.set_thinking_mode("adaptive")
        self.assertTrue(result["ok"])
        self.assertFalse(llm._thinking_force)

    def test_bad_mode_and_non_claude_rejected(self):
        agent = _bare_agent(_FakeClaudeLLM())
        self.assertFalse(agent.set_thinking_mode("sometimes")["ok"])
        agent2 = _bare_agent(llm=object())
        self.assertFalse(agent2.set_thinking_mode("forced")["ok"])


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_to_agent_and_replies(self):
        agent = _bare_agent(_FakeClaudeLLM())
        fake_self = SimpleNamespace(
            client_contexts={"uid-1": SimpleNamespace(agent_engine=agent)}
        )
        ws = SimpleNamespace(send_text=AsyncMock())

        await WebSocketHandler._handle_set_thinking_mode(
            fake_self, ws, "uid-1", {"type": "set-thinking-mode", "mode": "forced"}
        )

        payload = json.loads(ws.send_text.await_args.args[0])
        self.assertEqual(payload["type"], "thinking-mode-result")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["effective"], "forced")

    async def test_missing_agent_reports_error(self):
        fake_self = SimpleNamespace(client_contexts={})
        ws = SimpleNamespace(send_text=AsyncMock())

        await WebSocketHandler._handle_set_thinking_mode(
            fake_self, ws, "nope", {"mode": "forced"}
        )

        payload = json.loads(ws.send_text.await_args.args[0])
        self.assertFalse(payload["ok"])


class BridgeRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_resolves_with_server_result(self):
        bridge = OLVBridge.__new__(OLVBridge)
        bridge._ws = SimpleNamespace(send=AsyncMock())
        bridge._send_lock = asyncio.Lock()
        bridge._thinking_future = None

        async def feeder():
            await asyncio.sleep(0.01)
            await bridge._handle_incoming(
                {
                    "type": "thinking-mode-result",
                    "ok": True,
                    "requested": "forced",
                    "effective": "forced",
                    "model": "claude-opus-4-6",
                }
            )

        result, _ = await asyncio.gather(
            bridge.request_thinking_mode("forced"), feeder()
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["effective"], "forced")
        sent = json.loads(bridge._ws.send.await_args.args[0])
        self.assertEqual(sent, {"type": "set-thinking-mode", "mode": "forced"})

    async def test_disconnected_bridge_raises(self):
        bridge = OLVBridge.__new__(OLVBridge)
        bridge._ws = None
        with self.assertRaises(RuntimeError):
            await bridge.request_thinking_mode("forced")


class _FakeBridge:
    def set_proactive_callback(self, callback):
        self.proactive_callback = callback


class BotCommandTests(unittest.TestCase):
    def test_thinking_command_registered(self):
        bot = DiscordVTuberBot(
            bridge=_FakeBridge(),
            admin_user_id=42,
            project_root=Path.cwd(),
        )
        commands = {command.name: command for command in bot._tree.get_commands()}
        self.assertIn("thinking", commands)
        self.assertIn("forced", commands["thinking"].description)


if __name__ == "__main__":
    unittest.main()
