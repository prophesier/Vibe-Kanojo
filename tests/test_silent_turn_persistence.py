"""Silent turns (think-only / tool-only) persist and announce honestly.

あさひ 08-07: a think-only turn used to leave NO disk record, so a reload
collapsed into two adjacent user turns — the exact shape that seeded the
08-05 hallucinated-input incident. Now the empty-content record carries the
seed (verbatim replay on same-model restart) and Discord shows a precise
notice instead of "(no reply)".
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.conversations.single_conversation import _no_speech_reason
from src.open_llm_vtuber.discord_bot.bot import DiscordVTuberBot
from src.open_llm_vtuber.discord_bot.bridge import OLVBridge, TurnResult, _PendingTurn


def _seed(*block_types):
    return {
        "model": "claude-opus-5",
        "protocol": [
            {"role": "assistant", "content": [{"type": t} for t in block_types]}
        ],
    }


class NoSpeechReasonTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(_no_speech_reason(_seed("thinking")), "think-only")
        self.assertEqual(_no_speech_reason(_seed("thinking", "tool_use")), "think-only")
        self.assertEqual(_no_speech_reason(_seed("tool_use")), "tool-only")
        self.assertEqual(_no_speech_reason({"protocol": []}), "unknown")


class ReloadTests(unittest.TestCase):
    def _agent(self):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._format_timestamp = lambda ts: "[tag]"
        return agent

    def test_empty_ai_record_with_seed_is_kept(self):
        entry = self._agent()._msg_from_history_record(
            {
                "role": "ai",
                "content": "",
                "timestamp": "2026-08-07T19:07:10",
                "thinking_seed": _seed("thinking"),
            }
        )
        self.assertEqual(entry, {"role": "assistant", "content": ""})

    def test_empty_ai_record_without_seed_still_dropped(self):
        entry = self._agent()._msg_from_history_record(
            {"role": "ai", "content": "", "timestamp": "2026-08-07T19:07:10"}
        )
        self.assertIsNone(entry)


class BreakpointSkipTests(unittest.TestCase):
    def test_breakpoint_skips_empty_placeholder(self):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._llm = SimpleNamespace()
        agent._is_claude_llm = lambda: True
        messages = [
            {"role": "user", "content": "話しかけた"},
            {"role": "assistant", "content": ""},
        ]
        out = agent._attach_cache_breakpoint(messages)
        self.assertEqual(out[1], {"role": "assistant", "content": ""})
        self.assertIsInstance(out[0]["content"], list)
        self.assertIn("cache_control", out[0]["content"][-1])


class BridgeAndBotTests(unittest.IsolatedAsyncioTestCase):
    async def test_bridge_records_no_speech_reason(self):
        bridge = OLVBridge.__new__(OLVBridge)
        turn = _PendingTurn(request_id="r1", text="hi", callback=AsyncMock())
        bridge._current_turn = turn
        await bridge._handle_incoming(
            {"type": "turn-no-speech", "reason": "think-only"}
        )
        self.assertEqual(turn.result.no_speech_reason, "think-only")

    async def test_bridge_ignores_no_speech_without_pending_turn(self):
        bridge = OLVBridge.__new__(OLVBridge)
        bridge._current_turn = None
        await bridge._handle_incoming(
            {"type": "turn-no-speech", "reason": "think-only"}
        )  # must not raise

    async def _post(self, result):
        bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
        bot._safe_reply = AsyncMock(return_value=True)
        source = SimpleNamespace(channel=SimpleNamespace())
        await bot._post_reply(source, result)
        return bot._safe_reply.await_args.args[1]

    async def test_bot_precise_notice_for_think_only(self):
        notice = await self._post(
            TurnResult(request_id="r1", no_speech_reason="think-only")
        )
        self.assertEqual(notice, "(思考のみ、発話なし)")

    async def test_bot_precise_notice_for_tool_only(self):
        notice = await self._post(
            TurnResult(request_id="r2", no_speech_reason="tool-only")
        )
        self.assertEqual(notice, "(ツール実行のみ、発話なし)")

    async def test_bot_legacy_marker_for_genuinely_empty(self):
        notice = await self._post(TurnResult(request_id="r3"))
        self.assertEqual(notice, "(no reply)")


if __name__ == "__main__":
    unittest.main()
