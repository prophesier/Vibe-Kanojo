"""Tests for the thinking edge-mode batch (07-25): thinking-only turns are
committed with EMPTY text plus their verbatim transcript (never a synthesized
utterance — あさひ: rewriting the model's own output teaches it the pattern),
protocol-less replay omits such messages entirely, unsigned thinking never
enters replay, and invisible-thinking serialization keeps the required
"thinking" key."""

import unittest
from copy import deepcopy

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)


SIGNED_INVISIBLE = {"type": "thinking", "thinking": "", "signature": "sig"}
UNSIGNED = {"type": "thinking", "thinking": "cut off mid-reason", "signature": ""}
REDACTED = {"type": "redacted_thinking", "data": "opaque-blob"}


class _FakeClaudeLLM(ClaudeAsyncLLM):
    def __init__(self, rounds):
        self.model = "claude-opus-5"
        self._rounds = [list(r) for r in rounds]

    async def chat_completion(self, messages, system=None, tools=None):
        for event in self._rounds.pop(0):
            yield deepcopy(event)


def _bare_agent(llm) -> BasicMemoryAgent:
    agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
    agent._llm = llm
    agent._memory = []
    agent._memory_manager = None
    agent._build_system_for_llm = lambda: "system"
    agent._check_stateful_claims = lambda text: None
    agent._turn_inproc_calls = []
    agent._web_tools_config = {}
    agent._model_health_enabled = False
    agent._steam_enabled = False
    agent._memory_tools_enabled = False
    agent._tool_executor = None
    agent._mcp_tool_marker = lambda name: ""
    agent._last_thinking_seed = None
    return agent


def _thinking_only_rounds(block, stop_reason, thinking_tokens):
    final = {"role": "assistant", "content": [deepcopy(block)]}
    return [
        [
            {"type": "thinking_complete", "data": deepcopy(block)},
            {"type": "assistant_message_complete", "data": deepcopy(final)},
            {
                "type": "message_delta",
                "data": {
                    "delta": {"stop_reason": stop_reason},
                    "usage": {
                        "output_tokens": thinking_tokens,
                        "output_tokens_details": {"thinking_tokens": thinking_tokens},
                    },
                },
            },
            {"type": "message_stop"},
        ]
    ]


class ThinkingOnlyTurnTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, rounds):
        llm = _FakeClaudeLLM(rounds)
        agent = _bare_agent(llm)
        out = []
        async for item in agent._claude_tool_interaction_loop(
            [{"role": "user", "content": "hi"}], []
        ):
            out.append(item)
        return agent, out

    async def test_signed_thinking_only_commits_transcript_no_text(self):
        agent, out = await self._run(
            _thinking_only_rounds(SIGNED_INVISIBLE, "end_turn", 300)
        )
        self.assertEqual(out, [], "no synthesized utterance may be yielded")
        self.assertEqual(agent._memory[-1]["role"], "assistant")
        self.assertEqual(agent._memory[-1]["content"], "")
        self.assertIn("claude_protocol", agent._memory[-1])
        seed = agent.pop_thinking_seed()
        self.assertIsNotNone(seed, "signed thinking-only turn must stage a seed")
        self.assertEqual(seed["thinking_tokens"], 300)

    async def test_unsigned_thinking_turn_dropped(self):
        # max_tokens cut mid-thinking: the block has no signature, so the
        # transcript must never be stored or replayed — whole turn dropped.
        agent, out = await self._run(_thinking_only_rounds(UNSIGNED, "max_tokens", 500))
        self.assertEqual(out, [])
        self.assertEqual(agent._memory, [])
        self.assertIsNone(agent.pop_thinking_seed())

    async def test_redacted_only_turn_commits_with_protocol(self):
        agent, out = await self._run(_thinking_only_rounds(REDACTED, "end_turn", 400))
        self.assertEqual(out, [])
        self.assertEqual(agent._memory[-1]["content"], "")
        self.assertIn("claude_protocol", agent._memory[-1])
        seed = agent.pop_thinking_seed()
        self.assertIsNotNone(seed)
        self.assertEqual(
            seed["protocol"][-1]["content"][0]["type"], "redacted_thinking"
        )

    async def test_text_turn_untouched_by_empty_branch(self):
        block = {"type": "thinking", "thinking": "ponder", "signature": "sig"}
        final = {
            "role": "assistant",
            "content": [deepcopy(block), {"type": "text", "text": "reply"}],
        }
        rounds = [
            [
                {"type": "thinking_complete", "data": deepcopy(block)},
                {"type": "text_delta", "text": "reply"},
                {"type": "assistant_message_complete", "data": deepcopy(final)},
                {"type": "message_stop"},
            ]
        ]
        agent, out = await self._run(rounds)
        self.assertEqual(out, ["reply"])
        self.assertEqual(agent._memory[-1]["content"], "reply")


class ReplayableCheckTests(unittest.TestCase):
    def test_unsigned_thinking_not_replayable(self):
        self.assertFalse(
            BasicMemoryAgent._claude_thinking_blocks_replayable([UNSIGNED])
        )

    def test_signed_and_redacted_replayable(self):
        self.assertTrue(
            BasicMemoryAgent._claude_thinking_blocks_replayable(
                [SIGNED_INVISIBLE, REDACTED, {"type": "text", "text": "t"}]
            )
        )

    def test_empty_content_replayable(self):
        self.assertTrue(BasicMemoryAgent._claude_thinking_blocks_replayable([]))


class ConvertMessagesTests(unittest.TestCase):
    def _llm(self):
        return ClaudeAsyncLLM(model="claude-opus-5", llm_api_key="sk-test")

    def test_empty_assistant_without_protocol_omitted(self):
        # A thinking-only turn replayed where its transcript is unavailable
        # (cap-gated / reloaded seedless) must be omitted whole — an empty
        # text block is API-invalid and synthesized text is forbidden.
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"},
        ]
        out = self._llm()._convert_messages_format(msgs)
        self.assertEqual([m["role"] for m in out], ["user", "user"])

    def test_empty_assistant_with_protocol_expanded(self):
        proto = [{"role": "assistant", "content": [deepcopy(SIGNED_INVISIBLE)]}]
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "claude_protocol": deepcopy(proto),
            },
        ]
        out = self._llm()._convert_messages_format(msgs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["content"][0]["type"], "thinking")


class HistoryNoteRoutingTests(unittest.TestCase):
    def _agent_with_model(self, model):
        llm = _FakeClaudeLLM([])
        llm.model = model
        return _bare_agent(llm)

    def test_opus5_gets_lean_note(self):
        note = self._agent_with_model("claude-opus-5")._history_note()
        self.assertIs(note, BasicMemoryAgent._HISTORY_NOTE_LEAN)
        self.assertNotIn("[Thinking]", note)

    def test_opus46_keeps_full_note(self):
        note = self._agent_with_model("claude-opus-4-6")._history_note()
        self.assertIs(note, BasicMemoryAgent._HISTORY_NOTE)
        self.assertIn("[Thinking]", note)

    def test_lean_note_keeps_guardrails(self):
        note = BasicMemoryAgent._HISTORY_NOTE_LEAN
        for anchor in (
            "セッション開始",
            "自動検索",
            "タイムスタンプ",
            "web_fetch",
            # Full strict time rules restored 07-26 (time errors persisted).
            "時間に関する厳格なルール",
        ):
            self.assertIn(anchor, note)
        # 4.6-era nudges must be absent from the lean variant (and present in
        # the full one).
        for dropped in ("[Thinking]", "Strict rules on executing tools"):
            self.assertNotIn(dropped, note)
            self.assertIn(dropped, BasicMemoryAgent._HISTORY_NOTE)


class LeanCapabilityNoteTests(unittest.TestCase):
    def test_lean_alarm_note_keeps_when_to_use_only(self):
        note = BasicMemoryAgent._ALARM_CAPABILITY_NOTE_LEAN
        self.assertIn("set_alarm", note)
        self.assertNotIn("list_alarms", note)

    def test_lean_memory_note_drops_schema_mechanics(self):
        note = BasicMemoryAgent._MEMORY_CAPABILITY_NOTE_LEAN
        for kept in (
            "memory_search",
            "history_search",
            "memory_inject",
            "memory_delete",
            "consent",
        ):
            self.assertIn(kept, note)
        for cut in ("date_from", "user-tier", "resident list"):
            self.assertNotIn(cut, note)

    def test_lean_gate_by_model(self):
        llm = _FakeClaudeLLM([])
        llm.model = "claude-opus-5"
        self.assertTrue(_bare_agent(llm)._lean_prompt_active())
        llm2 = _FakeClaudeLLM([])
        llm2.model = "claude-opus-4-6"
        self.assertFalse(_bare_agent(llm2)._lean_prompt_active())


class _StubBlock:
    def __init__(self, data):
        self._data = data

    def model_dump(self, exclude_none=False):
        d = dict(self._data)
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


class _StubMessage:
    def __init__(self, blocks):
        self.content = blocks


class ReplaySerializerTests(unittest.TestCase):
    def test_none_thinking_field_kept_as_empty_string(self):
        # exclude_none must not strip the required "thinking" key off a
        # text-less signed block (invisible-thinking shape).
        msg = _StubMessage(
            [
                _StubBlock({"type": "thinking", "thinking": None, "signature": "sig"}),
                _StubBlock({"type": "text", "text": "hi", "citations": None}),
            ]
        )
        out = ClaudeAsyncLLM._assistant_message_for_replay(msg)
        self.assertEqual(out["content"][0]["thinking"], "")
        self.assertEqual(out["content"][0]["signature"], "sig")
        self.assertNotIn("citations", out["content"][1])

    def test_normal_thinking_block_unchanged(self):
        msg = _StubMessage(
            [_StubBlock({"type": "thinking", "thinking": "t", "signature": "s"})]
        )
        out = ClaudeAsyncLLM._assistant_message_for_replay(msg)
        self.assertEqual(out["content"][0]["thinking"], "t")


if __name__ == "__main__":
    unittest.main()
