"""Per-round tool-turn replay & caching (あさひ 08-09 rulings, probe-proven).

Three rulings: (1) thinking_replay_max_tokens applies PER ROUND, not per
turn; (2) the two over-cap dimensions are independent — an oversized tool
result is truncated on its own, an oversized round loses only its thinking
blocks (the _tool_thinking_replay_probe showed completed historical rounds
accept thinking removal; the immutable-latest-assistant 400 governs only the
active loop tail); (3) after a drop, the next user payload opens with a
one-shot notice so the missing precedent isn't imitated as "think less".

Plus the cache consequence: a round whose stability is knowable in-round
(thinking ≤ cap AND results truncation-stable) can never be invalidated by
the post-turn trim, so the message-level breakpoint MIGRATES onto its
tool_result mid-loop; the first unstable round stops the migration.
"""

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace

import src.open_llm_vtuber.agent.agents.basic_memory_agent as bma
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)


THINKING = {"type": "thinking", "thinking": "ponder", "signature": "sig"}
CC = {"type": "ephemeral", "ttl": "1h"}
LIMIT = BasicMemoryAgent._PROTOCOL_RESULT_MAX_CHARS
MARKER = BasicMemoryAgent._PROTOCOL_TRUNCATION_MARKER
NOTICE = BasicMemoryAgent._THINKING_DROP_NOTICE


class _RecordingLLM(ClaudeAsyncLLM):
    """Fake stream that also snapshots the messages list at every request —
    the loop's moving breakpoint is only observable through what each round's
    request actually contained."""

    def __init__(self, rounds):
        self.model = "claude-opus-5"
        self.thinking_replay_max_tokens = 2000
        self._rounds = [list(r) for r in rounds]
        self.seen_messages = []

    async def chat_completion(self, messages, system=None, tools=None):
        self.seen_messages.append(deepcopy(messages))
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
    return agent


def _usage(thinking_tokens):
    return {
        "type": "message_delta",
        "data": {
            "delta": {},
            "usage": {
                "output_tokens": thinking_tokens + 20,
                "output_tokens_details": {"thinking_tokens": thinking_tokens},
            },
        },
    }


def _tool_round(tool_id, url, thinking_tokens):
    content = [
        deepcopy(THINKING),
        {"type": "tool_use", "id": tool_id, "name": "web_fetch", "input": {"url": url}},
    ]
    return [
        {
            "type": "tool_use_complete",
            "data": {"id": tool_id, "name": "web_fetch", "input": {"url": url}},
        },
        {
            "type": "assistant_message_complete",
            "data": {"role": "assistant", "content": content},
        },
        _usage(thinking_tokens),
        {"type": "message_stop"},
    ]


def _final_round(text="done", thinking_tokens=100):
    content = [deepcopy(THINKING), {"type": "text", "text": text}]
    return [
        {"type": "text_delta", "text": text},
        {
            "type": "assistant_message_complete",
            "data": {"role": "assistant", "content": content},
        },
        _usage(thinking_tokens),
        {"type": "message_stop"},
    ]


def _marked_user(text="hi"):
    return {
        "role": "user",
        "content": [{"type": "text", "text": text, "cache_control": CC}],
    }


def _has_marker(msg):
    content = msg.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and "cache_control" in b for b in content
    )


def _thinking_types(msg):
    return [
        b.get("type")
        for b in msg.get("content", [])
        if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
    ]


async def _run_loop(agent, initial=None):
    async for _ in agent._claude_tool_interaction_loop(
        initial or [_marked_user()], [{"name": "web_fetch"}]
    ):
        pass


async def _fake_fetch(url, *, max_chars=20000):
    """web_fetch fake: /<n> in the URL → n chars of body text, so the URL
    itself picks whether the round's tool_result is truncation-stable."""
    n = int(url.rsplit("/", 1)[-1])
    return {"url": url, "title": "T", "text": "x" * n}


class MarkerMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, rounds, initial=None):
        llm = _RecordingLLM(rounds)
        llm._enable_web_fetch = True
        llm._max_web_fetches = 5
        agent = _bare_agent(llm)
        original = bma.web_fetch
        bma.web_fetch = _fake_fetch
        try:
            await _run_loop(agent, initial)
        finally:
            bma.web_fetch = original
        return agent, llm

    async def test_stable_round_moves_breakpoint_to_tool_result(self):
        agent, llm = await self._run(
            [_tool_round("t1", "https://e/100", 500), _final_round()]
        )
        # Request 2 saw the migrated marker: user message stripped, the
        # round's tool_result carries it on its last block.
        second = llm.seen_messages[1]
        self.assertEqual(len(second), 3)
        self.assertFalse(_has_marker(second[0]))
        self.assertTrue(_has_marker(second[2]))
        self.assertIn("cache_control", second[2]["content"][-1])
        # The stored protocol and the seed stay marker-free: replays must not
        # multiply breakpoints.
        protocol = agent._memory[-1]["claude_protocol"]
        seed = agent.pop_thinking_seed()
        for msg in protocol + seed["protocol"]:
            self.assertFalse(_has_marker(msg), msg)

    async def test_over_cap_round_never_marks_and_stays_off(self):
        agent, llm = await self._run(
            [
                _tool_round("t1", "https://e/100", 5000),
                _tool_round("t2", "https://e/100", 100),
                _final_round(),
            ]
        )
        # Round 1 over cap: marker stays on the user message for the whole
        # turn; round 2 is stable on its own but marking already stopped.
        for request in llm.seen_messages[1:]:
            self.assertTrue(_has_marker(request[0]))
            for msg in request[1:]:
                self.assertFalse(_has_marker(msg))

    async def test_oversized_result_stops_marking(self):
        agent, llm = await self._run(
            [
                _tool_round("t1", "https://e/1000", 100),
                _tool_round("t2", "https://e/100", 100),
                _final_round(),
            ]
        )
        for request in llm.seen_messages[1:]:
            self.assertTrue(_has_marker(request[0]))
            for msg in request[1:]:
                self.assertFalse(_has_marker(msg))

    async def test_image_in_prefix_disables_migration(self):
        # Mid-loop entries written on an image-bearing prefix can never be
        # re-read (the image is gone from next turn's replay, the prefix
        # diverges there), so migration is off for the whole turn even when
        # every round is stable. The marker stays parked on the leading text
        # block where _to_messages put it.
        initial = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look", "cache_control": CC},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,xxxx"},
                    },
                ],
            }
        ]
        agent, llm = await self._run(
            [_tool_round("t1", "https://e/100", 100), _final_round()],
            initial=initial,
        )
        for request in llm.seen_messages[1:]:
            self.assertIn("cache_control", request[0]["content"][0])
            for msg in request[1:]:
                self.assertFalse(_has_marker(msg))

    async def test_migration_advances_across_stable_rounds(self):
        agent, llm = await self._run(
            [
                _tool_round("t1", "https://e/100", 100),
                _tool_round("t2", "https://e/100", 200),
                _final_round(),
            ]
        )
        second = llm.seen_messages[1]
        self.assertFalse(_has_marker(second[0]))
        self.assertTrue(_has_marker(second[2]))
        third = llm.seen_messages[2]
        # Marker moved again: round-1 tool_result stripped, round-2 marked.
        self.assertFalse(_has_marker(third[2]))
        self.assertTrue(_has_marker(third[4]))


class TurnEndTrimTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_dimensions_and_seed_alignment(self):
        # Round 1: thinking over cap, result stable → thinking dropped, tool
        # machinery kept (the net-fidelity win over the old whole-turn gate).
        # Round 2: thinking under cap, result oversized → thinking kept,
        # result truncated. Final: under cap → kept.
        llm = _RecordingLLM(
            [
                _tool_round("t1", "https://e/100", 5000),
                _tool_round("t2", "https://e/1000", 100),
                _final_round(thinking_tokens=200),
            ]
        )
        llm._enable_web_fetch = True
        llm._max_web_fetches = 5
        agent = _bare_agent(llm)
        original = bma.web_fetch
        bma.web_fetch = _fake_fetch
        try:
            await _run_loop(agent)
        finally:
            bma.web_fetch = original

        protocol = agent._memory[-1]["claude_protocol"]
        # [assistant r1, tool_result r1, assistant r2, tool_result r2, final]
        self.assertEqual(len(protocol), 5)
        r1, res1, r2, res2, final = protocol
        self.assertEqual(_thinking_types(r1), [])
        self.assertTrue(
            any(b.get("type") == "tool_use" for b in r1["content"]),
            "over-cap round must keep its tool call",
        )
        self.assertEqual(_thinking_types(r2), ["thinking"])
        self.assertEqual(_thinking_types(final), ["thinking"])
        self.assertLessEqual(len(res1["content"][0]["content"]), LIMIT)
        truncated = res2["content"][0]["content"]
        self.assertTrue(truncated.endswith(MARKER))
        self.assertEqual(len(truncated), LIMIT + len(MARKER))
        # Seed: dropped round zeroed, kept rounds carry their billed counts.
        seed = agent.pop_thinking_seed()
        self.assertEqual(seed["round_thinking"], [0, 100, 200])
        self.assertEqual(seed["thinking_tokens"], 5300)
        self.assertTrue(agent._pending_thinking_drop_notice)


class TrimHelperTests(unittest.TestCase):
    def _protocol(self):
        return [
            {
                "role": "assistant",
                "content": [
                    deepcopy(THINKING),
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            },
            {
                "role": "assistant",
                "content": [deepcopy(THINKING), {"type": "text", "text": "done"}],
            },
        ]

    def test_interleaved_drop_keeps_later_signed_thinking(self):
        # The probe's P1 shape: first round's thinking dropped, second kept
        # with its signature.
        trimmed, kept, dropped = BasicMemoryAgent._trim_protocol_per_round(
            self._protocol(), [3000, 100], 2000
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, [0, 100])
        self.assertEqual(_thinking_types(trimmed[0]), [])
        self.assertEqual(_thinking_types(trimmed[2]), ["thinking"])
        self.assertEqual(trimmed[2]["content"][0]["signature"], "sig")

    def test_thinking_only_message_emptied_is_omitted(self):
        protocol = self._protocol()
        protocol[2] = {"role": "assistant", "content": [deepcopy(THINKING)]}
        trimmed, kept, dropped = BasicMemoryAgent._trim_protocol_per_round(
            protocol, [100, 9000], 2000
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(len(trimmed), 2)
        # Alignment survives the omission: only the kept assistant remains.
        self.assertEqual(kept, [100])

    def test_cap_zero_disables_and_redacted_dropped_with_thinking(self):
        trimmed, kept, dropped = BasicMemoryAgent._trim_protocol_per_round(
            self._protocol(), [9000, 9000], 0
        )
        self.assertEqual(dropped, 0)
        self.assertEqual(_thinking_types(trimmed[0]), ["thinking"])
        protocol = self._protocol()
        protocol[0]["content"].insert(1, {"type": "redacted_thinking", "data": "blob"})
        trimmed, _, dropped = BasicMemoryAgent._trim_protocol_per_round(
            protocol, [3000, 100], 2000
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(_thinking_types(trimmed[0]), [])

    def test_idempotent_on_retrimmed_seed(self):
        first, kept, _ = BasicMemoryAgent._trim_protocol_per_round(
            self._protocol(), [3000, 100], 2000
        )
        second, kept2, dropped2 = BasicMemoryAgent._trim_protocol_per_round(
            first, kept, 2000
        )
        self.assertEqual(second, first)
        self.assertEqual(kept2, kept)
        self.assertEqual(dropped2, 0)


class SeedReloadTests(unittest.TestCase):
    def _agent(self, cap):
        llm = _RecordingLLM([])
        llm.thinking_replay_max_tokens = cap
        agent = _bare_agent(llm)
        agent._memory = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        return agent

    def _seed_protocol(self):
        return [
            {
                "role": "assistant",
                "content": [
                    deepcopy(THINKING),
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            },
            {
                "role": "assistant",
                "content": [deepcopy(THINKING), {"type": "text", "text": "done"}],
            },
        ]

    def test_per_round_seed_over_old_whole_turn_total_still_attaches(self):
        # Whole-turn total (3000) exceeds the cap but every round is under it
        # — the old gate would have skipped this seed; per-round keeps it.
        agent = self._agent(cap=2000)
        seed = {
            "model": "claude-opus-5",
            "protocol": self._seed_protocol(),
            "thinking_tokens": 3000,
            "round_thinking": [1500, 1500],
        }
        agent._apply_thinking_seeds([(1, seed)], resuming=False)
        protocol = agent._memory[1]["claude_protocol"]
        self.assertEqual(_thinking_types(protocol[0]), ["thinking"])
        self.assertEqual(_thinking_types(protocol[2]), ["thinking"])

    def test_shrunken_cap_retrims_on_reload(self):
        agent = self._agent(cap=1000)
        seed = {
            "model": "claude-opus-5",
            "protocol": self._seed_protocol(),
            "thinking_tokens": 1600,
            "round_thinking": [1500, 100],
        }
        agent._apply_thinking_seeds([(1, seed)], resuming=False)
        protocol = agent._memory[1]["claude_protocol"]
        self.assertEqual(_thinking_types(protocol[0]), [])
        self.assertEqual(_thinking_types(protocol[2]), ["thinking"])
        # No notice on reload: these are old turns, not fresh precedent.
        self.assertFalse(agent._pending_thinking_drop_notice)

    def test_legacy_seed_keeps_whole_turn_skip(self):
        agent = self._agent(cap=2000)
        seed = {
            "model": "claude-opus-5",
            "protocol": self._seed_protocol(),
            "thinking_tokens": 3000,
        }
        agent._apply_thinking_seeds([(1, seed)], resuming=False)
        self.assertNotIn("claude_protocol", agent._memory[1])


class DropNoticeInjectionTests(unittest.TestCase):
    def _agent(self, memory=None):
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._past_history_cut = 0
        agent._memory = memory or []
        agent._is_claude_llm = lambda: True
        agent._openai_explicit_cache = lambda: False
        agent._to_text_prompt = lambda d: getattr(d, "text", "")
        agent._time_event_banner = lambda: ""
        agent._steam_pending_blocks = []
        agent._memory_pending_blocks = []
        agent._pending_rag_block = ""
        agent._pending_facts_block = ""
        agent._current_session_banner_added = True
        agent._memory_manager = None
        agent._added = []
        agent._add_message = lambda text, role, **kw: agent._added.append((role, text))
        return agent

    def _input(self, text=""):
        return SimpleNamespace(text=text, images=None, metadata=None)

    def test_notice_prepended_stored_verbatim_and_one_shot(self):
        agent = self._agent()
        agent._pending_thinking_drop_notice = True
        out = agent._to_messages(self._input("つづき"))
        sent = out[-1]["content"][-1]["text"]
        self.assertTrue(sent.startswith(NOTICE))
        self.assertIn("つづき", sent)
        # stored == sent (the notice-bearing message is also the breakpoint
        # holder, so byte identity is what keeps the cache chain unbroken).
        self.assertEqual(agent._added, [("user", sent)])
        self.assertIn("cache_control", out[-1]["content"][-1])
        self.assertFalse(agent._pending_thinking_drop_notice)
        # One-shot: the next turn is clean.
        agent2_out = agent._to_messages(self._input("その次"))
        self.assertNotIn(NOTICE, agent2_out[-1]["content"][-1]["text"])

    def test_banner_precedes_notice_when_both_fire(self):
        agent = self._agent()
        agent._current_session_banner_added = False
        agent._memory_manager = SimpleNamespace(
            _current_session_uid="2026-08-09_05-15-25_deadbeef"
        )
        agent._pending_thinking_drop_notice = True
        out = agent._to_messages(self._input("おはよう"))
        sent = out[-1]["content"][-1]["text"]
        self.assertLess(sent.index("セッション開始"), sent.index(NOTICE))
        self.assertLess(sent.index(NOTICE), sent.index("おはよう"))
        self.assertEqual(agent._added, [("user", sent)])

    def test_skip_memory_turn_does_not_consume_notice(self):
        agent = self._agent()
        agent._pending_thinking_drop_notice = True
        data = self._input("キャッシュ保温")
        data.metadata = {"skip_memory": True}
        out = agent._to_messages(data)
        self.assertNotIn(NOTICE, out[-1]["content"][-1]["text"])
        self.assertTrue(agent._pending_thinking_drop_notice)


class ToolResultStabilityMirrorTests(unittest.TestCase):
    def test_stability_identity_mirrors_post_turn_trim(self):
        # The in-loop judgement is `_truncate_tool_result_block(b) is b`;
        # these are the two sides it must agree with.
        short = {
            "type": "tool_result",
            "tool_use_id": "t",
            "content": json.dumps({"ok": True}),
        }
        self.assertIs(BasicMemoryAgent._truncate_tool_result_block(short), short)
        long = {**short, "content": "x" * (LIMIT + 1)}
        self.assertIsNot(BasicMemoryAgent._truncate_tool_result_block(long), long)


if __name__ == "__main__":
    unittest.main()
