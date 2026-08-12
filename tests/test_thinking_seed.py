import unittest
from copy import deepcopy

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)


THINKING = {"type": "thinking", "thinking": "ponder", "signature": "sig-1"}
TEXT = {"type": "text", "text": "reply"}
FINAL = {"role": "assistant", "content": [deepcopy(THINKING), deepcopy(TEXT)]}


class _FakeClaudeLLM(ClaudeAsyncLLM):
    def __init__(self, events):
        self.model = "claude-opus-4-6"
        # A flat list = one response; a list of lists = one response per call.
        self._rounds = (
            [list(e) for e in events]
            if events and isinstance(events[0], list)
            else [list(events)]
        )

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
    return agent


class SeedCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def _run_loop(self, events):
        agent = _bare_agent(_FakeClaudeLLM(events))
        async for _ in agent._claude_tool_interaction_loop(
            [{"role": "user", "content": "hi"}], []
        ):
            pass
        return agent

    async def test_thinking_turn_stages_seed(self):
        agent = await self._run_loop(
            [
                {"type": "thinking_complete", "data": deepcopy(THINKING)},
                {"type": "text_delta", "text": "reply"},
                {"type": "assistant_message_complete", "data": deepcopy(FINAL)},
                {"type": "message_stop"},
            ]
        )
        seed = agent.pop_thinking_seed()
        self.assertEqual(
            seed,
            {
                "model": "claude-opus-4-6",
                "protocol": [FINAL],
                "thinking_tokens": 0,
                # Per-round breakdown (08-09): one entry per protocol
                # assistant message, for reload re-trimming.
                "round_thinking": [0],
            },
        )
        self.assertIsNone(agent.pop_thinking_seed(), "pop must be one-shot")

    async def test_tool_turn_seed_keeps_the_full_transcript(self):
        """A tool turn's seed must be the WHOLE transcript (tool_use and
        truncated tool_result included) — replaying a tool turn without its
        tool machinery would rewrite history into a "results without calls"
        shape and teach fabrication (あさひ 07-24)."""
        thinking1 = {"type": "thinking", "thinking": "pick a tool", "signature": "s1"}
        tool_use = {
            "type": "tool_use",
            "id": "t1",
            "name": "memory_search",
            "input": {"query": "q"},
        }
        thinking2 = {"type": "thinking", "thinking": "use result", "signature": "s2"}
        final = {
            "role": "assistant",
            "content": [deepcopy(thinking2), {"type": "text", "text": "answer"}],
        }

        class _FakeExecutor:
            async def execute_tools(self, tool_calls, caller_mode):
                yield {
                    "type": "final_tool_results",
                    "results": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                    ],
                }

        agent = _bare_agent(
            _FakeClaudeLLM(
                [
                    [
                        {"type": "thinking_complete", "data": deepcopy(thinking1)},
                        {
                            "type": "tool_use_complete",
                            "data": {
                                "id": "t1",
                                "name": "memory_search",
                                "input": {"query": "q"},
                            },
                        },
                        {
                            "type": "assistant_message_complete",
                            "data": {
                                "role": "assistant",
                                "content": [deepcopy(thinking1), deepcopy(tool_use)],
                            },
                        },
                        {"type": "message_stop"},
                    ],
                    [
                        {"type": "thinking_complete", "data": deepcopy(thinking2)},
                        {"type": "text_delta", "text": "answer"},
                        {"type": "assistant_message_complete", "data": deepcopy(final)},
                        {"type": "message_stop"},
                    ],
                ]
            )
        )
        agent._memory_tools_enabled = False
        agent._model_health_enabled = False
        agent._steam_enabled = False
        agent._tool_executor = _FakeExecutor()
        agent._mcp_tool_marker = lambda name: ""

        async for _ in agent._claude_tool_interaction_loop(
            [{"role": "user", "content": "hi"}], [{"name": "memory_search"}]
        ):
            pass

        seed = agent.pop_thinking_seed()
        self.assertEqual(seed["model"], "claude-opus-4-6")
        self.assertEqual(
            [m["role"] for m in seed["protocol"]], ["assistant", "user", "assistant"]
        )
        self.assertEqual(
            seed["protocol"][0]["content"],
            [thinking1, tool_use],
            "round-1 message must keep its signed thinking AND the tool_use",
        )
        self.assertEqual(
            seed["protocol"][1]["content"][0]["tool_use_id"],
            "t1",
            "the (truncated) tool_result stays paired with its tool_use",
        )
        self.assertEqual(seed["protocol"][2], final)
        # Seed and in-session protocol are the same transcript.
        self.assertEqual(agent._memory[-1]["claude_protocol"], seed["protocol"])

    async def test_toolonly_turn_stages_seed_and_protocol(self):
        """03:19 regression (あさひ 07-26): a tool turn with ZERO thinking must
        still carry its transcript — dropping it made the model read its own
        honest turn as a fabricated claim and re-set an existing alarm."""
        tool_use = {
            "type": "tool_use",
            "id": "t1",
            "name": "memory_search",
            "input": {"query": "q"},
        }
        final = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}

        class _FakeExecutor:
            async def execute_tools(self, tool_calls, caller_mode):
                yield {
                    "type": "final_tool_results",
                    "results": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                    ],
                }

        agent = _bare_agent(
            _FakeClaudeLLM(
                [
                    [
                        {
                            "type": "tool_use_complete",
                            "data": {
                                "id": "t1",
                                "name": "memory_search",
                                "input": {"query": "q"},
                            },
                        },
                        {
                            "type": "assistant_message_complete",
                            "data": {
                                "role": "assistant",
                                "content": [deepcopy(tool_use)],
                            },
                        },
                        {"type": "message_stop"},
                    ],
                    [
                        {"type": "text_delta", "text": "done"},
                        {"type": "assistant_message_complete", "data": deepcopy(final)},
                        {"type": "message_stop"},
                    ],
                ]
            )
        )
        agent._memory_tools_enabled = False
        agent._model_health_enabled = False
        agent._steam_enabled = False
        agent._tool_executor = _FakeExecutor()
        agent._mcp_tool_marker = lambda name: ""

        async for _ in agent._claude_tool_interaction_loop(
            [{"role": "user", "content": "hi"}], [{"name": "memory_search"}]
        ):
            pass

        self.assertIn("claude_protocol", agent._memory[-1])
        seed = agent.pop_thinking_seed()
        self.assertIsNotNone(seed, "tool-only turn must persist its transcript")
        self.assertEqual(seed["thinking_tokens"], 0)
        self.assertEqual(
            [m["role"] for m in seed["protocol"]], ["assistant", "user", "assistant"]
        )

    async def test_thinking_less_turn_stages_nothing(self):
        agent = await self._run_loop(
            [
                {"type": "text_delta", "text": "reply"},
                {
                    "type": "assistant_message_complete",
                    "data": {"role": "assistant", "content": [deepcopy(TEXT)]},
                },
                {"type": "message_stop"},
            ]
        )
        self.assertIsNone(agent.pop_thinking_seed())


def _memory_with_assistants(n):
    memory = []
    for i in range(n):
        memory.append({"role": "user", "content": f"u{i}"})
        memory.append({"role": "assistant", "content": f"a{i}"})
    return memory


def _seed(model="claude-opus-4-6"):
    return {"model": model, "protocol": [deepcopy(FINAL)]}


class SeedApplyTests(unittest.TestCase):
    def test_high_rate_seeds_last_n(self):
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._memory = _memory_with_assistants(10)
        assistant_idxs = [
            i for i, m in enumerate(agent._memory) if m["role"] == "assistant"
        ]
        candidates = [(i, _seed()) for i in assistant_idxs]  # 10/10 thinking

        agent._apply_thinking_seeds(candidates, resuming=False)

        seeded = [i for i, m in enumerate(agent._memory) if "claude_protocol" in m]
        self.assertEqual(
            seeded, assistant_idxs[-BasicMemoryAgent._THINKING_SEED_COUNT :]
        )
        proto = agent._memory[seeded[-1]]["claude_protocol"]
        self.assertEqual(proto, _seed()["protocol"])

    def test_legacy_content_seed_still_applies(self):
        """07-23 evening seeds stored only the final message's blocks under a
        `content` key — they must still replay (wrapped as one message)."""
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._memory = _memory_with_assistants(2)
        legacy = {
            "model": "claude-opus-4-6",
            "content": [deepcopy(THINKING), deepcopy(TEXT)],
        }
        candidates = [(1, legacy), (3, legacy)]

        agent._apply_thinking_seeds(candidates, resuming=False)

        self.assertEqual(
            agent._memory[3]["claude_protocol"],
            [{"role": "assistant", "content": legacy["content"]}],
        )

    def test_low_rate_still_attaches_with_stats_warning(self):
        # あさひ 07-26: the 80% rate is stats/warning only — transcripts are
        # attached regardless (withholding them also erased tool calls and
        # made honest turns look fabricated).
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._memory = _memory_with_assistants(10)
        assistant_idxs = [
            i for i, m in enumerate(agent._memory) if m["role"] == "assistant"
        ]
        candidates = [(assistant_idxs[0], _seed())]  # 1/10 thinking

        agent._apply_thinking_seeds(candidates, resuming=True)

        self.assertIn("claude_protocol", agent._memory[assistant_idxs[0]])

    def test_only_last_10_root_assistant_turns_eligible(self):
        # The attach window is the last 10 ROOT assistant entries
        # (あさひ 07-26), not the last 10 seed-carrying candidates.
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._memory = _memory_with_assistants(12)
        assistant_idxs = [
            i for i, m in enumerate(agent._memory) if m["role"] == "assistant"
        ]
        candidates = [(i, _seed()) for i in assistant_idxs]

        agent._apply_thinking_seeds(candidates, resuming=False)

        for i in assistant_idxs[:2]:
            self.assertNotIn("claude_protocol", agent._memory[i])
        for i in assistant_idxs[2:]:
            self.assertIn("claude_protocol", agent._memory[i])

    def test_foreign_model_seed_skipped(self):
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._memory = _memory_with_assistants(4)
        assistant_idxs = [
            i for i, m in enumerate(agent._memory) if m["role"] == "assistant"
        ]
        candidates = [
            (assistant_idxs[0], _seed()),
            (assistant_idxs[1], _seed()),
            (assistant_idxs[2], _seed(model="claude-sonnet-5")),
            (assistant_idxs[3], _seed()),
        ]

        agent._apply_thinking_seeds(candidates, resuming=False)

        self.assertNotIn("claude_protocol", agent._memory[assistant_idxs[2]])
        self.assertIn("claude_protocol", agent._memory[assistant_idxs[3]])

    def test_non_claude_llm_is_noop(self):
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._llm = object()
        agent._memory = _memory_with_assistants(4)
        candidates = [(1, _seed()), (3, _seed()), (5, _seed()), (7, _seed())]

        agent._apply_thinking_seeds(candidates, resuming=False)

        self.assertFalse(any("claude_protocol" in m for m in agent._memory))


if __name__ == "__main__":
    unittest.main()
