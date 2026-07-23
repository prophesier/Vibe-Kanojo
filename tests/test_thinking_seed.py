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
        self._events = list(events)

    async def chat_completion(self, messages, system=None, tools=None):
        for event in self._events:
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
            {"model": "claude-opus-4-6", "content": FINAL["content"]},
        )
        self.assertIsNone(agent.pop_thinking_seed(), "pop must be one-shot")

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
    return {"model": model, "content": [deepcopy(THINKING), deepcopy(TEXT)]}


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
        self.assertEqual(proto, [{"role": "assistant", "content": _seed()["content"]}])

    def test_low_rate_seeds_nothing(self):
        agent = _bare_agent(_FakeClaudeLLM([]))
        agent._memory = _memory_with_assistants(10)
        assistant_idxs = [
            i for i, m in enumerate(agent._memory) if m["role"] == "assistant"
        ]
        candidates = [(assistant_idxs[0], _seed())]  # 1/10 < 30%

        agent._apply_thinking_seeds(candidates, resuming=True)

        self.assertFalse(any("claude_protocol" in m for m in agent._memory))

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
