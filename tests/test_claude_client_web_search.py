"""Client-side web_search on the Claude path (server tool retired 08-08).

Server web_search result blocks were replay bulk immune to protocol
truncation — one search parked 10-16k tokens in every later request of the
session. The brave/tavily client tool returns ordinary tool_results that the
400-char truncation caps once the turn ends.

Regression (08-09): web_tools.web_search returns a LIST (results on success,
[{"error": ...}] on failure) while every other in-proc tool returns a dict.
The loop's is_error line called .get on it and crashed the whole agent
stream ('list' object has no attribute 'get') — the full-loop tests below
pin the real shape end to end; the original suite only exercised
_run_claude_web_search in isolation and missed the seam.
"""

import json
import unittest
from copy import deepcopy
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.agent.agents import basic_memory_agent as bma
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)


def _agent(cfg=None):
    agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
    agent._turn_inproc_calls = []
    agent._web_tools_config = cfg or {
        "provider": "brave",
        "api_key": "test-key",
    }
    return agent


class SchemaTests(unittest.TestCase):
    def test_builder_shape(self):
        (tool,) = BasicMemoryAgent._build_web_search_tool_claude()
        self.assertEqual(tool["name"], "web_search")
        self.assertEqual(tool["input_schema"]["required"], ["query"])
        self.assertIn("query", tool["input_schema"]["properties"])


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_to_web_tools_with_config(self):
        agent = _agent()
        fake = AsyncMock(return_value=[{"title": "t", "url": "u", "snippet": "s"}])
        with patch.object(bma, "web_search", fake):
            marker, result = await agent._run_claude_web_search(
                {"query": "ゼルダ 祠"}, {"left": 5}
            )
        fake.assert_awaited_once_with(
            "ゼルダ 祠", provider="brave", api_key="test-key", max_results=5
        )
        self.assertIn("Web検索", marker)
        self.assertEqual(result[0]["title"], "t")
        self.assertEqual(agent._turn_inproc_calls, ["web_search"])

    async def test_budget_decrements_and_exhausts(self):
        agent = _agent()
        budget = {"left": 1}
        fake = AsyncMock(return_value=[{"title": "t"}])
        with patch.object(bma, "web_search", fake):
            await agent._run_claude_web_search({"query": "a"}, budget)
            self.assertEqual(budget["left"], 0)
            marker, result = await agent._run_claude_web_search({"query": "b"}, budget)
        self.assertIsNone(marker)
        self.assertEqual(result, {"error": "web search limit reached this turn"})
        fake.assert_awaited_once()  # second call never reached the backend


class _FakeClaudeLLM(ClaudeAsyncLLM):
    def __init__(self, rounds):
        self.model = "claude-opus-5"
        self.thinking_replay_max_tokens = 2000
        self._enable_web_search = True
        self._max_web_searches = 3
        self._rounds = [list(r) for r in rounds]

    async def chat_completion(self, messages, system=None, tools=None):
        for event in self._rounds.pop(0):
            yield deepcopy(event)


def _search_turn_rounds():
    tool_use = {
        "type": "tool_use",
        "id": "s1",
        "name": "web_search",
        "input": {"query": "q"},
    }
    return [
        [
            {"type": "tool_use_complete", "data": deepcopy(tool_use)},
            {
                "type": "assistant_message_complete",
                "data": {"role": "assistant", "content": [deepcopy(tool_use)]},
            },
            {"type": "message_stop"},
        ],
        [
            {"type": "text_delta", "text": "done"},
            {
                "type": "assistant_message_complete",
                "data": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                },
            },
            {"type": "message_stop"},
        ],
    ]


class FullLoopShapeTests(unittest.IsolatedAsyncioTestCase):
    """Drive the REAL tool loop with web_tools' actual return shapes."""

    async def _run(self, search_return):
        llm = _FakeClaudeLLM(_search_turn_rounds())
        agent = _agent()
        agent._llm = llm
        agent._memory = []
        agent._memory_manager = None
        agent._build_system_for_llm = lambda: "system"
        agent._check_stateful_claims = lambda text: None
        agent._model_health_enabled = False
        agent._steam_enabled = False
        agent._memory_tools_enabled = False
        agent._tool_executor = None
        agent._mcp_tool_marker = lambda name: ""

        fake = AsyncMock(return_value=search_return)
        with patch.object(bma, "web_search", fake):
            out = []
            async for item in agent._claude_tool_interaction_loop(
                [{"role": "user", "content": "search it"}],
                [{"name": "web_search"}],
            ):
                out.append(item)
        return agent, out

    async def test_success_list_shape_survives_the_loop(self):
        results = [{"title": "t", "url": "u", "snippet": "s"}]
        agent, out = await self._run(results)
        self.assertIn("done", out)
        protocol = agent._memory[-1]["claude_protocol"]
        result_block = protocol[1]["content"][0]
        self.assertFalse(result_block["is_error"])
        self.assertEqual(json.loads(result_block["content"]), results)

    async def test_error_list_shape_flags_is_error(self):
        agent, out = await self._run([{"error": "search failed: HTTP 429"}])
        self.assertIn("done", out)
        protocol = agent._memory[-1]["claude_protocol"]
        self.assertTrue(protocol[1]["content"][0]["is_error"])


class InprocErrorShapeTests(unittest.TestCase):
    def test_dict_and_list_shapes(self):
        f = BasicMemoryAgent._inproc_result_is_error
        self.assertTrue(f({"error": "x"}))
        self.assertTrue(f({"status": "error"}))
        self.assertFalse(f({"status": "ok", "data": 1}))
        self.assertTrue(f([{"error": "boom"}]))
        self.assertFalse(f([{"title": "t"}, {"title": "u"}]))
        self.assertFalse(f([]))
        self.assertFalse(f("plain string"))


if __name__ == "__main__":
    unittest.main()
