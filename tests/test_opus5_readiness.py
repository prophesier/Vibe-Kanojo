"""Tests for the Opus-5 readiness batch: client-side web_fetch routing,
oversized-thinking replay cap, safety-refusal cleanup, web_tools extraction
fixes."""

import json
import unittest
from copy import deepcopy

import httpx
from anthropic import AsyncAnthropic

import src.open_llm_vtuber.agent.agents.basic_memory_agent as bma
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)
from src.open_llm_vtuber.web_tools import _extract_main_text


THINKING = {"type": "thinking", "thinking": "ponder", "signature": "sig"}


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
    return agent


class NativeFetchRemovalTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_web_fetch_never_injected(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                400, json={"type": "error", "error": {"message": "stop"}}
            )

        llm = ClaudeAsyncLLM(
            model="claude-opus-5",
            llm_api_key="sk-test",
            enable_web_search=True,
            enable_web_fetch=True,
        )
        llm.client = AsyncAnthropic(
            api_key="sk-test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )
        try:
            async for _ in llm.chat_completion([{"role": "user", "content": "hi"}]):
                pass
        except Exception:
            pass

        tools = captured["body"].get("tools", [])
        types = [t.get("type", "") for t in tools]
        self.assertTrue(
            any(t.startswith("web_search") for t in types), f"search missing: {types}"
        )
        self.assertFalse(
            any(str(t).startswith("web_fetch") for t in types),
            f"native web_fetch must never be injected: {types}",
        )

    def test_replay_cap_resolution(self):
        llm = ClaudeAsyncLLM(
            model="claude-opus-5",
            llm_api_key="sk-test",
            thinking_replay_max_tokens=1234,
        )
        self.assertEqual(llm.thinking_replay_max_tokens, 1234)
        llm2 = ClaudeAsyncLLM(
            model="claude-opus-5",
            llm_api_key="sk-test",
            thinking_replay_max_tokens="garbage",
        )
        self.assertEqual(llm2.thinking_replay_max_tokens, 2000)


class ClaudeClientFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_routed_in_process_with_marker_and_budget(self):
        final = {
            "role": "assistant",
            "content": [deepcopy(THINKING), {"type": "text", "text": "done"}],
        }
        llm = _FakeClaudeLLM(
            [
                [
                    {
                        "type": "tool_use_complete",
                        "data": {
                            "id": "t1",
                            "name": "web_fetch",
                            "input": {"url": "https://example.com/a"},
                        },
                    },
                    {
                        "type": "assistant_message_complete",
                        "data": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "web_fetch",
                                    "input": {"url": "https://example.com/a"},
                                }
                            ],
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
        llm._enable_web_fetch = True
        llm._max_web_fetches = 1
        agent = _bare_agent(llm)

        fetched = {}

        async def fake_fetch(url, *, max_chars=20000):
            fetched["url"] = url
            fetched["max_chars"] = max_chars
            return {"url": url, "title": "T", "text": "body"}

        original = bma.web_fetch
        bma.web_fetch = fake_fetch
        try:
            out = []
            async for item in agent._claude_tool_interaction_loop(
                [{"role": "user", "content": "read it"}], [{"name": "web_fetch"}]
            ):
                out.append(item)
        finally:
            bma.web_fetch = original

        self.assertEqual(fetched["url"], "https://example.com/a")
        self.assertIn("\n🔗 *Web取得: https://example.com/a*\n", out)
        self.assertIn("done", out)

    async def test_fetch_budget_exhaustion(self):
        llm = _FakeClaudeLLM([[]])
        llm._max_web_fetches = 0
        agent = _bare_agent(llm)
        marker, result = await agent._run_claude_web_fetch(
            {"url": "https://example.com"}, {"left": 0}
        )
        self.assertIsNone(marker)
        self.assertIn("limit", result["error"])


class ThinkingReplayCapTests(unittest.IsolatedAsyncioTestCase):
    def _events(self, thinking_tokens):
        final = {
            "role": "assistant",
            "content": [deepcopy(THINKING), {"type": "text", "text": "reply"}],
        }
        return [
            [
                {"type": "thinking_complete", "data": deepcopy(THINKING)},
                {"type": "text_delta", "text": "reply"},
                {"type": "assistant_message_complete", "data": deepcopy(final)},
                {
                    "type": "message_delta",
                    "data": {
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {
                            "output_tokens": thinking_tokens + 50,
                            "output_tokens_details": {
                                "thinking_tokens": thinking_tokens
                            },
                        },
                    },
                },
                {"type": "message_stop"},
            ]
        ]

    async def _run(self, thinking_tokens, cap):
        llm = _FakeClaudeLLM(self._events(thinking_tokens))
        llm.thinking_replay_max_tokens = cap
        agent = _bare_agent(llm)
        async for _ in agent._claude_tool_interaction_loop(
            [{"role": "user", "content": "hi"}], []
        ):
            pass
        return agent

    async def test_over_cap_turn_not_carried(self):
        agent = await self._run(thinking_tokens=3000, cap=2000)
        self.assertNotIn("claude_protocol", agent._memory[-1])
        self.assertIsNone(agent.pop_thinking_seed())

    async def test_under_cap_turn_carried_with_count(self):
        agent = await self._run(thinking_tokens=500, cap=2000)
        self.assertIn("claude_protocol", agent._memory[-1])
        seed = agent.pop_thinking_seed()
        self.assertEqual(seed["thinking_tokens"], 500)

    async def test_cap_zero_disables(self):
        agent = await self._run(thinking_tokens=99999, cap=0)
        self.assertIn("claude_protocol", agent._memory[-1])

    def test_apply_skips_fat_seeds(self):
        llm = _FakeClaudeLLM([])
        llm.thinking_replay_max_tokens = 2000
        agent = _bare_agent(llm)
        agent._memory = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]
        proto = [{"role": "assistant", "content": [deepcopy(THINKING)]}]
        candidates = [
            (1, {"model": "claude-opus-5", "protocol": proto, "thinking_tokens": 100}),
            (3, {"model": "claude-opus-5", "protocol": proto, "thinking_tokens": 6000}),
        ]
        agent._apply_thinking_seeds(candidates, resuming=False)
        self.assertIn("claude_protocol", agent._memory[1])
        self.assertNotIn("claude_protocol", agent._memory[3])


class RefusalTests(unittest.IsolatedAsyncioTestCase):
    async def test_refusal_cleans_memory_and_disk_and_notifies(self):
        llm = _FakeClaudeLLM(
            [
                [
                    {
                        "type": "refusal",
                        "data": {"category": "cyber", "explanation": None},
                    },
                    {"type": "message_stop"},
                ]
            ]
        )
        agent = _bare_agent(llm)
        agent._memory = [{"role": "user", "content": "poisoned input"}]
        agent._conf_uid = "conf_x"
        agent._history_uid = "hist_y"

        calls = {}

        def fake_pop(conf_uid, history_uid, expected_role):
            calls["args"] = (conf_uid, history_uid, expected_role)
            return True

        original = bma.pop_last_message
        bma.pop_last_message = fake_pop
        try:
            out = []
            async for item in agent._claude_tool_interaction_loop(
                [{"role": "user", "content": "poisoned input"}], []
            ):
                out.append(item)
        finally:
            bma.pop_last_message = original

        self.assertEqual(agent._memory, [], "poisoned user input must be popped")
        self.assertEqual(calls["args"], ("conf_x", "hist_y", "human"))
        self.assertEqual(len(out), 1)
        self.assertIn("安全分類器", out[0])
        self.assertIn("cyber", out[0])


class WebToolsExtractionTests(unittest.TestCase):
    def test_meta_charset_bytes_decode(self):
        # Shift_JIS page declaring its charset only in <meta> — the old
        # resp.text path decoded this as UTF-8 mojibake.
        html = (
            "<html><head><meta charset='shift_jis'><title>テスト</title></head>"
            "<body><p>日本語の本文です。文字化けしないこと。</p></body></html>"
        ).encode("shift_jis")
        title, text = _extract_main_text(html)
        self.assertEqual(title, "テスト")
        self.assertIn("日本語の本文です", text)

    def test_empty_article_falls_back_to_body(self):
        html = (
            "<html><body><article></article>"
            "<div>" + "real body content here. " * 30 + "</div></body></html>"
        )
        _, text = _extract_main_text(html)
        self.assertIn("real body content", text)

    def test_table_content_extracted(self):
        rows = "".join(
            f"<tr><td>cell {i} with meaningful content</td></tr>" for i in range(30)
        )
        html = f"<html><body><table>{rows}</table></body></html>"
        _, text = _extract_main_text(html)
        self.assertIn("cell 3 with meaningful content", text)


if __name__ == "__main__":
    unittest.main()
