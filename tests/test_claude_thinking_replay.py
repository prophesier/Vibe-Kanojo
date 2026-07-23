import unittest
from copy import deepcopy
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
)


THINKING_1 = {
    "type": "thinking",
    "thinking": "choose a tool",
    "signature": "signature-1",
}
THINKING_2 = {
    "type": "thinking",
    "thinking": "use the result",
    "signature": "signature-2",
}
TOOL_USE = {
    "type": "tool_use",
    "id": "tool-1",
    "name": "memory_search",
    "input": {"query": "test"},
}
TOOL_RESULT = {
    "type": "tool_result",
    "tool_use_id": "tool-1",
    "content": [{"type": "text", "text": "result"}],
}
FINAL_TEXT = {"type": "text", "text": "final answer"}


class _FakeClaudeLLM(ClaudeAsyncLLM):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append(deepcopy(messages))
        events = self.responses.pop(0)
        for event in events:
            yield deepcopy(event)


class _FakeToolExecutor:
    async def execute_tools(self, tool_calls, caller_mode):
        yield {"type": "final_tool_results", "results": [deepcopy(TOOL_RESULT)]}


class _FakeBlock:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, exclude_none=True):
        return deepcopy(self.payload)


class ClaudeThinkingReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_preserves_protocol_boundaries_across_turns(self):
        first_assistant = {
            "role": "assistant",
            "content": [deepcopy(THINKING_1), deepcopy(TOOL_USE)],
        }
        final_assistant = {
            "role": "assistant",
            "content": [deepcopy(THINKING_2), deepcopy(FINAL_TEXT)],
        }
        llm = _FakeClaudeLLM(
            [
                [
                    {"type": "thinking_complete", "data": deepcopy(THINKING_1)},
                    {
                        "type": "tool_use_complete",
                        "data": {
                            "id": "tool-1",
                            "name": "memory_search",
                            "input": {"query": "test"},
                        },
                    },
                    {
                        "type": "assistant_message_complete",
                        "data": deepcopy(first_assistant),
                    },
                    {"type": "message_stop"},
                ],
                [
                    {"type": "thinking_complete", "data": deepcopy(THINKING_2)},
                    {"type": "text_delta", "text": "final answer"},
                    {
                        "type": "assistant_message_complete",
                        "data": deepcopy(final_assistant),
                    },
                    {"type": "message_stop"},
                ],
            ]
        )

        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._llm = llm
        agent._memory = []
        agent._memory_manager = None
        agent._memory_tools_enabled = False
        agent._model_health_enabled = False
        agent._steam_enabled = False
        agent._tool_executor = _FakeToolExecutor()
        agent._mcp_tool_marker = lambda name: ""
        agent._build_system_for_llm = lambda: "system"
        agent._check_stateful_claims = lambda text: None

        output = [
            item
            async for item in agent._claude_tool_interaction_loop(
                [{"role": "user", "content": "test memory search"}],
                [{"name": "memory_search"}],
            )
        ]

        self.assertEqual(output, ["final answer"])
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1][-2], first_assistant)
        self.assertEqual(
            llm.calls[1][-1],
            {"role": "user", "content": [TOOL_RESULT]},
        )

        stored = agent._memory[-1]
        protocol = stored["claude_protocol"]
        self.assertEqual(
            [message["role"] for message in protocol],
            ["assistant", "user", "assistant"],
        )
        self.assertEqual(
            [block["type"] for block in protocol[0]["content"]],
            ["thinking", "tool_use"],
        )
        self.assertEqual(
            [block["type"] for block in protocol[2]["content"]],
            ["thinking", "text"],
        )

        converted = llm._convert_messages_format(
            [
                {"role": "user", "content": "test memory search"},
                stored,
                {"role": "user", "content": "next turn"},
            ]
        )
        self.assertEqual(
            [message["role"] for message in converted],
            ["user", "assistant", "user", "assistant", "user"],
        )
        self.assertEqual(
            [block["type"] for block in converted[1]["content"]],
            ["thinking", "tool_use"],
        )
        self.assertEqual(
            [block["type"] for block in converted[3]["content"]],
            ["thinking", "text"],
        )

    def test_cache_breakpoint_does_not_modify_exact_assistant_protocol(self):
        llm = _FakeClaudeLLM([])
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._llm = llm
        protocol = [
            {
                "role": "assistant",
                "content": [deepcopy(THINKING_2), deepcopy(FINAL_TEXT)],
            }
        ]
        messages = [
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": "final answer",
                "claude_protocol": deepcopy(protocol),
            },
        ]

        marked = agent._attach_cache_breakpoint(messages)

        self.assertEqual(messages[0]["content"], "question")
        self.assertEqual(marked[1]["claude_protocol"], protocol)
        self.assertEqual(
            marked[0]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )

    def test_legacy_reconstructed_thinking_is_dropped(self):
        llm = _FakeClaudeLLM([])
        converted = llm._convert_messages_format(
            [
                {
                    "role": "assistant",
                    "content": "safe visible text",
                    "thinking_blocks": [deepcopy(THINKING_1), deepcopy(THINKING_2)],
                }
            ]
        )

        self.assertEqual(
            converted,
            [{"role": "assistant", "content": "safe visible text"}],
        )

    def test_sdk_snapshot_serialization_keeps_all_block_types(self):
        blocks = [
            _FakeBlock(THINKING_1),
            _FakeBlock(
                {
                    "type": "server_tool_use",
                    "id": "server-tool-1",
                    "name": "web_search",
                    "input": {"query": "test"},
                }
            ),
            _FakeBlock(
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "server-tool-1",
                    "content": [],
                }
            ),
            _FakeBlock(
                {
                    "type": "redacted_thinking",
                    "data": "opaque-redacted-data",
                }
            ),
            _FakeBlock(FINAL_TEXT),
        ]

        replay = ClaudeAsyncLLM._assistant_message_for_replay(
            SimpleNamespace(content=blocks)
        )

        self.assertEqual(replay["role"], "assistant")
        self.assertEqual(
            [block["type"] for block in replay["content"]],
            [
                "thinking",
                "server_tool_use",
                "web_search_tool_result",
                "redacted_thinking",
                "text",
            ],
        )


if __name__ == "__main__":
    unittest.main()
