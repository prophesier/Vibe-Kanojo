import unittest
from copy import deepcopy

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent


MARKER = BasicMemoryAgent._PROTOCOL_TRUNCATION_MARKER
LIMIT = BasicMemoryAgent._PROTOCOL_RESULT_MAX_CHARS


class ProtocolTruncationTests(unittest.TestCase):
    def test_long_string_result_is_capped(self):
        protocol = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "x" * (LIMIT + 500),
                        "is_error": False,
                    }
                ],
            }
        ]
        out = BasicMemoryAgent._truncate_protocol_tool_results(protocol)
        block = out[0]["content"][0]
        self.assertEqual(block["content"], "x" * LIMIT + MARKER)
        self.assertEqual(block["tool_use_id"], "t1")
        self.assertFalse(block["is_error"])

    def test_short_string_result_untouched(self):
        protocol = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            }
        ]
        out = BasicMemoryAgent._truncate_protocol_tool_results(protocol)
        self.assertEqual(out[0]["content"][0]["content"], "ok")

    def test_list_content_text_budget_and_image_stub(self):
        protocol = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "a" * (LIMIT - 10)},
                            {"type": "text", "text": "b" * 100},
                            {"type": "image", "source": {"type": "base64"}},
                        ],
                    }
                ],
            }
        ]
        out = BasicMemoryAgent._truncate_protocol_tool_results(protocol)
        blocks = out[0]["content"][0]["content"]
        self.assertEqual(blocks[0]["text"], "a" * (LIMIT - 10))
        self.assertEqual(blocks[1]["text"], "b" * 10 + MARKER)
        self.assertEqual(
            blocks[2], {"type": "text", "text": "[non-text tool output omitted]"}
        )

    def test_assistant_messages_never_touched(self):
        assistant = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "sig"},
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                {
                    "type": "web_fetch_tool_result",
                    "tool_use_id": "s1",
                    "content": {"url": "u", "content": "y" * (LIMIT + 999)},
                },
            ],
        }
        protocol = [deepcopy(assistant)]
        out = BasicMemoryAgent._truncate_protocol_tool_results(protocol)
        self.assertEqual(out[0], assistant)

    def test_original_protocol_not_mutated(self):
        original = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "x" * (LIMIT + 5),
                    }
                ],
            }
        ]
        snapshot = deepcopy(original)
        BasicMemoryAgent._truncate_protocol_tool_results(original)
        self.assertEqual(original, snapshot)


class ErrorPathCommitTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_text_survives_stream_error(self):
        class _ErrLLM:
            async def chat_completion(self, messages, system=None, tools=None):
                yield {"type": "text_delta", "text": "前半だけ話した。"}
                yield {"type": "error", "message": "boom"}

        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        agent._llm = _ErrLLM()
        agent._memory = []
        agent._memory_manager = None
        agent._build_system_for_llm = lambda: "system"
        agent._check_stateful_claims = lambda text: None

        output = [
            item
            async for item in agent._claude_tool_interaction_loop(
                [{"role": "user", "content": "hi"}], []
            )
        ]
        self.assertEqual(output, ["前半だけ話した。", "[Error from LLM: boom]"])
        self.assertEqual(
            agent._memory[-1],
            {"role": "assistant", "content": "前半だけ話した。"},
        )


if __name__ == "__main__":
    unittest.main()
