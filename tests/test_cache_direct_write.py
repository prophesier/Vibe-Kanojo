"""Direct-write cache breakpoint: the outgoing user message is marked.

あさひ 08-08: every new token used to ride uncached (fresh 1x) and only enter
the cache on the NEXT request's write (2x) — 3x total. Moving the breakpoint
onto the fresh user message makes it write on its own request (2x total).
Inside the tool loop the marker migrates per round when the round is judged
replay-stable in-round (08-09, per-round semantics — see
test_per_round_replay); unstable rounds stop the migration so bytes that
change post-turn are never cached.
"""

import base64
import unittest
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent


def _agent(memory=None):
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
    # Banner already placed (the steady-state case); the banner-pending
    # first-turn case overrides these in its own test.
    agent._current_session_banner_added = True
    agent._memory_manager = None
    agent._added = []
    agent._add_message = lambda text, role, **kw: agent._added.append((role, text))
    return agent


def _input(text="", images=None):
    return SimpleNamespace(text=text, images=images, metadata=None)


class DirectWriteBreakpointTests(unittest.TestCase):
    def test_fresh_user_message_carries_breakpoint(self):
        agent = _agent()
        agent._pending_rag_block = "【日記】昔の話"
        out = agent._to_messages(_input("こんにちは"))
        last = out[-1]
        self.assertEqual(last["role"], "user")
        self.assertIsInstance(last["content"], list)
        tail = last["content"][-1]
        self.assertIn("cache_control", tail)
        # RAG block folded above the user text — the marked bytes are the
        # exact stored/sent payload.
        self.assertIn("【日記】昔の話", tail["text"])
        self.assertIn("こんにちは", tail["text"])
        self.assertEqual(agent._added, [("user", tail["text"])])

    def test_history_protocol_entry_not_marked(self):
        history = [
            {"role": "user", "content": "前の入力"},
            {
                "role": "assistant",
                "content": "前の返事",
                "claude_protocol": [{"role": "assistant", "content": []}],
            },
        ]
        agent = _agent(memory=history)
        out = agent._to_messages(_input("次の入力"))
        self.assertEqual(len(out), 3)
        # Only the fresh user message is marked; history stays untouched.
        self.assertNotIsInstance(out[0]["content"], list)
        self.assertNotIsInstance(out[1]["content"], list)
        self.assertIn("cache_control", out[2]["content"][-1])

    def test_no_user_content_falls_back_to_newest_safe_history(self):
        history = [
            {"role": "user", "content": "話しかけた"},
            {
                "role": "assistant",
                "content": "返事",
                "claude_protocol": [{"role": "assistant", "content": []}],
            },
        ]
        agent = _agent(memory=history)
        out = agent._to_messages(_input(""))
        self.assertEqual(len(out), 2)
        # Protocol entry skipped → the preceding user message takes the marker.
        self.assertIn("cache_control", out[0]["content"][-1])

    def test_image_turn_marks_leading_text_block(self):
        # The image itself is never cached (it exists for exactly one
        # request), but the leading text payload is stored verbatim — so the
        # marker sits on the text block BEFORE the image (あさひ 08-09): the
        # whole prior prefix direct-writes and only the image rides fresh.
        # The old whole-message skip parked the marker on an already-cached
        # boundary and re-paid the previous turn's protocol replay as fresh.
        history = [{"role": "user", "content": "前の入力"}]
        agent = _agent(memory=history)
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 16).decode()
        data_url = f"data:image/png;base64,{png}"
        out = agent._to_messages(
            _input("見て", images=[SimpleNamespace(data=data_url)])
        )
        image_msg = out[-1]
        self.assertEqual(image_msg["content"][-1]["type"], "image_url")
        self.assertNotIn("cache_control", image_msg["content"][-1])
        self.assertIn("cache_control", image_msg["content"][0])
        # stored == sent for the cached span: the marked text block is the
        # exact stored payload.
        self.assertEqual(agent._added, [("user", image_msg["content"][0]["text"])])
        # No fallback marking on older history.
        self.assertEqual(out[0], {"role": "user", "content": "前の入力"})

    def test_first_turn_banner_injected_marked_and_stored_verbatim(self):
        # A fresh session's first user message carries the session banner in
        # the OUTGOING payload (あさひ 08-09: intended behavior), keeps its
        # breakpoint, and stores the exact sent bytes — the store-time-only
        # banner injection broke stored == sent and rewrote the whole span
        # on turn 2 (66393w/61678r case).
        agent = _agent()
        agent._current_session_banner_added = False
        agent._memory_manager = SimpleNamespace(
            _current_session_uid="2026-08-09_05-15-25_deadbeef"
        )
        out = agent._to_messages(_input("おはよう"))
        tail = out[-1]["content"][-1]
        self.assertIn("cache_control", tail)
        self.assertTrue(
            tail["text"].index("2026-08-09") < tail["text"].index("おはよう")
        )
        self.assertTrue(agent._current_session_banner_added)
        # stored == sent, banner included — _add_message must not re-inject.
        self.assertEqual(agent._added, [("user", tail["text"])])

    def test_skip_memory_turn_does_not_consume_banner(self):
        agent = _agent()
        agent._current_session_banner_added = False
        agent._memory_manager = SimpleNamespace(
            _current_session_uid="2026-08-09_05-15-25_deadbeef"
        )
        data = _input("キャッシュ保温")
        data.metadata = {"skip_memory": True}
        out = agent._to_messages(data)
        self.assertNotIn("2026-08-09_05-15-25", out[-1]["content"][-1]["text"])
        self.assertFalse(agent._current_session_banner_added)

    def test_banner_already_added_marks_normally(self):
        agent = _agent()
        agent._memory_manager = SimpleNamespace(
            _current_session_uid="2026-08-09_05-15-25_deadbeef"
        )
        out = agent._to_messages(_input("二言目"))
        self.assertIn("cache_control", out[-1]["content"][-1])
        self.assertNotIn("【", out[-1]["content"][-1]["text"])

    def test_image_only_first_turn_leaves_everything_unmarked(self):
        agent = _agent()
        png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 16).decode()
        out = agent._to_messages(
            _input("", images=[SimpleNamespace(data=f"data:image/png;base64,{png}")])
        )
        self.assertEqual(len(out), 1)
        for block in out[0]["content"]:
            self.assertNotIn("cache_control", block)


class BannerModelTagTests(unittest.TestCase):
    """The current-session banner names the running model (あさひ 08-10:
    facts can lag a model switch and confuse the character). In-memory only —
    the banner never reaches disk. Past-session banners stay model-free."""

    def test_current_banner_carries_model(self):
        agent = _agent()
        agent._llm = SimpleNamespace(model="claude-opus-5")
        banner = agent._session_header_text(
            "2026-08-10_12-00-00_deadbeef", is_current=True
        )
        self.assertIn("モデル: claude-opus-5", banner)
        self.assertTrue(banner.endswith("】"))

    def test_past_banner_is_model_free(self):
        agent = _agent()
        agent._llm = SimpleNamespace(model="claude-opus-5")
        banner = agent._session_header_text(
            "2026-08-01_12-00-00_deadbeef", is_current=False
        )
        self.assertNotIn("モデル", banner)

    def test_no_llm_degrades_to_plain_banner(self):
        agent = _agent()  # __new__-built, no _llm attribute at all
        banner = agent._session_header_text(
            "2026-08-10_12-00-00_deadbeef", is_current=True
        )
        self.assertNotIn("モデル", banner)
        self.assertIn("現在進行中のセッション開始", banner)

    def test_first_turn_outgoing_payload_carries_model_and_stores_verbatim(self):
        agent = _agent()
        agent._llm = SimpleNamespace(model="claude-opus-5")
        agent._current_session_banner_added = False
        agent._memory_manager = SimpleNamespace(
            _current_session_uid="2026-08-10_12-00-00_deadbeef"
        )
        out = agent._to_messages(_input("おはよう"))
        sent = out[-1]["content"][-1]["text"]
        self.assertIn("モデル: claude-opus-5", sent)
        # stored == sent still holds with the model tag in the banner.
        self.assertEqual(agent._added, [("user", sent)])


if __name__ == "__main__":
    unittest.main()
