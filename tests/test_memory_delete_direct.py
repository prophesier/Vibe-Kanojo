"""memory_delete fast path: user's own message asking for deletion skips the
confirm round-trip.

あさひ 08-09: the two-phase confirm doubled the work, and memory-edit turns
often think past the replay cap — the follow-up turn then lacks the fact_id
and the whole search-stage-confirm dance restarts. Direct execution is gated
on deletion-EXPLICIT phrasing only; generic approvals (はい/OK) still go
through show-and-confirm so a stray "OK" cannot authorize a deletion the
model decided on by itself.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent


def _agent(latest_user_text, importance="low"):
    agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
    agent._memory = [{"role": "user", "content": latest_user_text}]
    agent._pending_memory_deletes = {}
    agent._memory_manager = SimpleNamespace(
        find_fact=lambda fid: {"fact": "テスト記憶", "importance": importance},
        delete_fact_manual=AsyncMock(return_value={"status": "ok", "id": fid_ref[0]}),
    )
    return agent


fid_ref = ["f123"]


class DirectDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def _flow(self, text, importance="low", args=None):
        agent = _agent(text, importance)
        result = await agent._memory_delete_flow(args or {"fact_id": "f123"})
        return agent, result

    async def test_explicit_ja_delete_executes_directly(self):
        agent, result = await self._flow("あの散歩の記憶、消しておいて")
        agent._memory_manager.delete_fact_manual.assert_awaited_once_with("f123")
        self.assertEqual(result["status"], "ok")

    async def test_explicit_zh_delete_executes_directly(self):
        agent, result = await self._flow("把那条记忆删掉吧")
        agent._memory_manager.delete_fact_manual.assert_awaited_once()

    async def test_yappari_keshite_matches(self):
        agent, _ = await self._flow("やっぱり消して")
        agent._memory_manager.delete_fact_manual.assert_awaited_once()

    async def test_torikeshite_is_cancel_not_delete(self):
        agent, result = await self._flow("さっきの操作、取り消して")
        agent._memory_manager.delete_fact_manual.assert_not_awaited()
        self.assertEqual(result["status"], "pending_approval")

    async def test_generic_ok_stays_two_phase(self):
        agent, result = await self._flow("OK、それでいこう")
        agent._memory_manager.delete_fact_manual.assert_not_awaited()
        self.assertEqual(result["status"], "pending_approval")

    async def test_user_tier_guard_beats_fast_path(self):
        agent, result = await self._flow("その記憶消して", importance="user")
        agent._memory_manager.delete_fact_manual.assert_not_awaited()
        self.assertEqual(result["status"], "error")

    async def test_two_phase_confirm_still_works_with_generic_approval(self):
        agent = _agent("これ要らない気がする")
        first = await agent._memory_delete_flow({"fact_id": "f123"})
        self.assertEqual(first["status"], "pending_approval")
        # He answers with a generic approval; model re-calls with confirmed.
        agent._memory[-1] = {"role": "user", "content": "はい"}
        second = await agent._memory_delete_flow({"fact_id": "f123", "confirmed": True})
        agent._memory_manager.delete_fact_manual.assert_awaited_once_with("f123")
        self.assertEqual(second["status"], "ok")


if __name__ == "__main__":
    unittest.main()
