import asyncio
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent


class RagParallelTests(unittest.IsolatedAsyncioTestCase):
    async def test_diary_and_facts_retrieval_run_concurrently(self):
        """Handshake: each leg only finishes after the OTHER leg has started.
        Serial execution deadlocks (caught by the timeout); only concurrent
        execution can pass."""
        agent = BasicMemoryAgent.__new__(BasicMemoryAgent)
        diary_started = asyncio.Event()
        facts_started = asyncio.Event()
        order = []

        async def fake_diary(input_data):
            diary_started.set()
            await asyncio.wait_for(facts_started.wait(), timeout=1.0)
            order.append("diary")

        async def fake_facts(input_data):
            facts_started.set()
            await asyncio.wait_for(diary_started.wait(), timeout=1.0)
            order.append("facts")

        agent._maybe_inject_diary_rag = fake_diary
        agent._maybe_inject_facts_rag = fake_facts

        await asyncio.wait_for(agent._inject_memory_rags(object()), timeout=2.0)

        self.assertEqual(sorted(order), ["diary", "facts"])


if __name__ == "__main__":
    unittest.main()
