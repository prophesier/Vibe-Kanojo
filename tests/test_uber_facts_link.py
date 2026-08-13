"""uber_search × facts two-wave recall (あさひ 08-13).

Wave B string-matches result store titles against fact texts (three levels:
full title / adjacent pairs / single tokens with eval-tuned drops); wave A is
a no-judge semantic pass on the search keyword. Covers: the one-char store
name (the「◯◯丼 玄」pattern), branch mismatch, keyword-overlap drop,
ascii-short drop, prefix fallback, B-first merge with the reserved A slot, the
truncation that starts counting after the facts section, and the session
dedup ledger. All offline.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.memory.persistent_memory import PersistentMemoryManager

# Fictional stores mirroring the structural cases from the (private) 08-13
# eval: a one-char store name, a branch mismatch, a partially-recorded chain
# name, an ascii-short trap, and store-independent preference facts.
FACTS = [
    "ユーザーはUber Eatsの「うな丼処 玄」の相盛り丼を気に入っている",
    "ユーザーはUber Eatsで牛丼のヤマト 東店の牛カルビ弁当を高評価している",
    "ユーザーはケンタッキー（駅前店）を安定枠として繰り返し利用している",
    "ユーザーはUber Eatsで肉旨精肉店のカルビステーキ重を試し美味しいと評価した",
    "ユーザーはHeroes of Might and Magic系のゲームをよくプレイする",
    "ユーザーは玄米よりも白ご飯を好む",
    "ユーザーは日式中華料理は高くて美味しくないと感じている",
]


def _mgr(facts=FACTS, index=None):
    tmp = os.path.join(tempfile.mkdtemp(), "facts.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            [{"fact": t, "importance": "low", "updated": "2026-08-01"} for t in facts],
            f,
            ensure_ascii=False,
        )
    m = PersistentMemoryManager.__new__(PersistentMemoryManager)
    m._facts_path = tmp
    m._facts_index = index
    m._facts_rag_cfg = SimpleNamespace(similarity_threshold=0.6, lexical_weight=0.5)
    return m


def _fid(text):
    return PersistentMemoryManager._fact_id(text)


class WaveBMatcherTests(unittest.IsolatedAsyncioTestCase):
    async def _hits(self, titles, query="", mgr=None, cap=5):
        mgr = mgr or _mgr()
        return await mgr.uber_related_facts(query, titles, set(), cap)

    async def test_one_char_store_name_matches_via_full_title(self):
        # Real Uber titles do end in a single-char name token (08-13 probe);
        # the full-title level carries it.
        hits = await self._hits(["うな丼処 玄"], query="鰻")
        self.assertIn(_fid(FACTS[0]), [h["id"] for h in hits])

    async def test_one_char_token_alone_never_matches(self):
        # 玄 must not reach 玄米 facts on its own.
        hits = await self._hits(["謎の丼屋 玄"], query="丼")
        self.assertNotIn(_fid(FACTS[5]), [h["id"] for h in hits])

    async def test_branch_mismatch_still_hits(self):
        hits = await self._hits(["牛丼のヤマト 西店"], query="牛丼")
        self.assertIn(_fid(FACTS[1]), [h["id"] for h in hits])

    async def test_query_overlap_and_ascii_short_tokens_dropped(self):
        hits = await self._hits(
            ["ハラペコステーキ 南店", "HERO'S ステーキハウス 北口店 Hero's"],
            query="ステーキ",
        )
        ids = [h["id"] for h in hits]
        self.assertNotIn(_fid(FACTS[3]), ids)  # カルビステーキ fact stays out
        self.assertNotIn(_fid(FACTS[4]), ids)  # Hero ⊄ Heroes

    async def test_prefix_fallback_for_partial_recordings(self):
        hits = await self._hits(["ケンタッキーフライドチキン 駅前店"], query="チキン")
        self.assertIn(_fid(FACTS[2]), [h["id"] for h in hits])

    async def test_exclude_ids_respected(self):
        mgr = _mgr()
        hits = await mgr.uber_related_facts(
            "鰻", ["うな丼処 玄"], {_fid(FACTS[0])}, 5
        )
        self.assertEqual(hits, [])


class MergeTests(unittest.IsolatedAsyncioTestCase):
    async def test_b_first_and_reserved_a_slot(self):
        # Wave A returns the preference fact; wave B fills the cap → the top
        # A hit still keeps one guaranteed slot (あさひ: 偏好项与店名无关).
        index = SimpleNamespace(
            retrieve=AsyncMock(return_value=([{"id": _fid(FACTS[6])}], []))
        )
        mgr = _mgr(index=index)
        hits = await mgr.uber_related_facts(
            "中華",
            ["うな丼処 玄", "牛丼のヤマト 西店"],
            set(),
            2,
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["via"], "store")
        self.assertEqual(hits[-1]["id"], _fid(FACTS[6]))
        self.assertEqual(hits[-1]["via"], "topic")

    async def test_a_fills_when_b_underflows(self):
        index = SimpleNamespace(
            retrieve=AsyncMock(return_value=([{"id": _fid(FACTS[6])}], []))
        )
        mgr = _mgr(index=index)
        hits = await mgr.uber_related_facts("中華", [], set(), 3)
        self.assertEqual([h["id"] for h in hits], [_fid(FACTS[6])])


SEARCH_TEXT = (
    "「鰻」の検索結果（3件・飲食店）:\n"
    "- ［PR］うなぎ・ふぐ料理 鰻仙 中央店  ★4.4(200+件) · 配送¥100 · 22分\n"
    "    store_uuid: aaa\n"
    "- うな丼処 玄  ★4.7(97件) · 配送¥50(Uber One) · 11分\n"
    "    store_uuid: bbb\n"
    "- 名代 うな政 東口店  配送¥50 · 14分\n"
    "    store_uuid: ccc\n"
)


class AgentSeamTests(unittest.IsolatedAsyncioTestCase):
    def _agent(self, hits):
        a = BasicMemoryAgent.__new__(BasicMemoryAgent)
        a._session_injected_fact_ids = set()
        mgr = SimpleNamespace(
            uber_related_facts=AsyncMock(return_value=hits),
            injected_fact_ids=lambda: set(),
        )
        a._memory_manager = mgr
        return a

    def test_title_parsing(self):
        titles = BasicMemoryAgent._uber_store_titles(SEARCH_TEXT)
        self.assertEqual(
            titles,
            [
                "うなぎ・ふぐ料理 鰻仙 中央店",
                "うな丼処 玄",
                "名代 うな政 東口店",
            ],
        )

    async def test_prepend_ledger_and_cap_from_limit(self):
        hits = [
            {"id": "f1", "fact": "玄の丼が好き", "date": "2026-08-09", "via": "store"}
        ]
        agent = self._agent(hits)
        results = [{"type": "tool_result", "tool_use_id": "t1", "content": SEARCH_TEXT}]
        calls = [
            {"id": "t1", "name": "uber_search", "input": {"keyword": "鰻", "limit": 7}}
        ]
        await agent._augment_uber_search_results(results, calls)
        content = results[0]["content"]
        self.assertTrue(content.startswith(BasicMemoryAgent._UBER_FACTS_HEADER))
        self.assertIn(BasicMemoryAgent._UBER_FACTS_END, content)
        self.assertIn("玄の丼が好き", content)
        self.assertIn("「鰻」の検索結果", content)
        self.assertEqual(agent._session_injected_fact_ids, {"f1"})
        call = agent._memory_manager.uber_related_facts.await_args
        self.assertEqual(call.args[3], 4)  # cap = ceil(7/2)

    async def test_injected_ids_excluded_on_next_search(self):
        agent = self._agent([])
        agent._session_injected_fact_ids = {"seen"}
        results = [{"type": "tool_result", "tool_use_id": "t1", "content": SEARCH_TEXT}]
        calls = [{"id": "t1", "name": "uber_search", "input": {"keyword": "鰻"}}]
        await agent._augment_uber_search_results(results, calls)
        self.assertIn(
            "seen", agent._memory_manager.uber_related_facts.await_args.args[2]
        )

    async def test_openai_shape_and_non_uber_untouched(self):
        hits = [{"id": "f1", "fact": "x", "date": "", "via": "store"}]
        agent = self._agent(hits)
        results = [
            {"role": "tool", "tool_call_id": "t1", "content": SEARCH_TEXT},
            {"role": "tool", "tool_call_id": "t2", "content": "曲を再生した"},
        ]
        calls = [
            {"id": "t1", "name": "uber_search", "input": {"keyword": "鰻"}},
            {"id": "t2", "name": "play_music", "input": {}},
        ]
        await agent._augment_uber_search_results(results, calls)
        self.assertIn(BasicMemoryAgent._UBER_FACTS_HEADER, results[0]["content"])
        self.assertEqual(results[1]["content"], "曲を再生した")

    async def test_error_results_skipped(self):
        agent = self._agent([{"id": "f1", "fact": "x", "date": "", "via": "store"}])
        results = [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": SEARCH_TEXT,
                "is_error": True,
            }
        ]
        calls = [{"id": "t1", "name": "uber_search", "input": {"keyword": "鰻"}}]
        await agent._augment_uber_search_results(results, calls)
        self.assertNotIn(BasicMemoryAgent._UBER_FACTS_HEADER, results[0]["content"])


class FactsSectionTruncationTests(unittest.TestCase):
    def test_budget_counts_from_after_the_facts_section(self):
        limit = BasicMemoryAgent._PROTOCOL_RESULT_MAX_CHARS
        section = (
            f"{BasicMemoryAgent._UBER_FACTS_HEADER}\n- [2026-08-01] 事実\n"
            f"{BasicMemoryAgent._UBER_FACTS_END}"
        )
        long_tail = "\n" + "x" * (limit + 300)
        block = {
            "type": "tool_result",
            "tool_use_id": "t",
            "content": section + long_tail,
        }
        out = BasicMemoryAgent._truncate_tool_result_block(block)
        self.assertTrue(out["content"].startswith(section))
        self.assertIn(BasicMemoryAgent._PROTOCOL_TRUNCATION_MARKER, out["content"])
        # facts survive whole; the tail eats the whole 400-char budget
        self.assertEqual(
            len(out["content"]),
            len(section) + limit + len(BasicMemoryAgent._PROTOCOL_TRUNCATION_MARKER),
        )

    def test_short_tail_stays_identity(self):
        section = (
            f"{BasicMemoryAgent._UBER_FACTS_HEADER}\n- 事実\n"
            f"{BasicMemoryAgent._UBER_FACTS_END}\n短い結果"
        )
        block = {"type": "tool_result", "tool_use_id": "t", "content": section}
        self.assertIs(BasicMemoryAgent._truncate_tool_result_block(block), block)


if __name__ == "__main__":
    unittest.main()
