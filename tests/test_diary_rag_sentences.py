"""Sentence-level diary RAG (あさひ 08-13 redesign).

Covers the archived test list: stable sentence numbering, diff-injection with
既出 pointers, the 既出 mask reaching the judge prompt, diary-atomic packing
(over-budget stays whole, unpacked diaries stay re-pickable), judge-failure
skip, full-read replay exemption + in-round stability + per-turn cap, fully
injected diaries leaving the candidate pool, and the no-candidate no-op.
All offline — the judge/API layers are mocked.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.input_types import TextSource
from src.open_llm_vtuber.memory.persistent_memory import (
    PersistentMemoryManager,
    _split_sentences,
)
from src.open_llm_vtuber.memory.reranker import MemoryReranker, _compact_sent_ranges


def _fake_input(text="昔の話"):
    return SimpleNamespace(
        texts=[SimpleNamespace(content=text, source=TextSource.INPUT)]
    )


def _agent(hits, sentence_budget=8, ledger=None):
    """Minimal agent wired to a mocked memory manager returning ``hits``."""
    a = BasicMemoryAgent.__new__(BasicMemoryAgent)
    a._memory = []
    a._pending_rag_block = ""
    a._sliding_window_uids = set()
    a._diary_sent_ledger = ledger if ledger is not None else {}
    mgr = MagicMock()
    mgr.diary_rag_active = True
    mgr.diary_rag_config = SimpleNamespace(
        auto_inject=True, rerank_context_turns=0, sentence_budget=sentence_budget
    )
    mgr.injected_diary_uids.return_value = set()
    mgr.retrieve_diary_context = AsyncMock(return_value=(hits, [], []))
    a._memory_manager = mgr
    return a


def _hit(uid, sentences, n_total=6, date="2026-07-02"):
    sents = [f"{uid}の文{i + 1}。" for i in range(n_total)]
    return {
        "uid": uid,
        "date": date,
        "sents": sents,
        "sentences": sentences,
        "reason": "r",
    }


class SentenceNumberingTests(unittest.TestCase):
    def test_split_is_deterministic_and_stable(self):
        text = "今日は公園に行った。楽しかった！\n明日はどうする？また行こう。"
        first = _split_sentences(text)
        self.assertEqual(first, _split_sentences(text))
        self.assertEqual(len(first), 4)
        self.assertEqual(first[0], "今日は公園に行った。")

    def test_compact_ranges(self):
        self.assertEqual(_compact_sent_ranges([1, 2, 3, 5]), "s1-3, s5")
        self.assertEqual(_compact_sent_ranges([4]), "s4")
        self.assertEqual(_compact_sent_ranges([]), "")

    def test_fullwidth_bracket_aside_stays_whole(self):
        # ヒロ's diaries end in （……）asides — pysbd keeps them intact.
        s = _split_sentences("夜は一緒に遊んだ。（……長い一日だったが、良かった。）")
        self.assertEqual(len(s), 2)
        self.assertEqual(s[1], "（……長い一日だったが、良かった。）")

    def test_halfwidth_bracket_with_period_is_merged(self):
        # あさひ 08-13: the remembered landmine — a period inside halfwidth
        # parens used to shatter the span and orphan the closing bracket.
        s = _split_sentences("テストをした。(note: this failed once.) 結果は良好。")
        self.assertEqual(s[0], "テストをした。")
        self.assertEqual(s[1], "(note: this failed once.) 結果は良好。")

    def test_unclosed_bracket_merge_is_capped(self):
        # A typo'd never-closing bracket must not glue the whole diary.
        text = "（開きっぱなし。" + "次の文。" * 8
        s = _split_sentences(text)
        self.assertGreater(len(s), 1)

    def test_short_diary_id(self):
        short = BasicMemoryAgent._short_diary_id
        self.assertEqual(
            short("2026-07-18_14-50-27_fd4dd67a2b544c2aaed629dc4ad30933"),
            "fd4dd67a",
        )
        self.assertEqual(short("d1"), "d1")  # non-conforming → unshortened


class DiffInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_delta_only_with_pointer_to_prior(self):
        ledger = {"d1": {"got": {1, 2}, "total": 6}}
        agent = _agent([_hit("d1", [1, 2, 4])], ledger=ledger)
        await agent._maybe_inject_diary_rag(_fake_input())
        block = agent._pending_rag_block
        self.assertIn("s4: d1の文4。", block)
        self.assertNotIn("s1:", block)  # already in context — never re-sent
        self.assertNotIn("s2:", block)
        self.assertIn("（s1-2 は注入済み）", block)
        # Repeat injection: no id (the first block carried it), 続き label.
        self.assertIn("抜粋・続き", block)
        self.assertNotIn("id:", block)
        self.assertEqual(ledger["d1"]["got"], {1, 2, 4})

    async def test_first_appearance_carries_short_id_and_total(self):
        agent = _agent([_hit("d1", [2, 3])])
        await agent._maybe_inject_diary_rag(_fake_input())
        block = agent._pending_rag_block
        self.assertIn("（id: d1）", block)
        self.assertIn("全6句", block)
        self.assertNotIn("続き", block)

    async def test_all_covered_repick_is_noop(self):
        ledger = {"d1": {"got": {1, 2, 3}, "total": 6}}
        agent = _agent([_hit("d1", [1, 3])], ledger=ledger)
        await agent._maybe_inject_diary_rag(_fake_input())
        self.assertEqual(agent._pending_rag_block, "")
        self.assertEqual(ledger["d1"]["got"], {1, 2, 3})

    async def test_no_candidates_is_noop(self):
        agent = _agent([])
        await agent._maybe_inject_diary_rag(_fake_input())
        self.assertEqual(agent._pending_rag_block, "")
        self.assertEqual(agent._diary_sent_ledger, {})


class PackingTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_diary_over_budget_goes_in_whole(self):
        # 宁多不少: the packing point is a whole diary — never cut inside one.
        agent = _agent([_hit("d1", [1, 2, 3, 4, 5], n_total=5)], sentence_budget=3)
        await agent._maybe_inject_diary_rag(_fake_input())
        for n in range(1, 6):
            self.assertIn(f"s{n}:", agent._pending_rag_block)
        self.assertEqual(agent._diary_sent_ledger["d1"]["got"], {1, 2, 3, 4, 5})

    async def test_budget_stops_next_diary_and_leaves_it_unrecorded(self):
        hits = [_hit("d1", [1, 2, 3]), _hit("d2", [1, 2])]
        agent = _agent(hits, sentence_budget=3)
        await agent._maybe_inject_diary_rag(_fake_input())
        self.assertIn("id: d1", agent._pending_rag_block)
        self.assertNotIn("id: d2", agent._pending_rag_block)
        # d2 was NOT packed → not in the ledger → re-pickable next turn.
        self.assertNotIn("d2", agent._diary_sent_ledger)

    async def test_under_budget_packs_multiple_diaries(self):
        hits = [_hit("d1", [1]), _hit("d2", [2])]
        agent = _agent(hits, sentence_budget=8)
        await agent._maybe_inject_diary_rag(_fake_input())
        self.assertIn("id: d1", agent._pending_rag_block)
        self.assertIn("id: d2", agent._pending_rag_block)
        # Judge relevance order preserved at diary granularity.
        self.assertLess(
            agent._pending_rag_block.index("id: d1"),
            agent._pending_rag_block.index("id: d2"),
        )


class CandidatePoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_fully_injected_diary_excluded_partial_stays(self):
        ledger = {
            "full": {"got": {1, 2, 3}, "total": 3},
            "part": {"got": {1}, "total": 5},
        }
        agent = _agent([], ledger=ledger)
        await agent._maybe_inject_diary_rag(_fake_input())
        call = agent._memory_manager.retrieve_diary_context.await_args
        exclude = call.args[1]
        self.assertIn("full", exclude)
        self.assertNotIn("part", exclude)
        # Partial coverage rides along as the judge's 既出 mask.
        self.assertEqual(call.kwargs["injected_sents"]["part"], {1})


class JudgeSentenceModeTests(unittest.IsolatedAsyncioTestCase):
    def _reranker(self, payload):
        r = MemoryReranker(api_key="x", item_label="日記")
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
            ]
        )
        r._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=resp))
            )
        )
        return r

    async def test_mask_reaches_prompt_and_output_validated(self):
        payload = {
            "relevant": [
                # 9 out of range, "2" dupes 2 — survivors sorted ascending.
                {"index": 1, "sentences": [4, 2, 9, 2], "reason": "试"},
                {"index": 7, "sentences": [1], "reason": "范围外"},
            ]
        }
        r = self._reranker(payload)
        cands = [
            {
                "id": "d1",
                "date": "2026-07-02",
                "sents": ["甲。", "乙。", "丙。", "丁。"],
                "injected": [1, 2],
            }
        ]
        out = await r.rerank_sentences("q", cands, budget=8)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["sentences"], [2, 4])
        user_msg = r._client.chat.completions.create.await_args.kwargs["messages"][1][
            "content"
        ]
        self.assertIn("（注入済み: s1-2）", user_msg)
        self.assertIn("s1: 甲。　←注入済み", user_msg)
        self.assertIn("s3: 丙。", user_msg)

    async def test_judge_failure_returns_none(self):
        r = MemoryReranker(api_key="x", item_label="日記")
        r._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(side_effect=RuntimeError("boom"))
                )
            )
        )
        out = await r.rerank_sentences("q", [{"id": "d", "sents": ["a。"]}])
        self.assertIsNone(out)

    async def test_empty_sentence_pick_drops_the_diary(self):
        r = self._reranker({"relevant": [{"index": 1, "sentences": [], "reason": ""}]})
        out = await r.rerank_sentences(
            "q", [{"id": "d1", "date": "", "sents": ["a。"], "injected": []}]
        )
        self.assertEqual(out, [])


class FullReadTests(unittest.TestCase):
    def _agent(self, limit=2):
        a = BasicMemoryAgent.__new__(BasicMemoryAgent)
        a._diary_reads_this_turn = 0
        a._diary_sent_ledger = {}
        mgr = MagicMock()
        mgr.diary_rag_config = SimpleNamespace(full_reads_per_turn=limit)
        mgr.resolve_diary_uid.side_effect = lambda u: (u, [u])
        mgr.read_diary_full.return_value = {"date": "2026-07-02", "content": "a。b。"}
        mgr.diary_sentences.return_value = ["a。", "b。"]
        a._memory_manager = mgr
        return a

    def test_read_marks_ledger_full_and_result_persists(self):
        a = self._agent()
        res = a._memory_read_diary_query({"diary_uid": "d1"})
        self.assertEqual(res["status"], "ok")
        self.assertIn("残る", res["note"])
        self.assertEqual(a._diary_sent_ledger["d1"]["got"], {1, 2})
        self.assertEqual(a._ledger_full_uids(), {"d1"})

    def test_per_turn_cap(self):
        a = self._agent(limit=2)
        self.assertEqual(
            a._memory_read_diary_query({"diary_uid": "d1"})["status"], "ok"
        )
        self.assertEqual(
            a._memory_read_diary_query({"diary_uid": "d2"})["status"], "ok"
        )
        third = a._memory_read_diary_query({"diary_uid": "d3"})
        self.assertEqual(third["status"], "error")
        self.assertIn("上限2回", third["message"])


class ReplayExemptionTests(unittest.TestCase):
    def test_exempt_result_survives_replay_cap_others_truncated(self):
        long_text = "x" * (BasicMemoryAgent._PROTOCOL_RESULT_MAX_CHARS + 500)
        protocol = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "memory_read_diary"},
                    {"type": "tool_use", "id": "t2", "name": "memory_search"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": long_text},
                    {"type": "tool_result", "tool_use_id": "t2", "content": long_text},
                ],
            },
        ]
        out = BasicMemoryAgent._truncate_protocol_tool_results(protocol)
        results = out[1]["content"]
        self.assertEqual(results[0]["content"], long_text)  # exempt: verbatim
        self.assertIn(
            BasicMemoryAgent._PROTOCOL_TRUNCATION_MARKER, results[1]["content"]
        )

    def test_exempt_id_resolution_by_name(self):
        protocol = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "memory_read_diary"},
                    {"type": "tool_use", "id": "t2", "name": "uber_store"},
                ],
            }
        ]
        self.assertEqual(BasicMemoryAgent._protocol_exempt_result_ids(protocol), {"t1"})


class DiaryUidResolverTests(unittest.TestCase):
    def _mgr(self, stems):
        tmp = tempfile.mkdtemp()
        for s in stems:
            with open(os.path.join(tmp, f"{s}.json"), "w", encoding="utf-8") as f:
                json.dump({"date": s[:10], "content": "x。"}, f)
        m = PersistentMemoryManager.__new__(PersistentMemoryManager)
        m._diaries_dir = tmp
        return m

    def test_short_id_full_uid_and_ambiguity(self):
        full = "2026-07-18_14-50-27_fd4dd67a2b544c2aaed629dc4ad30933"
        m = self._mgr([full, "2026-07-19_10-00-00_aaaa000011112222aaaa000011112222"])
        self.assertEqual(m.resolve_diary_uid("fd4dd67a")[0], full)
        self.assertEqual(m.resolve_diary_uid(full)[0], full)
        uid, matches = m.resolve_diary_uid("2026-07-1")  # matches both
        self.assertIsNone(uid)
        self.assertEqual(len(matches), 2)
        self.assertEqual(m.resolve_diary_uid("deadbeef"), (None, []))


class IndexTextDriftTests(unittest.IsolatedAsyncioTestCase):
    async def test_changed_text_under_stable_id_is_reembedded(self):
        # Splitter changes shift positional chunk ids (uid#i) — the row must
        # be re-embedded, not silently kept with the stale vector/text.
        from src.open_llm_vtuber.memory.vector_index import VectorIndex

        idx = VectorIndex(os.path.join(tempfile.mkdtemp(), "v.json"), api_key="x")
        idx._embed = AsyncMock(return_value=[[1.0, 0.0]])
        await idx.ensure_indexed([{"id": "d#0", "text": "旧文。", "meta": {}}])
        idx._embed = AsyncMock(return_value=[[0.0, 1.0]])
        await idx.ensure_indexed([{"id": "d#0", "text": "新文。", "meta": {}}])
        idx._embed.assert_awaited_once()  # drift detected → re-embedded
        self.assertEqual(idx._texts["d#0"], "新文。")


if __name__ == "__main__":
    unittest.main()
