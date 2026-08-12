"""memory_update importance change (あさひ 08-09: ヒロ reported she couldn't
change a fact's tier — the capability was genuinely missing, not undocumented).

Rules: high/low freely changeable (text edit optional — importance-only calls
keep the content hash, so the id survives); ``user`` tier is the user's own in
BOTH directions — never assignable, never modifiable (content correction on
user-tier facts stays allowed, the 07-09 ruling).
"""

import unittest
from unittest.mock import AsyncMock

from src.open_llm_vtuber.memory.persistent_memory import PersistentMemoryManager
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent


def _mgr(facts):
    mgr = PersistentMemoryManager.__new__(PersistentMemoryManager)
    mgr._load_facts = lambda: facts
    mgr._saved = []
    mgr._save_facts = lambda f: mgr._saved.append([dict(x) for x in f])
    mgr._sync_facts_index = AsyncMock()
    return mgr


def _fid(mgr, text):
    return mgr._fact_id(text)


class UpdateImportanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_importance_only_change_keeps_text_and_id(self):
        facts = [{"fact": "散歩が好き", "importance": "low", "updated": "x"}]
        mgr = _mgr(facts)
        fid = _fid(mgr, "散歩が好き")
        result = await mgr.update_fact_manual(fid, importance="high")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["id"], fid)  # content hash unchanged
        self.assertEqual(mgr._saved[-1][0]["importance"], "high")
        self.assertEqual(mgr._saved[-1][0]["fact"], "散歩が好き")

    async def test_text_and_importance_together(self):
        facts = [{"fact": "旧い内容", "importance": "high", "updated": "x"}]
        mgr = _mgr(facts)
        result = await mgr.update_fact_manual(
            _fid(mgr, "旧い内容"), "新しい内容", importance="low"
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(mgr._saved[-1][0]["fact"], "新しい内容")
        self.assertEqual(mgr._saved[-1][0]["importance"], "low")

    async def test_user_tier_importance_is_untouchable(self):
        facts = [{"fact": "本人管理の記憶", "importance": "user", "updated": "x"}]
        mgr = _mgr(facts)
        result = await mgr.update_fact_manual(
            _fid(mgr, "本人管理の記憶"), importance="low"
        )
        self.assertEqual(result["status"], "error")
        self.assertFalse(mgr._saved)

    async def test_user_tier_text_edit_still_allowed(self):
        facts = [{"fact": "本人管理の記憶", "importance": "user", "updated": "x"}]
        mgr = _mgr(facts)
        result = await mgr.update_fact_manual(_fid(mgr, "本人管理の記憶"), "修正済み")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(mgr._saved[-1][0]["importance"], "user")

    async def test_user_tier_cannot_be_assigned(self):
        facts = [{"fact": "普通の記憶", "importance": "low", "updated": "x"}]
        mgr = _mgr(facts)
        result = await mgr.update_fact_manual(
            _fid(mgr, "普通の記憶"), importance="user"
        )
        self.assertEqual(result["status"], "error")

    async def test_neither_text_nor_importance_is_error(self):
        mgr = _mgr([{"fact": "x", "importance": "low", "updated": "x"}])
        result = await mgr.update_fact_manual(_fid(mgr, "x"))
        self.assertEqual(result["status"], "error")


class ApproveKeywordAdditionsTests(unittest.TestCase):
    def test_ii_and_keshitemoii_now_approve(self):
        for phrase in ("それでいい", "消してもいい", "うん、いい"):
            self.assertIsNotNone(
                BasicMemoryAgent._MEMORY_APPROVE_RE.search(phrase), phrase
            )

    def test_keshitemoii_is_direct(self):
        self.assertIsNotNone(
            BasicMemoryAgent._MEMORY_DELETE_DIRECT_RE.search("あれは消してもいいよ")
        )


if __name__ == "__main__":
    unittest.main()
