"""Session-frozen system-prompt blocks vs startup backfill (08-07 incident).

An overdue alarm fired within seconds of boot, so turn 1 shipped the header
BEFORE startup backfill settled; backfill then reset the facts snapshot and
the (unfrozen) diaries block re-read disk — turn 2 hit only 13% cache. The
rule these tests pin: once a turn has consumed a header block, it stays
byte-stable for the whole session, backfill or not.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber import websocket_handler as ws_module
from src.open_llm_vtuber.memory.persistent_memory import PersistentMemoryManager
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


def _bare_manager() -> PersistentMemoryManager:
    mgr = PersistentMemoryManager.__new__(PersistentMemoryManager)
    mgr._header_snapshot = None
    mgr._diaries_snapshot = None
    return mgr


class FactsSnapshotTests(unittest.TestCase):
    def test_snapshot_survives_disk_changes_once_frozen(self):
        mgr = _bare_manager()
        facts = [{"fact": "A", "importance": "high", "updated": "2026-08-01"}]
        mgr._header_facts = lambda: facts

        first = mgr._header_facts_frozen()
        facts.append({"fact": "B", "importance": "high", "updated": "2026-08-07"})
        second = mgr._header_facts_frozen()

        self.assertEqual(first, second)
        self.assertEqual(len(second), 1, "post-freeze facts must not leak in")

    def test_unconsumed_snapshot_picks_up_settled_facts(self):
        mgr = _bare_manager()
        facts = [{"fact": "A", "importance": "high", "updated": "2026-08-01"}]
        mgr._header_facts = lambda: facts

        facts.append({"fact": "B", "importance": "high", "updated": "2026-08-07"})
        self.assertEqual(len(mgr._header_facts_frozen()), 2)


class DiariesSnapshotTests(unittest.TestCase):
    def test_diaries_block_frozen_against_backfill_writes(self):
        mgr = _bare_manager()
        diaries = [{"date": "2026-08-05", "content": "昨日の日記"}]
        mgr._load_recent_diaries = lambda: diaries

        first = mgr.get_diaries_prompt()
        diaries.append({"date": "2026-08-06", "content": "起動後に書かれた日記"})
        second = mgr.get_diaries_prompt()

        self.assertEqual(first, second)
        self.assertNotIn("起動後に書かれた日記", second)

    def test_empty_diaries_freeze_as_empty(self):
        mgr = _bare_manager()
        calls = []

        def load():
            calls.append(1)
            return []

        mgr._load_recent_diaries = load
        self.assertEqual(mgr.get_diaries_prompt(), "")
        self.assertEqual(mgr.get_diaries_prompt(), "")
        self.assertEqual(len(calls), 1, "empty result must freeze too")


class BackfillResetGuardTests(unittest.TestCase):
    def test_settle_keeps_consumed_snapshot(self):
        """The 08-07 regression in miniature: freeze (turn 1) → backfill
        settles → snapshot must remain, byte-identical."""
        mgr = _bare_manager()
        facts = [{"fact": "A", "importance": "high", "updated": "2026-08-01"}]
        mgr._header_facts = lambda: facts

        frozen = mgr._header_facts_frozen()
        # What backfill_async's finally does now: keep a consumed snapshot.
        self.assertIsNotNone(mgr._header_snapshot)
        facts.append({"fact": "B", "importance": "high", "updated": "2026-08-07"})
        self.assertEqual(mgr._header_facts_frozen(), frozen)


def _bare_handler(started_ago: float = 30.0, with_memory: bool = True):
    handler = WebSocketHandler.__new__(WebSocketHandler)
    handler._server_started_at = time.time() - started_ago
    mm = SimpleNamespace() if with_memory else None
    handler.client_contexts = {"uid": SimpleNamespace(memory_manager=mm)}
    return handler


class AlarmBackfillGateTests(unittest.TestCase):
    def test_defers_before_settle_marker(self):
        handler = _bare_handler(started_ago=30)
        with patch.object(ws_module, "read_backfill_settled_at", return_value=None):
            self.assertFalse(handler._alarm_backfill_gate())

    def test_stale_marker_from_previous_run_defers(self):
        handler = _bare_handler(started_ago=30)
        stale = time.time() - 3600
        with patch.object(ws_module, "read_backfill_settled_at", return_value=stale):
            self.assertFalse(handler._alarm_backfill_gate())

    def test_fresh_settle_within_grace_defers_then_opens(self):
        handler = _bare_handler(started_ago=60)
        just_settled = time.time() - 2
        with patch.object(
            ws_module, "read_backfill_settled_at", return_value=just_settled
        ):
            self.assertFalse(handler._alarm_backfill_gate())
        past_grace = time.time() - 15
        with patch.object(
            ws_module, "read_backfill_settled_at", return_value=past_grace
        ):
            self.assertTrue(handler._alarm_backfill_gate())

    def test_fail_open_after_ceiling(self):
        handler = _bare_handler(started_ago=200)
        with patch.object(ws_module, "read_backfill_settled_at", return_value=None):
            self.assertTrue(handler._alarm_backfill_gate())

    def test_no_memory_manager_opens_immediately(self):
        handler = _bare_handler(started_ago=5, with_memory=False)
        self.assertTrue(handler._alarm_backfill_gate())


if __name__ == "__main__":
    unittest.main()
