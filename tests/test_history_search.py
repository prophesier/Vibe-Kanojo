"""Tests for history_search hardening (2026-07-25): comma-shatter defense,
the C1 half-coverage candidacy bar, and tool-marker sanitization of search
results."""

import unittest
from copy import deepcopy
from unittest import mock

import src.open_llm_vtuber.chat_history_manager as chm
from src.open_llm_vtuber.chat_history_manager import (
    search_history,
    split_search_keywords,
    strip_tool_markers,
)


def _msg(role, content, ts):
    return {"role": role, "content": content, "timestamp": ts}


def _run_search(keywords, messages):
    with (
        mock.patch.object(chm, "get_history_list", return_value=[{"uid": "s1"}]),
        mock.patch.object(chm, "get_history", return_value=deepcopy(messages)),
    ):
        return search_history("conf_t", keywords)


class SplitKeywordsTests(unittest.TestCase):
    def test_comma_joined_string(self):
        self.assertEqual(
            split_search_keywords("分布, 口調を真似, 移った"),
            ["分布", "口調を真似", "移った"],
        )

    def test_list_elements_with_separators(self):
        self.assertEqual(
            split_search_keywords(["a,b", " c ", "d、e"]),
            ["a", "b", "c", "d", "e"],
        )

    def test_clean_list_unchanged(self):
        self.assertEqual(split_search_keywords(["清潔"]), ["清潔"])

    def test_empty_inputs(self):
        self.assertEqual(split_search_keywords(None), [])
        self.assertEqual(split_search_keywords(""), [])
        self.assertEqual(split_search_keywords([",、"]), [])


class CommaStringDefenseTests(unittest.TestCase):
    def test_comma_string_not_char_shattered(self):
        messages = [
            _msg("human", "今日はラーメンを食べたよ", "2026-07-20 10:00:00"),
            _msg("ai", "いいね、体に悪そう", "2026-07-20 10:00:05"),
            _msg("human", "これは、ただの読点入りの文。", "2026-07-19 09:00:00"),
        ]
        out = _run_search("ラーメン, 焼肉", messages)
        self.assertEqual(out["status"], "ok")
        # Only the ラーメン message may hit — with the old char-shatter every
        # comma-bearing message scored.
        self.assertEqual(out["total_hits"], 1)
        self.assertIn("ラーメン", out["text"])
        self.assertNotIn("読点", out["text"])


class CandidacyBarTests(unittest.TestCase):
    def test_half_coverage_hits_and_stray_bigram_does_not(self):
        messages = [
            _msg("human", "アボカドを買ってきた", "2026-07-20 10:00:00"),
            _msg("human", "サンドイッチはまた今度", "2026-07-19 10:00:00"),
            _msg("human", "棚からドサッと落ちた", "2026-07-18 10:00:00"),
        ]
        # keyword アボカドサンド: 6 fragments. アボカド = 3/6 = 0.5 → hit;
        # サンドイッチ = 2/6 → below the bar; ドサッ = 1/6 → below the bar.
        out = _run_search(["アボカドサンド"], messages)
        self.assertEqual(out["total_hits"], 1)
        self.assertIn("アボカド", out["text"])
        self.assertNotIn("ドサッ", out["text"])

    def test_whole_substring_still_full_score(self):
        messages = [_msg("human", "海鮮もんじゃを予約した", "2026-07-20 10:00:00")]
        out = _run_search(["海鮮もんじゃ"], messages)
        self.assertEqual(out["total_hits"], 1)


class MarkerSanitizationTests(unittest.TestCase):
    _AI_WITH_MARKER = "了解。📝 *記憶検索: 猫の話* それでね、昨日の続きだけど"

    def test_marker_not_searchable(self):
        messages = [_msg("ai", self._AI_WITH_MARKER, "2026-07-20 10:00:00")]
        out = _run_search(["記憶検索"], messages)
        self.assertEqual(out["total_hits"], 0)

    def test_marker_not_shown_in_results(self):
        messages = [
            _msg("human", "昨日の続きの話をしよう", "2026-07-20 09:59:00"),
            _msg("ai", self._AI_WITH_MARKER, "2026-07-20 10:00:00"),
        ]
        out = _run_search(["続き"], messages)
        self.assertGreaterEqual(out["total_hits"], 1)
        self.assertNotIn("📝", out["text"])
        self.assertNotIn("記憶検索", out["text"])
        self.assertIn("続き", out["text"])

    def test_strip_helper_keeps_real_markdown(self):
        self.assertEqual(
            strip_tool_markers("これは**太字**のまま。🔍 *Web検索: 天気*"),
            "これは**太字**のまま。",
        )

    def test_time_marker_stripped(self):
        self.assertEqual(
            strip_tool_markers("🕐 *時刻確認: 2026-07-26 22:44（日）*いま10時44分だ。"),
            "いま10時44分だ。",
        )


if __name__ == "__main__":
    unittest.main()
