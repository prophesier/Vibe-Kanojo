"""Discord-only expressions (manual_<idx>.png) — placement and survival.

emotionMap keys mapped to out-of-range Live2D indices no-op on the frontend;
Discord resolves them to hand-placed ``manual_<idx>.png`` files that must
survive both /refresh-faces (which clears its own captures) and the atexit
cache wipe (directory-level keep-list, covered by inspection).
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from src.open_llm_vtuber.discord_bot.bot import DiscordVTuberBot
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "red").save(path)


class CaptureClearTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_clears_captures_but_keeps_manual_faces(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            _write_png(tmp / "3.png")
            _write_png(tmp / "manual_100.png")
            fake_self = SimpleNamespace(_discord_faces_dir=lambda uid: str(tmp))

            await WebSocketHandler._handle_expression_capture_begin(
                fake_self, SimpleNamespace(), "uid", {}
            )

            self.assertFalse((tmp / "3.png").exists())
            self.assertTrue((tmp / "manual_100.png").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ManualFaceSendTests(unittest.IsolatedAsyncioTestCase):
    _CONF_UID = "test_manual_faces_tmp"

    def setUp(self):
        self.faces_dir = Path("cache") / "discord_faces" / self._CONF_UID
        _write_png(self.faces_dir / "manual_100.png")
        _write_png(self.faces_dir / "7.png")

    def tearDown(self):
        shutil.rmtree(self.faces_dir, ignore_errors=True)

    def _bot(self):
        bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
        bot._last_face_index = None
        bot._full_config = SimpleNamespace(
            character_config=SimpleNamespace(conf_uid=self._CONF_UID)
        )
        return bot

    async def test_manual_face_is_sent_for_out_of_range_index(self):
        bot = self._bot()
        channel = SimpleNamespace(send=AsyncMock())

        await bot._maybe_send_face(channel, 100)

        channel.send.assert_awaited_once()
        sent_file = channel.send.await_args.kwargs["file"]
        self.assertEqual(sent_file.filename, "face_100.png")
        self.assertEqual(bot._last_face_index, 100)

    async def test_captured_face_still_falls_back(self):
        bot = self._bot()
        channel = SimpleNamespace(send=AsyncMock())

        await bot._maybe_send_face(channel, 7)

        channel.send.assert_awaited_once()
        self.assertEqual(channel.send.await_args.kwargs["file"].filename, "face_7.png")

    async def test_manual_face_wins_over_capture_with_same_index(self):
        _write_png(self.faces_dir / "manual_7.png")
        bot = self._bot()
        channel = SimpleNamespace(send=AsyncMock())

        await bot._maybe_send_face(channel, 7)

        channel.send.assert_awaited_once()

    async def test_missing_face_sends_nothing(self):
        bot = self._bot()
        channel = SimpleNamespace(send=AsyncMock())

        await bot._maybe_send_face(channel, 42)

        channel.send.assert_not_awaited()
        self.assertIsNone(bot._last_face_index)


if __name__ == "__main__":
    unittest.main()
