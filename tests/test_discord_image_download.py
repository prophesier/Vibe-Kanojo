"""Image-attachment download resilience (2026-07-31 IMG_8860.png incident).

Two phone uploads hung on the primary CDN url for aiohttp's full 300s default
timeout (str() of the TimeoutError is empty, so the old log line showed
nothing), then the turn was silently sent without the image. These tests pin
the fix: per-attempt timeout, media-proxy fallback, typed log output, and the
user-visible notification when an image never makes it.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.discord_bot import bot as bot_module
from src.open_llm_vtuber.discord_bot.bot import (
    DiscordVTuberBot,
    _collect_images,
    _read_attachment,
)


def _attachment(filename="img.png", content_type="image/png", read=None):
    return SimpleNamespace(filename=filename, content_type=content_type, read=read)


async def _raise_read(*, use_cached=False):
    raise RuntimeError("cdn down")


class _FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _bare_bot():
    bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
    bot._connection = SimpleNamespace(user=None)  # backs the self.user property
    bot._guild_ids = []
    bot._channel_ids = []
    bot._mentions_only = False
    bot._prefix = None
    bot._proactive_channel_id = 100  # matches _message() channel id -> no disk IO
    bot._bridge = SimpleNamespace(send_text=AsyncMock())
    bot._safe_reply = AsyncMock(return_value=True)
    return bot


def _message(content="", attachments=None, channel_id=100):
    channel = SimpleNamespace(
        id=channel_id,
        typing=lambda: _FakeTyping(),
        send=AsyncMock(),
    )
    return SimpleNamespace(
        author=SimpleNamespace(bot=False, id=7),
        guild=SimpleNamespace(id=1),
        channel=channel,
        content=content,
        attachments=attachments or [],
        mentions=[],
    )


class ReadAttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_cdn_hang_falls_back_to_media_proxy(self):
        calls = []

        async def read(*, use_cached=False):
            calls.append(use_cached)
            if not use_cached:
                await asyncio.sleep(3600)  # the 07-31 failure mode: hang, not error
            return b"bytes"

        with patch.object(bot_module, "_ATTACHMENT_TIMEOUT_S", 0.05):
            data = await _read_attachment(_attachment(read=read))

        self.assertEqual(data, b"bytes")
        self.assertEqual(calls, [False, True])

    async def test_cdn_error_falls_back_to_media_proxy(self):
        calls = []

        async def read(*, use_cached=False):
            calls.append(use_cached)
            if not use_cached:
                raise RuntimeError("cdn down")
            return b"bytes"

        data = await _read_attachment(_attachment(read=read))

        self.assertEqual(data, b"bytes")
        self.assertEqual(calls, [False, True])

    async def test_both_paths_failing_raises(self):
        with self.assertRaises(RuntimeError):
            await _read_attachment(_attachment(read=_raise_read))


class CollectImagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_success_failure_and_non_image(self):
        ok = _attachment("good.png", "image/png", read=AsyncMock(return_value=b"px"))
        bad = _attachment("bad.png", "image/png", read=_raise_read)
        skip = _attachment("doc.pdf", "application/pdf", read=AsyncMock())

        images, failed = await _collect_images([ok, bad, skip])

        self.assertEqual(len(images), 1)
        self.assertTrue(images[0]["data"].startswith("data:image/png;base64,"))
        self.assertEqual(failed, ["bad.png"])
        skip.read.assert_not_called()

    async def test_all_good_reports_no_failures(self):
        ok = _attachment("good.png", "image/png", read=AsyncMock(return_value=b"px"))

        images, failed = await _collect_images([ok])

        self.assertEqual(len(images), 1)
        self.assertEqual(failed, [])


class OnMessageNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_with_failed_image_notifies_and_still_delivers(self):
        bot = _bare_bot()
        bad = _attachment("IMG_8860.png", "image/png", read=_raise_read)
        msg = _message(content="チーズナンを試す", attachments=[bad])

        await bot.on_message(msg)

        bot._bridge.send_text.assert_awaited_once()
        self.assertIsNone(bot._bridge.send_text.await_args.kwargs["images"])
        bot._safe_reply.assert_awaited_once()
        notice = bot._safe_reply.await_args.args[1]
        self.assertIn("IMG_8860.png", notice)
        self.assertIn("画像抜き", notice)

    async def test_image_only_message_with_all_failed_notifies_and_skips_turn(self):
        bot = _bare_bot()
        bad = _attachment("IMG_8860.png", "image/png", read=_raise_read)
        msg = _message(content="", attachments=[bad])

        await bot.on_message(msg)

        bot._bridge.send_text.assert_not_awaited()
        bot._safe_reply.assert_awaited_once()
        notice = bot._safe_reply.await_args.args[1]
        self.assertIn("IMG_8860.png", notice)
        self.assertIn("送信されない", notice)

    async def test_healthy_images_produce_no_notification(self):
        bot = _bare_bot()
        ok = _attachment("good.png", "image/png", read=AsyncMock(return_value=b"px"))
        msg = _message(content="見て", attachments=[ok])

        await bot.on_message(msg)

        bot._bridge.send_text.assert_awaited_once()
        sent_images = bot._bridge.send_text.await_args.kwargs["images"]
        self.assertEqual(len(sent_images), 1)
        bot._safe_reply.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
