"""/wake Discord command: thin relay to scripts/wake_task.ps1.

The script is the single implementation (usable from a terminal with the
stack down); the bot only builds the argv, captures UTF-8 output, and relays.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.discord_bot.bot import DiscordVTuberBot


class _FakeProc:
    def __init__(self, out: bytes):
        self._out = out

    async def communicate(self):
        return self._out, None


class RunWakeScriptTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, action, lead, out=b"status: no wake task registered."):
        bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
        fake = AsyncMock(return_value=_FakeProc(out))
        with patch(
            "src.open_llm_vtuber.discord_bot.bot.asyncio.create_subprocess_exec", fake
        ):
            result = await bot._run_wake_script(action, lead)
        return result, fake.await_args.args

    async def test_status_argv_and_decode(self):
        result, argv = await self._run("status", 10)
        self.assertEqual(result, "status: no wake task registered.")
        self.assertIn("powershell", argv[0])
        self.assertTrue(str(argv[5]).endswith("wake_task.ps1"))
        self.assertEqual(argv[6], "status")
        self.assertNotIn("-LeadMinutes", argv)

    async def test_arm_passes_lead_minutes(self):
        _, argv = await self._run("arm", 25)
        self.assertEqual(argv[6], "arm")
        self.assertEqual(list(argv[7:9]), ["-LeadMinutes", "25"])

    async def test_utf8_output_decoded(self):
        result, _ = await self._run(
            "status", 10, out="detail: note=起きて".encode("utf-8")
        )
        self.assertIn("起きて", result)

    async def test_empty_output_placeholder(self):
        result, _ = await self._run("cancel", 10, out=b"")
        self.assertEqual(result, "(no output)")


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    unittest.main()
