import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.open_llm_vtuber.discord_bot.bot import DiscordVTuberBot


class _FakeBridge:
    def set_proactive_callback(self, callback):
        self.proactive_callback = callback


def _interaction(user_id: int = 42, channel_id: int = 100):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id),
        channel_id=channel_id,
        response=SimpleNamespace(send_message=AsyncMock()),
    )


class DiscordRestartCommandTests(unittest.IsolatedAsyncioTestCase):
    def test_restart_and_resume_commands_are_registered(self):
        bot = DiscordVTuberBot(
            bridge=_FakeBridge(),
            admin_user_id=42,
            project_root=Path.cwd(),
        )

        commands = {command.name: command for command in bot._tree.get_commands()}

        self.assertIn("restart", commands)
        self.assertIn("resume", commands)
        self.assertNotIn("pull", commands["restart"].description.lower())
        self.assertIn("previous session", commands["resume"].description.lower())

    def test_detached_launcher_passes_resume_without_spawning_for_real(self):
        bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
        bot._project_root = Path(r"D:\Code\projects\Vibe-Kanojo")
        restart_bat = bot._project_root / "restart.bat"

        with (
            patch("src.open_llm_vtuber.discord_bot.bot.sys.platform", "win32"),
            patch("src.open_llm_vtuber.discord_bot.bot.subprocess.Popen") as popen,
        ):
            launched = bot._spawn_detached_restart(restart_bat, resume=True)

        self.assertTrue(launched)
        command = popen.call_args.args[0]
        self.assertEqual(command[:5], ["cmd", "/c", "start", "", str(restart_bat)])
        self.assertEqual(command[5:], ["--resume"])

    def test_detached_launcher_keeps_restart_in_fresh_mode(self):
        bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
        bot._project_root = Path(r"D:\Code\projects\Vibe-Kanojo")
        restart_bat = bot._project_root / "restart.bat"

        with (
            patch("src.open_llm_vtuber.discord_bot.bot.sys.platform", "win32"),
            patch("src.open_llm_vtuber.discord_bot.bot.subprocess.Popen") as popen,
        ):
            launched = bot._spawn_detached_restart(restart_bat, resume=False)

        self.assertTrue(launched)
        self.assertEqual(
            popen.call_args.args[0],
            ["cmd", "/c", "start", "", str(restart_bat)],
        )

    async def test_spawn_failure_keeps_bot_online_and_clears_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "restart.bat").write_text("@echo off\n")
            bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
            bot._project_root = root
            bot._admin_user_id = 42
            bot._spawn_detached_restart = lambda script, resume=False: False
            bot.close = AsyncMock()
            interaction = _interaction()

            with patch("src.open_llm_vtuber.discord_bot.bot.sys.platform", "win32"):
                await bot._request_restart(interaction, resume=True)

            bot.close.assert_not_awaited()
            self.assertFalse(bot._restart_state_path().exists())
            message = interaction.response.send_message.await_args.args[0]
            self.assertIn("Failed to launch", message)

    async def test_resume_request_persists_mode_and_closes_after_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "restart.bat").write_text("@echo off\n")
            bot = DiscordVTuberBot.__new__(DiscordVTuberBot)
            bot._project_root = root
            bot._admin_user_id = 42
            bot._spawn_detached_restart = lambda script, resume=False: resume
            bot.close = AsyncMock()
            interaction = _interaction()

            with patch("src.open_llm_vtuber.discord_bot.bot.sys.platform", "win32"):
                await bot._request_restart(interaction, resume=True)

            state = json.loads(bot._restart_state_path().read_text())
            self.assertTrue(state["resume"])
            self.assertEqual(state["channel_id"], 100)
            bot.close.assert_awaited_once()
            message = interaction.response.send_message.await_args.args[0]
            self.assertIn("previous session", message)

    def test_batch_template_uses_local_code_and_resume_flag(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "restart.bat.example").read_text(encoding="utf-8")

        self.assertNotIn("git pull", text.lower())
        self.assertNotIn("GIT_BRANCH", text)
        self.assertNotIn("wt new-tab", text)
        self.assertIn('if /I "%~1"=="--resume"', text)
        self.assertIn('set "OLV_RESUME="', text)
        self.assertIn("python run_server.py %OLV_ARGS%", text)
        self.assertIn('start "OLV"', text)
        self.assertIn('start "Discord"', text)


if __name__ == "__main__":
    unittest.main()
