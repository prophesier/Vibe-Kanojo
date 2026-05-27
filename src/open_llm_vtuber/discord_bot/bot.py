"""Discord client that forwards channel messages to OLV and posts replies back.

This is the minimum text bridge — voice channel support is not implemented
here (see ``README.md``).
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional

import discord
from discord import app_commands
from loguru import logger

from .bridge import OLVBridge, TurnResult

_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


async def _collect_images(
    attachments: list[discord.Attachment],
) -> list[dict]:
    """Download image attachments and return base64-encoded OLV image dicts."""
    result = []
    for att in attachments:
        mime = (att.content_type or "").split(";")[0].strip()
        if mime not in _IMAGE_MIME_TYPES:
            continue
        try:
            data = await att.read()
            encoded = base64.b64encode(data).decode()
            result.append(
                {
                    "source": "upload",
                    # basic_memory_agent expects a data URL, not raw base64.
                    "data": f"data:{mime};base64,{encoded}",
                    "mime_type": mime,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to download attachment {att.filename!r}: {e}")
    return result


def _allowed(value: object, whitelist: Iterable[int]) -> bool:
    whitelist = list(whitelist)
    if not whitelist:
        return True
    try:
        return int(value) in whitelist  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _chunk_for_discord(text: str, limit: int = 1900) -> list[str]:
    """Split text into <2000-char chunks (Discord's per-message ceiling)."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return chunks


class DiscordVTuberBot(discord.Client):
    """Discord bot that bridges messages to an :class:`OLVBridge`."""

    def __init__(
        self,
        *,
        bridge: OLVBridge,
        guild_ids: Optional[Iterable[int]] = None,
        channel_ids: Optional[Iterable[int]] = None,
        respond_to_mentions_only: bool = False,
        command_prefix: Optional[str] = None,
        admin_user_id: int = 0,
        project_root: Optional[Path] = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self._bridge = bridge
        self._guild_ids = list(guild_ids or [])
        self._channel_ids = list(channel_ids or [])
        self._mentions_only = respond_to_mentions_only
        self._prefix = command_prefix or ""
        self._admin_user_id = int(admin_user_id or 0)
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._tree = app_commands.CommandTree(self)
        self._started_at = time.time()
        if self._admin_user_id:
            self._register_admin_commands()

    def _register_admin_commands(self) -> None:
        """Register slash commands restricted to the configured admin user."""

        @self._tree.command(
            name="restart",
            description="Pull latest code and restart OLV + Discord bot (admin only)",
        )
        async def restart(interaction: discord.Interaction) -> None:  # noqa: ARG001
            if interaction.user.id != self._admin_user_id:
                await interaction.response.send_message(
                    "Unauthorized.", ephemeral=True
                )
                return
            restart_bat = self._project_root / "restart.bat"
            if not restart_bat.exists():
                await interaction.response.send_message(
                    f"restart.bat not found at {restart_bat}. "
                    "Copy it from the repo root to your project root first.",
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(
                "🔄 Pulling latest code and restarting OLV + Discord bot. "
                "Expect ~10-20s of downtime.",
                ephemeral=True,
            )
            # Persist context so the post-restart bot can announce completion
            # in the same channel. Written before close() so it's saved even
            # if the bot is taskkilled before atexit hooks run.
            self._save_restart_pending(
                channel_id=interaction.channel_id,
                user_id=interaction.user.id,
            )
            self._spawn_detached_restart(restart_bat)
            # Bot will be killed by restart.bat shortly; close gracefully so
            # the PID file atexit cleanup runs.
            logger.info("Restart requested by admin; shutting down bot.")
            await self.close()

        @self._tree.command(
            name="logs",
            description="Show recent log lines from the bot or OLV (admin only)",
        )
        @app_commands.describe(
            target="Which log to tail (default: bot)",
            lines="How many recent lines (1-200, default 30)",
        )
        async def logs_cmd(
            interaction: discord.Interaction,
            target: Literal["bot", "olv", "both"] = "bot",
            lines: int = 30,
        ) -> None:
            if interaction.user.id != self._admin_user_id:
                await interaction.response.send_message(
                    "Unauthorized.", ephemeral=True
                )
                return
            lines = max(1, min(200, lines))
            await interaction.response.defer(ephemeral=True)

            targets: list[tuple[str, str]] = []
            if target in ("bot", "both"):
                targets.append(("Discord bot", "discord_"))
            if target in ("olv", "both"):
                targets.append(("OLV", "debug_"))

            chunks: list[str] = []
            for label, prefix in targets:
                path = self._find_latest_log(prefix)
                if path is None:
                    chunks.append(f"**{label}**: no log file found.")
                    continue
                tail = self._tail_lines(path, lines)
                chunks.append(
                    f"**{label}** ({path.name}, last {min(lines, tail.count(chr(10)) + 1)} lines):\n"
                    f"```\n{tail}\n```"
                )
            full = "\n\n".join(chunks)

            if len(full) <= 1900:
                await interaction.followup.send(full, ephemeral=True)
            else:
                buf = io.BytesIO(full.encode("utf-8"))
                await interaction.followup.send(
                    "(too long for inline; attached as file)",
                    ephemeral=True,
                    file=discord.File(buf, filename="logs.txt"),
                )

        @self._tree.command(
            name="status",
            description="Show OLV + Discord bot status (admin only)",
        )
        async def status_cmd(interaction: discord.Interaction) -> None:
            if interaction.user.id != self._admin_user_id:
                await interaction.response.send_message(
                    "Unauthorized.", ephemeral=True
                )
                return
            await interaction.response.defer(ephemeral=True)

            uptime = time.time() - self._started_at
            olv_pid = self._read_pid_file("olv")
            bot_pid = self._read_pid_file("discord")
            commit = self._current_commit()

            embed = discord.Embed(title="📊 Status", color=0x5865F2)
            embed.add_field(
                name="Discord bot",
                value=f"PID: {bot_pid or 'n/a'}\nUptime: {self._format_duration(uptime)}",
                inline=True,
            )
            embed.add_field(
                name="OLV",
                value=f"PID: {olv_pid or 'n/a (no PID file)'}",
                inline=True,
            )
            embed.add_field(
                name="Current commit",
                value=f"`{commit}`" if commit else "(unknown)",
                inline=False,
            )
            embed.set_footer(text=f"Reported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            await interaction.followup.send(embed=embed, ephemeral=True)

    def _find_latest_log(self, prefix: str) -> Optional[Path]:
        logs_dir = self._project_root / "logs"
        if not logs_dir.is_dir():
            return None
        candidates = sorted(
            logs_dir.glob(f"{prefix}*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _tail_lines(path: Path, n: int) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = 4096
                buf = b""
                while size > 0 and buf.count(b"\n") <= n:
                    read = min(block, size)
                    f.seek(size - read)
                    buf = f.read(read) + buf
                    size -= read
            lines = buf.decode("utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception as e:
            return f"(failed to read {path.name}: {e})"

    def _read_pid_file(self, name: str) -> Optional[int]:
        from ..pidfile import read_pid

        return read_pid(name, root=self._project_root)

    def _current_commit(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%h %s"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:120]
        except Exception:
            pass
        return None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _restart_state_path(self) -> Path:
        return self._project_root / "pids" / "restart_pending.json"

    def _save_restart_pending(self, *, channel_id: int, user_id: int) -> None:
        path = self._restart_state_path()
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "initiated_at": time.time(),
                }
            )
        )

    def _spawn_detached_restart(self, restart_bat: Path) -> None:
        """Launch restart.bat in a detached process so it outlives this bot."""
        if sys.platform == "win32":
            # `cmd /c start "" "path"` spawns a brand-new cmd window for the
            # bat. The window is independent of this Python process, so it
            # survives the bot's imminent shutdown. Keeping the window
            # visible (rather than DETACHED_PROCESS) lets the user see git
            # pull / launch output if something goes wrong.
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", str(restart_bat)],
                    cwd=str(self._project_root),
                    close_fds=True,
                )
                logger.info(f"Spawned restart script: {restart_bat}")
            except Exception as e:
                logger.exception(f"Failed to spawn restart script: {e}")
        else:
            # Non-Windows fallback: best-effort, expects a restart.sh script.
            restart_sh = self._project_root / "restart.sh"
            if restart_sh.exists():
                subprocess.Popen(
                    ["bash", str(restart_sh)],
                    start_new_session=True,
                    close_fds=True,
                    cwd=str(self._project_root),
                )
            else:
                logger.error("Non-Windows platform but restart.sh not found.")

    async def setup_hook(self) -> None:
        """Sync slash commands once on startup."""
        if not self._admin_user_id:
            return
        try:
            if self._guild_ids:
                # Per-guild sync is instant (no propagation delay).
                for gid in self._guild_ids:
                    guild = discord.Object(id=int(gid))
                    self._tree.copy_global_to(guild=guild)
                    await self._tree.sync(guild=guild)
                logger.info(
                    f"Slash commands synced to {len(self._guild_ids)} guild(s)."
                )
            else:
                await self._tree.sync()
                logger.info(
                    "Slash commands synced globally (may take up to 1h to propagate)."
                )
        except Exception as e:
            logger.warning(f"Failed to sync slash commands: {e}")

    async def on_ready(self) -> None:
        user = self.user
        logger.info(f"Discord bot ready as {user} (id={getattr(user, 'id', '?')})")
        # If we got here via /restart, announce completion in the originating
        # channel. This is a one-shot: the state file is deleted after the
        # attempt regardless of success.
        await self._maybe_announce_restart_complete()

    async def _maybe_announce_restart_complete(self) -> None:
        state_path = self._restart_state_path()
        if not state_path.exists():
            return
        data: dict = {}
        try:
            data = json.loads(state_path.read_text())
        except Exception as e:
            logger.warning(f"Failed to read restart_pending.json: {e}")
            try:
                state_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        channel_id = data.get("channel_id")
        user_id = data.get("user_id")
        elapsed = time.time() - float(data.get("initiated_at") or time.time())

        # Build a system-style embed. Bot-authored messages never re-enter
        # on_message (filtered by message.author.bot), so this notification
        # cannot trigger the OLV conversation pipeline or pollute chat history.
        embed = discord.Embed(
            title="✅ 再起動完了",
            description="OLV と Discord bot の再起動が完了し、両方とも稼働中です。",
            color=0x57F287,
        )
        embed.add_field(name="所要時間", value=f"{elapsed:.1f} 秒", inline=True)
        embed.set_footer(text="このメッセージはシステム通知であり、会話履歴には記録されません。")

        channel = None
        if channel_id:
            channel = self.get_channel(int(channel_id))
            if channel is None:
                try:
                    channel = await self.fetch_channel(int(channel_id))
                except Exception as e:
                    logger.warning(f"fetch_channel({channel_id}) failed: {e}")

        # Fallback: DM the admin if the original channel can't be reached.
        if channel is None and user_id:
            try:
                user = await self.fetch_user(int(user_id))
                channel = await user.create_dm()
            except Exception as e:
                logger.warning(f"DM fallback to {user_id} failed: {e}")

        if channel is not None:
            try:
                await channel.send(embed=embed)
                logger.info("Sent restart-complete notification.")
            except Exception as e:
                logger.warning(f"Failed to send restart-complete embed: {e}")
        else:
            logger.warning(
                "Could not resolve a channel to announce restart completion."
            )

        try:
            state_path.unlink(missing_ok=True)
        except Exception:
            pass

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.author == self.user:
            return
        if not _allowed(getattr(message.guild, "id", None), self._guild_ids):
            return
        if not _allowed(message.channel.id, self._channel_ids):
            return

        content = (message.content or "").strip()
        images = (
            await _collect_images(message.attachments) if message.attachments else []
        )

        if not content and not images:
            return

        if self._mentions_only:
            if self.user is None or self.user not in message.mentions:
                return
            content = self._strip_mention(content)

        if self._prefix:
            if not content.startswith(self._prefix):
                return
            content = content[len(self._prefix) :].strip()

        if not content and not images:
            return

        request_id = str(uuid.uuid4())
        logger.info(
            f"[discord→olv] guild={message.guild and message.guild.id} "
            f"channel={message.channel.id} user={message.author.id} "
            f"req={request_id} text={content!r} images={len(images)}"
        )

        async def _on_reply(result: TurnResult) -> None:
            await self._post_reply(message, result)

        try:
            async with message.channel.typing():
                await self._bridge.send_text(
                    content,
                    request_id=request_id,
                    on_reply=_on_reply,
                    images=images or None,
                    metadata={"skip_tts": True},
                )
        except Exception as e:
            logger.exception(f"Bridge send failed: {e}")
            await self._safe_reply(message, f"(bridge error: {e})")

    async def _post_reply(self, source: discord.Message, result: TurnResult) -> None:
        if result.error and not result.text:
            await self._safe_reply(source, f"(error: {result.error})")
            return
        if not result.text:
            await self._safe_reply(source, "(no reply)")
            return

        for chunk in _chunk_for_discord(result.text):
            await self._safe_reply(source, chunk)

        if result.error:
            logger.warning(
                f"Reply for req={result.request_id} had partial error: {result.error}"
            )

    async def _safe_reply(self, source: discord.Message, content: str) -> None:
        try:
            await source.channel.send(content)
        except discord.DiscordException as e:
            logger.warning(f"Failed to post Discord reply: {e}")

    def _strip_mention(self, content: str) -> str:
        if self.user is None:
            return content
        mention_forms = (f"<@{self.user.id}>", f"<@!{self.user.id}>")
        for form in mention_forms:
            if content.startswith(form):
                return content[len(form) :].strip()
        return content
