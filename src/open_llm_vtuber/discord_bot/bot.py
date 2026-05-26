"""Discord client that forwards channel messages to OLV and posts replies back.

This is the minimum text bridge — voice channel support is not implemented
here (see ``README.md``).
"""

from __future__ import annotations

import base64
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable, Optional

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
            self._spawn_detached_restart(restart_bat)
            # Bot will be killed by restart.bat shortly; close gracefully so
            # the PID file atexit cleanup runs.
            logger.info("Restart requested by admin; shutting down bot.")
            await self.close()

    def _spawn_detached_restart(self, restart_bat: Path) -> None:
        """Launch restart.bat in a detached process so it outlives this bot."""
        if sys.platform == "win32":
            flags = (
                subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
                | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            subprocess.Popen(
                [str(restart_bat)],
                creationflags=flags,
                close_fds=True,
                cwd=str(self._project_root),
                shell=True,
            )
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
