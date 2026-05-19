"""Discord client that forwards channel messages to OLV and posts replies back.

This is the minimum text bridge — voice channel support is not implemented
here (see ``README.md``).
"""

from __future__ import annotations

import uuid
from typing import Iterable, Optional

import discord
from loguru import logger

from .bridge import OLVBridge, TurnResult


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
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self._bridge = bridge
        self._guild_ids = list(guild_ids or [])
        self._channel_ids = list(channel_ids or [])
        self._mentions_only = respond_to_mentions_only
        self._prefix = command_prefix or ""

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
        if not content:
            return

        if self._mentions_only:
            if self.user is None or self.user not in message.mentions:
                return
            content = self._strip_mention(content)

        if self._prefix:
            if not content.startswith(self._prefix):
                return
            content = content[len(self._prefix) :].strip()

        if not content:
            return

        request_id = str(uuid.uuid4())
        logger.info(
            f"[discord→olv] guild={message.guild and message.guild.id} "
            f"channel={message.channel.id} user={message.author.id} "
            f"req={request_id} text={content!r}"
        )

        async def _on_reply(result: TurnResult) -> None:
            await self._post_reply(message, result)

        try:
            async with message.channel.typing():
                await self._bridge.send_text(
                    content, request_id=request_id, on_reply=_on_reply
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
