"""Discord bridge for Open-LLM-VTuber.

Connects a Discord bot to the OLV WebSocket backend so that chat messages in
allowed Discord channels are forwarded to the VTuber, and the VTuber's text
replies are posted back to the originating channel.

Voice channel bridging (TTS push + voice receive into ASR) is intentionally
left out of this minimum viable bridge — see ``README.md`` for the roadmap.

``DiscordVTuberBot`` is imported lazily because it depends on the optional
``discord`` extra (`uv sync --extra discord`). ``OLVBridge`` has no such
dependency and is always importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .bridge import OLVBridge

if TYPE_CHECKING:  # pragma: no cover
    from .bot import DiscordVTuberBot


def __getattr__(name: str):
    if name == "DiscordVTuberBot":
        from .bot import DiscordVTuberBot

        return DiscordVTuberBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OLVBridge", "DiscordVTuberBot"]
