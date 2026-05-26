from pydantic import Field
from typing import ClassVar, Dict, List, Optional

from .i18n import Description, I18nMixin


class DiscordConfig(I18nMixin):
    """Configuration for the Discord bridge."""

    token: str = Field("", alias="token")
    guild_ids: List[int] = Field(default_factory=list, alias="guild_ids")
    channel_ids: List[int] = Field(default_factory=list, alias="channel_ids")
    respond_to_mentions_only: bool = Field(False, alias="respond_to_mentions_only")
    command_prefix: Optional[str] = Field("", alias="command_prefix")
    olv_ws_url: Optional[str] = Field("", alias="olv_ws_url")
    admin_user_id: int = Field(0, alias="admin_user_id")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "token": Description(
            en=(
                "Discord bot token. Leave empty to read from the "
                "DISCORD_BOT_TOKEN environment variable instead."
            ),
            zh=("Discord bot 令牌。留空则从环境变量 DISCORD_BOT_TOKEN 读取。"),
        ),
        "guild_ids": Description(
            en=(
                "Whitelist of Discord guild (server) IDs the bot is allowed "
                "to respond in. Empty list = all guilds the bot is invited to."
            ),
            zh=(
                "允许响应的 Discord 服务器 (guild) ID 白名单。留空表示所有已加入的服务器。"
            ),
        ),
        "channel_ids": Description(
            en=(
                "Whitelist of channel IDs the bot is allowed to respond in. "
                "Empty list = no channel restriction."
            ),
            zh="允许响应的频道 ID 白名单。留空表示不限制频道。",
        ),
        "respond_to_mentions_only": Description(
            en=(
                "If true, the bot only replies when explicitly mentioned "
                "(@bot). Useful in busy channels."
            ),
            zh="若开启,仅在被 @ 时回复。适合人多的频道。",
        ),
        "command_prefix": Description(
            en=(
                "Optional prefix that user messages must start with to be "
                "forwarded (e.g. '!ai '). Leave empty to forward every "
                "matching message."
            ),
            zh="可选的消息前缀(如 '!ai '),只有以该前缀开头的消息会被转发。留空则转发所有消息。",
        ),
        "olv_ws_url": Description(
            en=(
                "Override the OLV WebSocket URL. Leave empty to derive it "
                "from system_config.host/port (ws://<host>:<port>/proxy-ws)."
            ),
            zh=(
                "覆盖 OLV WebSocket 地址。留空时根据 system_config.host/port 自动拼接 "
                "(ws://<host>:<port>/proxy-ws)。"
            ),
        ),
        "admin_user_id": Description(
            en=(
                "Discord user ID authorized to use admin slash commands like "
                "/restart. Set to 0 (default) to disable admin commands "
                "entirely. Enable Discord developer mode, right-click your "
                "profile, and 'Copy User ID' to obtain it."
            ),
            zh=(
                "可使用 /restart 等管理员斜杠命令的 Discord 用户 ID。设为 0（默认）"
                "则完全禁用管理命令。开启 Discord 开发者模式后，右键自己头像 → "
                "「复制用户 ID」获取。"
            ),
        ),
    }
