"""Run the Discord bridge: ``python -m src.open_llm_vtuber.discord_bot``."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from loguru import logger


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


async def _main() -> int:
    project_root = _project_root()
    sys.path.insert(0, str(project_root))

    from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config

    config_path = project_root / "conf.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    config_data = read_yaml(str(config_path))
    config = validate_config(config_data)

    discord_cfg = getattr(config, "discord_config", None)
    if discord_cfg is None:
        logger.error(
            "discord_config section missing from conf.yaml — copy the block "
            "from config_templates/conf.default.yaml and fill in your token."
        )
        return 1

    token = os.environ.get("DISCORD_BOT_TOKEN") or discord_cfg.token
    if not token:
        logger.error(
            "Discord bot token not set. Provide it via DISCORD_BOT_TOKEN env "
            "var or discord_config.token in conf.yaml."
        )
        return 1

    server_host = config.system_config.host if config.system_config else "localhost"
    server_port = config.system_config.port if config.system_config else 12393
    server_url = discord_cfg.olv_ws_url or f"ws://{server_host}:{server_port}/proxy-ws"

    from .bot import DiscordVTuberBot
    from .bridge import OLVBridge
    from ..pidfile import write_pid

    # Write PID so restart.bat can find and kill this process.
    write_pid("discord", root=project_root)

    bridge = OLVBridge(server_url)
    bot = DiscordVTuberBot(
        bridge=bridge,
        guild_ids=discord_cfg.guild_ids,
        channel_ids=discord_cfg.channel_ids,
        respond_to_mentions_only=discord_cfg.respond_to_mentions_only,
        command_prefix=discord_cfg.command_prefix,
        admin_user_id=discord_cfg.admin_user_id,
        project_root=project_root,
    )

    await bridge.start()

    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
    finally:
        try:
            await bot.close()
        except Exception:
            pass
        await bridge.stop()

    return 0


def main() -> None:
    try:
        rc = asyncio.run(_main())
    except KeyboardInterrupt:
        rc = 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
