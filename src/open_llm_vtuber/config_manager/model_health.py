from pydantic import Field
from typing import ClassVar, Dict, List

from .i18n import Description, I18nMixin


class ModelHealthConfig(I18nMixin):
    """Model-degradation monitor (standalone Discord alerts) + the character's
    self-check tool. Both read benchmark scores from aistupidlevel.info and the
    official Anthropic Statuspage. Independent of the LLM chat pipeline."""

    monitor_enabled: bool = Field(False, alias="monitor_enabled")
    watch_models: List[str] = Field(
        default_factory=lambda: [
            "claude-opus-4-8",
            "claude-fable-5",
            "claude-opus-4-6",
        ],
        alias="watch_models",
    )
    poll_seconds: int = Field(1200, alias="poll_seconds")
    # Detection sensitivity knobs.
    score_floor: float = Field(60.0, alias="score_floor")  # warning line
    current_drop_warn: float = Field(8.0, alias="current_drop_warn")
    alert_channel_id: int = Field(0, alias="alert_channel_id")
    mention_admin_on_alert: bool = Field(True, alias="mention_admin_on_alert")
    self_check_tool_enabled: bool = Field(True, alias="self_check_tool_enabled")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "monitor_enabled": Description(
            en="Enable the background degradation monitor in the Discord bot "
            "(polls aistupidlevel and posts a standalone alert on a state change).",
            zh="在 Discord bot 中启用后台降智监控（轮询 aistupidlevel，状态变化时推送独立告警）。",
        ),
        "watch_models": Description(
            en="Model name prefixes to monitor (matched against aistupidlevel names).",
            zh="要监控的模型名前缀（与 aistupidlevel 的 name 前缀匹配）。",
        ),
        "poll_seconds": Description(
            en="Monitor poll interval in seconds. Site data is hourly, so 1200 "
            "(20 min) is plenty; faster is pointless.",
            zh="监控轮询间隔（秒）。站点数据小时级，1200（20分钟）足够，更快无意义。",
        ),
        "score_floor": Description(
            en="Warning line: a score (combined or any axis) below this is flagged "
            "and always alerts, regardless of the site's status. Default 60.",
            zh="警戒线：分数（综合或任一维度）低于此值就标记并一律告警"
            "（不管站点 status）。默认 60。",
        ),
        "current_drop_warn": Description(
            en="Alert when currentScore is at least this many points below the "
            "model's real 7-day average. Lower = more sensitive but chattier. "
            "(A 7-day DOWN trend below average also alerts regardless.) Default 8.",
            zh="综合分低于该模型真实的7天均值达到此分数就告警。越小越灵敏、越吵。"
            "（另外，7天趋势为「下降」且低于均值时也会告警。）默认 8。",
        ),
        "alert_channel_id": Description(
            en="Channel ID to post alerts to. 0 (default) = the character's "
            "active/last channel, so you see it where you actually look.",
            zh="告警发送的频道 ID。0（默认）=角色当前/最后活跃的频道，方便你在常看的地方看到。",
        ),
        "mention_admin_on_alert": Description(
            en="Prefix alerts with an @mention of discord_config.admin_user_id "
            "so they ping you even when the channel is muted.",
            zh="告警前带上 @discord_config.admin_user_id，即使频道静音也能提醒到你。",
        ),
        "self_check_tool_enabled": Description(
            en="Give the character a check_model_status tool to look up its own "
            "(or a named model's) benchmark + official-status report.",
            zh="给角色一个 check_model_status 工具，可查自己（或指定模型）的基准分+官方状态报告。",
        ),
    }
