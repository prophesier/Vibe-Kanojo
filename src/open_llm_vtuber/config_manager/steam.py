from pydantic import Field
from typing import ClassVar, Dict

from .i18n import Description, I18nMixin


class SteamConfig(I18nMixin):
    """Steam integration: storefront lookups (search, prices, reviews, tags)
    plus reading the user's own library/wishlist via the Steam Web API.
    Storefront calls are keyless; library reading needs a free Web API key."""

    enabled: bool = Field(False, alias="enabled")
    web_api_key: str = Field("", alias="web_api_key")
    steamid64: str = Field("", alias="steamid64")
    cc: str = Field("jp", alias="cc")
    lang: str = Field("japanese", alias="lang")
    achievements_top_n: int = Field(40, alias="achievements_top_n")
    tags_top_n: int = Field(40, alias="tags_top_n")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable the Steam integration (store search/details tools and "
            "the library snapshot).",
            zh="启用 Steam 集成（商店搜索/详情工具与本地库快照）。",
        ),
        "web_api_key": Description(
            en="Steam Web API key — get one for free at "
            "https://steamcommunity.com/dev/apikey. Needed only for reading "
            "your own library/playtime/achievements; leave empty to disable "
            "library reading (storefront features still work). Secret: it is "
            "never printed or logged.",
            zh="Steam Web API 密钥——在 https://steamcommunity.com/dev/apikey "
            "免费申请。仅读取自己的库/游玩时长/成就时需要；留空则关闭库读取"
            "（商店功能不受影响）。属于机密：绝不会被打印或写入日志。",
        ),
        "steamid64": Description(
            en="Your 64-bit SteamID. Empty (default) = autodetect from the "
            "local Steam install (loginusers.vdf of the most recent login).",
            zh="你的 64 位 SteamID。留空（默认）= 从本机 Steam 安装自动检测"
            "（取 loginusers.vdf 中最近登录的账号）。",
        ),
        "cc": Description(
            en="Country code for storefront prices/availability (e.g. 'jp'). "
            "Prices are returned in that region's currency. Default jp.",
            zh="商店区域代码，决定价格/可购状态（如 'jp'）。价格按该区货币返回。"
            "默认 jp。",
        ),
        "lang": Description(
            en="Storefront language for descriptions, tags and review labels "
            "(Steam full language name, e.g. 'japanese'). Default japanese.",
            zh="商店语言，影响简介、标签与评测文案（Steam 完整语言名，"
            "如 'japanese'）。默认 japanese。",
        ),
        "achievements_top_n": Description(
            en="Fetch achievement progress for this many top-playtime games "
            "during snapshot enrichment. Higher = richer digest but more API "
            "calls. Default 40.",
            zh="快照增强时，为游玩时长前 N 的游戏抓取成就进度。越大信息越全、"
            "API 调用越多。默认 40。",
        ),
        "tags_top_n": Description(
            en="Fetch store-page community tags for this many top-playtime "
            "games during snapshot enrichment. Default 40.",
            zh="快照增强时，为游玩时长前 N 的游戏抓取商店页社区标签。默认 40。",
        ),
    }
