from pydantic import Field
from typing import ClassVar, Dict

from .i18n import Description, I18nMixin


class WeatherConfig(I18nMixin):
    """Weather tool: Japan Meteorological Agency (気象庁) forecasts,
    observations and advisories via its keyless JSON, with Open-Meteo for
    place-name lookup and locations outside Japan. No API key needed."""

    enabled: bool = Field(False, alias="enabled")
    home: str = Field("", alias="home")
    home_hint: str = Field("", alias="home_hint")
    open_meteo_fallback: bool = Field(True, alias="open_meteo_fallback")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable the in-process weather tool (JMA + Open-Meteo, keyless).",
            zh="启用进程内天气工具（气象厅 + Open-Meteo，无需密钥）。",
        ),
        "home": Description(
            en="Home area in Japanese, used when the model asks for the weather "
            "without naming a place, e.g. '東京都中央区' or '箱根町'. Prefix the "
            "prefecture when the municipality name exists in several "
            "prefectures (府中市 → 東京都府中市).",
            zh="默认地区（日语），模型不指定地点时使用，如 '東京都中央区' 或 "
            "'箱根町'。市区町村名在多个县重名时请带上都道府県（府中市 → "
            "東京都府中市）。",
        ),
        "home_hint": Description(
            en="How the tool describes the default area to the model, e.g. "
            "'home/school (Tokyo area)' when your usual places share one "
            "forecast — stops it from querying each place separately. Empty "
            "= 'the user's home'.",
            zh="工具向模型描述默认地区的说法，如 'home/school (Tokyo area)'——"
            "常去的地方共用一份预报时写明，免得模型逐个地点分别查询。留空 = "
            "'the user's home'。",
        ),
        "open_meteo_fallback": Description(
            en="When a place is not in the JMA area table, look it up with "
            "Open-Meteo geocoding (places outside Japan get Open-Meteo's own "
            "model forecast). Default true.",
            zh="地名不在气象厅区域表里时改用 Open-Meteo 地名检索（日本以外的"
            "地点直接用 Open-Meteo 的模式预报）。默认开启。",
        ),
    }
