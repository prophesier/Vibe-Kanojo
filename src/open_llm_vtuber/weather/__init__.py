"""Weather tool: JMA (気象庁) bosai JSON as the primary source, Open-Meteo
for geocoding and non-Japan places. See service.py."""

from .areas import Ambiguous, AreaIndex, Resolution
from .client import JmaClient, OpenMeteoClient, WeatherUnavailable
from .service import WeatherService

__all__ = [
    "Ambiguous",
    "AreaIndex",
    "JmaClient",
    "OpenMeteoClient",
    "Resolution",
    "WeatherService",
    "WeatherUnavailable",
]
