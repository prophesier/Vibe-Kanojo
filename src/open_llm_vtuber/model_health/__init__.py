"""Model-health: detect when the Anthropic models this project runs on are
degrading, and let the character check its own condition.

Two consumers share this package:
  - the Discord bot runs :class:`monitor.DegradationMonitor` on a timer and
    pushes a standalone alert (no LLM involved) when a watched model degrades;
  - the agent exposes a ``check_model_status`` tool built on :func:`report`.

Pure, provider-agnostic, network-only. Every call degrades gracefully — a
failure returns an "unavailable" result, never raises into the caller.
"""

from .aistupidlevel_client import AiStupidLevelClient, ModelScore, HistoryPoint
from .anthropic_status_client import AnthropicStatusClient, AnthropicStatus

__all__ = [
    "AiStupidLevelClient",
    "ModelScore",
    "HistoryPoint",
    "AnthropicStatusClient",
    "AnthropicStatus",
]
