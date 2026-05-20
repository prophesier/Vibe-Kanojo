from pydantic import Field
from typing import ClassVar, Dict
from .i18n import I18nMixin, Description


class PersistentMemoryConfig(I18nMixin):
    """Configuration for the persistent memory system."""

    enabled: bool = Field(False, alias="enabled")
    recent_sessions: int = Field(3, alias="recent_sessions")
    diary_count: int = Field(5, alias="diary_count")
    max_facts: int = Field(50, alias="max_facts")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable persistent memory (facts + session diaries)",
            zh="启用持久化记忆（事实库+会话日记）",
        ),
        "recent_sessions": Description(
            en="Number of recent session histories to load into context at session start",
            zh="每次新会话开始时加载到上下文的历史会话数量",
        ),
        "diary_count": Description(
            en="Number of recent session diaries to inject into system prompt",
            zh="注入到系统提示词的最近会话日记数量",
        ),
        "max_facts": Description(
            en="Maximum number of facts to retain in facts.json",
            zh="facts.json 中保留的最大事实条数",
        ),
    }
