"""
This module contains the pydantic model for the configurations of
different types of agents.
"""

from pydantic import BaseModel, Field
from typing import Dict, ClassVar, Optional, Literal, List
from .i18n import I18nMixin, Description
from .stateless_llm import StatelessLLMConfigs

# ======== Configurations for different Agents ========


class BasicMemoryAgentConfig(I18nMixin, BaseModel):
    """Configuration for the basic memory agent."""

    llm_provider: Literal[
        "stateless_llm_with_template",
        "openai_compatible_llm",
        "claude_llm",
        "llama_cpp_llm",
        "ollama_llm",
        "lmstudio_llm",
        "openai_llm",
        "gemini_llm",
        "zhipu_llm",
        "deepseek_llm",
        "groq_llm",
        "mistral_llm",
    ] = Field(..., alias="llm_provider")

    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")
    use_mcpp: Optional[bool] = Field(False, alias="use_mcpp")
    mcp_enabled_servers: Optional[List[str]] = Field([], alias="mcp_enabled_servers")

    # Client-side web tools (used by the OpenAI path; the Claude path uses
    # Anthropic's native server tools configured under claude_llm instead).
    enable_web_tools: Optional[bool] = Field(False, alias="enable_web_tools")
    web_search_provider: Literal["brave", "tavily"] = Field(
        "brave", alias="web_search_provider"
    )
    web_search_api_key: Optional[str] = Field("", alias="web_search_api_key")
    max_web_searches: Optional[int] = Field(3, alias="max_web_searches")
    max_web_fetches: Optional[int] = Field(3, alias="max_web_fetches")
    max_fetch_chars: Optional[int] = Field(20000, alias="max_fetch_chars")

    # Self-set alarms: the character can schedule a reminder for itself that
    # fires later (even across restarts) as a proactive message.
    enable_alarms: Optional[bool] = Field(True, alias="enable_alarms")

    # Claude-only prompt-cache keepalive: when the conversation has been idle
    # for ~this many minutes (Anthropic's cache TTL is 1h), nudge the character
    # to say something so the cache is refreshed instead of expiring.
    claude_cache_keepalive_minutes: Optional[int] = Field(
        55, alias="claude_cache_keepalive_minutes"
    )
    claude_cache_keepalive_max: Optional[int] = Field(
        6, alias="claude_cache_keepalive_max"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "llm_provider": Description(
            en="LLM provider to use for this agent",
            zh="Basic Memory Agent 智能体使用的大语言模型选项",
        ),
        "faster_first_response": Description(
            en="Whether to respond as soon as encountering a comma in the first sentence to reduce latency (default: True)",
            zh="是否在第一句回应时遇上逗号就直接生成音频以减少首句延迟（默认：True）",
        ),
        "segment_method": Description(
            en="Method for segmenting sentences: 'regex' or 'pysbd' (default: 'pysbd')",
            zh="分割句子的方法：'regex' 或 'pysbd'（默认：'pysbd'）",
        ),
        "use_mcpp": Description(
            en="Whether to use MCP (Model Context Protocol) for the agent (default: True)",
            zh="是否使用为智能体启用 MCP (Model Context Protocol) Plus（默认：False）",
        ),
        "mcp_enabled_servers": Description(
            en="List of MCP servers to enable for the agent",
            zh="为智能体启用 MCP 服务器列表",
        ),
        "enable_web_tools": Description(
            en=(
                "Enable client-side web search + fetch tools (OpenAI path). "
                "The model can search the web and read URL content. Search "
                "needs a free API key from Brave or Tavily; fetch is "
                "self-contained (no key, no cost)."
            ),
            zh=(
                "启用客户端网页搜索+抓取工具（OpenAI 路径）。模型可联网搜索并读取 "
                "URL 正文。搜索需要 Brave 或 Tavily 的免费 API key；抓取自带实现"
                "（无需 key、无成本）。"
            ),
        ),
        "web_search_provider": Description(
            en="Search backend: 'brave' (2000 free/mo) or 'tavily' (1000 free/mo).",
            zh="搜索后端：'brave'（每月免费 2000 次）或 'tavily'（每月免费 1000 次）。",
        ),
        "web_search_api_key": Description(
            en="API key for the chosen search provider. Empty disables search (fetch still works).",
            zh="所选搜索提供商的 API key。留空则禁用搜索（抓取仍可用）。",
        ),
        "max_web_searches": Description(
            en="Max search calls per reply.",
            zh="单次回复最多搜索次数。",
        ),
        "max_web_fetches": Description(
            en="Max URL fetches per reply.",
            zh="单次回复最多抓取 URL 次数。",
        ),
        "max_fetch_chars": Description(
            en="Truncate each fetched page to this many characters.",
            zh="每个抓取页面截断到的字符数。",
        ),
        "enable_alarms": Description(
            en=(
                "Let the character set self-reminders (alarms) that fire later "
                "as a proactive message. Persisted to a local file; survives "
                "restarts. (default: True)"
            ),
            zh=(
                "允许角色给自己设定闹钟提醒，到点后以主动消息触发。存本地文件，"
                "重启后仍有效。（默认：True）"
            ),
        ),
        "claude_cache_keepalive_minutes": Description(
            en=(
                "Claude only. If the conversation is idle this many minutes "
                "(Anthropic's prompt-cache TTL is 1h), nudge the character to "
                "say something so the cache is refreshed instead of expiring. "
                "0 disables. (default: 55)"
            ),
            zh=(
                "仅 Claude。会话空闲达到该分钟数时（Anthropic 提示缓存有效期 1 "
                "小时），提示角色随便说点什么以刷新缓存、避免过期。0 关闭。（默认：55）"
            ),
        ),
        "claude_cache_keepalive_max": Description(
            en=(
                "Max consecutive keepalive nudges before giving up (assume the "
                "user has left); any real user message resets the count. "
                "(default: 6)"
            ),
            zh=(
                "连续保活提示的最大次数，超过则放弃（判定用户已离开）；任何真实用户"
                "消息都会清零重新计。（默认：6）"
            ),
        ),
    }


class Mem0VectorStoreConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 vector store."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(
            en="Vector store provider (e.g., qdrant)", zh="向量存储提供者（如 qdrant）"
        ),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0LLMConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 LLM."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(en="LLM provider name", zh="语言模型提供者名称"),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0EmbedderConfig(I18nMixin, BaseModel):
    """Configuration for Mem0 embedder."""

    provider: str = Field(..., alias="provider")
    config: Dict = Field(..., alias="config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(en="Embedder provider name", zh="嵌入模型提供者名称"),
        "config": Description(
            en="Provider-specific configuration", zh="提供者特定配置"
        ),
    }


class Mem0Config(I18nMixin, BaseModel):
    """Configuration for Mem0."""

    vector_store: Mem0VectorStoreConfig = Field(..., alias="vector_store")
    llm: Mem0LLMConfig = Field(..., alias="llm")
    embedder: Mem0EmbedderConfig = Field(..., alias="embedder")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "vector_store": Description(en="Vector store configuration", zh="向量存储配置"),
        "llm": Description(en="LLM configuration", zh="语言模型配置"),
        "embedder": Description(en="Embedder configuration", zh="嵌入模型配置"),
    }


# =================================


class HumeAIConfig(I18nMixin, BaseModel):
    """Configuration for the Hume AI agent."""

    api_key: str = Field(..., alias="api_key")
    host: str = Field("api.hume.ai", alias="host")
    config_id: Optional[str] = Field(None, alias="config_id")
    idle_timeout: int = Field(15, alias="idle_timeout")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "api_key": Description(
            en="API key for Hume AI service", zh="Hume AI 服务的 API 密钥"
        ),
        "host": Description(
            en="Host URL for Hume AI service (default: api.hume.ai)",
            zh="Hume AI 服务的主机地址（默认：api.hume.ai）",
        ),
        "config_id": Description(
            en="Configuration ID for EVI settings", zh="EVI 配置 ID"
        ),
        "idle_timeout": Description(
            en="Idle timeout in seconds before disconnecting (default: 15)",
            zh="空闲超时断开连接的秒数（默认：15）",
        ),
    }


# =================================


class LettaConfig(I18nMixin, BaseModel):
    """Configuration for the Letta agent."""

    host: str = Field("localhost", alias="host")
    port: int = Field(8283, alias="port")
    id: str = Field(..., alias="id")
    faster_first_response: Optional[bool] = Field(True, alias="faster_first_response")
    segment_method: Literal["regex", "pysbd"] = Field("pysbd", alias="segment_method")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "host": Description(
            en="Host address for the Letta server", zh="Letta服务器的主机地址"
        ),
        "port": Description(
            en="Port number for the Letta server (default: 8283)",
            zh="Letta服务器的端口号（默认：8283）",
        ),
        "id": Description(
            en="Agent instance ID running on the Letta server",
            zh="指定Letta服务器上运行的Agent实例id",
        ),
    }


class AgentSettings(I18nMixin, BaseModel):
    """Settings for different types of agents."""

    basic_memory_agent: Optional[BasicMemoryAgentConfig] = Field(
        None, alias="basic_memory_agent"
    )
    mem0_agent: Optional[Mem0Config] = Field(None, alias="mem0_agent")
    hume_ai_agent: Optional[HumeAIConfig] = Field(None, alias="hume_ai_agent")
    letta_agent: Optional[LettaConfig] = Field(None, alias="letta_agent")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "basic_memory_agent": Description(
            en="Configuration for basic memory agent", zh="基础记忆代理配置"
        ),
        "mem0_agent": Description(en="Configuration for Mem0 agent", zh="Mem0代理配置"),
        "hume_ai_agent": Description(
            en="Configuration for Hume AI agent", zh="Hume AI 代理配置"
        ),
        "letta_agent": Description(
            en="Configuration for Letta agent", zh="Letta 代理配置"
        ),
    }


class AgentConfig(I18nMixin, BaseModel):
    """This class contains all of the configurations related to agent."""

    conversation_agent_choice: Literal[
        "basic_memory_agent", "mem0_agent", "hume_ai_agent", "letta_agent"
    ] = Field(..., alias="conversation_agent_choice")
    agent_settings: AgentSettings = Field(..., alias="agent_settings")
    llm_configs: StatelessLLMConfigs = Field(..., alias="llm_configs")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conversation_agent_choice": Description(
            en="Type of conversation agent to use", zh="要使用的对话代理类型"
        ),
        "agent_settings": Description(
            en="Settings for different agent types", zh="不同代理类型的设置"
        ),
        "llm_configs": Description(
            en="Pool of LLM provider configurations", zh="语言模型提供者配置池"
        ),
        "faster_first_response": Description(
            en="Whether to respond as soon as encountering a comma in the first sentence to reduce latency (default: True)",
            zh="是否在第一句回应时遇上逗号就直接生成音频以减少首句延迟（默认：True）",
        ),
        "segment_method": Description(
            en="Method for segmenting sentences: 'regex' or 'pysbd' (default: 'pysbd')",
            zh="分割句子的方法：'regex' 或 'pysbd'（默认：'pysbd'）",
        ),
    }
