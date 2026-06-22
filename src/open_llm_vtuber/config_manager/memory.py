from pydantic import Field
from typing import ClassVar, Dict
from .i18n import I18nMixin, Description


class DiaryRagConfig(I18nMixin):
    """Configuration for diary retrieval-augmented recall (long-tail memory)."""

    enabled: bool = Field(True, alias="enabled")
    openai_api_key: str = Field("", alias="openai_api_key")
    base_url: str = Field("", alias="base_url")
    embedding_model: str = Field("text-embedding-3-small", alias="embedding_model")
    similarity_threshold: float = Field(0.55, alias="similarity_threshold")
    topn_threshold: float = Field(0.70, alias="topn_threshold")
    max_retrievals_per_turn: int = Field(2, alias="max_retrievals_per_turn")
    lexical_weight: float = Field(0.5, alias="lexical_weight")
    rerank_enabled: bool = Field(True, alias="rerank_enabled")
    rerank_model: str = Field("gpt-4o-mini", alias="rerank_model")
    rerank_candidates: int = Field(12, alias="rerank_candidates")
    prefilter_floor: float = Field(0.3, alias="prefilter_floor")
    rerank_context_turns: int = Field(6, alias="rerank_context_turns")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable semantic recall of older diaries via embeddings",
            zh="启用基于向量的旧日记语义召回",
        ),
        "openai_api_key": Description(
            en="OpenAI API key for embeddings. Leave blank to reuse the openai_llm provider's key",
            zh="嵌入用的 OpenAI API key；留空则复用 openai_llm 的 key",
        ),
        "base_url": Description(
            en="Embeddings endpoint. Leave blank to reuse the openai_llm endpoint / official OpenAI",
            zh="嵌入端点；留空则复用 openai_llm 端点 / OpenAI 官方",
        ),
        "embedding_model": Description(
            en="Embedding model name",
            zh="嵌入模型名称",
        ),
        "similarity_threshold": Description(
            en="Minimum cosine similarity for any diary to be inserted (gate)",
            zh="任意日记被插入的最低余弦相似度（总闸）",
        ),
        "topn_threshold": Description(
            en="Stricter similarity a 2nd+ diary must clear to also be inserted",
            zh="追加第 2+ 篇日记需要达到的更严相似度",
        ),
        "max_retrievals_per_turn": Description(
            en="Maximum diaries inserted in a single turn",
            zh="单轮最多插入的日记数",
        ),
        "lexical_weight": Description(
            en="Weight of the keyword-overlap signal in hybrid scoring (0 = pure vector)",
            zh="混合打分里关键词重叠信号的权重（0=纯向量）",
        ),
        "rerank_enabled": Description(
            en="Let a cheap LLM judge which shortlisted diaries are actually relevant",
            zh="用一个便宜 LLM 判断候选日记里哪些真正相关",
        ),
        "rerank_model": Description(
            en="Model for the relevance judge (uses the diary_rag OpenAI key)",
            zh="相关性判官用的模型（复用 diary_rag 的 OpenAI key）",
        ),
        "rerank_candidates": Description(
            en="How many top hybrid candidates to shortlist for the LLM judge",
            zh="送给 LLM 判官的 hybrid 候选数量",
        ),
        "prefilter_floor": Description(
            en="Skip the LLM judge entirely when the best hybrid score is below this",
            zh="最高 hybrid 分低于此值时直接跳过 LLM 判官（省调用）",
        ),
        "rerank_context_turns": Description(
            en="How many recent conversation turns to give the LLM judge as context",
            zh="给 LLM 判官作为上下文的最近对话轮数",
        ),
    }


class FactsRagConfig(I18nMixin):
    """Diary's sibling: retrieval-augmented recall of low-importance facts.

    A separate, independent subsystem from diary RAG (own index, own config,
    injected after the diaries). Facts tagged ``user``/``llm`` stay in the system
    prompt header; ``low`` facts are recalled on demand via this index.
    """

    enabled: bool = Field(False, alias="enabled")
    similarity_threshold: float = Field(0.55, alias="similarity_threshold")
    topn_threshold: float = Field(0.70, alias="topn_threshold")
    max_retrievals_per_turn: int = Field(3, alias="max_retrievals_per_turn")
    lexical_weight: float = Field(0.5, alias="lexical_weight")
    rerank_enabled: bool = Field(True, alias="rerank_enabled")
    rerank_model: str = Field("gpt-4o-mini", alias="rerank_model")
    rerank_candidates: int = Field(12, alias="rerank_candidates")
    prefilter_floor: float = Field(0.3, alias="prefilter_floor")
    rerank_context_turns: int = Field(6, alias="rerank_context_turns")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "enabled": Description(
            en="Enable semantic recall of low-importance facts via embeddings",
            zh="启用基于向量的低优先级事实语义召回",
        ),
        "similarity_threshold": Description(
            en="Minimum cosine similarity for any fact to be inserted (gate)",
            zh="任意事实被插入的最低余弦相似度（总闸）",
        ),
        "topn_threshold": Description(
            en="Stricter similarity a 2nd+ fact must clear to also be inserted",
            zh="追加第 2+ 条事实需要达到的更严相似度",
        ),
        "max_retrievals_per_turn": Description(
            en="Maximum facts inserted in a single turn",
            zh="单轮最多插入的事实数",
        ),
        "lexical_weight": Description(
            en="Weight of the keyword-overlap signal in hybrid scoring (0 = pure vector)",
            zh="混合打分里关键词重叠信号的权重（0=纯向量）",
        ),
        "rerank_enabled": Description(
            en="Let a cheap LLM judge which shortlisted facts are actually relevant",
            zh="用一个便宜 LLM 判断候选事实里哪些真正相关",
        ),
        "rerank_model": Description(
            en="Model for the relevance judge (uses the diary_rag OpenAI key)",
            zh="相关性判官用的模型（复用 diary_rag 的 OpenAI key）",
        ),
        "rerank_candidates": Description(
            en="How many top hybrid candidates to shortlist for the LLM judge",
            zh="送给 LLM 判官的 hybrid 候选数量",
        ),
        "prefilter_floor": Description(
            en="Skip the LLM judge entirely when the best hybrid score is below this",
            zh="最高 hybrid 分低于此值时直接跳过 LLM 判官（省调用）",
        ),
        "rerank_context_turns": Description(
            en="How many recent conversation turns to give the LLM judge as context",
            zh="给 LLM 判官作为上下文的最近对话轮数",
        ),
    }


class PersistentMemoryConfig(I18nMixin):
    """Configuration for the persistent memory system."""

    enabled: bool = Field(False, alias="enabled")
    recent_sessions: int = Field(3, alias="recent_sessions")
    diary_count: int = Field(5, alias="diary_count")
    max_facts: int = Field(50, alias="max_facts")
    memory_llm_model: str = Field("", alias="memory_llm_model")
    memory_reasoning_effort: str = Field("", alias="memory_reasoning_effort")
    diary_rag: DiaryRagConfig = Field(default_factory=DiaryRagConfig, alias="diary_rag")
    facts_rag: FactsRagConfig = Field(default_factory=FactsRagConfig, alias="facts_rag")

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
        "memory_llm_model": Description(
            en="Model for memory tasks (diary/fact/consolidate); blank = reuse the chat model. Reuses the chat provider's key/endpoint",
            zh="记忆任务（日记/事实/合并）用的模型；留空则复用主对话模型。复用主 provider 的 key/端点",
        ),
        "memory_reasoning_effort": Description(
            en="reasoning_effort for memory tasks (none/low/medium/high). Blank = don't send it (use the model's default). gpt-5.1 defaults to 'none' (no reasoning) so set 'low' there; gpt-5.5 already defaults to 'medium' so blank is fine",
            zh="记忆任务的推理强度（none/low/medium/high）。留空=不发送（用模型默认）。gpt-5.1 默认 none（不推理），用它请设 low；gpt-5.5 默认就是 medium，留空即可",
        ),
        "diary_rag": Description(
            en="Diary retrieval-augmented recall settings",
            zh="日记向量召回（RAG）设置",
        ),
        "facts_rag": Description(
            en="Low-importance facts retrieval-augmented recall settings",
            zh="低优先级事实向量召回（RAG）设置",
        ),
    }
