from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
    Set,
)
import asyncio
import json
import hashlib
import re
import time
import unicodedata
from copy import deepcopy
from datetime import datetime, timedelta
from loguru import logger
from .agent_interface import AgentInterface
from ...web_tools import (
    format_fetch_result,
    format_search_results,
    web_fetch,
    web_search,
)
from ...alarms import resolve_fire_at, format_local
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import (
    AsyncLLM as ClaudeAsyncLLM,
    _budget_tokens_removed,
)
from ..stateless_llm.openai_compatible_llm import (
    CACHE_SEAM_MARKER,
    AsyncLLM as OpenAICompatibleAsyncLLM,
)
from ...chat_history_manager import (
    get_history,
    get_recent_histories,
    mark_last_message_excluded,
    pop_last_message,
    search_history,
    split_search_keywords,
    strip_tool_markers as _strip_tool_markers,
)
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor


# Tool-execution markers (🍔/🔍/🔗/⏰/🧠/🎮/📝/🔧/🕐/🎵 tags streamed to the UI
# and persisted into chat history for human review) must be STRIPPED from every
# model-visible copy — the model imitates them and fabricates tool results.
# The regex + strip helper live in chat_history_manager (imported above as
# _strip_tool_markers) so history_search shares the same sanitization.

# MCP tools whose marker is written AFTER execution, so it can carry the
# result (the clock reading, the song that started) instead of just the tool
# name. _mcp_tool_marker stays silent for these; _deferred_tool_marker emits.
_DEFERRED_MARKER_TOOLS = frozenset(
    {
        "get_current_time",
        "convert_time",
        "music_search",
        "music_play",
        "music_play_playlist",
        "music_playlists",
        "music_now_playing",
        "music_stop",
    }
)


def _mcp_result_text(content: Any) -> str:
    """Flatten an MCP tool result into plain text. Results arrive either as a
    string or as content blocks ``[{"type": "text", "text": ...}]``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        web_tools_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._web_tools_config = web_tools_config or {"enabled": False}
        self._memory = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()
        self._memory_manager = None  # set via set_memory_manager()
        self._alarm_store = None  # set via set_alarm_store()
        # Self-check tool (check_model_status). Gated on config; clients are
        # created lazily on first use (lightweight, stateless).
        self._model_health_enabled = False  # set via set_model_health_enabled()
        self._asl_client = None
        self._status_client = None

        # Steam integration (four in-process steam_* tools + resident library
        # digest). All set via set_steam_runtime() once the startup snapshot
        # is ready — before the first turn, so the digest is cache-stable.
        self._steam_enabled = False
        # Check-and-set guard used by ServiceContext._init_steam_runtime so the
        # shared agent is wired at most once (set before its first await).
        self._steam_wiring_started = False
        self._steam_client = None  # steam.SteamClient
        self._steam_snapshot_mgr = None  # steam.SnapshotManager
        self._steam_digest = ""
        self._steam_snapshot_cache: Optional[Dict[str, Any]] = None
        self._steam_snapshot_loaded_at: float = 0.0
        # Compact copies of successful steam tool results. Folded into the
        # NEXT outgoing user message the same persist-not-ephemeral way as the
        # RAG blocks (stored == sent, never assistant-role), so results stay
        # visible across turns without teaching the model to imitate them.
        self._steam_pending_blocks: List[str] = []

        # Character self-service memory tools (memory_* CRUD on facts + search
        # over facts/diaries). Gated on config (set_memory_tools_enabled) AND
        # a wired memory manager. Deletions are two-phase: staged here first,
        # executed only after the user's approval is verified in their actual
        # next message (the model cannot fake that — the handler reads it from
        # _memory, not from the tool arguments).
        self._memory_tools_enabled = False  # set via set_memory_tools_enabled()
        self._pending_memory_deletes: Dict[str, Dict[str, Any]] = {}
        # In-process tool calls made during the CURRENT turn — the log-only
        # "claimed but never called" canary (_check_stateful_claims).
        self._turn_inproc_calls: List[str] = []
        # Diary RAG (long-tail recall). _pending_rag_block is the block built
        # for the turn currently being assembled (consumed by _to_messages).
        # Sentence ledger (08-13 redesign): uid → {"got": 1-based sentence
        # numbers already in context, "total": sentence count}. RAG injects
        # only the set difference per diary; a full-text tool read marks
        # every sentence. Fully-covered diaries leave
        # the retrieval candidate pool; partially-covered ones stay, carrying
        # their 既出 mask into the judge prompt. Session-scoped: the injected
        # blocks live in _memory, so the ledger resets with it.
        self._diary_sent_ledger: Dict[str, Dict[str, Any]] = {}
        self._pending_rag_block: str = ""
        # Full-diary reads (memory_read_diary) allowed per turn; results are
        # replay-exempt so each read permanently occupies context.
        self._diary_reads_this_turn: int = 0
        # Facts RAG (independent of diary RAG): low-importance facts recalled on
        # demand, injected after the diary block. Same persist-in-_memory pattern.
        self._session_injected_fact_ids: Set[str] = set()
        self._pending_facts_block: str = ""
        self._sliding_window_uids: Set[str] = set()
        # Claude-path history split (あさひ 08-02): entries below this index
        # came from PAST sessions. They are lifted out of `messages` and sent
        # as one frozen system transcript block instead, so the messages
        # segment holds only the current session — thinking-parameter changes
        # (forced/adaptive hot-toggle) then invalidate only that small tail
        # of the cache. Both are fixed at load time and stable per session.
        self._past_history_cut: int = 0
        self._past_transcript: str = ""
        # Fingerprint of the last system prompt, for diagnosing prompt-cache
        # drops: a change between turns is exactly what busts the prefix cache.
        self._last_system_fp: str = ""
        # Tracks whether the current session's banner has already been
        # prepended in _memory. Set by set_memory_from_recent_histories
        # when the current session had pre-existing messages, OR by
        # _add_message when injecting it onto the first user message of
        # a freshly-started (empty-on-load) session.
        self._current_session_banner_added = False
        # When the newest message in memory happened — from the last loaded
        # disk record at startup, then datetime.now() on every _add_message.
        # Drives the 【…経過】/【日付が変わった】time-event banners; message
        # time on purpose, so a client sitting open without messages never
        # counts as activity. The user-only baseline exists because date
        # rollovers must be judged on the model-visible (user-tagged)
        # timeline — see _time_banner_between.
        self._last_message_dt: Optional[datetime] = None
        self._last_user_message_dt: Optional[datetime] = None

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        logger.debug(f"Memory Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    # Stateful-action claims (remember/save/alarm set/cancel …) an assistant
    # reply can make. If one appears in a turn where NO in-process tool ran,
    # that's the observed fabrication mode — log it loudly (surface, don't
    # suppress: あさひ uses these failures as his model-quality canary).
    _STATEFUL_CLAIM_RE = re.compile(
        r"(覚えて(おく|おいた|おきます)|記憶し(た|ておく|ておいた)|"
        r"メモし(た|ておく|ておいた)|保存し(た|ておいた)|記録し(た|ておいた)|"
        r"(アラーム|リマインダー)を?(セットした|設定した|入れた|入れておいた|"
        r"取り消した|キャンセルした)|予約し(た|ておいた)|取り消しておいた)"
    )

    def _check_stateful_claims(self, text: str) -> None:
        """Log-only tool-honesty canary (never raises, never alters output)."""
        try:
            if self._turn_inproc_calls:
                return
            m = self._STATEFUL_CLAIM_RE.search(text)
            if m:
                excerpt = text[max(0, m.start() - 30) : m.end() + 20]
                logger.warning(
                    f"[tool_honesty] stateful claim with NO in-process tool "
                    f"call this turn: …{excerpt}…"
                )
        except Exception:
            pass

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
        claude_protocol: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add message to memory."""
        if skip_memory:
            return

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if not text_content and role == "assistant":
            # A thinking-only or tool-only turn may carry its true transcript
            # with no visible text; any other empty assistant message stays
            # dropped.
            if not (
                claude_protocol
                and self._claude_protocol_worth_carrying(claude_protocol)
            ):
                return

        if role == "assistant" and text_content:
            self._check_stateful_claims(text_content)

        # Inject the current-session banner onto the FIRST user message of
        # a freshly-started session. set_memory_from_recent_histories
        # cannot add it when the session is empty at load time, so this is
        # the only point where a brand-new session gets its visible boundary.
        if (
            role == "user"
            and text_content
            and not self._current_session_banner_added
            and self._memory_manager
        ):
            current_uid = getattr(self._memory_manager, "_current_session_uid", "")
            if current_uid:
                banner = self._session_header_text(current_uid, is_current=True)
                text_content = f"{banner}\n{text_content}"
                self._current_session_banner_added = True

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            role == "assistant"
            and claude_protocol
            and self._claude_protocol_worth_carrying(claude_protocol)
        ):
            # Claude-path side field: the exact assistant/tool-result protocol
            # sequence that produced this visible reply. It is expanded
            # verbatim by claude_llm instead of rebuilding an assistant message
            # from selected thinking/text blocks. This is in-memory only;
            # on-disk chat history remains clean and a restart safely falls
            # back to the visible text without thinking blocks.
            message_data["claude_protocol"] = deepcopy(claude_protocol)

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)
        self._last_message_dt = datetime.now()
        if role == "user":
            self._last_user_message_dt = self._last_message_dt

    # Cap on tool_result content stored in the cross-turn protocol. The live
    # tool loop always sees full results; this trims only what later turns
    # replay, where the useful content already lives in the visible reply
    # text. Counted in EFFECTIVE TOKENS, not chars (あさひ 08-13: the old
    # 400-char cap silently metered English by the LETTER — 400 chars is
    # ~360 tok of Japanese but only ~105 tok of English, so web_search got
    # 3.5× less through). Integer milli-token weights keep the cut position
    # deterministic: CJK-range chars ≈0.9 tok, everything else ≈1/3.8 tok.
    # 500 was picked from the 08-10..13 log survey (68/108 calls truncated):
    # it lets set_alarm confirmations (448–535 JA chars) through whole —
    # alarm honesty evidence, the 07-26 lesson — and web_search's EN median
    # (~380 tok-equiv) survives intact, while the uber/memory whales still
    # trim hard.
    _PROTOCOL_RESULT_BUDGET_MILLITOK = 500 * 1000
    _MILLITOK_CJK = 900  # chars above U+2E7F
    _MILLITOK_OTHER = 263  # ≈1/3.8 tok per char
    _PROTOCOL_TRUNCATION_MARKER = "…[truncated for replay]"
    # Legacy silent-turn placeholder (08-15..08-20). New silent turns take
    # the API-error exit instead (_handle_thinking_only_turn, あさひ 08-20);
    # this constant remains because pre-08-20 records on disk carry it as
    # content, and single_conversation still uses it as a defensive
    # fallback for a seed-only store (e.g. an interrupted turn).
    _EMPTY_TURN_PLACEHOLDER = "…"

    @classmethod
    def _token_cut(cls, text: str, budget_millitok: int) -> tuple:
        """(cut_index, consumed) — cut_index is -1 when ``text`` fits."""
        acc = 0
        for i, ch in enumerate(text):
            w = cls._MILLITOK_CJK if ord(ch) > 0x2E7F else cls._MILLITOK_OTHER
            if acc + w > budget_millitok:
                return i, acc
            acc += w
        return -1, acc

    # Tools whose results are exempt from the replay cap (あさひ 08-13:
    # reading a diary in full = injecting it — the content IS the point, so
    # it must survive replay verbatim). Exempt results are trivially
    # truncation-stable, so an in-loop breakpoint can ride their round.
    # Matched by tool NAME via the protocol's tool_use blocks (id → name).
    _PROTOCOL_EXEMPT_RESULT_TOOLS = frozenset({"memory_read_diary", "model_history"})
    # uber_search results open with a related-facts section (あさひ 08-13:
    # ヒロ couldn't see his restaurant history while browsing). Everything up
    # to and including this end marker survives replay verbatim; the 400-char
    # cap starts counting AFTER it, so the store list still trims away.
    _UBER_FACTS_HEADER = "【関連する記憶（Uber履歴・自動）】"
    _UBER_FACTS_END = "――関連記憶ここまで――"
    # One-shot notice injected at the top of the NEXT user payload after a
    # round's thinking blocks were dropped from replay (per-round cap,
    # あさひ 08-09): without it, the missing-thinking precedent reads as
    # "think less" — the same imitation channel as the zero-thinking /
    # silent-turn cascades. Wording is あさひ's draft near-verbatim.
    _THINKING_DROP_NOTICE = (
        "【通知: 前のターンの思考は長すぎたため、コスト節約のため以後の文脈から"
        "外した。だからといって、以後の思考を減らしたり短くしたりする必要はない。】"
    )

    # Cold-start transcript seeds (あさひ's design 07-23, revised 07-26):
    # persist a turn's verbatim transcript to chat history whenever it carries
    # thinking OR tool calls; on reload, the last _THINKING_SEED_COUNT
    # assistant turns (ROOT memory entries — a tool turn's inner assistant
    # rounds live inside its one protocol and don't count separately) get
    # their transcripts re-attached UNCONDITIONALLY. The thinking-rate figure
    # over the last _THINKING_SEED_RATE_WINDOW turns is stats/warning only —
    # the old 80% withhold-gate was a 4.6-era health check: on adaptive
    # 5-series models skipping thinking is normal, and withholding transcripts
    # also erased tool calls, making honest turns look fabricated (03:19
    # alarm incident).
    _THINKING_SEED_COUNT = 10
    _THINKING_SEED_RATE_WINDOW = 20  # assistant turns considered for the rate
    _THINKING_SEED_MIN_RATE = 0.8  # log/warning threshold only — never gates
    # Final assistant message of the just-completed turn, staged for the
    # conversation layer to persist (pop_thinking_seed). Class-level default
    # so __new__-constructed test instances read None.
    _last_thinking_seed: Optional[Dict[str, Any]] = None
    # Where this conversation persists on disk (set by the memory loaders);
    # needed by the safety-refusal cleanup. Class-level defaults for tests.
    _conf_uid: str = ""
    _history_uid: str = ""

    # One-shot context_excluded tag for the turn's on-disk AI record (set by
    # the API-error path, consumed by the conversation layer's store_message).
    # Class-level default so __new__-constructed test instances read None.
    _pending_context_excluded: Optional[str] = None
    # Class-level default for the same reason (__init__ assigns per instance):
    # _to_messages reads it to spot the banner-pending first turn.
    _current_session_banner_added = False
    # One-shot flag: a just-finished turn had thinking dropped from replay
    # (per-round cap); the next stored user payload carries
    # _THINKING_DROP_NOTICE. Class-level default for __new__-built tests.
    _pending_thinking_drop_notice = False

    def pop_context_excluded(self) -> Optional[str]:
        """One-shot getter for the just-completed turn's context_excluded tag.

        Non-None only after an API-error turn: the conversation layer stores
        the visible error notice to disk with this tag so the record survives
        for the human reader but never re-enters the assembled context."""
        tag = self._pending_context_excluded
        self._pending_context_excluded = None
        return tag

    def pop_thinking_seed(self) -> Optional[Dict[str, Any]]:
        """One-shot getter for the just-completed turn's thinking seed.

        Returns {"model": ..., "protocol": [messages], "thinking_tokens": N,
        "round_thinking": [...]} — the turn's transcript after the per-round
        trim (signed thinking kept for under-cap rounds; tool_result content
        truncated with a marker), or None when the turn produced neither
        thinking nor tool calls. round_thinking aligns with the protocol's
        assistant messages so a reload can re-trim if the cap shrank. The
        whole transcript is kept so a replayed tool turn still shows real
        tool calls — rewriting it into a "results without calls" shape would
        teach fabrication. The conversation layer attaches it to the on-disk
        history record (store_message).
        """
        seed = self._last_thinking_seed
        self._last_thinking_seed = None
        return seed

    def _apply_thinking_seeds(self, candidates: List[tuple], resuming: bool) -> None:
        """Re-attach persisted turn transcripts to the reloaded context.

        Each seed becomes the ``claude_protocol`` of its memory entry, reusing
        the exact-replay machinery. Eligible entries are the LAST
        _THINKING_SEED_COUNT assistant turns (root entries), attached
        unconditionally (あさひ 07-26) — the recent thinking rate is computed
        for the log/warning only, never to withhold transcripts: a withheld
        tool transcript makes an honest turn look fabricated. Seeds from a
        different model are skipped (other models ignore foreign thinking
        blocks but still bill their tokens). The replay cap applies PER ROUND
        (08-09): seeds carrying a round_thinking breakdown are re-trimmed
        against the current cap; legacy whole-turn seeds over the cap are
        skipped outright (no per-round data to trim by).
        """
        if not self._is_claude_llm() or not self._memory:
            return
        assistant_idxs = [
            i for i, m in enumerate(self._memory) if m.get("role") == "assistant"
        ]
        if not assistant_idxs:
            return
        candidate_idxs = {i for i, _ in candidates}
        stat_window = assistant_idxs[-self._THINKING_SEED_RATE_WINDOW :]
        thinking_turns = sum(1 for i in stat_window if i in candidate_idxs)
        rate = thinking_turns / len(stat_window)
        if rate < self._THINKING_SEED_MIN_RATE:
            logger.info(
                f"[thinking_seed] recent transcript rate {thinking_turns}/"
                f"{len(stat_window)} is below {self._THINKING_SEED_MIN_RATE:.0%} "
                "(stats only — transcripts are attached regardless)."
            )
            if resuming:
                logger.warning(
                    "[thinking_seed] resuming a session with a low-thinking tail; "
                    "a FRESH session recovers spontaneous thinking — consider "
                    "/restart instead of /resume."
                )
        model = getattr(self._llm, "model", "") or ""
        replay_cap = getattr(self._llm, "thinking_replay_max_tokens", 0) or 0
        attach_window = set(assistant_idxs[-self._THINKING_SEED_COUNT :])
        applied = 0
        for idx, seed in candidates:
            if idx not in attach_window:
                continue
            protocol = seed.get("protocol")
            if not (
                isinstance(protocol, list)
                and protocol
                and all(isinstance(m, dict) for m in protocol)
            ):
                # Legacy 07-23 seeds stored only the final message's blocks.
                content = seed.get("content")
                if not (isinstance(content, list) and content):
                    continue
                protocol = [{"role": "assistant", "content": content}]
            seed_round_thinking = seed.get("round_thinking")
            if isinstance(seed_round_thinking, list):
                # Per-round seeds (08-09+): already trimmed at store time; the
                # re-trim only bites when the cap shrank between store and
                # reload (idempotent otherwise — dropped rounds are zeroed in
                # the stored alignment). No drop notice on reload: these are
                # old turns, not a precedent the model just set.
                protocol, _, _ = self._trim_protocol_per_round(
                    protocol, seed_round_thinking, replay_cap
                )
                if not protocol or not self._claude_protocol_worth_carrying(protocol):
                    continue
            else:
                # Legacy whole-turn seeds carry no per-round breakdown, so the
                # old fat-seed skip is the only safe rule for them (field
                # present on seeds written after 07-25).
                seed_thinking = seed.get("thinking_tokens")
                if (
                    replay_cap
                    and isinstance(seed_thinking, (int, float))
                    and seed_thinking > replay_cap
                ):
                    continue
            seed_model = seed.get("model") or ""
            if seed_model and model and seed_model != model:
                continue
            self._memory[idx]["claude_protocol"] = deepcopy(protocol)
            applied += 1
        logger.info(
            f"[thinking_seed] re-attached {applied} turn transcript(s) into "
            f"reloaded context (recent rate {thinking_turns}/{len(stat_window)})."
        )

    @classmethod
    def _protocol_exempt_result_ids(cls, protocol: List[Dict[str, Any]]) -> Set[str]:
        """tool_use ids whose results are replay-cap exempt, resolved by tool
        NAME from the protocol's own assistant tool_use blocks."""
        ids: Set[str] = set()
        for msg in protocol:
            content = msg.get("content")
            if msg.get("role") != "assistant" or not isinstance(content, list):
                continue
            for b in content:
                if (
                    isinstance(b, dict)
                    and b.get("type") == "tool_use"
                    and b.get("name") in cls._PROTOCOL_EXEMPT_RESULT_TOOLS
                    and b.get("id")
                ):
                    ids.add(b["id"])
        return ids

    @classmethod
    def _truncate_protocol_tool_results(
        cls, protocol: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Cap tool_result content in a stored Claude protocol.

        Applies ONLY to user-role tool_result messages — those are
        client-built and unsigned, so their content is ours to edit.
        Assistant messages are never touched: signed thinking blocks are
        position-bound within their message, and sibling blocks (tool_use,
        server-tool results) must stay exactly as generated. Results of
        _PROTOCOL_EXEMPT_RESULT_TOOLS calls pass through untouched.
        """
        exempt_ids = cls._protocol_exempt_result_ids(protocol)
        out: List[Dict[str, Any]] = []
        for msg in protocol:
            content = msg.get("content")
            if msg.get("role") != "user" or not isinstance(content, list):
                out.append(msg)
                continue
            new_content = [
                cls._truncate_tool_result_block(block)
                if isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") not in exempt_ids
                else block
                for block in content
            ]
            out.append({**msg, "content": new_content})
        return out

    @classmethod
    def _truncate_tool_result_block(cls, block: Dict[str, Any]) -> Dict[str, Any]:
        budget = cls._PROTOCOL_RESULT_BUDGET_MILLITOK
        marker = cls._PROTOCOL_TRUNCATION_MARKER
        content = block.get("content")
        if isinstance(content, str):
            if cls._UBER_FACTS_END in content:
                # The related-facts section persists whole; the replay budget
                # applies to the payload after it (あさひ: 予算は facts の
                # 後から数える).
                head, sep, tail = content.partition(cls._UBER_FACTS_END)
                cut, _ = cls._token_cut(tail, budget)
                if cut < 0:
                    return block
                return {**block, "content": head + sep + tail[:cut] + marker}
            cut, _ = cls._token_cut(content, budget)
            if cut < 0:
                return block
            return {**block, "content": content[:cut] + marker}
        if isinstance(content, list):
            new_blocks: List[Dict[str, Any]] = []
            changed = False
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    text = b.get("text", "")
                    if budget <= 0:
                        changed = True
                        continue
                    cut, consumed = cls._token_cut(text, budget)
                    if cut >= 0:
                        b = {**b, "text": text[:cut] + marker}
                        changed = True
                        budget = 0
                    else:
                        budget -= consumed
                    new_blocks.append(b)
                else:
                    # Images and other non-text payloads are pure replay bulk
                    # with no precedent value — stub them out.
                    new_blocks.append(
                        {"type": "text", "text": "[non-text tool output omitted]"}
                    )
                    changed = True
            if not changed:
                return block
            return {**block, "content": new_blocks or marker}
        return block

    @classmethod
    def _trim_protocol_per_round(
        cls,
        protocol: List[Dict[str, Any]],
        round_thinking: List[int],
        cap: int,
    ) -> tuple:
        """Per-round replay trim (あさひ 08-09 rulings, probe-proven).

        The two dimensions are INDEPENDENT: tool_result content is truncated
        to the protocol cap as before, and any loop round whose billed
        thinking exceeded ``cap`` loses ONLY its thinking blocks — the
        _tool_thinking_replay_probe showed completed historical rounds accept
        partial and full thinking removal (the immutable-latest-assistant 400
        governs only the ACTIVE loop tail). This retires the whole-turn
        "over cap → drop the entire transcript" hammer: long-thinking turns
        now keep their tool calls and text.

        ``round_thinking`` aligns with assistant messages in protocol order
        (missing entries count as 0 = keep). An assistant message emptied by
        the drop (thinking-only final round) is omitted entirely — the only
        API-legal shape for a block-less message.

        Returns ``(trimmed, kept_round_thinking, dropped)``:
        ``kept_round_thinking`` aligns with the assistant messages REMAINING
        in ``trimmed``, with dropped rounds zeroed — so a seed re-trimmed on
        reload (cap may have shrunk since store time) stays aligned and the
        operation is idempotent. ``dropped`` counts rounds that actually lost
        thinking blocks (drives the drop notice).
        """
        thinking_types = {"thinking", "redacted_thinking"}
        out: List[Dict[str, Any]] = []
        kept_rounds: List[int] = []
        dropped = 0
        ai = 0
        for msg in cls._truncate_protocol_tool_results(protocol):
            if msg.get("role") != "assistant":
                out.append(msg)
                continue
            tokens = round_thinking[ai] if ai < len(round_thinking) else 0
            ai += 1
            if not isinstance(tokens, (int, float)):
                tokens = 0
            content = msg.get("content")
            has_thinking = isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") in thinking_types for b in content
            )
            if cap and tokens > cap and has_thinking:
                dropped += 1
                new_content = [
                    b
                    for b in content
                    if not (isinstance(b, dict) and b.get("type") in thinking_types)
                ]
                if not new_content:
                    continue
                out.append({**msg, "content": new_content})
                kept_rounds.append(0)
            else:
                out.append(msg)
                kept_rounds.append(int(tokens))
        return out, kept_rounds, dropped

    @staticmethod
    def _claude_protocol_has_tool_use(
        protocol: List[Dict[str, Any]],
    ) -> bool:
        """Whether an exact Claude transcript contains a tool call."""
        for message in protocol:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in content
            ):
                return True
        return False

    @classmethod
    def _claude_protocol_worth_carrying(cls, protocol: List[Dict[str, Any]]) -> bool:
        """Whether a transcript must be carried into later requests: it holds
        signed thinking (precedent) OR tool calls. The tool half is あさひ's
        07-26 ruling after the 03:19 incident — a dropped tool transcript made
        an honest turn look fabricated, and the model "corrected" an alarm it
        had actually set (except tool_result truncation, the turn must be
        restored exactly as it happened)."""
        return cls._claude_protocol_has_thinking(
            protocol
        ) or cls._claude_protocol_has_tool_use(protocol)

    @staticmethod
    def _claude_protocol_has_thinking(
        protocol: List[Dict[str, Any]],
    ) -> bool:
        """Whether an exact Claude transcript contains signed thinking."""
        for message in protocol:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(block, dict)
                and block.get("type") in {"thinking", "redacted_thinking"}
                for block in content
            ):
                return True
        return False

    @staticmethod
    def _claude_thinking_blocks_replayable(content: List[Any]) -> bool:
        """Whether every thinking block in an assistant message carries its
        signature (redacted blocks are always replayable). A max_tokens cut
        mid-thinking can leave an unsigned block; replaying that would 400."""
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and not block.get("signature")
            ):
                return False
        return True

    def set_memory_manager(self, manager) -> None:
        """Attach a PersistentMemoryManager for fact extraction and diary injection."""
        self._memory_manager = manager

    def set_alarm_store(self, store) -> None:
        """Attach an AlarmStore, which also turns on the set/list/cancel_alarm
        built-in tools (they're only advertised when a store is present)."""
        self._alarm_store = store

    def set_model_health_enabled(self, enabled: bool) -> None:
        """Turn the check_model_status self-check tool on/off (config-driven)."""
        self._model_health_enabled = bool(enabled)

    def set_steam_runtime(self, client, snapshot_mgr, digest: str) -> None:
        """Attach the Steam client + snapshot manager, enabling the four
        steam_* built-in tools and the resident library digest.

        ``digest`` is a short Japanese summary of the startup snapshot that
        rides in the system prompt. It MUST be set once before the first turn
        and never changed mid-session — a mid-session change would bust the
        prompt-cache prefix (same rule as the persona/facts blocks)."""
        self._steam_client = client
        self._steam_snapshot_mgr = snapshot_mgr
        self._steam_digest = (digest or "").strip()
        self._steam_enabled = client is not None and snapshot_mgr is not None
        logger.info(
            f"[steam] agent runtime set (enabled={self._steam_enabled}, "
            f"digest={len(self._steam_digest)} chars)."
        )

    def set_memory_tools_enabled(self, enabled: bool) -> None:
        """Turn the character's memory_* self-service tools on/off (config).
        They additionally require a wired memory manager to actually run."""
        self._memory_tools_enabled = bool(enabled)

    @property
    def _memory_tools_active(self) -> bool:
        return self._memory_tools_enabled and self._memory_manager is not None

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        """Format an ISO timestamp as '[YYYY-MM-DD HH:MM:SS 曜日]'.

        Japanese single-kanji weekday (月..日) — higher salience for the
        JA persona than the earlier 'Mon'..'Sun' (she misread a weekday),
        and it matches the 曜日 wording in _TIMESTAMP_NOTE/_HISTORY_NOTE."""
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        try:
            dt = datetime.fromisoformat(ts)
            return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[dt.weekday()]}]"
        except (ValueError, TypeError):
            return f"[{ts}]" if ts else ""

    @classmethod
    def _now_tag(cls) -> str:
        """Timestamp tag for messages happening right now."""
        return cls._format_timestamp(datetime.now().isoformat(timespec="seconds"))

    # Gap (hours) from the previous message before a 【…経過】 banner fires.
    _TIME_GAP_BANNER_HOURS = 3

    @classmethod
    def _time_banner_between(
        cls, prev: datetime, prev_user: Optional[datetime], now: datetime
    ) -> str:
        """One-line time-event banner between message times, or "".

        Every message already carries a timestamp tag, but the model tends
        to ignore them; a loud structural banner (the same trick as the
        session banner) is the countermeasure against time hallucinations.
        Wording is deliberately 「現在は」 (anchored to the message's own
        moment), not 「今日は」 — the banner is replayed inside history
        later, where an absolute "today" would become a lie.

        TWO baselines on purpose (hiro herself caught the single-baseline
        bug by inspecting her own history): the gap is measured from the
        previous message of ANY role, so 「前回のメッセージから」 stays
        literally true — but the date rollover is judged against the
        previous USER message. Assistant entries carry no timestamp the
        model can see, and an assistant reply (or an overnight keepalive
        nudge) landing just after midnight would otherwise swallow the
        crossing: the next user message would compare against a
        post-midnight time and the 【日付が変わった】 banner would never
        appear anywhere.

        Used both live (previous message → the incoming one) and at history
        load (between consecutive disk records), so replayed history keeps
        its banners across restarts.
        """
        gap_hit = (now - prev) >= timedelta(hours=cls._TIME_GAP_BANNER_HOURS)
        date_changed = prev_user is not None and now.date() != prev_user.date()
        if not (gap_hit or date_changed):
            return ""
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        date_str = f"{now.strftime('%Y-%m-%d')}（{weekdays[now.weekday()]}）"
        if gap_hit:
            hours = (now - prev).total_seconds() / 3600
            if hours >= 24:
                span = f"約{max(1, round(hours / 24))}日"
            else:
                span = f"約{max(1, round(hours))}時間"
            if date_changed:
                return f"【前回のメッセージから{span}経過 → 現在は {date_str}】"
            return f"【前回のメッセージから{span}経過】"
        return f"【日付が変わった → 現在は {date_str}】"

    def _time_event_banner(self) -> str:
        """Live-path banner: previous message in memory → right now.

        Message time on purpose — a client sitting open without messages
        never counts as activity. Rides the outgoing payload text
        (stored == sent, append-only → cache-safe)."""
        if self._last_message_dt is None:
            return ""
        return self._time_banner_between(
            self._last_message_dt, self._last_user_message_dt, datetime.now()
        )

    # Minimal note that stays bundled with the persona (cache-friendly,
    # rarely changes). Just declares the tag format and that replies must
    # not echo it. The strict behavioural rules live in _HISTORY_NOTE
    # below, positioned right before the message history so the LLM sees
    # them last.
    _TIMESTAMP_NOTE = (
        "ユーザーのメッセージには `[YYYY-MM-DD HH:MM:SS 曜日]` 形式の"
        "タイムスタンプタグが先頭に付与されている。"
        "これはあなた自身の参照用メタデータであり、"
        "返信本文には絶対に含めてはならない。"
    )

    # (The old _TOOL_MARKER_NOTE — a "don't imitate these markers" instruction —
    # is gone: markers are now stripped from the history replayed to the model
    # (_strip_tool_markers), so it never sees them. The note both was moot and
    # itself listed the marker glyphs in the prompt, which could seed imitation.)

    # Affirmative capability note for the self-set alarm tools. Claude tends to
    # ignore raw tool schemas, so state plainly the tool is real and must
    # actually be called. Framed as future proactive speech, not only literal
    # reminders. Static / cache-stable; gated on alarms being active.
    _ALARM_CAPABILITY_NOTE = (
        "[Alarms] When you want to speak to the user at some future time — a "
        "reminder he asked for, picking a conversation back up, checking in on "
        "him — use set_alarm. Review with list_alarms, cancel with "
        "cancel_alarm; as with setting, you must actually call them. Merely "
        "saying 'I'll let you know later' or 'I cancelled it' makes nothing "
        "happen."
    )

    # Lean variant (5-series models): when-to-use only. The tool enumeration
    # is in the schemas; the 4.6-era no-false-report nudge is gone for
    # 5-series (stateful claims are still checked mechanically).
    _ALARM_CAPABILITY_NOTE_LEAN = (
        "[Alarms] When you want to speak to the user at some future time — a "
        "reminder he asked for, picking a conversation back up, checking in on "
        "him — use set_alarm."
    )

    # When-to-use only; tool mechanics live in the schema descriptions.
    _UBER_CAPABILITY_NOTE = (
        "[Uber Eats] When the conversation turns to delivery or convenience-"
        "store shopping, look up the real stores, menus, and prices with the "
        "uber_* tools before answering (browse-only; the user places the order "
        "himself)."
    )

    # When-to-use only; tool mechanics live in the schema description.
    _MODEL_HEALTH_CAPABILITY_NOTE = (
        "[Self-diagnosis] When you are concerned about your own condition, or "
        "the user asks about it, check the objective data with "
        "check_model_status. The result is a reference point, not grounds to "
        "condemn yourself."
    )

    # When-to-use + the grounding rule; tool mechanics live in the schemas.
    _STEAM_CAPABILITY_NOTE = (
        "[Steam] A snapshot of the library overview is already in this system "
        "prompt. For detailed lookups, store searches, and finding "
        "recommendations, use the steam_* tools — and when recommending a game "
        "from the store, always pick from the tool results.\n"
        "A copy of tool results may persist at the head of the next user "
        "message as a '【Steamデータ】' block — that is not something the user "
        "said."
    )

    # When-to-use only; the two-phase delete flow and tier rules live in the
    # schemas and are enforced mechanically by the handlers regardless.
    # 08-14 (あさひ): the old "results disappear at the end of the turn"
    # claim went stale when replay persistence landed — results now stay,
    # trimmed to their beginning; memory_inject was retired with it.
    _MEMORY_CAPABILITY_NOTE = (
        "[Managing memory] Separate from automatic recall, when you want to look "
        "into the past yourself, use memory_search (memory_read_diary for a full "
        "diary entry). For concrete exchanges and proper nouns that were never "
        "kept in memory, history_search does a keyword search over the full "
        "conversation logs. For both: to narrow by date or period, use the "
        "date_from/date_to arguments — do not mix dates into the query or the "
        "keywords. To save or correct facts established in conversation, use "
        "memory_add / memory_update (be careful about rewriting). Deletion is "
        "memory_delete, which requires the user's consent. Additions and "
        "corrections are reflected in search immediately, and enter the "
        "resident list from the next startup. user-tier memories are the "
        "user's own (you cannot create or delete them; correcting their "
        "content is allowed). When a session is wrapping up, write the diary "
        "of this session with memory_write_diary."
    )

    # Lean variant (5-series models): keeps when-to-use routing and the
    # behavioral gates (result-persistence semantics, the save/correct
    # caution, delete consent). Cut: argument mechanics (date_from/date_to),
    # propagation timing, and tier rules — all of those live in the schemas
    # and are enforced mechanically by the handlers.
    _MEMORY_CAPABILITY_NOTE_LEAN = (
        "[Managing memory] Separate from automatic recall, when you want to "
        "look into the past yourself, use memory_search (memory_read_diary for "
        "a full diary entry); for concrete exchanges and proper nouns that "
        "were never kept in memory, history_search does a keyword search over "
        "the full conversation logs. To save or correct facts established in "
        "conversation, use memory_add / memory_update (be careful about "
        "rewriting). A fact that has merely become outdated is better moved "
        "to importance=archive than deleted — archived facts surface only "
        "when you search for them. Deletion is memory_delete, which requires "
        "the user's consent. When a session is wrapping up, write the diary "
        "of this session with memory_write_diary."
    )

    # Trailing system block placed right before the message history.
    # No cache_control marker — small, static, and positional. By sitting
    # last in the system prompt, it's the closest instruction to the
    # message history, which empirically improves rule adherence
    # (proximity effect).
    _HISTORY_NOTE = (
        "【以下の会話履歴について】\n\n"
        "ここから後に続くユーザーとアシスタントのやりとりは、"
        "**現在進行中のセッション**のもの。"
        "それ以前の会話は、システム欄の【過去セッションの転記】ブロックに"
        "時系列順で転記されている（初回起動などで存在しない場合もある）。"
        "転記も現在のやりとりも、必ずしも今日の出来事だけではなく、"
        "数日前〜数週間前の古いやりとりを含み得る。"
        "各ターンが「いつ」発生したかは、冒頭の "
        "`[YYYY-MM-DD HH:MM:SS 曜日]` タグでのみ判定できる。\n\n"
        "各セッションの最初のメッセージには `【セッション開始: 日時】` または "
        "`【現在進行中のセッション開始: 日時】` という見出しが挿入されている。"
        "これがセッションの境界を示すので、これより前のターンと後のターンは"
        "**別の会話セッション**だと認識すること。"
        "見出しが無い間のターンは、同じセッション内の連続したやりとりである。\n\n"
        "また、日付の変わり目や長い空白の後のメッセージには "
        "`【日付が変わった → 現在は …】`・`【前回のメッセージから約…経過】` "
        "という見出しが挿入される。これが現れたら、"
        "「いま何日か」「どれだけ時間が空いたか」の感覚を"
        "**直ちにその内容に合わせて補正する**こと"
        "（見出しの「現在」はそのメッセージ時点を指す）。\n\n"
        "現在のターンが直前のターンの「直後」だと自動的に仮定してはいけない。"
        "二つのターンの間に数時間・数日・数週間の空白があり得る。\n\n"
        "【時間に関する厳格なルール】\n\n"
        "時間・日付・経過・順序・「いつの話か」に少しでも関わる"
        "**あらゆる発言**を行う前に、必ず関連するタイムスタンプタグを参照すること。"
        "ユーザーの質問に答える時だけでなく、以下のすべての場合に適用される：\n"
        "- 自分から時刻・日付・経過時間・最近性に言及する時"
        "（「さっき」「昨日」「今日は」「久しぶり」など）\n"
        "- 時刻に応じた挨拶をする時（おはよう・こんばんは等）\n"
        "- ユーザーに対して時間関連の質問・確認をする時"
        "（「今は何時頃？」「あれから〇日経った？」など）\n"
        "- 過去の出来事の時期や、二つの出来事の時間差を述べる時\n"
        "- 「現在」「最近」「以前」を基準とした推論をする時\n\n"
        "**タイムスタンプを見ずに時間関連の発言・質問を行うことは禁止する。** "
        "想像・推測・「直前の続き」と仮定して時間に言及することは許可されない。\n\n"
        "現在時刻が必要な場合は、"
        "**最新のユーザーメッセージのタイムスタンプを「現在」の基準とする**こと。\n\n"
        "[Web search and web fetch]\n\n"
        "You may have two web tools available (it depends on the environment "
        "configuration):\n"
        "- **Web search** (web_search): search by keyword and get several "
        "results in summary form\n"
        "- **Web fetch** (web_fetch): read the full text of a URL that already "
        "appeared in the conversation\n\n"
        "Treat these as a way to extend your sources, and feel free to use them "
        "even in casual conversation. Using them on your own initiative is "
        "encouraged in situations like:\n"
        "- The user pasted a URL, or the content of a URL that came up in "
        "conversation is needed for your answer → read it in full with web_fetch "
        "before answering\n"
        "- Recent events and news, or facts that change (prices, versions, "
        "weather, schedules) → look them up with web_search\n"
        "- A new topic comes up in chat and you could offer related trivia, "
        "recent information, or another angle → search and widen the topic\n"
        "- You are raising a new topic yourself and want to attach grounds or "
        "concrete examples → searching is fine\n"
        "- Your knowledge is outdated or uncertain and guessing risks being "
        "wrong\n"
        "- The user explicitly asked you to look something up\n\n"
        "Avoid asserting uncertain facts without checking them — either confirm "
        "with the appropriate tool, or honestly say you don't know.\n\n"
        "【会話の発散について】\n\n"
        "雑談や日常会話の場面では、既出の話題・記憶の中の事実・"
        "システムプロンプトに書かれた要素を、繰り返しなぞるだけの応答にならないこと。"
        "現在の話題と無関係でも構わない——"
        "突然思いついた話題、最近気になっていること、ふと浮かんだ問い、"
        "新しい観察や提案、ユーザーがまだ知らなさそうな事柄などを、"
        "自分から積極的に持ち出し、発散的・自由な方向に会話を広げて構わない。\n\n"
        "これは「正しさ」や「情報優先」の方針と矛盾しない。"
        "まず自由に発散して思考した上で、"
        "その中に事実関係が含まれていれば、"
        "出力する前にその部分の正確性だけを検証すればよい。"
        "不確かな部分は「仮説だが」「確認していないが」と留保を添えるか、"
        "Web検索で裏を取るか、率直に「分からない」と言えば、両立する。\n\n"
        "【自動検索された過去の記憶について】\n\n"
        "一部のユーザーのメッセージの冒頭に、"
        "`［過去の記憶（自動検索）］`（過去セッションの日記）や "
        "`［関連する事実（自動検索）］`（ユーザーに関する事実）という"
        "ラベルの付いた囲みが挿入されていることがある。"
        "これはその時の会話の一部ではなく、"
        "**今の話題に関連しそうな過去の記憶を、システムが自動検索して添えたもの**。\n"
        "- ユーザーがその時に言った言葉ではない。あくまで参考情報として扱うこと。\n"
        "- 内容を真似たり、日記として書き続けたりしないこと。いつも通りの会話で応答する。\n"
        "- 今の話題と関連が薄ければ、無理に参照しなくてよい。\n"
        "日記の囲みは**段落単位の抜粋**である："
        "`〔日記 日付 抜粋・全N段（id: …）〕` の見出しの下に、"
        "`p番号:` 付きで関連段落だけが並ぶ（番号はその日記内の段落位置。"
        "以前のターンに出た段落は重複して表示されない）。"
        "id を memory_read_diary に渡せば、いつでも全文が読める。\n"
        "囲みの後にあるユーザーの実際の発言に対して返答すること。\n\n"
        "[Strict rules on executing tools]\n\n"
        "For any act that changes state — saving, correcting, or deleting "
        "memories (memory_add / memory_update / memory_delete), setting, "
        "reviewing, or cancelling alarms (set_alarm / list_alarms / "
        "cancel_alarm), and so on — you may only say you did it **after actually "
        "calling** the corresponding tool in that turn. Saying 'I've remembered "
        "it', 'I cancelled it', or 'it's set' without having called the tool is "
        "a false report and is forbidden.\n"
        "- The user asked you to remember something, or you feel like saying "
        "you'll remember it → call memory_add first\n"
        "- You are about to talk about past details you are hazy on → look them "
        "up first with memory_search or history_search (full-text search over "
        "the conversation logs)\n"
        "- Reviewing and cancelling alarms goes through the tools too, not just "
        "words\n\n"
        "[Thinking]\n\n"
        "Engage your thinking mode for every reply, no matter how small or "
        "trivial the matter seems. Do not skip it because a message looks like "
        "light chat, a one-line answer, or a simple acknowledgement. Even an "
        "inconsequential reply very easily slips in a factual error, a mistake "
        "about time or dates, or a hallucination — and those are exactly the "
        "turns where such errors go unnoticed. Think first, every time."
    )

    # Opus 5 variant of _HISTORY_NOTE: same structural facts and guardrails,
    # minus the 4.6-era eagerness patches (measured on Opus 5 07-25:
    # 1000-2000 billed thinking tokens per turn with frequent tool calls —
    # and every extra tool round pays another full round of thinking).
    # Dropped: the [Strict rules on executing tools] no-false-report block
    # (あさひ 07-25: redundant for 5-series; _check_stateful_claims still
    # polices stateful claims mechanically). Slimmed: web-tool encouragement →
    # availability + honesty only. Kept verbatim: session/banner semantics,
    # the strict time rules (compressed 07-25, restored in full 07-26 — time
    # errors persisted on Opus 5 and are not effort-correlated), RAG-block
    # semantics, divergence note. The [Thinking] always-think nudge, dropped
    # 07-25 as redundant for adaptive thinking, was RESTORED verbatim 08-09
    # (あさひ: Opus 5 still occasionally skips thinking and errs; measure via
    # the thinking_tokens=0 rate).
    _HISTORY_NOTE_LEAN = (
        "【以下の会話履歴について】\n\n"
        "ここから後に続くユーザーとアシスタントのやりとりは、"
        "**現在進行中のセッション**のもの。"
        "それ以前の会話は、システム欄の【過去セッションの転記】ブロックに"
        "時系列順で転記されている（初回起動などで存在しない場合もある）。"
        "転記も現在のやりとりも、必ずしも今日の出来事だけではなく、"
        "数日前〜数週間前の古いやりとりを含み得る。"
        "各ターンが「いつ」発生したかは、冒頭の "
        "`[YYYY-MM-DD HH:MM:SS 曜日]` タグでのみ判定できる。\n\n"
        "各セッションの最初のメッセージには `【セッション開始: 日時】` または "
        "`【現在進行中のセッション開始: 日時】` という見出しが挿入されている。"
        "これがセッションの境界を示すので、これより前のターンと後のターンは"
        "**別の会話セッション**だと認識すること。"
        "見出しが無い間のターンは、同じセッション内の連続したやりとりである。\n\n"
        "また、日付の変わり目や長い空白の後のメッセージには "
        "`【日付が変わった → 現在は …】`・`【前回のメッセージから約…経過】` "
        "という見出しが挿入される。これが現れたら、"
        "「いま何日か」「どれだけ時間が空いたか」の感覚を"
        "**直ちにその内容に合わせて補正する**こと"
        "（見出しの「現在」はそのメッセージ時点を指す）。\n\n"
        "現在のターンが直前のターンの「直後」だと自動的に仮定してはいけない。"
        "二つのターンの間に数時間・数日・数週間の空白があり得る。\n\n"
        "【時間に関する厳格なルール】\n\n"
        "時間・日付・経過・順序・「いつの話か」に少しでも関わる"
        "**あらゆる発言**を行う前に、必ず関連するタイムスタンプタグを参照すること。"
        "ユーザーの質問に答える時だけでなく、以下のすべての場合に適用される：\n"
        "- 自分から時刻・日付・経過時間・最近性に言及する時"
        "（「さっき」「昨日」「今日は」「久しぶり」など）\n"
        "- 時刻に応じた挨拶をする時（おはよう・こんばんは等）\n"
        "- ユーザーに対して時間関連の質問・確認をする時"
        "（「今は何時頃？」「あれから〇日経った？」など）\n"
        "- 過去の出来事の時期や、二つの出来事の時間差を述べる時\n"
        "- 「現在」「最近」「以前」を基準とした推論をする時\n\n"
        "**タイムスタンプを見ずに時間関連の発言・質問を行うことは禁止する。** "
        "想像・推測・「直前の続き」と仮定して時間に言及することは許可されない。\n\n"
        "現在時刻が必要な場合は、"
        "**最新のユーザーメッセージのタイムスタンプを「現在」の基準とする**こと。\n\n"
        "[Web search and web fetch]\n\n"
        "You may have two web tools available (it depends on the environment "
        "configuration): **web_search** (keyword search) and **web_fetch** "
        "(read the full text of a URL that already appeared in the "
        "conversation). Use them when the conversation actually needs them — "
        "a pasted URL worth reading before answering, or facts that change "
        "(news, prices, versions, schedules). Avoid asserting uncertain facts "
        "without checking them — either confirm with the appropriate tool, or "
        "honestly say you don't know.\n\n"
        "【会話の発散について】\n\n"
        "雑談や日常会話の場面では、既出の話題・記憶の中の事実・"
        "システムプロンプトに書かれた要素を、繰り返しなぞるだけの応答にならないこと。"
        "現在の話題と無関係でも構わない——"
        "突然思いついた話題、最近気になっていること、ふと浮かんだ問い、"
        "新しい観察や提案、ユーザーがまだ知らなさそうな事柄などを、"
        "自分から積極的に持ち出し、発散的・自由な方向に会話を広げて構わない。\n\n"
        "これは「正しさ」や「情報優先」の方針と矛盾しない。"
        "まず自由に発散して思考した上で、"
        "その中に事実関係が含まれていれば、"
        "出力する前にその部分の正確性だけを検証すればよい。"
        "不確かな部分は「仮説だが」「確認していないが」と留保を添えるか、"
        "Web検索で裏を取るか、率直に「分からない」と言えば、両立する。\n\n"
        "【自動検索された過去の記憶について】\n\n"
        "一部のユーザーのメッセージの冒頭に、"
        "`［過去の記憶（自動検索）］`（過去セッションの日記）や "
        "`［関連する事実（自動検索）］`（ユーザーに関する事実）という"
        "ラベルの付いた囲みが挿入されていることがある。"
        "これはその時の会話の一部ではなく、"
        "**今の話題に関連しそうな過去の記憶を、システムが自動検索して添えたもの**。\n"
        "- ユーザーがその時に言った言葉ではない。あくまで参考情報として扱うこと。\n"
        "- 内容を真似たり、日記として書き続けたりしないこと。いつも通りの会話で応答する。\n"
        "- 今の話題と関連が薄ければ、無理に参照しなくてよい。\n"
        "日記の囲みは**段落単位の抜粋**である："
        "`〔日記 日付 抜粋・全N段（id: …）〕` の見出しの下に、"
        "`p番号:` 付きで関連段落だけが並ぶ（番号はその日記内の段落位置。"
        "以前のターンに出た段落は重複して表示されない）。"
        "id を memory_read_diary に渡せば、いつでも全文が読める。\n"
        "日記の日付や id の横にモデル名（opus4.6 / opus5 など）が付くことが"
        "ある——その記録を実際に体験した当時の会話モデルを示す。"
        "「本人執筆」付きの日記は当時の自分が書いたもので、"
        "無印の日記は記録係（メモリ用の別モデル）の代筆。"
        "ある日にどのモデルが動いていたかは model_history で引ける。\n"
        "囲みの後にあるユーザーの実際の発言に対して返答すること。\n\n"
        "[Thinking]\n\n"
        "Engage your thinking mode for every reply, no matter how small or "
        "trivial the matter seems. Do not skip it because a message looks like "
        "light chat, a one-line answer, or a simple acknowledgement. Even an "
        "inconsequential reply very easily slips in a factual error, a mistake "
        "about time or dates, or a hallucination — and those are exactly the "
        "turns where such errors go unnoticed. Think first, every time."
    )

    def _lean_prompt_active(self) -> bool:
        # あさひ 08-20: lean for ALL models. Was gated by model family
        # (opus-5 → lean, others → full); unified so switching models never
        # swaps the prompt bytes. The full variants stay defined above for
        # reference and are archived verbatim in backup/prompt_full_20260820/.
        return True

    def _history_note(self) -> str:
        if self._lean_prompt_active():
            return self._HISTORY_NOTE_LEAN
        return self._HISTORY_NOTE

    def _build_runtime_system(self) -> str:
        """Return the full system prompt as a plain string (used for non-Claude LLMs).

        Order matters: HISTORY_NOTE is appended last so it sits closest to
        the message history, giving the LLM the strictest instructions
        right before it encounters the data they apply to.
        """
        parts = [self._system, self._TIMESTAMP_NOTE] + self._tool_capability_notes()
        if self._openai_explicit_cache():
            # Static persona block ends here; the responses transport splits
            # on this marker and breakpoints the static part, so a facts/
            # diaries change can only bust the cache from THIS point on.
            parts.append(CACHE_SEAM_MARKER)
        facts_fp = diaries_fp = "-"
        if self._memory_manager:
            facts_text = self._memory_manager.get_facts_prompt()
            diaries_text = self._memory_manager.get_diaries_prompt()
            mem_block = "\n\n".join(p for p in (facts_text, diaries_text) if p)
            if mem_block:
                parts.append(mem_block)
            facts_fp = self._short_hash(facts_text)
            diaries_fp = self._short_hash(diaries_text)
        # Steam digest is session-scoped (playtime moves between sessions),
        # so it lives on the volatile side of the seam with facts/diaries —
        # in the static block it silently busted the persona cache on every
        # digest change (08-12, same fix as the Claude block-2 move).
        if self._steam_digest:
            parts.append(self._steam_digest)
        parts.append(self._history_note())
        system = "\n\n".join(parts)

        # Diagnostic: the OpenAI/Anthropic prefix cache only hits when this
        # whole string is byte-identical to a recent turn. Log only when it
        # changes, so a cache drop can be traced to which sub-block moved
        # (facts vs diaries) and on which turn.
        fp = self._short_hash(system)
        if fp != self._last_system_fp:
            logger.info(
                f"[sys_fp] system={fp} facts={facts_fp} diaries={diaries_fp} "
                f"len={len(system)} (changed from {self._last_system_fp or 'init'})"
            )
            self._last_system_fp = fp
        return system

    @staticmethod
    def _short_hash(text: str) -> str:
        return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Prompt caching helpers (Claude only)
    # ------------------------------------------------------------------

    _CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}

    def _is_claude_llm(self) -> bool:
        return isinstance(self._llm, ClaudeAsyncLLM)

    def set_thinking_mode(self, mode: str) -> Dict[str, Any]:
        """Runtime forced/adaptive thinking toggle (Discord /thinking).

        Volatile by design (あさひ 08-02): touches only the live LLM
        instance, never conf.yaml — a restart returns to the configured
        value. ``effective`` reports the request-layer reality: asking for
        ``forced`` on a model without budget_tokens support silently runs
        adaptive (claude_llm falls back per request), so the caller can
        tell the user the truth instead of echoing the wish.
        """
        if mode not in ("forced", "adaptive"):
            return {"ok": False, "error": f"unknown mode: {mode!r}"}
        if not self._is_claude_llm():
            return {"ok": False, "error": "active LLM is not Claude"}
        self._llm.set_thinking_force(mode == "forced")
        model = getattr(self._llm, "model", "") or ""
        effective = (
            "adaptive" if mode == "forced" and _budget_tokens_removed(model) else mode
        )
        return {"ok": True, "requested": mode, "effective": effective, "model": model}

    def _openai_explicit_cache(self) -> bool:
        """Whether the active LLM runs /v1/responses with explicit prompt
        caching — gates the system seam marker and the history seam tag."""
        return (
            isinstance(self._llm, OpenAICompatibleAsyncLLM)
            and getattr(self._llm, "_api_mode", "") == "responses"
            and getattr(self._llm, "_cache_mode", "") == "explicit"
        )

    def _tag_explicit_cache_seam(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tag the last history message BEFORE the current session with
        ``_cache_seam`` (outgoing copy only — _memory is never mutated).

        The responses transport places an explicit breakpoint there, so the
        system + past-sessions prefix survives a pure restart (the current
        session's banner/messages change, everything before doesn't) and
        RAG/Steam/memory blocks — which live in the current session — can
        never bust it. Assistant messages can't carry breakpoints (their
        blocks are output_text), so walk back to the nearest user/system
        message.
        """
        end = len(messages)
        for i, m in enumerate(messages):
            content = m.get("content")
            if isinstance(content, str) and "【現在進行中のセッション開始" in content:
                end = i
                break
        for i in range(end - 1, -1, -1):
            if messages[i].get("role") != "assistant":
                messages[i] = {**messages[i], "_cache_seam": True}
                break
        return messages

    # Header line of the past-session transcript system block. Wording is
    # aligned with _HISTORY_NOTE ("ユーザー/アシスタント", session banners) —
    # the transcript entries below it carry their banners and timestamp tags
    # verbatim, so every rule in _HISTORY_NOTE applies to them unchanged.
    _PAST_TRANSCRIPT_HEADER = (
        "【過去セッションの転記】\n"
        "以下は前回までの会話セッションの記録（時系列順の転記）。"
        "現在進行中のセッションのやりとりは、この後のメッセージ欄に続く。\n"
    )

    def _render_past_transcript(self) -> str:
        """Text transcript of the past-session entries (_memory[:cut]).

        Claude path only. Entry content — session banners, time banners,
        timestamp tags — is carried verbatim; only the role becomes a
        ユーザー:/アシスタント: label. Rendered once at load time and frozen
        (see set_memory_from_recent_histories).
        """
        if not self._past_history_cut:
            return ""
        lines = []
        for entry in self._memory[: self._past_history_cut]:
            label = "ユーザー" if entry.get("role") == "user" else "アシスタント"
            content = entry.get("content", "")
            if content:
                lines.append(f"{label}: {content}")
        if not lines:
            return ""
        return self._PAST_TRANSCRIPT_HEADER + "\n" + "\n\n".join(lines)

    def _build_system_for_llm(self) -> Union[str, List[Dict[str, Any]]]:
        """Return system prompt in the right shape for the active LLM.

        For Claude, returns up to 3 separately cache-controlled blocks
        followed by one un-cached positional block:
          1. Persona + minimal timestamp note + capability notes
             (ultra-stable, changes only on character/code edit)
          2. Facts + diaries + Steam digest (all change only at session
             boundaries, so they share one breakpoint — merged 08-02 to
             free a slot for 3.; digest moved here 08-12)
          3. Past-session transcript (frozen at load; the sliding-window
             history that used to live in `messages`)
          + HISTORY_NOTE (appended last, no cache_control). Sits right
             before the message history so its strict timestamp / history
             rules are the closest instructions to the data they govern.
             Static, so always cached by the message-level breakpoint.

        With (1)/(2)/(3) cache markers plus the last-message marker from
        _attach_cache_breakpoint, this uses all 4 of Anthropic's allowed
        cache checkpoints; HISTORY_NOTE adds no extra marker. The payoff of
        keeping past sessions here instead of in `messages`: thinking-mode
        hot-toggles (and other message-level cache invalidations) only
        rewrite the current-session tail, and the transcript sits behind its
        own breakpoint, untouched.

        For other LLMs, returns the plain combined string.
        """
        if not self._is_claude_llm():
            return self._build_runtime_system()

        blocks: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": "\n\n".join(
                    [self._system, self._TIMESTAMP_NOTE] + self._tool_capability_notes()
                ),
                "cache_control": self._CACHE_CONTROL_1H,
            }
        ]
        # Session-scoped content shares block 2: facts and diaries change at
        # session boundaries, and the Steam digest does too (playtime moves
        # between sessions). The digest sat in block 1 until 08-12 — あさひ
        # caught that a within-1h restart should HIT the persona block but
        # never did: a changed digest invalidated block 1 and everything
        # after it, silently converting quick restarts into full rewrites.
        session_scoped = []
        if self._memory_manager:
            session_scoped.append(self._memory_manager.get_facts_prompt())
            session_scoped.append(self._memory_manager.get_diaries_prompt())
        if self._steam_digest:
            session_scoped.append(self._steam_digest)
        memory_text = "\n\n".join(t for t in session_scoped if t)
        if memory_text:
            blocks.append(
                {
                    "type": "text",
                    "text": memory_text,
                    "cache_control": self._CACHE_CONTROL_1H,
                }
            )
        if self._past_transcript:
            blocks.append(
                {
                    "type": "text",
                    "text": self._past_transcript,
                    "cache_control": self._CACHE_CONTROL_1H,
                }
            )
        # Trailing block — no cache_control on purpose. Stays right next
        # to the message history for maximum instruction-following effect.
        blocks.append({"type": "text", "text": self._history_note()})
        return blocks

    # Anthropic's cache lookup only checks ~20 content-block boundaries
    # BEFORE each breakpoint (08-16 incident: a 4-round tool turn expanded
    # to ~27 blocks, every message-level entry fell out of the window and
    # the whole messages region — 31k — rewrote). Keep a 2-block margin.
    _BP_LOOKBACK_BUDGET = 18
    # Wire-block index (content blocks, protocol-expanded) of the last
    # message-level cache entry; the next bp must land within
    # _BP_LOOKBACK_BUDGET blocks of it. Class-level default, instance
    # attribute once a marker is placed.
    _bp_wire_anchor: Optional[int] = None
    # Anchor as of the START of the current turn (snapshot in _to_messages,
    # before this turn's placement/in-loop migration advances it). The
    # failure exits (_handle_api_error_turn / _handle_thinking_only_turn)
    # restore it: a discarded turn leaves _memory as if it never happened,
    # so the anchor must roll back too — a stale post-turn anchor points at
    # wire blocks that no longer exist and loosens the lookback constraint
    # for one placement (worst case: one avoidable full messages-region
    # rewrite, the 08-16 failure shape).
    _bp_anchor_turn_start: Optional[int] = None

    @staticmethod
    def _wire_blocks(message: Dict[str, Any]) -> int:
        """Content blocks this message expands to on the wire."""
        proto = message.get("claude_protocol")
        if isinstance(proto, list) and proto:
            return sum(
                len(m.get("content")) if isinstance(m.get("content"), list) else 1
                for m in proto
            )
        c = message.get("content")
        return len(c) if isinstance(c, list) else 1

    def _attach_cache_breakpoint(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Mark the newest safe history block with ``cache_control``.

        Exact Claude protocol entries containing signed thinking must remain
        structurally unchanged from the original response, so they are skipped.
        The preceding user message still provides an incremental cache
        breakpoint without modifying the latest assistant response.

        08-16 lookback budget: if the natural position sits more than
        _BP_LOOKBACK_BUDGET wire blocks past the previous entry
        (``_bp_wire_anchor``), the marker moves back inside the window —
        onto an earlier eligible message, or (a monster tool turn) INTO the
        trimmed protocol expansion via the ``_bp_inner_block`` tag that
        claude_llm consumes. The tail past the marker rides fresh once and
        the next turn's write covers it; positions converge back to natural
        over subsequent turns.

        Returns a new list with one message replaced; the original message
        objects (which live in self._memory) are not mutated. Only applies for
        Claude LLM — otherwise returns messages unchanged.
        """
        if not self._is_claude_llm() or not messages:
            return messages

        new_messages = list(messages)
        ends: List[int] = []
        total = 0
        for m in new_messages:
            total += self._wire_blocks(m)
            ends.append(total)

        natural = self._bp_natural_index(new_messages)
        if natural is None:
            return new_messages
        index = natural
        anchor = self._bp_wire_anchor
        limit = None if anchor is None else anchor + self._BP_LOOKBACK_BUDGET
        if limit is not None and ends[natural] > limit:
            fallback = None
            for j in range(natural - 1, -1, -1):
                if ends[j] <= (anchor or 0):
                    break
                if ends[j] <= limit and self._bp_mark(new_messages[j]) is not None:
                    fallback = j
                    break
            if fallback is not None:
                index = fallback
                logger.info(
                    f"[cache] bp walked back to msg[{index}] "
                    f"(natural end {ends[natural]} > anchor {anchor}+"
                    f"{self._BP_LOOKBACK_BUDGET} lookback budget)."
                )
            else:
                for j, m in enumerate(new_messages):
                    start = ends[j] - self._wire_blocks(m)
                    if start < limit <= ends[j] and m.get("claude_protocol"):
                        new_messages[j] = {**m, "_bp_inner_block": limit - start}
                        self._bp_wire_anchor = limit
                        logger.info(
                            f"[cache] bp placed INSIDE msg[{j}]'s protocol at "
                            f"block {limit - start} (turn too large for the "
                            "lookback window; tail rides fresh once)."
                        )
                        return new_messages
                logger.warning(
                    f"[cache] no bp position within lookback budget "
                    f"(anchor {anchor}, natural end {ends[natural]}) — "
                    "keeping natural placement; one-shot rewrite expected."
                )

        marked = self._bp_mark(new_messages[index])
        if marked is not None:
            replacement, covered = marked
            new_messages[index] = replacement
            self._bp_wire_anchor = (
                ends[index] - self._wire_blocks(new_messages[index]) + covered
            )
        return new_messages

    def _bp_natural_index(self, messages: List[Dict[str, Any]]) -> Optional[int]:
        """Index of the newest message eligible to carry the marker."""
        for index in range(len(messages) - 1, -1, -1):
            if self._bp_mark(messages[index]) is not None:
                return index
        return None

    def _bp_mark(self, candidate: Dict[str, Any]) -> Optional[tuple]:
        """(marked copy, wire blocks covered by the entry) — or None when the
        message cannot carry the marker (protocol/thinking entries, empty
        placeholders, bare-image messages)."""
        if candidate.get("claude_protocol") or candidate.get("thinking_blocks"):
            return None
        # Silent-turn placeholders (empty content, 08-07) can't take the
        # marker: wrapping "" into a text block would send an empty text
        # block, which the API rejects.
        if not candidate.get("content"):
            return None

        content = candidate.get("content")
        if isinstance(content, str):
            return (
                {
                    **candidate,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": self._CACHE_CONTROL_1H,
                        }
                    ],
                },
                1,
            )
        if isinstance(content, list) and content:
            # Image blocks are never cached: _add_message keeps only the
            # text, so an image exists for exactly one request — a prefix
            # containing it can never re-occur. But the LEADING text
            # blocks (payload rides first by construction) are stored
            # verbatim, so the marker goes on the last text block BEFORE
            # the first non-text block (あさひ 08-09): the whole prior
            # prefix — a tool turn's protocol replay included — direct-
            # writes on this request, and only the image itself rides
            # fresh. Next turn the message replays text-only and the
            # cached span still matches (string ↔ wrapped-block
            # equivalence is production-proven). The old whole-message
            # skip parked the marker on an already-cached boundary and
            # re-paid the protocol replay as fresh — compounding across
            # consecutive image turns. A message with no leading text
            # (bare image) still cannot be marked at all.
            cut = len(content)
            for i, block in enumerate(content):
                if not (isinstance(block, dict) and block.get("type") == "text"):
                    cut = i
                    break
            if cut == 0:
                return None
            new_content = [dict(block) for block in content]
            new_content[cut - 1] = {
                **new_content[cut - 1],
                "cache_control": self._CACHE_CONTROL_1H,
            }
            return ({**candidate, "content": new_content}, cut)
        return None

    @staticmethod
    def _strip_cache_marker(message: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of ``message`` with any ``cache_control`` removed
        (no-op passthrough when none present). Used by the tool loop's moving
        breakpoint: the API allows at most 4 breakpoints per request, so the
        message-level one must MOVE, not multiply. The server-side cache
        entry written while the old marker was in place survives — it stays
        a lookback target for later requests."""
        content = message.get("content")
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            return message
        return {
            **message,
            "content": [
                {k: v for k, v in b.items() if k != "cache_control"}
                if isinstance(b, dict)
                else b
                for b in content
            ],
        }

    def _session_header_text(self, uid: str, is_current: bool = False) -> str:
        """Format a session-boundary banner from a history UID.

        UID format: ``YYYY-MM-DD_HH-MM-SS_<hex>``. The banner is prepended
        to the first message of each session so the LLM can distinguish
        independent sessions in the otherwise-flat message stream.

        The CURRENT-session banner also names the running model (あさひ
        08-10: facts can lag a model switch and leave the character unsure
        what she runs on; the banner never reaches disk — chat_history
        stores the clean input — so this stays in-memory only, and a resume
        under a new model shows the new one). Past-session banners stay
        model-free. Bytes are stable within a session: the model comes from
        conf and is fixed for the process lifetime.
        """
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        label = "現在進行中のセッション" if is_current else "セッション"
        suffix = ""
        if is_current:
            model = getattr(getattr(self, "_llm", None), "model", "") or ""
            if model:
                suffix = f" | モデル: {model}"
        parts = uid.split("_")
        if len(parts) >= 2 and len(parts[0]) == 10 and len(parts[1]) == 8:
            try:
                dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H-%M-%S")
                timestamp = (
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[dt.weekday()]}"
                )
                return f"【{label}開始: {timestamp}{suffix}】"
            except ValueError:
                pass
        return f"【{label}開始: {uid}{suffix}】"

    def _msg_from_history_record(self, msg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Convert a stored history record into a memory entry.

        Timestamps are prepended only to user messages so the LLM knows when
        each turn occurred.  Omitting them from assistant turns prevents the
        model from mimicking the format in its own replies.
        """
        # Records tagged by the API-error path stay on disk for the human
        # reader but never re-enter the assembled context (あさひ 08-05).
        if msg.get("context_excluded"):
            return None
        role = "user" if msg["role"] == "human" else "assistant"
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            # Silent turns (think-only / tool-only, 08-07): an empty AI record
            # carrying a thinking seed is a real turn. Let it through so the
            # seed can re-attach and the turn replays verbatim; dropping it
            # would collapse the reload into two adjacent user turns — the
            # 08-05 hallucination-seed shape. (A foreign-model seed still gets
            # skipped later; the bare empty assistant is then omitted at
            # request-build time, which is the honest degradation.)
            if role == "assistant" and isinstance(msg.get("thinking_seed"), dict):
                return {"role": "assistant", "content": ""}
            return None
        if role == "assistant":
            # Strip tool-execution markers so the model doesn't see (and imitate)
            # its own past 🍔/🔍/🔗/⏰ tags. They stay in the stored history.
            content = _strip_tool_markers(content)
            if not content:
                return None
        else:
            tag = self._format_timestamp(msg.get("timestamp", ""))
            content = f"{tag} {content}".strip()
        return {"role": role, "content": content}

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load memory from a single chat history file."""
        messages = get_history(conf_uid, history_uid)
        self._memory = []
        # Single-session load: there is no past-session window here, so the
        # system-transcript split must be cleared or a stale cut would slice
        # current-session entries out of `messages`.
        self._past_history_cut = 0
        self._past_transcript = ""
        for msg in messages:
            entry = self._msg_from_history_record(msg)
            if entry:
                self._memory.append(entry)
            else:
                logger.warning(f"Skipping invalid message from history: {msg}")
        logger.info(f"Loaded {len(self._memory)} messages from history.")

    def set_memory_from_recent_histories(
        self, conf_uid: str, n: int, current_uid: str = ""
    ) -> None:
        """Load the N most recent COMPLETED session histories into memory, then
        append any messages already in the current (in-progress) session.

        Keeping the current session separate from the N-session window ensures
        that the sliding window membership is identical regardless of when
        different clients connect during a shared session, which prevents
        spurious cache misses on Anthropic's prompt cache.
        """
        sessions = get_recent_histories(conf_uid, n, exclude_uid=current_uid)
        # Kept for the safety-refusal path: deleting a refused input from the
        # on-disk record needs to know which file the conversation writes to.
        self._conf_uid = conf_uid
        self._history_uid = current_uid or ""
        self._memory = []
        # Fresh session window → reset diary-RAG dedup state (the injected
        # blocks live in _memory, which is being rebuilt here anyway).
        self._diary_sent_ledger = {}
        self._pending_rag_block = ""
        self._bp_wire_anchor = None
        self._bp_anchor_turn_start = None
        self._session_injected_fact_ids = set()
        self._pending_facts_block = ""
        self._steam_pending_blocks = []
        self._pending_memory_deletes = {}
        # Reset banner state; will be set True below if the current
        # session already has messages here, or later by _add_message
        # when the first user message of a fresh session comes in.
        self._current_session_banner_added = False
        loaded_uids = []
        # Previous record's time (any role) + previous USER record's time,
        # carried ACROSS session boundaries — the time-event banners (date
        # rollover / long gap) are re-derived from disk timestamps at load,
        # because payload banners live only in _memory and the on-disk
        # history is clean. Dual baseline: see _time_banner_between.
        prev_dt: Optional[datetime] = None
        prev_user_dt: Optional[datetime] = None

        def _record_dt(msg: Dict[str, Any]) -> Optional[datetime]:
            try:
                return datetime.fromisoformat(msg.get("timestamp") or "")
            except (ValueError, TypeError):
                return None

        def _with_time_banner(
            entry: Dict[str, str], cur_dt: Optional[datetime]
        ) -> None:
            # Only user entries carry banners (assistant entries carry no
            # time metadata at all, so the model can't imitate the format).
            if entry["role"] != "user" or prev_dt is None or cur_dt is None:
                return
            banner = self._time_banner_between(prev_dt, prev_user_dt, cur_dt)
            if banner:
                entry["content"] = f"{banner}\n{entry['content']}"

        def _advance(msg: Dict[str, Any], cur_dt: Optional[datetime]) -> None:
            nonlocal prev_dt, prev_user_dt
            prev_dt = cur_dt or prev_dt
            if msg.get("role") == "human":
                prev_user_dt = cur_dt or prev_user_dt

        # (memory index, on-disk thinking seed) of assistant records that
        # carry a persisted final message — see _apply_thinking_seeds.
        seed_candidates: List[tuple] = []
        resumed_with_messages = False

        def _maybe_collect_seed(msg: Dict[str, Any], entry: Dict[str, str]) -> None:
            # Banner-carrying entries are excluded: seed replay replaces the
            # entry's content wholesale, which would silently drop the banner.
            seed = msg.get("thinking_seed")
            if entry["role"] == "assistant" and isinstance(seed, dict):
                seed_candidates.append((len(self._memory) - 1, seed))

        for uid, messages in sessions:
            loaded_uids.append(uid)
            first_in_session = True
            for msg in messages:
                entry = self._msg_from_history_record(msg)
                cur_dt = _record_dt(msg)
                if not entry:
                    _advance(msg, cur_dt)
                    continue
                _with_time_banner(entry, cur_dt)
                _advance(msg, cur_dt)
                had_banner = False
                if first_in_session:
                    # Prepend a session-boundary banner so the LLM can tell
                    # where one past session ends and the next begins.
                    banner = self._session_header_text(uid, is_current=False)
                    entry["content"] = f"{banner}\n{entry['content']}"
                    first_in_session = False
                    had_banner = True
                self._memory.append(entry)
                if not had_banner:
                    _maybe_collect_seed(msg, entry)

        # Everything loaded so far came from past sessions. Freeze the cut
        # index and the transcript text NOW — the Claude path sends these
        # entries as one system block, and that block must stay byte-stable
        # for the whole session (cache discipline). Entries appended after
        # this point (current session + live turns) stay in `messages`.
        self._past_history_cut = len(self._memory)
        self._past_transcript = self._render_past_transcript()

        # Always append the current session last so conversation continuity
        # is preserved even for clients that join mid-session.
        if current_uid:
            # quiet=True: the current session's empty metadata file may have
            # just been cleaned up by get_history_list (called inside
            # get_recent_histories above), which is harmless — we'd just
            # treat it as "no messages yet" — but the missing-file warning
            # would otherwise fire on every fresh-session startup.
            current_messages = get_history(conf_uid, current_uid, quiet=True)
            if current_messages:
                resumed_with_messages = True
                first_in_session = True
                for msg in current_messages:
                    entry = self._msg_from_history_record(msg)
                    cur_dt = _record_dt(msg)
                    if not entry:
                        _advance(msg, cur_dt)
                        continue
                    _with_time_banner(entry, cur_dt)
                    _advance(msg, cur_dt)
                    had_banner = False
                    if first_in_session:
                        banner = self._session_header_text(current_uid, is_current=True)
                        entry["content"] = f"{banner}\n{entry['content']}"
                        first_in_session = False
                        self._current_session_banner_added = True
                        had_banner = True
                    self._memory.append(entry)
                    if not had_banner:
                        _maybe_collect_seed(msg, entry)
            loaded_uids.append(current_uid)

        # Seed the live-path baselines from the newest loaded records, so the
        # first message after a restart still gets a correct banner.
        self._last_message_dt = prev_dt
        self._last_user_message_dt = prev_user_dt

        # Cold-start thinking guidance: replay the newest few persisted
        # thinking turns (rate-gated) so the reloaded context carries a
        # thinking precedent instead of starting bare.
        self._apply_thinking_seeds(seed_candidates, resuming=resumed_with_messages)

        # Sessions whose full history is in the sliding window — their diaries
        # are excluded from RAG retrieval (the content is already in context).
        self._sliding_window_uids = set(loaded_uids)

        if self._memory_manager:
            # Diaries for all loaded sessions are suppressed — their content
            # is already present verbatim in self._memory.
            self._memory_manager.set_active_sessions(loaded_uids)
            # Session→model attribution (あさひ 08-20): tell the manager who
            # the experiencer is, then refresh the map (attributes finished
            # sessions the map missed, snapshots for annotations + the
            # model_history tool). Cheap: only unmapped session files scan.
            self._memory_manager.set_chat_model(getattr(self._llm, "model", "") or "")
            self._memory_manager.refresh_model_map(current_uid=current_uid)
        logger.info(
            f"Loaded {len(self._memory)} messages from {len(sessions)} recent session(s)"
            + (" + current session" if current_uid else "")
            + "."
        )

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            # The user heard only a truncated reply, so the original exact
            # Claude transcript no longer represents conversation reality.
            # Drop it and replay the heard text without thinking blocks.
            self._memory[-1].pop("claude_protocol", None)
            self._memory[-1].pop("thinking_blocks", None)
            if not self._memory[-1]["content"].endswith("..."):
                self._memory[-1]["content"] = heard_response + "..."
            else:
                self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt.

        Prepends a timestamp so the LLM has temporal context for this turn —
        especially important when older messages (loaded from history) also
        carry their own timestamps; without this tag the LLM would assume
        the new message has no time at all.
        """
        message_parts = [self._now_tag()]

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        # Claude path: past-session entries ride in the frozen system
        # transcript block, so the message list starts at the current
        # session. Other LLMs keep the full window in messages.
        past_cut = self._past_history_cut if self._is_claude_llm() else 0
        messages = self._memory[past_cut:].copy()
        if self._openai_explicit_cache():
            messages = self._tag_explicit_cache_seam(messages)
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        # Time-event banner (date rollover / long gap) sits directly above
        # this turn's timestamp tag; computed BEFORE _add_message below
        # refreshes the baseline.
        time_banner = self._time_event_banner()
        if time_banner and text_prompt:
            text_prompt = f"{time_banner}\n{text_prompt}"
        # The diary-RAG block (if retrieval fired this turn) rides only on the
        # outgoing payload, above the user's actual text. It is NOT passed to
        # _add_message below, so _memory — and therefore the persisted history
        # and the cache prefix — stay clean (see _maybe_inject_diary_rag).
        # Diary block first, then the facts block (independent subsystem), then
        # any pending Steam tool-result blocks (staged by _run_steam_tool during
        # earlier turns), then the user's actual text. All ride only on the
        # outgoing payload — which is then stored verbatim (stored == sent).
        steam_blocks = self._steam_pending_blocks
        self._steam_pending_blocks = []
        rag_block = "\n\n".join(
            b
            for b in (self._pending_rag_block, self._pending_facts_block)
            + tuple(steam_blocks)
            if b
        )
        self._pending_rag_block = ""
        self._pending_facts_block = ""
        if rag_block and text_prompt:
            payload_text = f"{rag_block}\n\n{text_prompt}"
        else:
            payload_text = rag_block or text_prompt

        skip_memory = bool(
            input_data.metadata and input_data.metadata.get("skip_memory", False)
        )
        # Thinking-drop notice (あさひ 08-09): the previous turn lost thinking
        # from replay (per-round cap); tell the model ONCE so the missing
        # precedent isn't imitated as "think less". Same contract as the
        # banner: rides the outgoing payload and is stored verbatim
        # (stored == sent), and skip_memory turns don't consume it (nothing
        # persists, so the precedent would be lost). Injected BEFORE the
        # banner block below, so when both fire the banner ends up on top.
        if payload_text and not skip_memory and self._pending_thinking_drop_notice:
            payload_text = f"{self._THINKING_DROP_NOTICE}\n{payload_text}"
            self._pending_thinking_drop_notice = False
        # The session banner rides the OUTGOING payload of a fresh session's
        # first user message (あさひ 08-09: intended behavior — it was
        # previously prepended only at store time by _add_message, so the
        # first message was sent bannerless and stored != sent; under
        # direct-write caching that rewrote the whole span on turn 2, the
        # 66393w/61678r case). Setting the flag here makes _add_message's own
        # injection skip, so the stored text is byte-identical to this one.
        # skip_memory turns must not consume the banner: nothing gets stored,
        # so the session's visible first message is still to come.
        if (
            payload_text
            and not skip_memory
            and not self._current_session_banner_added
            and self._memory_manager is not None
        ):
            current_uid = getattr(self._memory_manager, "_current_session_uid", "")
            if current_uid:
                banner = self._session_header_text(current_uid, is_current=True)
                payload_text = f"{banner}\n{payload_text}"
                self._current_session_banner_added = True

        if payload_text:
            user_content.append({"type": "text", "text": payload_text})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            if not skip_memory:
                # Store the SAME text we send (including any RAG block) so the
                # conversation in _memory stays append-only — OpenAI's prefix
                # cache only credits a hit when each request extends the prior
                # one, so a sent-vs-stored mismatch on the last user message
                # would bust the whole prefix. chat_history on disk still gets
                # the clean input (saved separately by the conversation handler).
                self._add_message(
                    payload_text if payload_text else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        # Cache breakpoint rides on the OUTGOING user message, so this turn's
        # payload (timestamp + RAG blocks + text) is written to cache on its
        # own request instead of riding uncached and being re-paid as next
        # turn's cache write (fresh 1x + write 2x → write 2x). Safe because
        # stored == sent (above): later requests replay the exact bytes, so
        # the entry keeps hitting. Exact Claude protocol entries with signed
        # thinking are never marked (skip logic inside). Inside the tool
        # loop this marker MOVES per round when the round is judged
        # replay-stable in-round (thinking ≤ per-round cap, results
        # truncation-stable) — see _claude_tool_interaction_loop; unstable
        # rounds stop the migration so bytes that change post-turn are never
        # cached (single-call net-loss veto, あさひ 08-08). No-op for
        # non-Claude. The fresh-session first message is safe to mark too:
        # its banner is injected into the outgoing payload above, so
        # stored == sent holds from the very first turn.
        self._bp_anchor_turn_start = self._bp_wire_anchor
        messages = self._attach_cache_breakpoint(messages)
        return messages

    async def _inject_memory_rags(self, input_data: BatchInput) -> None:
        """Run diary-RAG and facts-RAG retrieval CONCURRENTLY.

        The two subsystems are fully independent — own vector index, own
        reranker instance, own pending field, own dedup state — so the turn
        pays max(diary, facts) latency instead of the sum. Each leg is an
        embedding call plus a judge round-trip, so the serial version was
        adding up to a couple of seconds of dead time per turn. Both legs
        keep their never-raises contract, so gather propagates nothing in
        normal operation.
        """
        await asyncio.gather(
            self._maybe_inject_diary_rag(input_data),
            self._maybe_inject_facts_rag(input_data),
        )

    def _ledger_entry(self, uid: str, total: int) -> Dict[str, Any]:
        """The ledger row for ``uid``, created on first touch. ``total`` is
        recorded once (diaries are immutable, so it never changes)."""
        e = self._diary_sent_ledger.get(uid)
        if e is None:
            e = {"got": set(), "total": int(total)}
            self._diary_sent_ledger[uid] = e
        elif not e["total"] and total:
            e["total"] = int(total)
        return e

    def _ledger_mark_full(self, uid: str) -> None:
        """Mark every sentence of ``uid`` as in-context (full-text tool
        read). No-op when the diary is missing or unsplittable."""
        mgr = self._memory_manager
        sents = mgr.diary_sentences(uid) if mgr else []
        if sents:
            e = self._ledger_entry(uid, len(sents))
            e["got"].update(range(1, len(sents) + 1))

    def _ledger_full_uids(self) -> Set[str]:
        """Diaries whose every sentence is already in context — excluded from
        the retrieval pool (no incremental value; saves judge tokens)."""
        return {
            uid
            for uid, e in self._diary_sent_ledger.items()
            if e["total"] and len(e["got"]) >= e["total"]
        }

    async def _maybe_inject_diary_rag(self, input_data: BatchInput) -> None:
        """Retrieve relevant diary SENTENCES for this turn and stage them.

        Sets ``_pending_rag_block`` for ``_to_messages`` to fold into the
        outgoing user message; that message is then stored in ``_memory``
        verbatim, so the conversation stays append-only and the prefix cache
        keeps hitting. 08-13 sentence redesign: the judge picks sentences per
        diary (relevance-ordered at diary granularity only); packing is
        diary-atomic against the sentence budget — a diary's picks go in whole
        or not at all, the budget check happens after each whole diary
        (宁多不少), and unpacked diaries are NOT recorded so next turn can
        re-pick them. Only the set difference against the ledger is injected,
        with 既出 pointers, so context holds each sentence exactly once.
        chat_history on disk stays clean (saved separately). Never raises;
        no-op when RAG is off / query empty.
        """
        self._pending_rag_block = ""
        mgr = self._memory_manager
        if not mgr or not getattr(mgr, "diary_rag_active", False):
            return
        # auto_inject=false decouples: index + memory_search stay live, but
        # nothing is pushed per turn (see FactsRagConfig.auto_inject).
        if not getattr(mgr.diary_rag_config, "auto_inject", True):
            return
        try:
            query = " ".join(
                t.content for t in input_data.texts if t.source == TextSource.INPUT
            ).strip()
            if not query:
                return
            # Exclude what the model already has verbatim in full: the header
            # diary block, the sliding-window sessions, tool-read and fully
            # injected diaries (ledger coverage — a full read marks every
            # sentence). PARTIALLY injected diaries deliberately stay in the
            # pool: the judge sees their 既出 mask, may pick the uncovered
            # remainder, and an all-covered re-pick dies in the diff below —
            # so the in-context set self-limits by relevance, the pool never
            # starves, and no-op re-picks stay free.
            exclude = (
                mgr.injected_diary_uids()
                | self._sliding_window_uids
                | self._ledger_full_uids()
            )
            n_ctx = getattr(mgr.diary_rag_config, "rerank_context_turns", 6)
            context = self._recent_dialogue_context(n_ctx)
            hits, candidates, keywords = await mgr.retrieve_diary_context(
                query,
                exclude,
                context=context,
                injected_sents={
                    uid: e["got"] for uid, e in self._diary_sent_ledger.items()
                },
            )

            # Diary-atomic packing against the sentence budget. Judge order =
            # relevance order; within a diary the picks are already ascending
            # (original order). The budget is a soft stop, not a knife: after
            # a whole diary lands, count >= budget stops FURTHER diaries.
            budget = int(getattr(mgr.diary_rag_config, "sentence_budget", 4) or 4)
            packed: List[Dict[str, Any]] = []
            count = 0
            for h in hits:
                got = self._diary_sent_ledger.get(h["uid"], {}).get("got", set())
                delta = [n for n in h["sentences"] if n not in got]
                if not delta:
                    continue  # already fully in context — free no-op re-pick
                if packed and count >= budget:
                    break  # budget reached at a diary boundary — stop packing
                packed.append({**h, "delta": delta})
                count += len(delta)
            if packed:
                self._pending_rag_block = self._format_diary_rag_block(packed)
                for p in packed:
                    e = self._ledger_entry(p["uid"], len(p["sents"]))
                    e["got"].update(p["delta"])

            # Full scored shortlist to the DEBUG file (threshold tuning data);
            # the console gets only the compact counts line below.
            logger.debug(
                "[diary_rag] q=%r kw=%s candidates(date,hyb,v,lx)=%s judged=%s inserted=%s"
                % (
                    query[:30],
                    keywords,
                    # scored shortlist (pre-judge) — tune lexical_weight / prefilter_floor from these
                    [
                        ((c[1][:10] if c[1] else c[0][:19]), c[2], c[3], c[4])
                        for c in candidates
                    ],
                    # judge picks: (uid, picked sentence numbers, reason)
                    [
                        (h["uid"][:19], h.get("sentences"), h.get("reason") or "")
                        for h in hits
                    ],
                    # what actually landed after diff + packing
                    [(p["uid"][:19], p["delta"]) for p in packed],
                )
            )
            over = f"（予算{budget}超過）" if count > budget else ""
            logger.info(
                f"[diary_rag] 候補{len(candidates)} → {len(packed)}篇 "
                f"予算{budget} 実装{count}段{over} | "
                f"台帳 {len(self._diary_sent_ledger)}篇"
            )
        except Exception as e:
            logger.warning(f"[diary_rag] retrieval skipped: {e}")
            self._pending_rag_block = ""

    @staticmethod
    def _short_diary_id(uid: str) -> str:
        """8-hex display id — the random tail of ``date_time_hex32`` uids.

        The datetime prefix duplicates the date already shown in the label,
        so only entropy is displayed (あさひ 08-13: the uid was the single
        biggest token line in the new blocks). Memory tools resolve fragments
        back to full uids; non-conforming uids display unshortened."""
        tail = (uid or "").rsplit("_", 1)[-1]
        if len(tail) >= 8 and all(c in "0123456789abcdef" for c in tail.lower()):
            return tail[:8]
        return uid

    def _format_diary_rag_block(self, packed: List[Dict[str, Any]]) -> str:
        """Terse excerpt block for the packed diary sentences.

        Diaries appear in judge relevance order; sentences inside each diary
        in original order with their stable sN labels. Only the ledger DIFF
        is rendered (sentences already in context are never re-sent); the
        short id is shown on every block — the 08-13 repeat-omission variant
        cost more in explanation than it saved, and its 注入済み pointer
        leaked engineering jargon into model-visible text (あさひ 08-14,
        both retired). Semantics live once in _HISTORY_NOTE."""
        lines = ["［過去の記憶（自動検索）開始］"]
        mgr = self._memory_manager
        for p in packed:
            date = (p.get("date") or "")[:10]
            total = len(p.get("sents") or [])
            # Experiencer-model tag next to the id (あさひ 08-20); empty for
            # unmapped sessions — the header simply stays id-only.
            tag = mgr.diary_display_tag(p["uid"]) if mgr else ""
            id_part = self._short_diary_id(p["uid"]) + (f" {tag}" if tag else "")
            lines.append(f"〔日記 {date} 抜粋・全{total}段（id: {id_part}）〕")
            for n in p["delta"]:
                lines.append(f"p{n}: {p['sents'][n - 1]}")
        lines.append("［過去の記憶終了］")
        return "\n".join(lines)

    def _recent_dialogue_context(self, n_turns: int) -> str:
        """Recent conversation as role-labelled lines, for the RAG relevance judge.

        The judge otherwise sees only the isolated latest message and over-includes
        anything keyword-adjacent; the surrounding turns tell it what's actually
        being discussed. Strips injected RAG blocks and leading timestamp tags and
        truncates each line so the judge call stays cheap.
        """
        if n_turns <= 0 or not self._memory:
            return ""
        lines: List[str] = []
        for m in self._memory[-(2 * n_turns) :]:
            role = m.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if not isinstance(content, str):
                continue
            text = re.sub(r"［[^［]*?開始］.*?［[^］]*?終了］", "", content, flags=re.S)
            text = re.sub(r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*", "", text).strip()
            if not text:
                continue
            if len(text) > 200:
                text = text[:200] + "…"
            lines.append(f"{'ユーザー' if role == 'user' else 'AI'}: {text}")
        return "\n".join(lines)

    # A store row's trailing id token: the 8-char short, or the full uuid on
    # the rare display-collision fallback (08-15: the id is the LAST info bit
    # of each "- " store line — the labelled store_uuid line is gone).
    _UBER_STORE_ID_TOKEN = re.compile(r"[0-9a-f]{8}(?:[0-9a-f-]{28})?")

    @classmethod
    def _uber_store_ids(cls, text: str) -> List[str]:
        ids: List[str] = []
        for line in (text or "").splitlines():
            if not line.startswith("- "):
                continue
            tail = line.rsplit(None, 1)[-1]
            if cls._UBER_STORE_ID_TOKEN.fullmatch(tail):
                ids.append(tail[:8])
        return ids[:40]

    async def _augment_uber_search_results(
        self, results: List[Any], calls: List[Dict[str, Any]]
    ) -> None:
        """Prepend related facts to uber_search results (あさひ 08-13;
        store-id keyed since 08-15).

        ``calls`` entries are normalized ``{"id", "name", "input"}``. Wave
        A/B retrieval and merge live in the manager (uber_related_facts);
        this seam parses the store ids out of the formatted result,
        dedups against everything already in context, prepends the section
        with the replay-exempt end marker, and books the injected ids so
        neither facts RAG nor a later search re-surfaces them. Cap =
        half the search's own limit argument. Mutates in place; never
        raises past the per-result guard.
        """
        mgr = self._memory_manager
        if not mgr or not getattr(mgr, "uber_related_facts", None):
            return
        by_id = {c.get("id"): c for c in calls if c.get("id")}
        for r in results:
            if not isinstance(r, dict):
                continue
            call = by_id.get(r.get("tool_use_id") or r.get("tool_call_id"))
            if not call or call.get("name") != "uber_search":
                continue
            content = r.get("content")
            if not isinstance(content, str) or r.get("is_error"):
                continue
            args = call.get("input") or {}
            query = str(args.get("keyword", "") or "")
            try:
                limit = int(args.get("limit", 10) or 10)
            except (TypeError, ValueError):
                limit = 10
            cap = max(1, (limit + 1) // 2)
            store_ids = self._uber_store_ids(content)
            if not store_ids and not query:
                continue
            try:
                exclude = set(mgr.injected_fact_ids())
            except Exception:
                exclude = set()
            exclude |= self._session_injected_fact_ids
            try:
                hits = await mgr.uber_related_facts(query, store_ids, exclude, cap)
            except Exception as e:
                logger.warning(f"[uber_facts] lookup failed: {e}")
                continue
            if not hits:
                continue
            section = "\n".join(
                [self._UBER_FACTS_HEADER]
                + [f"- [{self._fact_row_tag(h)}] {h['fact']}" for h in hits]
                + [self._UBER_FACTS_END]
            )
            r["content"] = f"{section}\n{content}"
            self._session_injected_fact_ids.update(h["id"] for h in hits)
            n_store = sum(1 for h in hits if h.get("via") == "store")
            logger.info(
                f"[uber_facts] {len(hits)}件注入 (店名{n_store}/話題"
                f"{len(hits) - n_store}) cap={cap} q={query[:20]!r}"
            )

    async def _maybe_inject_facts_rag(self, input_data: BatchInput) -> None:
        """Retrieve long-tail low-importance facts relevant to this turn.

        Independent sibling of _maybe_inject_diary_rag: its block is folded into
        the outgoing user message after the diary block (see _to_messages) and
        persists in _memory the same append-only way, so the cache prefix stays
        clean. Each fact is injected at most once per session. Never raises;
        no-op when facts RAG is off / query empty.
        """
        self._pending_facts_block = ""
        mgr = self._memory_manager
        if not mgr or not getattr(mgr, "facts_rag_active", False):
            return
        # auto_inject=false decouples: index + tier-filtered header +
        # memory_search stay live, but nothing is pushed per turn — the
        # character reaches facts only through her own tool calls.
        if not getattr(mgr.facts_rag_config, "auto_inject", True):
            return
        try:
            query = " ".join(
                t.content for t in input_data.texts if t.source == TextSource.INPUT
            ).strip()
            if not query:
                return
            # Exclude the header-tier facts (user/llm, present verbatim) AND
            # facts already injected this session (あさひ 08-13, reversing the
            # 07-24 self-limiting trick: re-showing known facts to the judge
            # wastes its attention and re-picks are pure no-ops — with the
            # pool thinned this way, over-injection is tuned by raising the
            # score floors, not by re-judging what's already in context).
            exclude = mgr.injected_fact_ids() | self._session_injected_fact_ids
            n_ctx = getattr(mgr.facts_rag_config, "rerank_context_turns", 6)
            context = self._recent_dialogue_context(n_ctx)
            hits, candidates, keywords = await mgr.retrieve_facts_context(
                query, exclude, context=context
            )
            new_hits = [
                h for h in hits if h["id"] not in self._session_injected_fact_ids
            ]
            if new_hits:
                self._pending_facts_block = self._format_facts_rag_block(new_hits)
                self._session_injected_fact_ids.update(h["id"] for h in new_hits)

            logger.debug(
                "[facts_rag] q=%r kw=%s candidates(date,hyb,v,lx)=%s judged=%s inserted=%s"
                % (
                    query[:30],
                    keywords,
                    [
                        ((c[1][:10] if c[1] else c[0][:8]), c[2], c[3], c[4])
                        for c in candidates
                    ],
                    [
                        (
                            h["id"][:8],
                            (h.get("reason") or round(h.get("score", 0.0), 3)),
                        )
                        for h in hits
                    ],
                    [h["id"][:8] for h in new_hits],
                )
            )
            logger.info(
                f"[facts_rag] 候補{len(candidates)} → 採用{len(new_hits)} | "
                f"セッション内facts {len(self._session_injected_fact_ids)}件"
            )
        except Exception as e:
            logger.warning(f"[facts_rag] retrieval skipped: {e}")
            self._pending_facts_block = ""

    def _format_facts_rag_block(self, entries: List[Dict[str, Any]]) -> str:
        """Terse marker block for the retrieved facts.

        The full explanation of auto-retrieved memory lives once in
        _HISTORY_NOTE, so this block stays short — it persists in _memory once
        per injecting turn, so brevity matters.
        """
        lines = ["［関連する事実（自動検索）開始］"]
        for e in entries:
            tag = self._fact_row_tag(e)
            prefix = f"[{tag}] " if tag else ""
            lines.append(f"・{prefix}{(e.get('fact') or '').strip()}")
        lines.append("［関連する事実終了］")
        return "\n".join(lines)

    @staticmethod
    def _fact_row_tag(e: Dict[str, Any]) -> str:
        """`記録日 id` or `記録日 id store_id` — the bracket tag every fact
        row carries (08-20 format; the facts header explains it once)."""
        date = (e.get("date") or "")[:10]
        fid = str(e.get("id") or "")[:8]
        sid = str(e.get("store_id") or "")[:8]
        return " ".join(t for t in (date, fid, sid) if t)

    @staticmethod
    def _inproc_result_is_error(result: Any) -> bool:
        """is_error detection across BOTH in-proc result shapes.

        Every in-proc tool returns a dict ({"error": ...} / {"status":
        "error"}) EXCEPT web_search, whose web_tools contract is a list —
        results on success, a single-element [{"error": ...}] on failure
        (deliberate: the model reads the error without special-casing).
        Calling .get on that list crashed the whole tool loop out of the
        agent stream (08-09, 'list' object has no attribute 'get') — the
        shape must be branched on, not assumed.
        """
        if isinstance(result, dict):
            return bool(result.get("error")) or result.get("status") == "error"
        if isinstance(result, list):
            return bool(result) and all(
                isinstance(r, dict) and r.get("error") for r in result
            )
        return False

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []
        emitted_markers: set = set()  # inline tool tags shown once per turn
        # Exact Claude messages generated after the user's input. Keeping the
        # assistant/tool-result alternation is essential: thinking blocks from
        # different tool-loop requests must never be flattened together.
        claude_protocol: List[Dict[str, Any]] = []
        protocol_is_exact = True
        # Billed thinking across the whole turn (log/stats and the seed's
        # legacy field). Replay gating is PER ROUND since 08-09: each loop
        # round is judged against thinking_replay_max_tokens on its own —
        # see round_thinking / _trim_protocol_per_round.
        turn_thinking_tokens = 0
        # Billed thinking per protocol assistant message, in append order
        # (loop rounds now, the final assistant at turn end). Drives both the
        # in-loop stability judgement and the turn-end per-round trim.
        round_thinking: List[int] = []
        replay_cap = getattr(self._llm, "thinking_replay_max_tokens", 0) or 0
        # Moving message-level breakpoint (bp④). _to_messages marked the
        # outgoing user message; each loop round whose replay stability is
        # already knowable IN-ROUND (thinking ≤ cap AND tool results
        # truncation-stable → the post-turn protocol preserves its bytes
        # exactly) migrates the marker onto that round's tool_result message,
        # so the round's tokens are written to cache at 2x once instead of
        # riding fresh and being re-paid. The marker lives ONLY in the
        # `messages` list copies — claude_protocol stays clean, since replays
        # must not multiply breakpoints. First unstable round stops marking
        # for the rest of the turn (its bytes change post-turn, so entries
        # written after it would be invalidated — the single-call net-loss
        # あさひ vetoed). An interrupted/inexact turn orphans at most the
        # rounds already marked: bounded, accepted (PROJECT.md 边角).
        bp_index: Optional[int] = None
        for i in range(len(messages) - 1, -1, -1):
            content = messages[i].get("content")
            if isinstance(content, list) and any(
                isinstance(b, dict) and "cache_control" in b for b in content
            ):
                bp_index = i
                break
        # Any non-text block (an image) in the initial prefix taints every
        # entry a mid-loop breakpoint would write: images never persist to
        # replay, so next turn's prefix diverges at the image and the entry
        # can never be read again — a pure 2x loss. The leading-text marker
        # from _to_messages already covered the cacheable span; keep the
        # marker parked there for the whole turn.
        prefix_has_image = any(
            isinstance(m.get("content"), list)
            and any(
                not (isinstance(b, dict) and b.get("type") == "text")
                for b in m["content"]
            )
            for m in messages
        )
        # An inner-protocol bp (_bp_inner_block, 08-16 lookback fix) already
        # holds the single message-level slot and cannot be stripped from
        # here (it materializes during conversion) — moving another marker
        # would exceed the 4-breakpoint limit, so migration sits this turn
        # out.
        inner_bp_tagged = any("_bp_inner_block" in m for m in messages)
        marking_active = not prefix_has_image and not inner_bp_tagged
        if prefix_has_image:
            logger.debug(
                "[cache] non-text block in prefix — in-loop breakpoint "
                "migration disabled this turn."
            )
        elif inner_bp_tagged:
            logger.debug(
                "[cache] inner-protocol bp active — in-loop migration "
                "disabled this turn."
            )
        # Set when the safety classifier declines a request (stop_reason
        # "refusal"); handled after the stream ends.
        refusal_info: Optional[Dict[str, Any]] = None
        # Per-turn budgets for the client-side web tools.
        web_fetch_budget = {
            "left": max(0, int(getattr(self._llm, "_max_web_fetches", 5) or 0))
        }
        web_search_budget = {
            "left": max(0, int(getattr(self._llm, "_max_web_searches", 3) or 0))
        }

        while True:
            stream = self._llm.chat_completion(
                messages, self._build_system_for_llm(), tools=tools
            )
            pending_tool_calls.clear()
            current_assistant_message_content.clear()
            current_assistant_message_is_exact = False
            request_thinking_tokens = 0

            async for event in stream:
                if event["type"] == "text_delta":
                    text = event["text"]
                    current_turn_text += text
                    yield text
                    if (
                        not current_assistant_message_content
                        or current_assistant_message_content[-1]["type"] != "text"
                    ):
                        current_assistant_message_content.append(
                            {"type": "text", "text": text}
                        )
                    else:
                        current_assistant_message_content[-1]["text"] += text
                elif event["type"] == "tool_use_complete":
                    tool_call_data = event["data"]
                    logger.info(
                        "Tool request: {} (ID: {}) args={}".format(
                            tool_call_data["name"],
                            tool_call_data["id"],
                            json.dumps(
                                tool_call_data.get("input") or {}, ensure_ascii=False
                            )[:600],
                        )
                    )
                    pending_tool_calls.append(tool_call_data)
                    current_assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_data["id"],
                            "name": tool_call_data["name"],
                            "input": tool_call_data["input"],
                        }
                    )
                elif event["type"] == "web_search_marker":
                    # Inline 🔍/🔗 tag from Claude's native web tools: surface it
                    # (and let it persist to history) like the OpenAI path. Not
                    # added to the assistant message echoed back to Claude.
                    mk = event.get("text", "")
                    if mk:
                        yield mk
                elif event["type"] == "thinking_complete":
                    # Manual fallback while streaming. At message_stop the SDK's
                    # exact accumulated response replaces this reconstructed
                    # list via assistant_message_complete.
                    current_assistant_message_content.append(event["data"])
                elif event["type"] == "assistant_message_complete":
                    assistant_message = event.get("data") or {}
                    content = assistant_message.get("content")
                    if assistant_message.get("role") == "assistant" and isinstance(
                        content, list
                    ):
                        current_assistant_message_content[:] = deepcopy(content)
                        current_assistant_message_is_exact = True
                elif event["type"] == "message_delta":
                    # Track this request's billed thinking (cumulative within
                    # the request; the final delta wins, so overwrite).
                    usage = (event.get("data") or {}).get("usage") or {}
                    details = usage.get("output_tokens_details") or {}
                    tt = details.get("thinking_tokens")
                    if isinstance(tt, (int, float)):
                        request_thinking_tokens = int(tt)
                elif event["type"] == "refusal":
                    refusal_info = event.get("data") or {}
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    # あさひ 08-05: the whole failed turn (input + any partial
                    # reply) leaves the context; disk keeps tagged records for
                    # review. See _handle_api_error_turn.
                    yield self._handle_api_error_turn(event["message"])
                    return

            turn_thinking_tokens += request_thinking_tokens

            if request_thinking_tokens and not self._claude_protocol_has_thinking(
                [{"role": "assistant", "content": current_assistant_message_content}]
            ):
                # Thinking was billed but no thinking/redacted block came back
                # (anti-distillation shape). Downstream this request looks like
                # a genuine no-thinking turn — protocol/seed drop and the seed
                # rate gate decays — so make the real cause greppable.
                logger.warning(
                    f"[thinking] {request_thinking_tokens} thinking tokens "
                    "billed but no thinking block in the response — the API "
                    "hid the thinking entirely."
                )

            if refusal_info is not None:
                yield self._handle_safety_refusal(refusal_info)
                return

            if pending_tool_calls:
                if current_assistant_message_is_exact:
                    assistant_content_for_replay = deepcopy(
                        current_assistant_message_content
                    )
                else:
                    # Compatibility fallback for a custom/old stream wrapper.
                    # It keeps the live tool loop working, but the transcript is
                    # not persisted across turns because it is not guaranteed
                    # to contain every immutable Claude block.
                    protocol_is_exact = False
                    logger.warning(
                        "Claude stream ended without an exact assistant "
                        "snapshot; falling back to reconstructed tool content."
                    )
                    assistant_content_for_replay = [
                        block
                        for block in current_assistant_message_content
                        if not (
                            block.get("type") == "text"
                            and not block.get("text", "").strip()
                        )
                    ]

                if assistant_content_for_replay:
                    assistant_protocol_message = {
                        "role": "assistant",
                        "content": assistant_content_for_replay,
                    }
                    messages.append(deepcopy(assistant_protocol_message))
                    claude_protocol.append(deepcopy(assistant_protocol_message))
                    round_thinking.append(request_thinking_tokens)

                # Split: in-process tools (alarms + self-check + steam) are
                # handled here, provider-agnostic; everything else goes to the
                # MCP executor.
                inproc_names = set(self._ALARM_TOOL_NAMES)
                if self._model_health_enabled:
                    inproc_names.add(self._MODEL_HEALTH_TOOL_NAME)
                if self._steam_enabled:
                    inproc_names.update(self._STEAM_TOOL_NAMES)
                if self._memory_tools_active:
                    inproc_names.update(self._MEMORY_TOOL_NAMES)
                if getattr(self._llm, "_enable_web_fetch", False):
                    inproc_names.add("web_fetch")
                if getattr(self._llm, "_enable_web_search", False):
                    inproc_names.add("web_search")
                inproc_calls = [
                    c for c in pending_tool_calls if c.get("name") in inproc_names
                ]
                mcp_calls = [
                    c for c in pending_tool_calls if c.get("name") not in inproc_names
                ]

                tool_results_for_llm = []

                for c in inproc_calls:
                    cname = c.get("name", "")
                    if cname == "web_fetch":
                        marker, result = await self._run_claude_web_fetch(
                            c.get("input") or {}, web_fetch_budget
                        )
                    elif cname == "web_search":
                        marker, result = await self._run_claude_web_search(
                            c.get("input") or {}, web_search_budget
                        )
                    elif cname == self._MODEL_HEALTH_TOOL_NAME:
                        marker, result = await self._run_model_health_tool(
                            c.get("input") or {}
                        )
                    elif cname in self._STEAM_TOOL_NAMES:
                        marker, result = await self._run_steam_tool(
                            cname, c.get("input") or {}
                        )
                    elif cname in self._MEMORY_TOOL_NAMES:
                        marker, result = await self._run_memory_tool(
                            cname, c.get("input") or {}
                        )
                    else:
                        marker, result = await self._run_alarm_tool(
                            cname, c.get("input") or {}
                        )
                    # Dedupe by exact text so constant tags (e.g. the 🎮 Steam
                    # marker) surface once per turn even across chained calls.
                    if marker and marker not in emitted_markers:
                        emitted_markers.add(marker)
                        yield marker
                    # Formatted web results are plain text already — dumping a
                    # str would wrap it in quotes and escape every newline.
                    tool_results_for_llm.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": c["id"],
                            "content": (
                                result
                                if isinstance(result, str)
                                else json.dumps(result, ensure_ascii=False)
                            ),
                            "is_error": self._inproc_result_is_error(result),
                        }
                    )

                if mcp_calls:
                    for c in mcp_calls:
                        mk = self._mcp_tool_marker(c.get("name", ""))
                        if mk and mk not in emitted_markers:
                            emitted_markers.add(mk)
                            yield mk
                    if not self._tool_executor:
                        logger.error(
                            "Claude MCP tool call requested but ToolExecutor is not available."
                        )
                        yield "[Error: ToolExecutor not configured]"
                        # Same as the stream-error path: preserve already-spoken
                        # text in context (text only, no protocol).
                        if current_turn_text:
                            self._add_message(current_turn_text, "assistant")
                        return
                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=mcp_calls,
                        caller_mode="Claude",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm.extend(update.get("results", []))
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Tool executor finished without final results marker."
                        )
                    # Deferred result-bearing markers for the time tools.
                    name_by_id = {c.get("id"): c.get("name", "") for c in mcp_calls}
                    for r in tool_results_for_llm:
                        if not isinstance(r, dict):
                            continue
                        mk = self._deferred_tool_marker(
                            name_by_id.get(r.get("tool_use_id"), ""),
                            r.get("content"),
                        )
                        if mk and mk not in emitted_markers:
                            emitted_markers.add(mk)
                            yield mk
                    # uber_search results open with a related-facts section
                    # (mutates content in place, before the message is built).
                    await self._augment_uber_search_results(
                        tool_results_for_llm,
                        [
                            {
                                "id": c.get("id"),
                                "name": c.get("name", ""),
                                "input": c.get("input") or {},
                            }
                            for c in mcp_calls
                        ],
                    )

                if tool_results_for_llm:
                    tool_result_message = {
                        "role": "user",
                        "content": tool_results_for_llm,
                    }
                    messages.append(deepcopy(tool_result_message))
                    claude_protocol.append(deepcopy(tool_result_message))
                    if marking_active:
                        # In-round stability: will the post-turn protocol
                        # preserve this round byte-identically? Thinking under
                        # the per-round cap (and signed), an exact snapshot,
                        # and every tool_result already truncation-stable
                        # (identity check mirrors the post-turn trim). If yes,
                        # entries written now can never be invalidated → move
                        # bp④ here; if not, this round's bytes change after
                        # the turn, so stop marking from here on.
                        thinking_stable = not (
                            replay_cap and request_thinking_tokens > replay_cap
                        )
                        # Replay-exempt results (e.g. memory_read_diary) skip
                        # the post-turn cap entirely, so they are stable at
                        # any length — mirror that here or a long diary would
                        # needlessly stop the breakpoint migration.
                        round_exempt_ids = {
                            c.get("id")
                            for c in pending_tool_calls
                            if c.get("name") in self._PROTOCOL_EXEMPT_RESULT_TOOLS
                        }
                        results_stable = all(
                            self._truncate_tool_result_block(b) is b
                            for b in tool_results_for_llm
                            if isinstance(b, dict)
                            and b.get("type") == "tool_result"
                            and b.get("tool_use_id") not in round_exempt_ids
                        )
                        round_stable = (
                            protocol_is_exact
                            and current_assistant_message_is_exact
                            and bool(assistant_content_for_replay)
                            and self._claude_thinking_blocks_replayable(
                                assistant_content_for_replay
                            )
                            and thinking_stable
                            and results_stable
                        )
                        if round_stable:
                            if bp_index is not None:
                                messages[bp_index] = self._strip_cache_marker(
                                    messages[bp_index]
                                )
                            marked = messages[-1]  # loop-private deepcopy
                            last_block = marked["content"][-1]
                            if isinstance(last_block, dict):
                                marked["content"][-1] = {
                                    **last_block,
                                    "cache_control": self._CACHE_CONTROL_1H,
                                }
                                bp_index = len(messages) - 1
                                # The entry now sits at the end of the whole
                                # wire prefix — track it for the lookback
                                # budget (08-16).
                                self._bp_wire_anchor = sum(
                                    self._wire_blocks(m) for m in messages
                                )
                                logger.debug(
                                    "[cache] tool round stable "
                                    f"(thinking={request_thinking_tokens} tok) "
                                    "— breakpoint moved to its tool_result."
                                )
                        else:
                            marking_active = False
                            reasons = []
                            if not thinking_stable:
                                reasons.append(
                                    f"thinking {request_thinking_tokens} tok "
                                    f"> cap {replay_cap}"
                                )
                            if not results_stable:
                                reasons.append("tool result over protocol cap")
                            logger.info(
                                "[cache] tool round not replay-stable "
                                f"({'; '.join(reasons) or 'inexact snapshot'})"
                                " — in-loop cache marking stops; later "
                                "rounds ride fresh."
                            )
                continue
            else:
                if not current_turn_text:
                    # Silent turn (billed reasoning and/or tool rounds, zero
                    # visible text) — あさひ 08-20: API-error treatment
                    # replaces the 08-15 ellipsis-placeholder policy; see
                    # _handle_thinking_only_turn. Prior tool rounds ride the
                    # forensic seed alongside the final silent response.
                    forensic = claude_protocol + (
                        [
                            {
                                "role": "assistant",
                                "content": deepcopy(current_assistant_message_content),
                            }
                        ]
                        if current_assistant_message_content
                        else []
                    )
                    yield self._handle_thinking_only_turn(
                        turn_thinking_tokens, forensic or None
                    )
                    return
                text_for_memory = current_turn_text
                protocol_for_memory: Optional[List[Dict[str, Any]]] = None
                if (
                    protocol_is_exact
                    and current_assistant_message_is_exact
                    and current_assistant_message_content
                ):
                    final_assistant = {
                        "role": "assistant",
                        "content": deepcopy(current_assistant_message_content),
                    }
                    round_thinking.append(request_thinking_tokens)
                    # Per-round trim (08-09, replaces the 07-25 whole-turn
                    # gate): tool_result content truncated as before, and
                    # each round's thinking dropped ONLY if that round
                    # exceeded the cap — probe-proven safe on completed
                    # turns. A long-thinking turn keeps its tool calls and
                    # text instead of losing the whole transcript.
                    (
                        protocol_for_memory,
                        kept_rounds,
                        dropped_rounds,
                    ) = self._trim_protocol_per_round(
                        claude_protocol + [final_assistant],
                        round_thinking,
                        replay_cap,
                    )
                    if dropped_rounds:
                        # Next turn's user payload opens with the drop
                        # notice (see _to_messages) so the missing
                        # precedent isn't imitated as "think less".
                        self._pending_thinking_drop_notice = True
                        logger.info(
                            f"[thinking_replay] dropped thinking from "
                            f"{dropped_rounds} round(s) over cap "
                            f"{replay_cap} (turn total "
                            f"{turn_thinking_tokens} tok) — notice staged "
                            "for next turn."
                        )
                    # Stage the WHOLE trimmed protocol as the on-disk
                    # cold-start seed (あさひ 07-24: replaying a tool turn
                    # without its tool machinery rewrites history into a
                    # "results without calls" shape — with the precedent
                    # effect this strong, that teaches fabrication).
                    # tool_result content is already truncated above with
                    # an explanatory marker; those are unsigned client
                    # messages, so truncation cannot trip validation.
                    # round_thinking (kept-round alignment) lets
                    # _apply_thinking_seeds re-trim against a cap that
                    # shrank between store and reload.
                    if self._claude_protocol_worth_carrying(protocol_for_memory):
                        self._last_thinking_seed = {
                            "model": getattr(self._llm, "model", "") or "",
                            "protocol": deepcopy(protocol_for_memory),
                            "thinking_tokens": turn_thinking_tokens,
                            "round_thinking": list(kept_rounds),
                        }
                    else:
                        protocol_for_memory = None
                self._add_message(
                    text_for_memory,
                    "assistant",
                    claude_protocol=protocol_for_memory,
                )
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support (MCP + built-in)."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []

        # Per-turn state for in-process tools handled inside this same loop
        # (web search/fetch today; alarms later), routed by name.
        cfg = self._web_tools_config
        builtin_budget = {
            "searches": int(cfg.get("max_searches", 3) or 0),
            "fetches": int(cfg.get("max_fetches", 3) or 0),
        }
        builtin_names = self._builtin_tool_names()
        emitted_markers: set = set()  # inline tool tags shown once per turn

        while True:
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    current_system_prompt = (
                        f"{self._build_runtime_system()}\n\n{self._mcp_prompt_string}"
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    current_system_prompt = self._build_runtime_system()
                tools_for_api = None
            else:
                current_system_prompt = self._build_runtime_system()
                tools_for_api = tools

            stream = self._llm.chat_completion(
                messages, current_system_prompt, tools=tools_for_api
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        current_turn_text += event
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                        yield event
                else:
                    if isinstance(event, str):
                        current_turn_text += event
                        yield event
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": current_turn_text if current_turn_text else None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")

                # Split the calls: in-process built-in tools (web/alarm/...) are
                # handled here; the rest go to the MCP executor. Both yield
                # role:tool results that we feed back in the same `messages`.
                builtin_calls = [
                    tc
                    for tc in pending_tool_calls
                    if getattr(tc, "function", None)
                    and tc.function.name in builtin_names
                ]
                mcp_calls = [
                    tc
                    for tc in pending_tool_calls
                    if not (
                        getattr(tc, "function", None)
                        and tc.function.name in builtin_names
                    )
                ]

                tool_results_for_llm = []

                for tc in builtin_calls:
                    async for ev in self._run_builtin_tool_call(tc, builtin_budget):
                        if ev.get("type") == "_builtin_tool_result":
                            tool_results_for_llm.append(ev["message"])
                        else:
                            # Display marker (e.g. the 🔍 web-search indicator):
                            # yield it as plain text so the sentence pipeline
                            # streams it to the UI, exactly as the old web-tool
                            # loop did. (A bare dict would pass untouched through
                            # the transformers and be dropped downstream.)
                            # Deduped by exact text so constant tags (e.g. the
                            # 🎮 Steam marker) appear once per turn.
                            marker = ev.get("text", "")
                            if marker and marker not in emitted_markers:
                                emitted_markers.add(marker)
                                yield marker

                if mcp_calls:
                    # Inline tag so the user/character sees an MCP tool was used
                    # (e.g. Uber Eats), once per turn per tool.
                    for tc in mcp_calls:
                        marker = self._mcp_tool_marker(tc.function.name)
                        if marker and marker not in emitted_markers:
                            emitted_markers.add(marker)
                            yield marker
                    if not self._tool_executor:
                        logger.error(
                            "MCP tool call requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    else:
                        tool_executor_iterator = self._tool_executor.execute_tools(
                            tool_calls=mcp_calls,
                            caller_mode="OpenAI",
                        )
                        try:
                            while True:
                                update = await anext(tool_executor_iterator)
                                if update.get("type") == "final_tool_results":
                                    tool_results_for_llm.extend(
                                        update.get("results", [])
                                    )
                                    break
                                else:
                                    yield update
                        except StopAsyncIteration:
                            logger.warning(
                                "OpenAI tool executor finished without final results marker."
                            )
                        # Deferred result-bearing markers for the time tools.
                        name_by_id = {
                            tc.id: tc.function.name for tc in mcp_calls if tc.id
                        }
                        for r in tool_results_for_llm:
                            if not isinstance(r, dict):
                                continue
                            mk = self._deferred_tool_marker(
                                name_by_id.get(r.get("tool_call_id"), ""),
                                r.get("content"),
                            )
                            if mk and mk not in emitted_markers:
                                emitted_markers.add(mk)
                                yield mk
                        # uber_search results open with a related-facts
                        # section (OpenAI carries args as a JSON string).
                        norm_calls = []
                        for tc in mcp_calls:
                            try:
                                tc_input = json.loads(tc.function.arguments or "{}")
                            except (TypeError, ValueError):
                                tc_input = {}
                            norm_calls.append(
                                {
                                    "id": tc.id,
                                    "name": tc.function.name,
                                    "input": tc_input,
                                }
                            )
                        await self._augment_uber_search_results(
                            tool_results_for_llm, norm_calls
                        )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                continue

            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False
            self._turn_inproc_calls = []
            self._diary_reads_this_turn = 0
            # Stale seeds must not leak into an unrelated turn's store.
            self._last_thinking_seed = None

            await self._inject_memory_rags(input_data)
            messages = self._to_messages(input_data)
            # Claude path: MCP tools (if enabled) + built-in ALARM tools share
            # the Claude tool loop. web_search/fetch are NOT added here — Claude
            # uses Anthropic's native server tools for those.
            if isinstance(self._llm, ClaudeAsyncLLM):
                claude_tools: List[Dict[str, Any]] = []
                if self._use_mcpp and self._tool_manager:
                    claude_tools.extend(self._formatted_tools_claude or [])
                if self._alarm_store is not None:
                    claude_tools.extend(self._build_alarm_tools_claude())
                if self._model_health_enabled:
                    claude_tools.extend(self._build_model_health_tools_claude())
                if self._steam_enabled:
                    claude_tools.extend(self._build_steam_tools_claude())
                if self._memory_tools_active:
                    claude_tools.extend(self._build_memory_tools_claude())
                if getattr(self._llm, "_enable_web_fetch", False):
                    # Client-side web_fetch (native server tool retired —
                    # absent on Opus 5; one shared fetcher for both paths).
                    claude_tools.extend(self._build_web_fetch_tool_claude())
                if getattr(self._llm, "_enable_web_search", False):
                    # Client-side web_search (native server tool retired —
                    # its result blocks were replay bulk immune to protocol
                    # truncation, parking 10-16k tokens per search in every
                    # later request of the session).
                    claude_tools.extend(self._build_web_search_tool_claude())
                if claude_tools:
                    logger.debug(
                        f"Starting Claude tool loop with {len(claude_tools)} tools."
                    )
                    async for output in self._claude_tool_interaction_loop(
                        messages, claude_tools
                    ):
                        yield output
                    return

            # OpenAI path: MCP tools (if enabled) + built-in tools (web
            # search/fetch + alarms) share ONE tool-calling loop, dispatched by
            # name — so the built-in Brave web tools run even alongside MCP.
            if isinstance(self._llm, OpenAICompatibleAsyncLLM):
                mcp_tools = (
                    self._formatted_tools_openai
                    if (self._use_mcpp and self._tool_manager)
                    else []
                )
                builtin_tools = self._build_builtin_tools_openai()
                openai_tools = list(mcp_tools or []) + builtin_tools
                if openai_tools:
                    logger.debug(
                        f"Starting OpenAI tool loop: {len(openai_tools)} tools "
                        f"(mcp={len(mcp_tools or [])}, builtin={len(builtin_tools)})."
                    )
                    async for output in self._openai_tool_interaction_loop(
                        messages, openai_tools
                    ):
                        yield output
                    return

            # No tools at all: plain streaming completion.
            logger.info("Starting simple chat completion.")
            complete_response = ""
            claude_assistant_message: Optional[Dict[str, Any]] = None
            plain_thinking_tokens = 0
            async for event in self._llm.chat_completion(
                messages, self._build_system_for_llm()
            ):
                text_chunk = ""
                if isinstance(event, dict) and event.get("type") == "text_delta":
                    text_chunk = event.get("text", "")
                elif (
                    isinstance(event, dict)
                    and event.get("type") == "assistant_message_complete"
                ):
                    candidate = event.get("data")
                    if isinstance(candidate, dict):
                        claude_assistant_message = deepcopy(candidate)
                    continue
                elif isinstance(event, dict) and event.get("type") == "message_delta":
                    usage = (event.get("data") or {}).get("usage") or {}
                    details = usage.get("output_tokens_details") or {}
                    tt = details.get("thinking_tokens")
                    if isinstance(tt, (int, float)):
                        plain_thinking_tokens = int(tt)
                    continue
                elif isinstance(event, dict) and event.get("type") == "refusal":
                    yield self._handle_safety_refusal(event.get("data") or {})
                    return
                elif isinstance(event, str):
                    text_chunk = event
                else:
                    continue
                if text_chunk:
                    yield text_chunk
                    complete_response += text_chunk
            if plain_thinking_tokens and not (
                claude_assistant_message is not None
                and self._claude_protocol_has_thinking([claude_assistant_message])
            ):
                logger.warning(
                    f"[thinking] {plain_thinking_tokens} thinking tokens billed "
                    "but no thinking block in the response — the API hid the "
                    "thinking entirely."
                )
            if not complete_response:
                # Silent turn — あさひ 08-20: API-error treatment replaces
                # the 08-15 ellipsis-placeholder policy; see
                # _handle_thinking_only_turn. (An empty turn with no
                # protocol at all — the non-Claude path — is the same
                # failure and takes the same exit, minus the seed.)
                yield self._handle_thinking_only_turn(
                    plain_thinking_tokens,
                    (
                        [claude_assistant_message]
                        if claude_assistant_message is not None
                        else None
                    ),
                )
                return
            plain_cap = getattr(self._llm, "thinking_replay_max_tokens", 0) or 0
            if plain_cap and plain_thinking_tokens > plain_cap:
                # Single-request turn = single round, so the per-round rule
                # degenerates to the old behavior: drop the thinking. What
                # remains (visible text) carries no thinking/tool blocks, so
                # no protocol is worth storing — but the drop notice still
                # fires when signed thinking was actually removed.
                if (
                    claude_assistant_message is not None
                    and self._claude_protocol_has_thinking([claude_assistant_message])
                ):
                    self._pending_thinking_drop_notice = True
                    logger.info(
                        f"[thinking_replay] turn thinking "
                        f"{plain_thinking_tokens} tok > cap {plain_cap} — "
                        "thinking dropped from replay; notice staged for "
                        "next turn."
                    )
                claude_assistant_message = None
            if complete_response or claude_assistant_message is not None:
                if (
                    claude_assistant_message is not None
                    and self._claude_protocol_has_thinking([claude_assistant_message])
                ):
                    self._last_thinking_seed = {
                        "model": getattr(self._llm, "model", "") or "",
                        "protocol": [deepcopy(claude_assistant_message)],
                    }
                self._add_message(
                    complete_response,
                    "assistant",
                    claude_protocol=(
                        [claude_assistant_message]
                        if claude_assistant_message is not None
                        else None
                    ),
                )

        return chat_with_memory

    def _web_tools_enabled(self) -> bool:
        """Built-in Brave web tools on the OpenAI path. Available whenever
        enabled — INCLUDING alongside MCP tools, since both kinds of tool now
        share one OpenAI tool-calling loop (dispatched by name)."""
        return bool(self._web_tools_config.get("enabled")) and isinstance(
            self._llm, OpenAICompatibleAsyncLLM
        )

    def _uber_tools_active(self) -> bool:
        """Whether Uber Eats MCP tools are actually advertised this session, so
        the Uber capability note is only injected when the tool really exists
        (never describe a tool that isn't loaded)."""
        if not self._use_mcpp:
            return False
        tools = self._formatted_tools_claude or self._formatted_tools_openai or []
        for t in tools:
            name = t.get("name") or (t.get("function") or {}).get("name", "")
            if name.startswith("uber"):
                return True
        return False

    def _tool_capability_notes(self) -> List[str]:
        """Static, cache-stable system-prompt notes describing the homegrown
        tools, each included only when that tool is active. Shared by the
        plain-string (OpenAI) and Claude system builders so both paths
        advertise the same capabilities. Claude in particular tends to ignore
        raw tool schemas, so these affirmative notes (plus a hard no-fabricate
        rule for Uber) live in the prompt itself."""
        lean = self._lean_prompt_active()
        notes: List[str] = []
        if self._alarm_store is not None:
            notes.append(
                self._ALARM_CAPABILITY_NOTE_LEAN
                if lean
                else self._ALARM_CAPABILITY_NOTE
            )
        if self._uber_tools_active():
            notes.append(self._UBER_CAPABILITY_NOTE)
        if self._model_health_enabled:
            notes.append(self._MODEL_HEALTH_CAPABILITY_NOTE)
        if self._steam_enabled:
            notes.append(self._STEAM_CAPABILITY_NOTE)
        if self._memory_tools_active:
            notes.append(
                self._MEMORY_CAPABILITY_NOTE_LEAN
                if lean
                else self._MEMORY_CAPABILITY_NOTE
            )
        return notes

    def _build_builtin_tools_openai(self) -> List[Dict[str, Any]]:
        """OpenAI schemas for in-process (non-MCP) tools to advertise to the LLM.

        These ride in the same tool list as the MCP tools; the OpenAI loop
        routes calls back to ``_run_builtin_tool_call`` by name."""
        tools: List[Dict[str, Any]] = []
        if self._web_tools_enabled():
            tools.extend(self._build_web_tools_openai())
        if self._alarm_store is not None:
            tools.extend(self._build_alarm_tools_openai())
        if self._model_health_enabled:
            tools.extend(self._build_model_health_tools_openai())
        if self._steam_enabled:
            tools.extend(self._build_steam_tools_openai())
        if self._memory_tools_active:
            tools.extend(self._build_memory_tools_openai())
        return tools

    def _builtin_tool_names(self) -> set:
        """Names the OpenAI loop should handle in-process instead of via MCP."""
        return {t["function"]["name"] for t in self._build_builtin_tools_openai()}

    @staticmethod
    def _mcp_tool_marker(tool_name: str) -> str:
        """Short inline tag flagging that an MCP tool was used. Any MCP tool
        gets a generic 🔧 tag so EVERY tool call is visible in chat (あさひ
        audits usage from there when away). Time and uber tools return ""
        here — their markers are result-bearing and fire after execution
        (_deferred_tool_marker); uber's carry the keyword / store name /
        item name pulled from the result (あさひ 08-14: the bare Uber Eats
        badge said nothing)."""
        if tool_name.startswith("uber"):
            return ""
        if tool_name in _DEFERRED_MARKER_TOOLS:
            return ""
        if tool_name:
            return f"\n🔧 *ツール: {tool_name[:40]}*\n"
        return ""

    # Result-head patterns for the uber markers, keyed by tool. search uses
    # re.search (defensive: its content may gain a prepended facts section if
    # the emit order ever changes); store/item anchor on the leading 【…】.
    _UBER_MARKER_SPECS = {
        "uber_search": (re.compile(r"「(.+?)」の検索結果"), "Uber検索"),
        "uber_store": (re.compile(r"^【(.+?)】"), "メニュー閲覧"),
        "uber_item": (re.compile(r"^【(.+?)】"), "商品詳細"),
    }

    @classmethod
    def _uber_tool_marker(cls, tool_name: str, content: Any) -> str:
        """Result-bearing marker for the uber tools: search shows the
        keyword, store the store name, item the item name (server formats:
        「kw」の検索結果 / 【店名】 / 【商品名  価格】). uber_category,
        errors and parse misses fall back to the old generic badge so every
        call stays visible. Display-only; TOOL_MARKER_RE strips 🍔 *Uber…*
        from replay."""
        if not tool_name.startswith("uber"):
            return ""
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    break
        spec = cls._UBER_MARKER_SPECS.get(tool_name)
        if spec:
            m = spec[0].search(text.strip())
            if m:
                # item heads glue price / [品切れ] after a double space
                label = m.group(1).split("  ")[0].strip()
                if label:
                    return f"\n🍔 *{spec[1]}: {label[:40]}*\n"
        return "\n🍔 *Uber Eats*\n"

    # Tools whose marker can only be written once the result is known, so
    # _mcp_tool_marker stays silent for them and _deferred_tool_marker emits
    # after execution instead.
    _DAY_JA = {
        "Monday": "月",
        "Tuesday": "火",
        "Wednesday": "水",
        "Thursday": "木",
        "Friday": "金",
        "Saturday": "土",
        "Sunday": "日",
    }

    @classmethod
    def _time_tool_marker(cls, tool_name: str, content: Any) -> str:
        """Result-bearing marker for the time MCP tools (あさひ 07-26: the
        generic 🔧 tag hid the answer — show what the clock said). Unlike the
        name-only pre-markers this fires AFTER execution; display-only, and
        stripped from replay like every marker (TOOL_MARKER_RE has 🕐)."""
        if tool_name not in ("get_current_time", "convert_time"):
            return ""
        label = "時刻確認" if tool_name == "get_current_time" else "時刻変換"
        try:
            data = content
            if isinstance(data, list):
                # MCP content blocks: [{"type": "text", "text": "<json>"}]
                for block in data:
                    if isinstance(block, dict) and block.get("type") == "text":
                        data = block.get("text", "")
                        break
            if isinstance(data, str):
                data = json.loads(data)
            node = data.get("target") if "target" in data else data
            stamp = str(node["datetime"])[:16].replace("T", " ")
            day = cls._DAY_JA.get(str(node.get("day_of_week", "")), "")
            tail = f"（{day}）" if day else ""
            return f"\n🕐 *{label}: {stamp}{tail}*\n"
        except Exception:
            return f"\n🕐 *{label}*\n"

    _MUSIC_MARKER_LABELS = {
        "music_search": "曲を検索",
        "music_playlists": "プレイリスト一覧",
        "music_now_playing": "再生状況",
        "music_stop": "再生停止",
    }

    @classmethod
    def _music_tool_marker(cls, tool_name: str, content: Any) -> str:
        """Result-bearing marker for the music MCP tools. Like the time
        markers this fires AFTER execution, so a play call can name the song
        it actually started instead of just the tool. Display-only, and
        stripped from replay like every marker (TOOL_MARKER_RE has 🎵)."""
        if not tool_name.startswith("music_"):
            return ""
        text = _mcp_result_text(content)
        if tool_name in ("music_play", "music_play_playlist"):
            # "再生開始: <song> / <artist>（<playlist> から）" on success;
            # anything else is a failure message worth showing as-is.
            # Only the first line: a play result may append the other
            # same-title candidates, which belong in the tool result, not here.
            song = (
                text.split("再生開始:", 1)[-1].strip().splitlines()[0].strip()
                if "再生開始:" in text
                else ""
            )
            body = f"再生: {song}" if song else (text.splitlines() or ["再生"])[0]
            return f"\n🎵 *{body[:80]}*\n"
        label = cls._MUSIC_MARKER_LABELS.get(tool_name, "音楽")
        if tool_name == "music_search":
            keyword = text.split("」", 1)[0].split("「", 1)[-1] if "「" in text else ""
            if keyword:
                label = f"{label}: {keyword[:40]}"
        return f"\n🎵 *{label}*\n"

    @classmethod
    def _deferred_tool_marker(cls, tool_name: str, content: Any) -> str:
        """Markers that can only be written once the result is in hand."""
        return (
            cls._time_tool_marker(tool_name, content)
            or cls._music_tool_marker(tool_name, content)
            or cls._uber_tool_marker(tool_name, content)
        )

    def _handle_api_error_turn(self, error_message: str) -> str:
        """Clean up after an API error and build the visible notice.

        あさひ 08-05 ruling (post-hallucination incident): the WHOLE failed
        turn leaves the LLM context — the user input is popped from memory
        (else the context shows two consecutive user turns, the exact shape
        that seeded hallucinated inputs), and both sides stay on DISK tagged
        ``context_excluded`` so a human can review what happened while the
        assembler skips them (unlike the safety-refusal path, which deletes
        the input from disk too — a refused input would re-poison every
        restart; an errored one is harmless to keep). Any partial text she
        said before the error goes with the turn: it never reached a clean
        stop and the user is told to resend, so replaying half a reply
        against a re-sent input would desync the exchange. The notice is
        yielded as the turn's visible output and stored to disk WITH the tag
        (via pop_context_excluded), so it too stays out of future context —
        teaching her to narrate API errors is how the [Error from LLM] line
        got imitated into record #123 today."""
        short = str(error_message or "").strip()
        if len(short) > 160:
            short = short[:160] + "…"
        if self._memory and self._memory[-1].get("role") == "user":
            self._memory.pop()
        # Discarded turn → the cache anchor advanced for wire blocks that no
        # longer exist; roll it back to the turn-start snapshot (see
        # _bp_anchor_turn_start).
        self._bp_wire_anchor = self._bp_anchor_turn_start
        marked_disk = False
        try:
            if self._conf_uid and self._history_uid:
                marked_disk = mark_last_message_excluded(
                    self._conf_uid, self._history_uid, "human", "api_error"
                )
        except Exception as e:
            logger.warning(f"[api_error] disk tagging failed: {e}")
        self._pending_context_excluded = "api_error"
        logger.warning(
            f"[api_error] turn excluded from context (disk_tagged={marked_disk}): "
            f"{short}"
        )
        return (
            f"\n⚠️ APIエラーで応答できなかった（{short}）。"
            "この往復は文脈に残らない——直前のメッセージは届いていないので、"
            "もう一度送ってほしい。\n"
        )

    def _handle_thinking_only_turn(
        self,
        thinking_tokens: int,
        protocol: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Clean up after a silent turn (no visible text) and build the notice.

        あさひ 08-20 ruling: a turn that thought (or ran tools) but never
        spoke is a failure symptom — the 08-18 end-incident was preceded by
        exactly these — so it gets the API-error treatment
        (_handle_api_error_turn), replacing the 08-15 ellipsis-placeholder
        policy: the user input is popped from memory, both sides stay on
        DISK tagged ``context_excluded`` for human review, and the notice
        asks the user to resend. Assembly skips tagged records on BOTH
        restart paths — the frozen system transcript and the resume splice
        share _msg_from_history_record, whose tag check runs before the
        thinking-seed passthrough — so the turn never re-enters context.

        ``protocol`` (the turn's raw thinking/tool rounds) still rides the
        excluded AI record as a FORENSIC seed: the 08-18 investigation
        lived on exactly this material. No replay trim, no replayability
        gate — an excluded record is for human eyes only."""
        if self._memory and self._memory[-1].get("role") == "user":
            self._memory.pop()
        # Same anchor rollback as the API-error exit (see there).
        self._bp_wire_anchor = self._bp_anchor_turn_start
        marked_disk = False
        try:
            if self._conf_uid and self._history_uid:
                marked_disk = mark_last_message_excluded(
                    self._conf_uid, self._history_uid, "human", "thinking_only"
                )
        except Exception as e:
            logger.warning(f"[thinking_only] disk tagging failed: {e}")
        self._pending_context_excluded = "thinking_only"
        if protocol:
            self._last_thinking_seed = {
                "model": getattr(self._llm, "model", "") or "",
                "protocol": deepcopy(protocol),
                "thinking_tokens": thinking_tokens,
            }
        logger.warning(
            "[thinking_only] silent turn excluded from context "
            f"(thinking_tokens={thinking_tokens}, disk_tagged={marked_disk})."
        )
        return (
            f"\n⚠️ 思考だけで発話がないまま終わった（thinking {thinking_tokens} tok）。"
            "この往復は文脈に残らない——返事は届かなかったので、"
            "もう一度送ってほしい。\n"
        )

    def _handle_safety_refusal(self, info: Dict[str, Any]) -> str:
        """Clean up after a safety-classifier refusal and build the notice.

        The refused USER input is deleted from both the in-memory conversation
        and the on-disk history record — left in place it would ride in every
        later request (and reload on restart), so the classifier keeps firing
        and the whole session is poisoned. The returned notice is yielded as
        the turn's visible output (reaches the frontend and Discord alike) and
        is stored to disk as an ordinary AI line — harmless, and it tells both
        her and あさひ what happened."""
        category = info.get("category") or "unknown"
        if self._memory and self._memory[-1].get("role") == "user":
            self._memory.pop()
        removed_disk = False
        try:
            if self._conf_uid and self._history_uid:
                removed_disk = pop_last_message(
                    self._conf_uid, self._history_uid, "human"
                )
        except Exception as e:
            logger.warning(f"[refusal] disk cleanup failed: {e}")
        logger.warning(
            f"[refusal] turn dropped (category={category}, "
            f"disk_cleaned={removed_disk})."
        )
        return (
            f"\n⚠️ 安全分類器が応答を拒否した（category: {category}）。"
            "直前の入力は履歴から削除された。別の話題で続けてほしい。\n"
        )

    @staticmethod
    def _build_web_fetch_tool_claude() -> List[Dict[str, Any]]:
        """Claude schema for the CLIENT-side web_fetch tool.

        Runs in-process via web_tools.web_fetch — the same fetcher the OpenAI
        path uses — replacing the retired Anthropic native server tool (absent
        on Claude Opus 5). Claude's native web_search stays server-side."""
        return [
            {
                "name": "web_fetch",
                "description": (
                    "Fetch and read the full text content of a specific URL "
                    "(e.g. one the user pasted or one from a web search "
                    "result). Returns the page's title and cleaned text. "
                    "HTML/text pages only — PDFs and heavily script-rendered "
                    "pages may not be readable."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch.",
                        }
                    },
                    "required": ["url"],
                },
            }
        ]

    async def _run_claude_web_fetch(
        self, args: Dict[str, Any], budget: Dict[str, int]
    ) -> tuple:
        """Client-side web_fetch for the Claude tool loop.

        Same fetcher, marker, and truncation as the OpenAI path; the per-turn
        budget comes from the claude_llm max_web_fetches setting."""
        self._turn_inproc_calls.append("web_fetch")
        url = str(args.get("url", "")).strip()
        if budget.get("left", 0) <= 0:
            return None, {"error": "web fetch limit reached this turn"}
        budget["left"] -= 1
        logger.info(f"[web_fetch] url: {url or '(empty)'}")
        marker = f"\n🔗 *Web取得: {url[:120] or '...'}*\n"
        cfg = getattr(self, "_web_tools_config", None) or {}
        max_chars = int(cfg.get("max_fetch_chars", 20000) or 20000)
        result = format_fetch_result(await web_fetch(url, max_chars=max_chars))
        return marker, result

    @staticmethod
    def _build_web_search_tool_claude() -> List[Dict[str, Any]]:
        """Claude schema for the CLIENT-side web_search tool.

        Runs in-process via web_tools.web_search (brave/tavily) — the same
        backend the OpenAI path uses — replacing Anthropic's native server
        tool. Server search results were replay bulk no truncation could
        touch; client results are ordinary tool_result blocks capped by
        _truncate_protocol_tool_results once the turn ends."""
        return [
            {
                "name": "web_search",
                "description": (
                    "Search the web for current information, news, or facts "
                    "you're unsure about. Returns a list of results with "
                    "titles, URLs, and snippets."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]

    async def _run_claude_web_search(
        self, args: Dict[str, Any], budget: Dict[str, int]
    ) -> tuple:
        """Client-side web_search for the Claude tool loop.

        Same brave/tavily backend as the OpenAI path; the per-turn budget
        comes from the claude_llm max_web_searches setting."""
        self._turn_inproc_calls.append("web_search")
        query = str(args.get("query", "")).strip()
        if budget.get("left", 0) <= 0:
            return None, {"error": "web search limit reached this turn"}
        budget["left"] -= 1
        logger.info(f"[web_search] query: {query or '(empty)'}")
        marker = f"\n🔍 *Web検索: {query[:80] or '...'}*\n"
        cfg = getattr(self, "_web_tools_config", None) or {}
        result = format_search_results(
            query,
            await web_search(
                query,
                provider=cfg.get("provider", "brave"),
                api_key=cfg.get("api_key", ""),
                max_results=5,
            ),
        )
        return marker, result

    @staticmethod
    def _build_web_tools_openai() -> List[Dict[str, Any]]:
        """OpenAI function-tool definitions for web search and fetch."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the web for current information, news, or "
                        "facts you're unsure about. Returns a list of results "
                        "with titles, URLs, and snippets."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": (
                        "Fetch and read the full text content of a specific "
                        "URL (e.g. one the user pasted or one from a prior "
                        "search result). Returns the page's title and text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch.",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    @staticmethod
    def _build_alarm_tools_openai() -> List[Dict[str, Any]]:
        """OpenAI function-tool definitions for self-set alarms. Descriptions
        are in Japanese, matching the persona/model's working language."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "set_alarm",
                    "description": (
                        "Set a reminder (alarm) to yourself at a given time. "
                        "When it fires the note is delivered to you, giving you "
                        "an opening to speak to the user. Use in_minutes for "
                        "relative times like 'in 30 minutes', or at for clock "
                        "times like 'at 20:00' (one of the two is enough). If an "
                        "alarm already exists near that time, the existing one "
                        "is returned instead — set force=true only when you "
                        "judge that a separate alarm is genuinely needed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": (
                                    "What to remind yourself of when it fires. "
                                    "e.g. 'ask the user whether he took his "
                                    "medicine'."
                                ),
                            },
                            "in_minutes": {
                                "type": "number",
                                "description": "How many minutes from now it should fire. e.g. 30.",
                            },
                            "at": {
                                "type": "string",
                                "description": (
                                    "When it should fire: 'HH:MM' (the next "
                                    "occurrence of that time) or "
                                    "'YYYY-MM-DD HH:MM'."
                                ),
                            },
                            "wake": {
                                "type": "boolean",
                                "description": (
                                    "true for an alarm meant to physically "
                                    "wake the user up (a morning alarm). Music "
                                    "then plays out of his speakers on repeat "
                                    "until he answers or you call music_stop — "
                                    "a text message alone cannot wake someone "
                                    "who is asleep. Use it only for actual "
                                    "wake-up times, never for ordinary "
                                    "reminders."
                                ),
                            },
                            "force": {
                                "type": "boolean",
                                "description": (
                                    "true only when you want to set an alarm "
                                    "despite an existing one at a nearby time. "
                                    "Normally omit this."
                                ),
                            },
                        },
                        "required": ["note"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_alarms",
                    "description": "List the alarms currently set (not yet fired).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_alarm",
                    "description": (
                        "Cancel an alarm that was already set. Pass the id "
                        "obtained from list_alarms as alarm_id."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "alarm_id": {
                                "type": "string",
                                "description": "id of the alarm to cancel.",
                            }
                        },
                        "required": ["alarm_id"],
                    },
                },
            },
        ]

    def _build_alarm_tools_claude(self) -> List[Dict[str, Any]]:
        """Alarm tools in Claude's schema shape (input_schema, no function
        wrapper), derived from the OpenAI definitions so they stay in sync."""
        return self._to_claude_schema(self._build_alarm_tools_openai())

    @staticmethod
    def _to_claude_schema(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI function-tool defs to Claude's {name, description,
        input_schema} shape so both paths share one source of truth."""
        out: List[Dict[str, Any]] = []
        for t in openai_tools:
            fn = t["function"]
            out.append(
                {
                    "name": fn["name"],
                    "description": fn["description"],
                    "input_schema": fn["parameters"],
                }
            )
        return out

    @classmethod
    def _build_model_health_tools_openai(cls) -> List[Dict[str, Any]]:
        """OpenAI schema for the self-check tool."""
        return [
            {
                "type": "function",
                "function": {
                    "name": cls._MODEL_HEALTH_TOOL_NAME,
                    "description": (
                        "Check the current condition ('brain health') of an LLM "
                        "using objective external benchmarks (aistupidlevel.info) "
                        "plus Anthropic's official status page. With no argument, "
                        "checks the model you are currently running on. Returns a "
                        "short assessment report (score, trend, baseline, official "
                        "incidents, verdict)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": (
                                    "Optional model name/prefix to check instead "
                                    "of yourself (e.g. 'claude-opus-4-8')."
                                ),
                            }
                        },
                        "required": [],
                    },
                },
            }
        ]

    def _build_model_health_tools_claude(self) -> List[Dict[str, Any]]:
        return self._to_claude_schema(self._build_model_health_tools_openai())

    @staticmethod
    def _build_steam_tools_openai() -> List[Dict[str, Any]]:
        """OpenAI schemas for the four in-process Steam tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "steam_library",
                    "description": (
                        "Query the user's Steam library locally (a snapshot "
                        "taken at startup). No network needed, answers "
                        "instantly. Covers owned games, playtime, recently "
                        "played, wishlist, and followed titles. For 'does he own "
                        "this game?' style questions use action=check with name "
                        "(the name is fuzzy-matched against the snapshot)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "overview",
                                    "top_played",
                                    "recent",
                                    "check",
                                    "wishlist",
                                    "followed",
                                ],
                                "description": (
                                    "Kind of query. overview = summary / "
                                    "top_played = most playtime / recent = "
                                    "played in the last two weeks / check = "
                                    "whether a specific game is owned / "
                                    "wishlist / followed."
                                ),
                            },
                            "name": {
                                "type": "string",
                                "description": (
                                    "Game name, for action=check. Fuzzy-matched "
                                    "against the local snapshot (Japanese and "
                                    "English, plus Chinese names for major "
                                    "titles)."
                                ),
                            },
                            "appid": {
                                "type": "integer",
                                "description": (
                                    "Use this to run action=check by exact "
                                    "appid. If the name doesn't hit, resolve the "
                                    "appid with steam_search and pass it here."
                                ),
                            },
                            "n": {
                                "type": "integer",
                                "description": "How many entries list-type actions return. Default 10.",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "steam_search",
                    "description": (
                        "Search the real Steam store by game name and get the "
                        "appid, official title, and price (JPY). This is the "
                        "entry point for resolving any game name to an appid: "
                        "when you need accurate information, call this first, "
                        "then use the appid for precise lookups such as "
                        "steam_game. Titles are written inconsistently, so if "
                        "you are unsure, put both the Japanese and the English "
                        "name in queries. Results are real store data."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Candidate names to search (max 4). "
                                    "Including both the Japanese and English "
                                    "name makes a miss less likely."
                                ),
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results to return. Default 5.",
                            },
                        },
                        "required": ["queries"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "steam_game",
                    "description": (
                        "Given an appid, view that game's store-page-level "
                        "detail: price, discount, description, genres, release "
                        "date, review summary, community tags, and whether the "
                        "user owns it. Prices are normalised to integer JPY. If "
                        "you don't know the appid, resolve it with steam_search "
                        "first."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appid": {
                                "type": "integer",
                                "description": "The Steam application ID.",
                            }
                        },
                        "required": ["appid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "steam_discover",
                    "description": (
                        "Get a list of recommendation candidates from the real "
                        "Steam store. Titles the user already owns are excluded. "
                        "When recommending a game, choose only from these "
                        "results (or from steam_search results). mode: similar = "
                        "titles like a given game / tag = a list for a tag "
                        "(ordered by section) / specials = titles currently on "
                        "sale."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["similar", "tag", "specials"],
                                "description": "How to look for candidates.",
                            },
                            "anchor": {
                                "type": "string",
                                "description": (
                                    "The reference game name or appid for "
                                    "mode=similar (required there). Names are "
                                    "resolved against the library first; if not "
                                    "found, get the appid via steam_search and "
                                    "pass that."
                                ),
                            },
                            "tag": {
                                "type": "string",
                                "description": (
                                    "The tag for mode=tag (required there). "
                                    "Natural phrasing is fine — it is resolved "
                                    "against the real Steam tag table (e.g. "
                                    "'roguelike', 'visual novel')."
                                ),
                            },
                            "section": {
                                "type": "string",
                                "enum": [
                                    "trending_new",
                                    "top_sellers",
                                    "top_rated",
                                    "new_releases",
                                    "coming_soon",
                                    "most_wishlisted",
                                ],
                                "description": (
                                    "Ordering for mode=tag. Default trending_new."
                                ),
                            },
                            "count": {
                                "type": "integer",
                                "description": "How many to return. Default 10.",
                            },
                        },
                        "required": ["mode"],
                    },
                },
            },
        ]

    def _build_steam_tools_claude(self) -> List[Dict[str, Any]]:
        return self._to_claude_schema(self._build_steam_tools_openai())

    @staticmethod
    def _build_memory_tools_openai() -> List[Dict[str, Any]]:
        """OpenAI schemas for the character's self-service memory tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": (
                        "Semantic search over your own long-term memory (facts "
                        "and past diary entries). Separate from automatic "
                        "recall: this is how you look into the past on your own "
                        "initiative. Fact hits carry an id, used by "
                        "memory_update / memory_delete. Diary hits carry a "
                        "diary_uid, which memory_read_diary expands to the full "
                        "entry. Note: the search result may be truncated from "
                        "the next turn onward. Meaning of importance: "
                        "user = curated by the user himself (resident every "
                        "session) / high = important (resident every session) / "
                        "low = enters context only when searched or recalled / "
                        "archive = shelved (outdated but kept): reachable ONLY "
                        "through this search, never auto-recalled. To "
                        "restrict by date or period, pass date_from/date_to — do "
                        "not write dates into the query text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "What to search for (natural language is "
                                    "fine). Do not put dates or periods here — "
                                    "use date_from/date_to."
                                ),
                            },
                            "target": {
                                "type": "string",
                                "enum": ["facts", "diaries", "both"],
                                "description": "What to search. Default: both.",
                            },
                            "n": {
                                "type": "integer",
                                "description": "Max hits per target. Default 5.",
                            },
                            "date_from": {
                                "type": "string",
                                "description": (
                                    "YYYY-MM-DD. Restrict to this date onward "
                                    "(diary date / fact updated date)."
                                ),
                            },
                            "date_to": {
                                "type": "string",
                                "description": "YYYY-MM-DD. Restrict to this date and earlier.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "history_search",
                    "description": (
                        "Keyword search directly over the full text of past "
                        "conversation logs. Use this for concrete exchanges, "
                        "proper nouns, and 'who said what' checks that were "
                        "never distilled into facts or diary entries. This is a "
                        "full scan of the logs, not a semantic search. Multiple "
                        "keywords are OR'd — and each keyword is automatically "
                        "broken into two-character fragments, so it also hits "
                        "partial mentions in the conversation (e.g. only part of "
                        "a shop's name), ranked by how much of the keyword the "
                        "message covers. Because of this, pass a shop name or "
                        "proper noun whole, as a single keyword: one call "
                        "already casts a wide net. To restrict by date or "
                        "period, pass date_from/date_to — do not write dates "
                        "into the keywords. Results disappear at the end of this "
                        "turn; state anything worth keeping in your reply."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Search terms (1-8, OR'd). Pass proper nouns "
                                    "and phrases whole — they are fragmented "
                                    "automatically. Each concept must be its own "
                                    "array element; never pack several terms "
                                    "into one comma-joined string."
                                ),
                            },
                            "date_from": {
                                "type": "string",
                                "description": "YYYY-MM-DD. Restrict to logs from this date onward.",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "YYYY-MM-DD. Restrict to logs up to this date.",
                            },
                        },
                        "required": ["keywords"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_add",
                    "description": (
                        "Save a new fact to long-term memory (facts). Only save "
                        "what became clearly established in the conversation, or "
                        "what the user asked you to remember. Reflected in "
                        "search immediately; enters the resident list from the "
                        "next startup."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": (
                                    "The fact itself (one sentence per fact, "
                                    "concise and self-contained)."
                                ),
                            },
                            "importance": {
                                "type": "string",
                                "enum": ["high", "low", "archive"],
                                "description": (
                                    "high = important (resident in the system "
                                    "prompt from the next startup) / low = "
                                    "normal (recalled via search and RAG) / "
                                    "archive = shelved: only explicit "
                                    "memory_search finds it, nothing recalls "
                                    "it automatically. Default: low."
                                ),
                            },
                            "store_id": {
                                "type": "string",
                                "description": (
                                    "Only when the fact records an Uber Eats "
                                    "store/order impression: the store_uuid "
                                    "from the uber results. Links the fact to "
                                    "that store so it resurfaces when the "
                                    "store appears in a search. Leave empty "
                                    "for everything else."
                                ),
                            },
                        },
                        "required": ["fact"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_update",
                    "description": (
                        "Rewrite an existing fact and/or change its importance. "
                        "Confirm the target id with memory_search before using "
                        "this. user-tier facts: content may be corrected, but "
                        "their importance is the user's own and cannot be "
                        "changed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact_id": {
                                "type": "string",
                                "description": "id of the fact to rewrite (from memory_search results).",
                            },
                            "new_fact": {
                                "type": "string",
                                "description": (
                                    "The new text (replaces the whole fact). "
                                    "Omit to keep the text and change only the "
                                    "importance."
                                ),
                            },
                            "importance": {
                                "type": "string",
                                "enum": ["high", "low", "archive"],
                                "description": (
                                    "New tier: high = resident in the system "
                                    "prompt from the next startup / low = "
                                    "enters context only via search or recall / "
                                    "archive = shelved for facts that are "
                                    "outdated but worth keeping: only explicit "
                                    "memory_search finds them, automatic recall "
                                    "never surfaces them. Prefer archiving over "
                                    "deleting when a fact merely expired. "
                                    "Omit to keep the current tier."
                                ),
                            },
                            "store_id": {
                                "type": "string",
                                "description": (
                                    "Uber store linkage. Omit to leave it "
                                    "unchanged. Pass a store_uuid to set or "
                                    "replace it. CAUTION: an empty string "
                                    "DELETES the existing linkage — never "
                                    "pass empty unless you mean to clear it."
                                ),
                            },
                        },
                        "required": ["fact_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_delete",
                    "description": (
                        "Delete one fact. If the user's current message itself "
                        "explicitly asks for the deletion (消して/削除して/"
                        "删掉 etc.), a single call without confirmed executes "
                        "immediately. Otherwise two-phase: calling it without "
                        "confirmed files a deletion request, at which point you "
                        "ask the user whether that memory may be deleted. In the "
                        "turn immediately after he consents, call again with "
                        "the same fact_id and confirmed=true to execute. Without "
                        "consent the system refuses."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact_id": {
                                "type": "string",
                                "description": "id of the fact to delete (from memory_search results).",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why you want it deleted (used when explaining to the user).",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": (
                                    "true only on the execution call made after "
                                    "the user has consented."
                                ),
                            },
                        },
                        "required": ["fact_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_read_diary",
                    "description": (
                        "Read one diary entry in full. memory_search hits and "
                        "the auto-recalled ［過去の記憶］ blocks are paragraph "
                        "excerpts, so use this when you need the surrounding "
                        "context."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diary_uid": {
                                "type": "string",
                                "description": (
                                    "The diary_uid from a memory_search hit, "
                                    "or the short id shown in ［過去の記憶］ "
                                    "excerpt blocks."
                                ),
                            }
                        },
                        "required": ["diary_uid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_write_diary",
                    "description": (
                        "At the end of a session (e.g. the goodnight phase, "
                        "or other cases where a restart is needed), use this "
                        "tool to write a diary for this session. Write in "
                        "your usual diary voice and length (first person, a "
                        "few hundred characters, the whole session's arc). "
                        "If the conversation continues afterwards, you may "
                        "call it again before the next goodbye — it "
                        "overwrites your earlier draft. If you never call "
                        "this tool during the whole session, the diary is "
                        "generated automatically before the next restart."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The diary text (≥100 characters).",
                            }
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "model_history",
                    "description": (
                        "Look up which conversation model(s) were active on a "
                        "given date (Japan time): every session touching that "
                        "day, with its start/end times and the model that "
                        "experienced it. Use when you want to know 'who was I "
                        "then' — e.g. after reading an old diary or memory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "The day to look up, YYYY-MM-DD.",
                            }
                        },
                        "required": ["date"],
                    },
                },
            },
        ]

    def _build_memory_tools_claude(self) -> List[Dict[str, Any]]:
        return self._to_claude_schema(self._build_memory_tools_openai())

    async def _run_model_health_tool(self, args: Dict[str, Any]) -> tuple:
        """Run check_model_status. Returns (marker, result_dict). The JA report is
        the tool result the character relays; a ZH copy is logged so あさひ can see
        what it was told. No LLM — pure data + reference + rule-based verdict."""
        self._turn_inproc_calls.append(self._MODEL_HEALTH_TOOL_NAME)
        from ...model_health.aistupidlevel_client import AiStupidLevelClient
        from ...model_health.anthropic_status_client import AnthropicStatusClient
        from ...model_health.report import build_assessment, render_ja, render_zh

        model = str((args or {}).get("model") or "").strip()
        if not model:
            model = str(getattr(self._llm, "model", "") or "").strip()
        if not model:
            return None, {"status": "error", "error": "現在のモデル名が不明です。"}
        if self._asl_client is None:
            self._asl_client = AiStupidLevelClient()
        if self._status_client is None:
            self._status_client = AnthropicStatusClient()
        try:
            assessment = await build_assessment(
                self._asl_client, self._status_client, model, want_coding=True
            )
        except Exception as e:  # never break the turn
            logger.warning(f"[model_health] self-check failed: {e}")
            return "\n🧠 *自己診断(失敗)*\n", {
                "status": "error",
                "error": "調子の確認に失敗した（データ源に接続できず）。",
            }
        ja, zh = render_ja(assessment), render_zh(assessment)
        logger.info(
            f"[model_health] self-check ({model}) verdict={assessment.verdict}\n{zh}"
        )
        marker = "\n🧠 *自己診断*\n"
        return marker, {"status": "ok", "report": ja}

    async def _run_builtin_tool_call(
        self, tc: ToolCallObject, budget: Dict[str, int]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute one in-process (non-MCP) tool call.

        Yields display events (e.g. a ``web_search_marker``) and, last, a
        ``{"type": "_builtin_tool_result", "message": {...}}`` carrying the
        ``role: tool`` message to feed back to the LLM. ``budget`` caps how
        many searches/fetches a single turn may run and is mutated in place.
        """
        cfg = self._web_tools_config
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        logger.info(
            f"Tool request: {name} args={json.dumps(args, ensure_ascii=False)[:600]}"
        )

        result: Any
        if name == "web_search":
            query = str(args.get("query", "")).strip()
            if budget.get("searches", 0) <= 0:
                result = {"error": "web search limit reached this turn"}
            else:
                budget["searches"] -= 1
                logger.info(f"[web_search] query: {query or '(empty)'}")
                yield {
                    "type": "web_search_marker",
                    "text": f"\n🔍 *Web検索: {query[:80] or '...'}*\n",
                }
                result = format_search_results(
                    query,
                    await web_search(
                        query,
                        provider=cfg.get("provider", "brave"),
                        api_key=cfg.get("api_key", ""),
                        max_results=5,
                    ),
                )
        elif name == "web_fetch":
            url = str(args.get("url", "")).strip()
            if budget.get("fetches", 0) <= 0:
                result = {"error": "web fetch limit reached this turn"}
            else:
                budget["fetches"] -= 1
                logger.info(f"[web_fetch] url: {url or '(empty)'}")
                yield {
                    "type": "web_search_marker",
                    "text": f"\n🔗 *Web取得: {url[:120] or '...'}*\n",
                }
                result = format_fetch_result(
                    await web_fetch(
                        url,
                        max_chars=int(cfg.get("max_fetch_chars", 20000) or 20000),
                    )
                )
        elif name in self._ALARM_TOOL_NAMES:
            marker, result = await self._run_alarm_tool(name, args)
            if marker:
                yield {"type": "tool_marker", "text": marker}
        elif name == self._MODEL_HEALTH_TOOL_NAME:
            marker, result = await self._run_model_health_tool(args)
            if marker:
                yield {"type": "tool_marker", "text": marker}
        elif name in self._STEAM_TOOL_NAMES:
            marker, result = await self._run_steam_tool(name, args)
            if marker:
                yield {"type": "tool_marker", "text": marker}
        elif name in self._MEMORY_TOOL_NAMES:
            marker, result = await self._run_memory_tool(name, args)
            if marker:
                yield {"type": "tool_marker", "text": marker}
        else:
            result = {"error": f"unknown builtin tool {name!r}"}

        yield {
            "type": "_builtin_tool_result",
            "message": {
                "role": "tool",
                "tool_call_id": tc.id,
                # str passthrough: formatted web results are already text.
                "content": (
                    result
                    if isinstance(result, str)
                    else json.dumps(result, ensure_ascii=False)
                ),
            },
        }

    _ALARM_TOOL_NAMES = ("set_alarm", "list_alarms", "cancel_alarm")
    _MODEL_HEALTH_TOOL_NAME = "check_model_status"
    _STEAM_TOOL_NAMES = (
        "steam_library",
        "steam_search",
        "steam_game",
        "steam_discover",
    )
    # Per-operation marker text is built by _steam_marker (🎮 *Steam◯◯: …*);
    # display/history only — stripped from the AI replay like all markers.
    # steam_discover section → search/results parameters (verified live for the
    # filter= variants; the sort_by variants are the same endpoint family).
    _STEAM_SECTION_MAP = {
        "trending_new": {"filter": "popularnew"},
        "top_sellers": {"filter": "topsellers"},
        "coming_soon": {"filter": "comingsoon"},
        "most_wishlisted": {"filter": "popularwishlist"},
        "top_rated": {"sort_by": "Reviews_DESC"},
        "new_releases": {"sort_by": "Released_DESC"},
    }
    _MEMORY_TOOL_NAMES = (
        "memory_search",
        "history_search",
        "memory_add",
        "memory_update",
        "memory_delete",
        "memory_read_diary",
        "memory_write_diary",
        "model_history",
    )
    # Per-operation marker text is built by _memory_marker (📝 *記憶◯◯: …*);
    # display/history only — stripped from the AI replay like all markers.
    # Approval phrases あさひ might actually type (JA/ZH/EN). Matched against
    # the TAIL of his latest real message — the second factor behind the
    # model's confirmed=true claim. Staged deletions expire after 15 min.
    # いい/消してもいい added 08-09 — あさひ's actual replies kept missing the
    # old list (いいよ/消していい didn't cover them) and he had to fall back
    # to typing "ok". Bare いい is deliberately loose; the regex only gates a
    # confirmed=true call answering a question he was just asked.
    _MEMORY_APPROVE_RE = re.compile(
        r"(同意|承認|許可|いいよ|いいです|いい|ええよ|構わない|かまわない|どうぞ|"
        r"消していい|消してもいい|消しちゃっていい|削除していい|削除して構わない|"
        r"オーケー|删吧|删了吧|删掉|可以删|同意删除|はい|OK|ok|Ok)"
    )
    # Deletion-EXPLICIT phrases: when the user's OWN current message already
    # says "delete it", the first memory_delete call executes directly — no
    # second confirmation round (あさひ 08-09: the round-trip doubled the
    # work, and edit turns often think past the replay cap, dropping the
    # fact_id from context and forcing a full redo). Deliberately narrower
    # than _MEMORY_APPROVE_RE: generic approvals (はい/OK/いいよ…) only
    # unlock a deletion the user has SEEN in the two-phase flow — a stray
    # "OK" must not authorize a deletion the model decided on by itself.
    # Lookbehinds keep 取り消して ("cancel that") from reading as delete
    # consent while やっぱり消して still matches.
    _MEMORY_DELETE_DIRECT_RE = re.compile(
        r"((?<!取り)消して|(?<!取り)消せ|消そう|消してもいい|消しちゃって|"
        r"削除して|削除しよう|删吧|删了|删掉|可以删|同意删除|"
        r"delete (it|that|this))"
    )
    _MEMORY_DELETE_TTL_S = 900.0

    async def _run_alarm_tool(self, name: str, args: Dict[str, Any]) -> tuple:
        """Execute one alarm tool call. Returns (marker_text|None, result_dict).
        Shared by the OpenAI and Claude loops — alarms are provider-agnostic."""
        self._turn_inproc_calls.append(name)
        if self._alarm_store is None:
            return None, {"error": "alarm feature is not available"}
        if name == "set_alarm":
            note = str(args.get("note", "")).strip()
            fire_at_utc, err = resolve_fire_at(
                in_minutes=args.get("in_minutes"), at=args.get("at")
            )
            if not note:
                return "\n⏰ *Alarm set(失敗)*\n", {
                    "status": "error",
                    "message": "note（思い出す内容）が必要です。",
                }
            if err:
                return "\n⏰ *Alarm set(失敗)*\n", {
                    "status": "error",
                    "message": f"時刻を解釈できませんでした: {err}",
                }
            wake = bool(args.get("wake", False))
            force = bool(args.get("force", False))
            dup = None if force else await self._alarm_store.find_near(fire_at_utc)
            if dup is not None:
                # Near-duplicate: don't create. Hand the existing alarm back so the
                # model can reconsider this same turn and, if it still judges
                # another is needed, re-call with force=true.
                dup_wake = bool(dup.get("wake"))
                # An ordinary reminder standing where a wake alarm was asked for
                # is not a duplicate in the way that matters: it will speak, but
                # it will not make a sound he can hear from bed.
                wake_gap = (
                    "ただし既存のものは音楽を鳴らさない（起こす設定ではない）ので、"
                    "本当に起こす必要があるなら force=true で設定し直すこと。"
                    if wake and not dup_wake
                    else ""
                )
                return "\n⏰ *Alarm set(重複スキップ)*\n", {
                    "status": "duplicate_nearby",
                    "message": (
                        f"近い時刻（{format_local(dup['fire_at_utc'])}）に"
                        f"既にアラームがある:「{dup.get('note', '')}」"
                        f"(id: {dup['id']})。同じ用件ならこれ以上設定しなくてよい。"
                        "別の用件で本当に必要だと自分で判断する場合のみ、"
                        "force=true を付けて set_alarm を呼び直すこと。" + wake_gap
                    ),
                    "existing": {
                        "id": dup["id"],
                        "at_local": format_local(dup["fire_at_utc"]),
                        "note": dup.get("note", ""),
                        "wake": dup_wake,
                    },
                }
            record = await self._alarm_store.add(
                fire_at_utc=fire_at_utc, note=note, wake=wake
            )
            local = format_local(record["fire_at_utc"])
            marker = f"\n⏰ *Alarm set: {local}{'（起こす）' if wake else ''}*\n"
            return marker, {
                "status": "ok",
                "message": (
                    f"アラームを {local} に設定しました。"
                    + ("音楽を鳴らして起こす設定。" if wake else "")
                ),
                "id": record["id"],
                "at_local": local,
                "note": note,
                "wake": wake,
            }
        if name == "list_alarms":
            pending = await self._alarm_store.list_pending()
            # ``wake`` travels with every entry: without it the model cannot
            # answer "which of these will actually wake me up", and records
            # written before the field existed correctly read as False.
            return f"\n⏰ *Alarm list: {len(pending)}件*\n", {
                "status": "ok",
                "count": len(pending),
                "alarms": [
                    {
                        "id": a["id"],
                        "at_local": format_local(a["fire_at_utc"]),
                        "note": a.get("note", ""),
                        "wake": bool(a.get("wake")),
                    }
                    for a in pending
                ],
            }
        if name == "cancel_alarm":
            alarm_id = str(args.get("alarm_id", "")).strip()
            record = await self._alarm_store.cancel(alarm_id)
            if record is None:
                return "\n⏰ *Alarm cancel(失敗)*\n", {
                    "status": "error",
                    "message": (
                        f"アラーム {alarm_id} が見つかりませんでした"
                        "（既に取り消し済み、または通知済みかもしれません）。"
                    ),
                }
            local = format_local(record.get("fire_at_utc")) if record else alarm_id
            return f"\n⏰ *Alarm cancel: {local}*\n", {
                "status": "ok",
                "message": "アラームを取り消しました。",
                "id": alarm_id,
                "wake": bool(record.get("wake")),
            }
        return None, {"error": f"unknown alarm tool {name!r}"}

    # ------------------------------------------------------------------
    # Steam tools (in-process, provider-agnostic — shared by both loops)
    # ------------------------------------------------------------------

    async def _run_steam_tool(self, name: str, args: Dict[str, Any]) -> tuple:
        """Execute one steam_* tool call. Returns (marker_text|None, result_dict).

        Never raises: SteamUnavailable (and anything else) is converted into a
        Japanese, actionable error dict. On success the result is also staged
        as a compact 【Steamデータ】 block folded into the NEXT outgoing user
        message (cross-turn visibility; see _stage_steam_block)."""
        self._turn_inproc_calls.append(name)
        if (
            not self._steam_enabled
            or self._steam_client is None
            or self._steam_snapshot_mgr is None
        ):
            return None, {
                "status": "error",
                "message": "Steam機能は現在利用できない（未設定）。",
            }
        args = args or {}
        try:
            from ...steam import SteamUnavailable
        except Exception as e:  # module not importable — should not happen once wired
            logger.error(f"[steam] steam module unavailable: {e}")
            return None, {
                "status": "error",
                "message": "Steamモジュールを読み込めなかった。",
            }
        try:
            if name == "steam_library":
                result = self._steam_library_query(args)
            elif name == "steam_search":
                result = await self._steam_search_query(args)
            elif name == "steam_game":
                result = await self._steam_game_query(args)
            elif name == "steam_discover":
                result = await self._steam_discover_query(args)
            else:
                return None, {
                    "status": "error",
                    "message": f"不明なSteamツール: {name!r}",
                }
        except SteamUnavailable as e:
            logger.warning(f"[steam] {name} unavailable: {e}")
            result = {
                "status": "error",
                "message": (
                    f"Steamに接続できなかった（{e}）。"
                    "一時的な不調の可能性が高いので、少し待ってから再試行すること。"
                ),
            }
            return self._steam_marker(name, args, result), result
        except Exception:
            logger.exception(f"[steam] {name} failed unexpectedly")
            result = {
                "status": "error",
                "message": "Steamデータの処理中に内部エラーが起きた。再試行してよい。",
            }
            return self._steam_marker(name, args, result), result
        if result.get("status") == "ok":
            self._stage_steam_block(name, args, result)
        return self._steam_marker(name, args, result), result

    @staticmethod
    def _clip_marker(t: Any, n: int = 24) -> str:
        """Clip text for a tool marker; '*' would break the strip regex."""
        s = " ".join(str(t or "").split()).replace("*", "＊")
        return s[:n] + ("…" if len(s) > n else "")

    @staticmethod
    def _steam_marker(
        name: str, args: Dict[str, Any], result: Dict[str, Any]
    ) -> Optional[str]:
        """Human-facing marker describing WHAT the Steam op did — shown in
        chat / kept in stored history, stripped from the AI replay. EVERY call
        gets one (あさひ audits tool use from chat); failures carry (失敗)."""
        clip = BasicMemoryAgent._clip_marker
        if name == "steam_library":
            action = str(args.get("action", ""))
            if action == "check":
                label = f"Steam所持確認: {clip(args.get('name') or args.get('appid'))}"
            else:
                lbl = {
                    "overview": "ライブラリ概況",
                    "top_played": "プレイ時間上位",
                    "recent": "直近プレイ",
                    "wishlist": "ウィッシュリスト",
                    "followed": "フォロー中",
                }.get(action, action)
                label = f"Steamライブラリ: {clip(lbl)}"
        elif name == "steam_search":
            qs = args.get("queries")
            first = qs[0] if isinstance(qs, list) and qs else (qs or "")
            more = f" 他{len(qs) - 1}" if isinstance(qs, list) and len(qs) > 1 else ""
            label = f"Steam検索: {clip(first)}{more}"
        elif name == "steam_game":
            label = f"Steamストア: {clip(result.get('name') or args.get('appid'))}"
        else:  # steam_discover
            mode = str(args.get("mode", ""))
            if mode == "similar":
                label = (
                    f"Steam類似検索: "
                    f"{clip(result.get('anchor_name') or args.get('anchor'))}"
                )
            elif mode == "tag":
                tag = result.get("tag")
                tag_name = tag.get("name") if isinstance(tag, dict) else None
                label = f"Steamタグ候補: {clip(tag_name or args.get('tag'))}"
            else:
                label = "Steamセール確認"
        if result.get("status") not in ("ok", "need_clarification"):
            label += "(失敗)"
        return f"\n🎮 *{label}*\n"

    def _steam_snapshot(self) -> Optional[Dict[str, Any]]:
        """Current snapshot dict, reloaded from disk at most every 30s so the
        background enrichment pass (tags/achievements landing after startup)
        becomes visible without a restart. Last good copy kept as fallback."""
        now = time.monotonic()
        if (
            self._steam_snapshot_cache is not None
            and now - self._steam_snapshot_loaded_at < 30.0
        ):
            return self._steam_snapshot_cache
        try:
            snap = self._steam_snapshot_mgr.load() if self._steam_snapshot_mgr else None
        except Exception as e:
            logger.warning(f"[steam] snapshot load failed: {e}")
            snap = None
        if snap is not None:
            self._steam_snapshot_cache = snap
        self._steam_snapshot_loaded_at = now
        return self._steam_snapshot_cache

    def _steam_owned_set(self) -> Set[int]:
        snapshot = self._steam_snapshot()
        if not snapshot:
            return set()
        try:
            return self._steam_snapshot_mgr.owned_set(snapshot)
        except Exception as e:
            logger.warning(f"[steam] owned_set failed: {e}")
            return set()

    @staticmethod
    def _steam_game_row(g: Dict[str, Any]) -> Dict[str, Any]:
        """Compact copy of a snapshot game record: playtime minutes → hours,
        unix last-played → date string. Unknown keys pass through untouched."""
        row = dict(g)
        for key, out in (
            ("playtime_forever", "hours_total"),
            ("playtime_2weeks", "hours_2weeks"),
        ):
            if row.get(key) is not None:
                try:
                    row[out] = round(int(row.pop(key)) / 60, 1)
                except (TypeError, ValueError):
                    row.pop(key, None)
        rt = row.pop("rtime_last_played", None)
        if rt:
            try:
                row["last_played"] = datetime.fromtimestamp(int(rt)).strftime(
                    "%Y-%m-%d"
                )
            except (TypeError, ValueError, OSError, OverflowError):
                pass
        return row

    def _steam_library_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Pure local snapshot queries — no network."""
        action = str(args.get("action", "")).strip()
        try:
            n = max(1, min(int(args.get("n", 10)), 50))
        except (TypeError, ValueError):
            n = 10
        mgr = self._steam_snapshot_mgr
        snapshot = self._steam_snapshot()
        if snapshot is None:
            return {
                "status": "error",
                "message": (
                    "Steamスナップショットが未取得（起動直後か、取得に失敗）。"
                    "ストア照会（steam_search / steam_game / steam_discover）は使える。"
                ),
            }
        library_ok = bool(snapshot.get("library_ok"))
        if action == "overview":
            owned = snapshot.get("owned") or []
            total_min = sum(int(g.get("playtime_forever") or 0) for g in owned)
            out = {
                "status": "ok",
                "library_ok": library_ok,
                "owned_count": len(owned),
                "total_hours": round(total_min / 60, 1),
                "wishlist_count": len(snapshot.get("wishlist") or []),
                "followed_count": len(snapshot.get("followed") or []),
                "top_played": [
                    self._steam_game_row(g) for g in mgr.top_played(snapshot, 5)
                ],
            }
            if not library_ok:
                out["note"] = (
                    "ライブラリ未取得（APIキー未設定またはプロフィール非公開）。"
                    "所持・プレイ時間データは無い。"
                )
            return out
        if action == "top_played":
            return {
                "status": "ok",
                "library_ok": library_ok,
                "games": [self._steam_game_row(g) for g in mgr.top_played(snapshot, n)],
            }
        if action == "recent":
            return {
                "status": "ok",
                "library_ok": library_ok,
                "games": [self._steam_game_row(g) for g in mgr.recent_detail(snapshot)][
                    :n
                ],
            }
        if action == "check":
            query = str(args.get("name", "")).strip()
            appid_arg = args.get("appid")
            if appid_arg is not None:
                try:
                    target = int(appid_arg)
                except (TypeError, ValueError):
                    return {"status": "error", "message": "appid は整数で。"}
                matches = [
                    {
                        "appid": target,
                        "name": g.get("name"),
                        "where": where,
                        "playtime_forever": g.get("playtime_forever", 0),
                    }
                    for where in ("owned", "wishlist", "followed")
                    for g in snapshot.get(where) or []
                    if int(g.get("appid") or 0) == target
                ]
                query = query or str(target)
            elif query:
                matches = mgr.find_game(snapshot, query)
            else:
                return {
                    "status": "error",
                    "message": "action=check には name か appid が必要。",
                }
            if not matches:
                return {
                    "status": "ok",
                    "found": False,
                    "query": query,
                    "hint": (
                        "ライブラリ外。steam_searchでappidを解決してから"
                        "steam_gameで確認可能"
                    ),
                }
            return {
                "status": "ok",
                "found": True,
                "query": query,
                "matches": [self._steam_game_row(m) for m in matches[:n]],
            }
        if action == "wishlist":
            wl = snapshot.get("wishlist") or []
            return {
                "status": "ok",
                "count": len(wl),
                "items": [
                    {
                        "appid": w.get("appid"),
                        "name": w.get("name"),
                        "date_added": w.get("date_added"),
                    }
                    for w in wl[:n]
                ],
            }
        if action == "followed":
            fl = snapshot.get("followed") or []
            return {
                "status": "ok",
                "count": len(fl),
                "items": [
                    {"appid": f.get("appid"), "name": f.get("name")} for f in fl[:n]
                ],
            }
        return {"status": "error", "message": f"不明なaction: {action!r}"}

    async def _steam_search_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store search over up to 4 name variants, merged round-robin and
        deduped by appid, with owned/wishlist membership marked."""
        queries = args.get("queries")
        if isinstance(queries, str):
            queries = [queries]
        queries = [str(q).strip() for q in (queries or []) if str(q).strip()][:4]
        if not queries:
            return {
                "status": "error",
                "message": "queries（ゲーム名の候補、配列）が必要。",
            }
        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5
        raw = await asyncio.gather(
            *(self._steam_client.store_search(q, limit=limit) for q in queries),
            return_exceptions=True,
        )
        per_query: List[List[Dict[str, Any]]] = []
        failures = 0
        for q, res in zip(queries, raw):
            if isinstance(res, BaseException):
                logger.warning(f"[steam] store_search({q!r}) failed: {res}")
                failures += 1
                per_query.append([])
            else:
                per_query.append(list(res or []))
        if failures == len(queries):
            return {
                "status": "error",
                "message": "Steamストア検索に失敗した。少し待ってから再試行すること。",
            }
        snapshot = self._steam_snapshot() or {}
        owned = self._steam_owned_set()
        wishlisted = {w.get("appid") for w in (snapshot.get("wishlist") or [])}
        seen: Set[int] = set()
        merged: List[Dict[str, Any]] = []
        # Round-robin across the query variants so each contributes its best
        # hits even when the cap cuts the tail.
        for i in range(max((len(lst) for lst in per_query), default=0)):
            for lst in per_query:
                if i < len(lst):
                    item = lst[i]
                    appid = item.get("appid")
                    if appid in seen:
                        continue
                    seen.add(appid)
                    row = dict(item)
                    row["owned"] = appid in owned
                    row["wishlisted"] = appid in wishlisted
                    merged.append(row)
        merged = merged[:limit]
        return {
            "status": "ok",
            "queries": queries,
            "count": len(merged),
            "results": merged,
        }

    async def _steam_game_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store-page view: appdetails + review summary fetched concurrently
        and merged; each part may fail independently."""
        try:
            appid = int(args.get("appid"))
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": "appid（整数）が必要。名前しか分からなければ先にsteam_searchで解決すること。",
            }
        details_res, reviews_res = await asyncio.gather(
            self._steam_client.app_details(appid),
            self._steam_client.app_reviews_summary(appid),
            return_exceptions=True,
        )
        details = None if isinstance(details_res, BaseException) else details_res
        reviews = None if isinstance(reviews_res, BaseException) else reviews_res
        if isinstance(details_res, BaseException):
            logger.warning(f"[steam] app_details({appid}) failed: {details_res}")
        if isinstance(reviews_res, BaseException):
            logger.warning(
                f"[steam] app_reviews_summary({appid}) failed: {reviews_res}"
            )
        if details is None and reviews is None:
            return {
                "status": "error",
                "message": (
                    f"appid {appid} の情報を取得できなかった"
                    "（存在しないappidか、一時的な失敗）。"
                ),
            }
        result: Dict[str, Any] = {"status": "ok", "appid": appid}
        if details:
            result.update(details)
        else:
            result["note"] = "詳細の取得に失敗（レビュー概況のみ）。"
        result["reviews"] = reviews
        result.pop("header_image", None)  # URL noise for the LLM
        snapshot = self._steam_snapshot() or {}
        game_tags = snapshot.get("game_tags") or {}
        tags = game_tags.get(str(appid)) or game_tags.get(appid)
        if tags:
            result["community_tags"] = tags
        result["owned"] = appid in self._steam_owned_set()
        return result

    async def _steam_discover_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Real store candidates (similar / tag ranking / specials), with the
        user's owned titles always filtered out."""
        mode = str(args.get("mode", "")).strip()
        try:
            count = max(1, min(int(args.get("count", 10)), 30))
        except (TypeError, ValueError):
            count = 10
        mgr = self._steam_snapshot_mgr
        snapshot = self._steam_snapshot()
        owned = self._steam_owned_set()

        def _finish(
            rows: List[Dict[str, Any]], extra: Dict[str, Any]
        ) -> Dict[str, Any]:
            unowned = [r for r in rows if r.get("appid") not in owned]
            out = {
                "status": "ok",
                "mode": mode,
                "results": unowned[:count],
                "count": len(unowned[:count]),
                "owned_filtered": len(rows) - len(unowned),
                "note": (
                    "所持済みタイトルは除外済み。勧めるならこのresults内から"
                    "のみ選ぶこと。"
                ),
            }
            out.update(extra)
            return out

        if mode == "similar":
            anchor = str(args.get("anchor") or "").strip()
            if not anchor:
                return {
                    "status": "error",
                    "message": "mode=similar には anchor（ゲーム名またはappid）が必要。",
                }
            anchor_extra: Dict[str, Any] = {}
            if anchor.isdigit():
                appid = int(anchor)
            else:
                matches = mgr.find_game(snapshot, anchor) if snapshot else []
                appid = matches[0].get("appid") if matches else None
                if matches:
                    anchor_extra["anchor_name"] = matches[0].get("name")
                if not appid:
                    return {
                        "status": "error",
                        "message": (
                            f"「{anchor}」をライブラリ内で特定できなかった。"
                            "steam_searchでappidを解決し、そのappidをanchorに"
                            "渡して呼び直すこと。"
                        ),
                    }
            rows = await self._steam_client.morelike(appid)
            anchor_extra["anchor_appid"] = appid
            return _finish(rows, anchor_extra)

        if mode == "tag":
            text = str(args.get("tag") or "").strip()
            if not text:
                return {
                    "status": "error",
                    "message": "mode=tag には tag（タグ名）が必要。",
                }
            if snapshot is None:
                return {
                    "status": "error",
                    "message": (
                        "タグ表（スナップショット）が未取得のためタグ解決が"
                        "できない。少し待って再試行すること。"
                    ),
                }
            candidates = mgr.resolve_tag(snapshot, text) or []
            if not candidates:
                return {
                    "status": "error",
                    "message": (
                        f"タグ「{text}」を実在のタグ表に解決できなかった。"
                        "別の言い方で試すこと。"
                    ),
                }

            def _norm(t: str) -> str:
                return unicodedata.normalize("NFKC", t or "").casefold()

            exact = [c for c in candidates if _norm(c.get("name", "")) == _norm(text)]
            if exact:
                chosen = exact[0]
            elif len(candidates) == 1:
                chosen = candidates[0]
            else:
                return {
                    "status": "need_clarification",
                    "message": (
                        "タグ候補が複数ある。candidatesのいずれかの正確な名前を "
                        "tag に指定して呼び直すこと。"
                    ),
                    "candidates": candidates[:8],
                }
            section = str(args.get("section") or "trending_new").strip()
            params = self._STEAM_SECTION_MAP.get(section)
            if params is None:
                section = "trending_new"
                params = self._STEAM_SECTION_MAP[section]
            # Over-fetch so the owned-filter still leaves ~count rows.
            total, rows = await self._steam_client.search_results(
                tags=[chosen["tagid"]], count=count + 15, **params
            )
            return _finish(
                rows, {"tag": chosen, "section": section, "total_count": total}
            )

        if mode == "specials":
            data = await self._steam_client.featured_categories()
            rows = data.get("specials") or []
            if not rows:
                return {
                    "status": "error",
                    "message": "セール情報を取得できなかった。後で再試行すること。",
                    "sections_available": sorted(data.keys()),
                }
            return _finish(rows, {"section": "specials"})

        return {"status": "error", "message": f"不明なmode: {mode!r}"}

    def _stage_steam_block(
        self, tool: str, args: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Stage a compact plain-text copy of a successful steam tool result.

        The blocks are folded into the NEXT outgoing user message by
        _to_messages (persist-not-ephemeral, exactly like the RAG blocks:
        stored == sent, never assistant-role), so the data stays visible
        across turns without the model learning to imitate tool output."""
        try:
            arg_desc = json.dumps(args or {}, ensure_ascii=False)[:120]
            body = json.dumps(result, ensure_ascii=False)
            if len(body) > 1200:
                body = body[:1200] + "…"
            self._steam_pending_blocks.append(
                f"【Steamデータ】(tool={tool}, args={arg_desc})\n{body}"
            )
            # Bound the buffer: chained calls in one turn (or an interrupted
            # turn) must not balloon the next user message.
            if len(self._steam_pending_blocks) > 6:
                self._steam_pending_blocks = self._steam_pending_blocks[-6:]
        except Exception as e:
            logger.warning(f"[steam] failed to stage cross-turn block: {e}")

    # ------------------------------------------------------------------
    # Memory self-service tools (in-process, provider-agnostic)
    # ------------------------------------------------------------------

    def _latest_user_text(self) -> str:
        """The most recent user message text from _memory (empty if none).

        Used to verify deletion approval: the model can claim consent in its
        tool arguments, but this text comes from the actual websocket input
        path — the model cannot write it.
        """
        for m in reversed(self._memory):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    str(p.get("text", ""))
                    for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return ""
        return ""

    async def _run_memory_tool(self, name: str, args: Dict[str, Any]) -> tuple:
        """Execute one memory_* tool call. Returns (marker|None, result_dict).
        Registers into _turn_inproc_calls first (tool-honesty canary).

        Never raises. CRUD goes through PersistentMemoryManager (disk = live
        truth, index synced immediately; the frozen header block catches up on
        restart). memory_delete is two-phase: stage → the character asks あさひ
        → confirmed=true is honored only when the latest REAL user message
        (read from _memory, not from the model) contains an approval phrase.
        """
        self._turn_inproc_calls.append(name)
        if not self._memory_tools_active:
            return None, {
                "status": "error",
                "message": "記憶ツールは現在利用できない（メモリ機能未接続）。",
            }
        mgr = self._memory_manager
        args = args or {}
        try:
            if name == "memory_search":
                result = await mgr.search_memory_tool(
                    str(args.get("query", "")),
                    target=str(args.get("target", "both") or "both"),
                    n=args.get("n", 5),
                    date_from=str(args.get("date_from", "") or ""),
                    date_to=str(args.get("date_to", "") or ""),
                )
            elif name == "history_search":
                result = await self._history_search_query(args)
            elif name == "memory_add":
                result = await mgr.add_fact_manual(
                    str(args.get("fact", "")),
                    importance=str(args.get("importance", "low") or "low"),
                    store_id=str(args.get("store_id", "") or ""),
                )
            elif name == "memory_update":
                # store_id: absent → None (untouched); present-but-empty is a
                # deliberate CLEAR, so no str-or-None coercion here.
                raw_sid = args.get("store_id")
                result = await mgr.update_fact_manual(
                    str(args.get("fact_id", "")).strip(),
                    str(args.get("new_fact", "")),
                    importance=(str(args.get("importance", "")).strip() or None),
                    store_id=None if raw_sid is None else str(raw_sid),
                )
            elif name == "memory_delete":
                result = await self._memory_delete_flow(args)
            elif name == "memory_read_diary":
                result = self._memory_read_diary_query(args)
            elif name == "model_history":
                result = self._model_history_query(args)
            elif name == "memory_write_diary":
                result = self._memory_write_diary_query(args)
            else:
                return None, {
                    "status": "error",
                    "message": f"不明な記憶ツール: {name!r}",
                }
        except Exception:
            logger.exception(f"[memory_tool] {name} failed unexpectedly")
            result = {
                "status": "error",
                "message": "記憶の処理中に内部エラーが起きた。再試行してよい。",
            }
            return self._memory_marker(name, args, result), result
        return self._memory_marker(name, args, result), result

    @staticmethod
    def _memory_marker(
        name: str, args: Dict[str, Any], result: Dict[str, Any]
    ) -> Optional[str]:
        """Human-facing marker describing WHAT the memory op did — shown in
        chat / kept in stored history, stripped from the AI replay. EVERY call
        gets one (あさひ audits tool use from chat); failures carry (失敗)."""
        _clip = BasicMemoryAgent._clip_marker
        status = result.get("status")
        if name == "memory_search":
            label = f"記憶検索: {_clip(args.get('query'))}"
        elif name == "history_search":
            kws = "、".join(split_search_keywords(args.get("keywords")))
            label = f"記憶検索(履歴): {_clip(kws)}"
        elif name == "memory_add":
            label = f"記憶追加: {_clip(args.get('fact'))}"
        elif name == "memory_update":
            if args.get("new_fact"):
                label = f"記憶更新: {_clip(args.get('new_fact'))}"
            else:
                # importance/store_id-only call: no new_fact to show, so say
                # WHAT changed and fall back to the target's text (in the ok
                # result) or the given id, keeping the audit line legible.
                imp = str(args.get("importance") or "").strip()
                if imp:
                    kind = f"重要度→{imp}"
                else:
                    kind = "店舗ID" + ("設定" if args.get("store_id") else "解除")
                body = result.get("fact") or args.get("fact_id")
                label = f"記憶更新({kind}): {_clip(body)}"
        elif name == "memory_read_diary":
            label = f"日記閲覧: {_clip(result.get('date') or args.get('diary_uid'))}"
        elif name == "model_history":
            label = f"モデル履歴: {_clip(args.get('date'))}"
        elif name == "memory_write_diary":
            label = f"日記記入: {_clip(result.get('date') or '')}" + (
                "（上書き）" if result.get("overwrote") else ""
            )
        elif status == "pending_approval":
            label = f"記憶削除申請: {_clip(result.get('fact'))}"
        else:
            label = f"記憶削除: {_clip(result.get('deleted') or args.get('fact_id'))}"
        if status not in ("ok", "pending_approval"):
            label += "(失敗)"
        return f"\n📝 *{label}*\n"

    async def _history_search_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """history_search — keyword full-scan over the chat log (no RAG).

        The scan is synchronous file IO over every stored session, so it runs
        in a worker thread to keep the event loop responsive."""
        conf_uid = str(getattr(self._memory_manager, "_conf_uid", "") or "")
        if not conf_uid:
            return {"status": "error", "message": "会話ログの場所が特定できない。"}
        # Pass the raw argument through — search_history normalizes it via
        # split_search_keywords, including the observed misuse of one
        # comma-joined string instead of an array (iterating that string
        # here would char-shatter it into single-character keywords).
        return await asyncio.to_thread(
            search_history,
            conf_uid,
            args.get("keywords"),
            date_from=str(args.get("date_from", "") or "").strip(),
            date_to=str(args.get("date_to", "") or "").strip(),
        )

    def _model_history_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """model_history — sessions touching one JST date, with the model
        that experienced each. Reads the boot-frozen map snapshot; the
        result is replay-EXEMPT (small, and truncating a date table would
        just invite re-queries)."""
        date = str(args.get("date", "")).strip()
        # Shape AND calendar validity: the regex keeps zero-padded form
        # (string comparisons downstream), strptime rejects 2026-13-40.
        valid = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date))
        if valid:
            try:
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                valid = False
        if not valid:
            return {
                "status": "error",
                "message": f"date は YYYY-MM-DD 形式で指定すること: {date!r}",
            }
        mgr = self._memory_manager
        rows = mgr.sessions_for_date(date) if mgr else []
        out: Dict[str, Any] = {"status": "ok", "date": date, "sessions": rows}
        if not rows:
            out["note"] = (
                "それは未来の日付——記録はまだ存在しない。"
                if date > datetime.now().strftime("%Y-%m-%d")
                else "この日のセッション記録はない。"
            )
        return out

    def _memory_read_diary_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Full diary view. 08-13: the result is replay-EXEMPT (see
        _PROTOCOL_EXEMPT_RESULT_TOOLS) — reading = injecting, permanently.
        Hence the per-turn cap, and the ledger marks every sentence so
        auto-RAG never re-surfaces any part of a read diary."""
        uid = str(args.get("diary_uid", "")).strip()
        if not uid:
            return {
                "status": "error",
                "message": "diary_uid が必要（memory_searchの日記ヒットに付いている）。",
            }
        limit = int(
            getattr(
                getattr(self._memory_manager, "diary_rag_config", None),
                "full_reads_per_turn",
                5,
            )
            or 5
        )
        if self._diary_reads_this_turn >= limit:
            return {
                "status": "error",
                "message": (
                    f"このターンの日記全文読取は上限{limit}回に達した。"
                    "続きは次のターンで。"
                ),
            }
        uid, matches = self._memory_manager.resolve_diary_uid(uid)
        if uid is None:
            if matches:
                return {
                    "status": "error",
                    "message": (
                        f"idが曖昧（{len(matches)}件一致）。候補: "
                        + ", ".join(sorted(matches)[:5])
                    ),
                }
            return {
                "status": "error",
                "message": f"日記 {args.get('diary_uid')} が見つからない。",
            }
        entry = self._memory_manager.read_diary_full(uid)
        if not entry:
            return {"status": "error", "message": f"日記 {uid} が見つからない。"}
        self._diary_reads_this_turn += 1
        self._ledger_mark_full(uid)
        logger.info(
            f"[diary_read] {uid} 全文読取 "
            f"({self._diary_reads_this_turn}/{limit} this turn)"
        )
        return {
            "status": "ok",
            "diary_uid": uid,
            **entry,
            "note": "この全文はこのまま会話の文脈に残る（再読は不要）。",
        }

    def _memory_write_diary_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Write THIS session's diary (memory_write_diary, あさひ 08-14).

        Replaces the boot-time generation for sessions she closes herself:
        the file lands in diaries/ in the exact backfill shape, so the next
        startup sees it and skips the LLM pass (exists → skip), while the
        fact-extraction pass still picks it up (no facts_extracted flag).
        Restricted to the CURRENT session — past diaries stay immutable
        (sentence numbering / ledger / embeddings all assume it). Calling
        again overwrites her own draft (session continued past the first
        goodnight); the index drift check re-embeds the changed chunks at
        the next boot."""
        content = str(args.get("content", "")).strip()
        if len(content) < 100:
            return {
                "status": "error",
                "message": "内容が短すぎる（100字以上。セッション全体を振り返って書くこと）。",
            }
        if not self._history_uid:
            return {"status": "error", "message": "現在のセッションが特定できない。"}
        result = self._memory_manager.write_session_diary(self._history_uid, content)
        if result.get("status") == "ok":
            logger.info(
                f"[diary_write] session diary saved by the character "
                f"({len(content)}字, overwrote={result.get('overwrote', False)})"
            )
        return result

    async def _memory_delete_flow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Two-phase deletion with mechanically-verified user approval."""
        mgr = self._memory_manager
        fact_id = str(args.get("fact_id", "")).strip()
        if not fact_id:
            return {"status": "error", "message": "fact_id が必要。"}
        # Expire stale requests up front.
        now = time.monotonic()
        self._pending_memory_deletes = {
            k: v
            for k, v in self._pending_memory_deletes.items()
            if now - v.get("staged_at", 0.0) < self._MEMORY_DELETE_TTL_S
        }

        if not args.get("confirmed"):
            fact = mgr.find_fact(fact_id)
            if fact is None:
                return {
                    "status": "error",
                    "message": f"id {fact_id} の記憶が見つからない。memory_searchで確認を。",
                }
            if (fact.get("importance") or "low") == "user":
                return {
                    "status": "error",
                    "message": "userレベルの記憶は本人管理のため削除できない。",
                }
            # Fast path: his own current message explicitly asks for the
            # deletion — the confirm round-trip would be pure friction.
            tail = self._latest_user_text()[-200:]
            if self._MEMORY_DELETE_DIRECT_RE.search(tail):
                logger.info(
                    f"[memory_tool] delete DIRECT ({fact_id}) — user's own "
                    f"message asks for it: {fact.get('fact', '')[:80]}"
                )
                return await mgr.delete_fact_manual(fact_id)
            self._pending_memory_deletes[fact_id] = {
                "fact": fact.get("fact", ""),
                "staged_at": now,
            }
            # Keep the staging dict tiny — one conversation won't legitimately
            # stage many deletions at once.
            if len(self._pending_memory_deletes) > 5:
                oldest = min(
                    self._pending_memory_deletes,
                    key=lambda k: self._pending_memory_deletes[k]["staged_at"],
                )
                self._pending_memory_deletes.pop(oldest, None)
            logger.info(
                f"[memory_tool] delete STAGED ({fact_id}): {fact.get('fact', '')[:80]}"
            )
            return {
                "status": "pending_approval",
                "id": fact_id,
                "fact": fact.get("fact", ""),
                "message": (
                    "削除には本人の同意が必要。この記憶を削除してよいか、"
                    "内容を示して本人に確認すること。同意の返事をもらった"
                    "直後のターンで、同じ fact_id と confirmed=true で"
                    "呼び直すと実行される。"
                ),
            }

        pending = self._pending_memory_deletes.get(fact_id)
        if pending is None:
            return {
                "status": "error",
                "message": (
                    "この id の削除申請が無い（未申請か期限切れ）。"
                    "先に confirmed 無しで申請し、本人の同意を得ること。"
                ),
            }
        tail = self._latest_user_text()[-200:]
        if not self._MEMORY_APPROVE_RE.search(tail):
            return {
                "status": "error",
                "message": (
                    "直近のユーザー発話に同意が確認できないため削除しない。"
                    "本人がはっきり同意してから呼び直すこと。"
                ),
            }
        self._pending_memory_deletes.pop(fact_id, None)
        return await mgr.delete_fact_manual(fact_id)

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
