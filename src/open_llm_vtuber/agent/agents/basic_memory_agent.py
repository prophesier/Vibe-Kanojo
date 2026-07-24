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
from ...web_tools import web_search, web_fetch
from ...alarms import resolve_fire_at, format_local
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import (
    CACHE_SEAM_MARKER,
    AsyncLLM as OpenAICompatibleAsyncLLM,
)
from ...chat_history_manager import (
    get_history,
    get_recent_histories,
    pop_last_message,
    search_history,
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


# Tool-execution marker lines that get streamed to the UI and persisted into
# chat history for human review (see _mcp_tool_marker, _run_builtin_tool /
# _run_alarm_tool, and claude_llm's native web tags). They must be STRIPPED from
# the copy of history replayed to the model: the model imitates them, emitting
# the marker (and inventing the tool's result) without actually calling the tool
# — the failure a "don't imitate" system note could not suppress. Live chat and
# the stored history keep them; only the model's replay copy is cleaned.
#
# The markers are EMITTED as "\n<glyph> *…*\n", but the sentence/TTS pipeline
# collapses those newlines before the turn is persisted, so in stored history
# they sit INLINE, glued to the reply text (e.g. "…では試す。🔍 *Web検索: …*[neutral]
# ……"). So match/remove them as an inline substring — NOT line-anchored (an
# earlier line-anchored version matched zero real markers and the model kept
# imitating them).
#
# Each alternative is keyed to its exact glyph + label, and the variable
# query/url is bound with [^*\n]* so removal stops at the marker's OWN closing
# '*'. That is deliberately conservative: replies contain real markdown like
# "**Zero Escape**", so a greedy ".*\*" would swallow reply text — instead, a
# query/url that itself contains '*' (rare) is left as-is rather than risk
# eating real content. Keep in sync with the emitters (currently: 🍔 Uber /
# 🔍 Web検索 / 🔗 Web取得 / ⏰ Alarm set / 🧠 自己診断 / 🎮 Steam / 📝 記憶).
_TOOL_MARKER_RE = re.compile(
    r"[ \t]*(?:"
    r"🍔[ \t]*\*Uber Eats\*"
    r"|🔍[ \t]*\*Web検索:[^*\n]*\*"
    r"|🔗[ \t]*\*Web取得:[^*\n]*\*"
    r"|⏰[ \t]*\*Alarm[^*\n]*\*"
    r"|🧠[ \t]*\*自己診断[^*\n]*\*"
    r"|🎮[ \t]*\*Steam[^*\n]*\*"
    r"|📝[ \t]*\*記憶[^*\n]*\*"
    r"|🔧[ \t]*\*ツール:[^*\n]*\*"
    r")"
)


def _strip_tool_markers(text: str) -> str:
    """Remove tool-execution markers from an assistant turn before it is replayed
    to the model (it imitates them — emits the marker and fabricates the tool's
    result without calling it). Markers stay in stored history / live chat, so
    they remain human-searchable; only the model's replay copy is cleaned.

    Strips the marker wherever it sits inline (the TTS pipeline collapses the
    newlines it was emitted with); the bounded variable part can't eat the
    reply's own '**bold**'. See :data:`_TOOL_MARKER_RE`."""
    if "*" not in text:  # every marker contains '*' — cheap fast path
        return text
    return re.sub(r"\n{3,}", "\n\n", _TOOL_MARKER_RE.sub("", text)).strip()


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
        # Memory blocks pinned via memory_inject — folded into the NEXT
        # outgoing user message exactly like the RAG/steam blocks
        # (persist-not-ephemeral), so they stay visible across turns.
        self._memory_pending_blocks: List[str] = []

        # Diary RAG (long-tail recall). The in-context list is ephemeral: it
        # holds retrieved diaries with a per-turn TTL and is injected only into
        # the outgoing user message — never persisted to _memory or history, so
        # the prompt cache prefix stays clean. _pending_rag_block is the block
        # built for the turn currently being assembled (consumed by _to_messages).
        # Diaries injected via RAG this session — persisted in _memory, so they
        # are excluded from further retrieval (each diary appears at most once).
        self._session_injected_uids: Set[str] = set()
        self._pending_rag_block: str = ""
        # Facts RAG (independent of diary RAG): low-importance facts recalled on
        # demand, injected after the diary block. Same persist-in-_memory pattern.
        self._session_injected_fact_ids: Set[str] = set()
        self._pending_facts_block: str = ""
        self._sliding_window_uids: Set[str] = set()
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
            and self._claude_protocol_has_thinking(claude_protocol)
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
    # text. Chars, not tokens: cheap and close enough (JA≈1, EN≈4 chars/tok).
    _PROTOCOL_RESULT_MAX_CHARS = 400
    _PROTOCOL_TRUNCATION_MARKER = "…[truncated for replay]"

    # Cold-start thinking seeds (あさひ's design, 07-23): persist each turn's
    # final assistant message ([thinking..., text], verbatim) to chat history;
    # on reload, replay the newest N of them as thinking precedent — but only
    # when the recent tail was actually thinking (rate gate), because seeding
    # a slump just reinforces it, while a bare fresh session thinks
    # spontaneously on its own.
    _THINKING_SEED_COUNT = 10
    _THINKING_SEED_RATE_WINDOW = 20  # assistant turns considered for the rate
    # Seed only a genuinely healthy tail — anything below 80% recent thinking
    # is already slipping, and replaying a slipping tail reinforces the slump.
    _THINKING_SEED_MIN_RATE = 0.8
    # Final assistant message of the just-completed turn, staged for the
    # conversation layer to persist (pop_thinking_seed). Class-level default
    # so __new__-constructed test instances read None.
    _last_thinking_seed: Optional[Dict[str, Any]] = None
    # Where this conversation persists on disk (set by the memory loaders);
    # needed by the safety-refusal cleanup. Class-level defaults for tests.
    _conf_uid: str = ""
    _history_uid: str = ""

    def pop_thinking_seed(self) -> Optional[Dict[str, Any]]:
        """One-shot getter for the just-completed turn's thinking seed.

        Returns {"model": ..., "protocol": [messages]} — the turn's full
        transcript verbatim (signed thinking included; tool_result content
        truncated with a marker), or None when the turn produced no thinking.
        The whole transcript is kept so a replayed tool turn still shows real
        tool calls — rewriting it into a "results without calls" shape would
        teach fabrication. The conversation layer attaches it to the on-disk
        history record (store_message).
        """
        seed = self._last_thinking_seed
        self._last_thinking_seed = None
        return seed

    def _apply_thinking_seeds(self, candidates: List[tuple], resuming: bool) -> None:
        """Replay the newest persisted thinking turns into reloaded context.

        Each seed becomes a length-1 ``claude_protocol`` on its memory entry,
        reusing the exact-replay machinery. Gated on the recent thinking RATE:
        below the threshold nothing is seeded (don't reinforce a slump), and a
        resumed session gets a log hint that a fresh session recovers
        spontaneous thinking. Seeds from a different model are skipped (other
        models ignore foreign thinking blocks but still bill their tokens).
        """
        if not self._is_claude_llm() or not self._memory:
            return
        assistant_idxs = [
            i for i, m in enumerate(self._memory) if m.get("role") == "assistant"
        ]
        window = assistant_idxs[-self._THINKING_SEED_RATE_WINDOW :]
        if not window:
            return
        candidate_idxs = {i for i, _ in candidates}
        thinking_turns = sum(1 for i in window if i in candidate_idxs)
        rate = thinking_turns / len(window)
        if rate < self._THINKING_SEED_MIN_RATE:
            logger.info(
                f"[thinking_seed] recent thinking rate {thinking_turns}/{len(window)} "
                f"is below {self._THINKING_SEED_MIN_RATE:.0%} — no seeds injected."
            )
            if resuming:
                logger.warning(
                    "[thinking_seed] resuming a session with a low-thinking tail; "
                    "a FRESH session recovers spontaneous thinking — consider "
                    "/restart instead of /resume."
                )
            return
        model = getattr(self._llm, "model", "") or ""
        replay_cap = getattr(self._llm, "thinking_replay_max_tokens", 0) or 0
        applied = 0
        for idx, seed in candidates[-self._THINKING_SEED_COUNT :]:
            # Fat-thinking seeds are skipped for the same reason they were
            # never stored live (field present on seeds written after 07-25).
            seed_thinking = seed.get("thinking_tokens")
            if (
                replay_cap
                and isinstance(seed_thinking, (int, float))
                and seed_thinking > replay_cap
            ):
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
            seed_model = seed.get("model") or ""
            if seed_model and model and seed_model != model:
                continue
            self._memory[idx]["claude_protocol"] = deepcopy(protocol)
            applied += 1
        logger.info(
            f"[thinking_seed] seeded {applied} thinking turn(s) into reloaded "
            f"context (recent rate {thinking_turns}/{len(window)})."
        )

    @classmethod
    def _truncate_protocol_tool_results(
        cls, protocol: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Cap tool_result content in a stored Claude protocol.

        Applies ONLY to user-role tool_result messages — those are
        client-built and unsigned, so their content is ours to edit.
        Assistant messages are never touched: signed thinking blocks are
        position-bound within their message, and sibling blocks (tool_use,
        server-tool results) must stay exactly as generated.
        """
        out: List[Dict[str, Any]] = []
        for msg in protocol:
            content = msg.get("content")
            if msg.get("role") != "user" or not isinstance(content, list):
                out.append(msg)
                continue
            new_content = [
                cls._truncate_tool_result_block(block)
                if isinstance(block, dict) and block.get("type") == "tool_result"
                else block
                for block in content
            ]
            out.append({**msg, "content": new_content})
        return out

    @classmethod
    def _truncate_tool_result_block(cls, block: Dict[str, Any]) -> Dict[str, Any]:
        limit = cls._PROTOCOL_RESULT_MAX_CHARS
        marker = cls._PROTOCOL_TRUNCATION_MARKER
        content = block.get("content")
        if isinstance(content, str):
            if len(content) <= limit:
                return block
            return {**block, "content": content[:limit] + marker}
        if isinstance(content, list):
            new_blocks: List[Dict[str, Any]] = []
            budget = limit
            changed = False
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    text = b.get("text", "")
                    if budget <= 0:
                        changed = True
                        continue
                    if len(text) > budget:
                        b = {**b, "text": text[:budget] + marker}
                        changed = True
                    budget -= min(len(text), budget)
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
    _MEMORY_CAPABILITY_NOTE = (
        "[Managing memory] Separate from automatic recall, when you want to look "
        "into the past yourself, use memory_search (memory_read_diary for a full "
        "diary entry). For concrete exchanges and proper nouns that were never "
        "kept in memory, history_search does a keyword search over the full "
        "conversation logs. For both: to narrow by date or period, use the "
        "date_from/date_to arguments — do not mix dates into the query or the "
        "keywords. Search results disappear at the end of the turn; pin what you "
        "want to keep referring to with memory_inject. To save or correct facts "
        "established in conversation, use memory_add / memory_update (only when "
        "the user asked, or the correction is unambiguous; be careful about "
        "rewriting). Deletion is memory_delete, which requires the user's "
        "consent. Additions and corrections are reflected in search immediately, "
        "and enter the resident list from the next startup. user-tier memories "
        "are the user's own (you cannot create or delete them; correcting their "
        "content is allowed)."
    )

    # Trailing system block placed right before the message history.
    # No cache_control marker — small, static, and positional. By sitting
    # last in the system prompt, it's the closest instruction to the
    # message history, which empirically improves rule adherence
    # (proximity effect).
    _HISTORY_NOTE = (
        "【以下の会話履歴について】\n\n"
        "ここから後に続くユーザーとアシスタントのやりとりは、"
        "**複数の過去セッションが時系列順に連結されたもの**。"
        "必ずしも今日の出来事だけではなく、数日前〜数週間前の古いやりとりと、"
        "直近のやりとりが一つのストリームに混在している。"
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

    def _build_runtime_system(self) -> str:
        """Return the full system prompt as a plain string (used for non-Claude LLMs).

        Order matters: HISTORY_NOTE is appended last so it sits closest to
        the message history, giving the LLM the strictest instructions
        right before it encounters the data they apply to.
        """
        parts = [self._system, self._TIMESTAMP_NOTE] + self._tool_capability_notes()
        # Resident Steam library digest — set once before the first turn and
        # never changed mid-session, so it is as cache-stable as the notes.
        if self._steam_digest:
            parts.append(self._steam_digest)
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
        parts.append(self._HISTORY_NOTE)
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

    def _build_system_for_llm(self) -> Union[str, List[Dict[str, Any]]]:
        """Return system prompt in the right shape for the active LLM.

        For Claude, returns up to 3 separately cache-controlled blocks
        followed by one un-cached positional block:
          1. Persona + minimal timestamp note (ultra-stable, changes only
             on character edit)
          2. Facts (changes only on fact extraction)
          3. Diaries (changes only when a new diary is generated)
          + HISTORY_NOTE (appended last, no cache_control). Sits right
             before the message history so its strict timestamp / history
             rules are the closest instructions to the data they govern.
             Static, so always cached by the message-level breakpoint.

        With (1)/(2)/(3) cache markers plus the last-message marker from
        _attach_cache_breakpoint, this uses all 4 of Anthropic's allowed
        cache checkpoints; HISTORY_NOTE adds no extra marker.

        For other LLMs, returns the plain combined string.
        """
        if not self._is_claude_llm():
            return self._build_runtime_system()

        blocks: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": "\n\n".join(
                    [self._system, self._TIMESTAMP_NOTE]
                    + self._tool_capability_notes()
                    # Steam digest: set once before the first turn, immutable
                    # for the session, so it belongs in this cached block.
                    + ([self._steam_digest] if self._steam_digest else [])
                ),
                "cache_control": self._CACHE_CONTROL_1H,
            }
        ]
        if self._memory_manager:
            facts_text = self._memory_manager.get_facts_prompt()
            if facts_text:
                blocks.append(
                    {
                        "type": "text",
                        "text": facts_text,
                        "cache_control": self._CACHE_CONTROL_1H,
                    }
                )
            diaries_text = self._memory_manager.get_diaries_prompt()
            if diaries_text:
                blocks.append(
                    {
                        "type": "text",
                        "text": diaries_text,
                        "cache_control": self._CACHE_CONTROL_1H,
                    }
                )
        # Trailing block — no cache_control on purpose. Stays right next
        # to the message history for maximum instruction-following effect.
        blocks.append({"type": "text", "text": self._HISTORY_NOTE})
        return blocks

    def _attach_cache_breakpoint(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Mark the newest safe history block with ``cache_control``.

        Exact Claude protocol entries containing signed thinking must remain
        structurally unchanged from the original response, so they are skipped.
        The preceding user message still provides an incremental cache
        breakpoint without modifying the latest assistant response.

        Returns a new list with one message replaced; the original message
        objects (which live in self._memory) are not mutated. Only applies for
        Claude LLM — otherwise returns messages unchanged.
        """
        if not self._is_claude_llm() or not messages:
            return messages

        new_messages = list(messages)
        for index in range(len(new_messages) - 1, -1, -1):
            candidate = new_messages[index]
            if candidate.get("claude_protocol") or candidate.get("thinking_blocks"):
                continue

            content = candidate.get("content")
            if isinstance(content, str):
                replacement = {
                    **candidate,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": self._CACHE_CONTROL_1H,
                        }
                    ],
                }
            elif isinstance(content, list) and content:
                new_content = [dict(block) for block in content]
                new_content[-1] = {
                    **new_content[-1],
                    "cache_control": self._CACHE_CONTROL_1H,
                }
                replacement = {**candidate, "content": new_content}
            else:
                continue

            new_messages[index] = replacement
            break
        return new_messages

    @staticmethod
    def _session_header_text(uid: str, is_current: bool = False) -> str:
        """Format a session-boundary banner from a history UID.

        UID format: ``YYYY-MM-DD_HH-MM-SS_<hex>``. The banner is prepended
        to the first message of each session so the LLM can distinguish
        independent sessions in the otherwise-flat message stream.
        """
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        label = "現在進行中のセッション" if is_current else "セッション"
        parts = uid.split("_")
        if len(parts) >= 2 and len(parts[0]) == 10 and len(parts[1]) == 8:
            try:
                dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H-%M-%S")
                timestamp = (
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[dt.weekday()]}"
                )
                return f"【{label}開始: {timestamp}】"
            except ValueError:
                pass
        return f"【{label}開始: {uid}】"

    def _msg_from_history_record(self, msg: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Convert a stored history record into a memory entry.

        Timestamps are prepended only to user messages so the LLM knows when
        each turn occurred.  Omitting them from assistant turns prevents the
        model from mimicking the format in its own replies.
        """
        role = "user" if msg["role"] == "human" else "assistant"
        content = msg.get("content")
        if not isinstance(content, str) or not content:
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
        self._session_injected_uids = set()
        self._pending_rag_block = ""
        self._session_injected_fact_ids = set()
        self._pending_facts_block = ""
        self._steam_pending_blocks = []
        self._pending_memory_deletes = {}
        self._memory_pending_blocks = []
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
        messages = self._memory.copy()
        # Cache breakpoint goes on the newest safe historical block. Exact
        # Claude assistant transcripts with signed thinking are skipped, so
        # their preceding user block becomes the breakpoint instead. The fresh
        # user input appended below stays uncached. No-op for non-Claude.
        messages = self._attach_cache_breakpoint(messages)
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
        memory_blocks = self._memory_pending_blocks
        self._memory_pending_blocks = []
        rag_block = "\n\n".join(
            b
            for b in (self._pending_rag_block, self._pending_facts_block)
            + tuple(steam_blocks)
            + tuple(memory_blocks)
            if b
        )
        self._pending_rag_block = ""
        self._pending_facts_block = ""
        if rag_block and text_prompt:
            payload_text = f"{rag_block}\n\n{text_prompt}"
        else:
            payload_text = rag_block or text_prompt
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

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

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

    async def _maybe_inject_diary_rag(self, input_data: BatchInput) -> None:
        """Retrieve long-tail diaries relevant to this turn and stage them.

        Sets ``_pending_rag_block`` for ``_to_messages`` to fold into the
        outgoing user message; that message is then stored in ``_memory``
        verbatim, so the conversation stays append-only and the OpenAI prefix
        cache keeps hitting. Each diary is injected at most once per session
        (``_session_injected_uids``) — afterwards it lives in history, so it is
        excluded from further retrieval. chat_history on disk stays clean
        (saved separately). Never raises; no-op when RAG is off / query empty.
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
            # Exclude what the model already has verbatim: the injected diary
            # block, the sliding-window sessions, and diaries already injected
            # earlier this session (they persist in _memory).
            # Deliberately do NOT exclude diaries already injected via RAG this
            # session. Letting the judge re-see them means it re-picks the
            # genuinely most-relevant ones (which stay at the top) instead of
            # being forced to reach for new, similar diaries every turn — those
            # re-picks then drop out below as already-present, so the in-context
            # set self-limits by relevance without a hard cap. Header diaries and
            # sliding-window sessions stay excluded (already present verbatim).
            exclude = mgr.injected_diary_uids() | self._sliding_window_uids
            n_ctx = getattr(mgr.diary_rag_config, "rerank_context_turns", 6)
            context = self._recent_dialogue_context(n_ctx)
            hits, candidates, keywords = await mgr.retrieve_diary_context(
                query, exclude, context=context
            )
            # Inject only the picks not already in context (already-injected
            # re-picks are no-ops — they're still present from earlier turns).
            new_hits = [h for h in hits if h["uid"] not in self._session_injected_uids]
            if new_hits:
                self._pending_rag_block = self._format_diary_rag_block(new_hits)
                self._session_injected_uids.update(h["uid"] for h in new_hits)

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
                    # what the judge picked (may include already-injected → no-op)
                    [
                        (
                            h["uid"][:19],
                            (h.get("reason") or round(h.get("score", 0.0), 3)),
                        )
                        for h in hits
                    ],
                    [h["uid"][:19] for h in new_hits],
                )
            )
            logger.info(
                f"[diary_rag] 候補{len(candidates)} → 採用{len(new_hits)} | "
                f"セッション内日記 {len(self._session_injected_uids)}件"
            )
        except Exception as e:
            logger.warning(f"[diary_rag] retrieval skipped: {e}")
            self._pending_rag_block = ""

    def _format_diary_rag_block(self, entries: List[Dict[str, Any]]) -> str:
        """Terse marker block for the retrieved diaries (chronological).

        The full explanation of what these blocks are lives once in
        _HISTORY_NOTE, so each injected block stays short — it now persists in
        the conversation history (once per injecting turn), so brevity matters.
        """
        lines = ["［過去の記憶（自動検索）開始］"]
        for e in sorted(entries, key=lambda x: x.get("date", "")):
            lines.append(f"〔{(e.get('date') or '')[:10]} のセッション〕")
            lines.append((e.get("content") or "").strip())
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
            # Exclude only the header-tier facts (user/llm), which are present
            # verbatim. Do NOT exclude facts already RAG-injected this session —
            # same self-limiting trick as diaries: the judge re-picks the best
            # ones and they drop out below as no-ops, so it isn't forced to keep
            # surfacing new similar facts.
            exclude = mgr.injected_fact_ids()
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
            date = (e.get("date") or "")[:10]
            prefix = f"[{date}] " if date else ""
            lines.append(f"・{prefix}{(e.get('fact') or '').strip()}")
        lines.append("［関連する事実終了］")
        return "\n".join(lines)

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
        # Billed thinking across the whole turn (all loop rounds) — compared
        # against thinking_replay_max_tokens to decide whether this turn's
        # transcript may enter later requests at all.
        turn_thinking_tokens = 0
        # Set when the safety classifier declines a request (stop_reason
        # "refusal"); handled after the stream ends.
        refusal_info: Optional[Dict[str, Any]] = None
        # Per-turn budget for the client-side web_fetch tool.
        web_fetch_budget = {
            "left": max(0, int(getattr(self._llm, "_max_web_fetches", 5) or 0))
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
                    yield f"[Error from LLM: {event['message']}]"
                    # Keep whatever she already said in context — text only, no
                    # protocol (the turn is incomplete, so the transcript can't
                    # be replayed safely). The user heard this text; dropping it
                    # would make her forget her own words next turn.
                    if current_turn_text:
                        self._add_message(current_turn_text, "assistant")
                    return

            turn_thinking_tokens += request_thinking_tokens

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
                    tool_results_for_llm.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": c["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                            "is_error": bool(result.get("error"))
                            or result.get("status") == "error",
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

                if tool_results_for_llm:
                    tool_result_message = {
                        "role": "user",
                        "content": tool_results_for_llm,
                    }
                    messages.append(deepcopy(tool_result_message))
                    claude_protocol.append(deepcopy(tool_result_message))
                continue
            else:
                if current_turn_text:
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
                        # Truncate stored tool_result content (user-role
                        # messages only — assistant messages stay verbatim).
                        protocol_for_memory = self._truncate_protocol_tool_results(
                            claude_protocol + [final_assistant]
                        )
                        # Stage the WHOLE truncated protocol as the on-disk
                        # cold-start seed (あさひ 07-24: replaying a tool turn
                        # without its tool machinery rewrites history into a
                        # "results without calls" shape — with the precedent
                        # effect this strong, that teaches fabrication).
                        # tool_result content is already truncated above with
                        # an explanatory marker; those are unsigned client
                        # messages, so truncation cannot trip validation.
                        if self._claude_protocol_has_thinking(protocol_for_memory):
                            self._last_thinking_seed = {
                                "model": getattr(self._llm, "model", "") or "",
                                "protocol": deepcopy(protocol_for_memory),
                                "thinking_tokens": turn_thinking_tokens,
                            }
                    # Oversized-thinking gate (あさひ 07-25): a turn whose
                    # billed thinking exceeds the cap is NOT carried into
                    # later requests at all — neither in-session nor as a
                    # seed — so a single reasoning spree can't squat in the
                    # cache prefix for the rest of the session.
                    replay_cap = (
                        getattr(self._llm, "thinking_replay_max_tokens", 0) or 0
                    )
                    if (
                        protocol_for_memory is not None
                        and replay_cap
                        and turn_thinking_tokens > replay_cap
                    ):
                        logger.info(
                            f"[thinking_replay] turn thinking "
                            f"{turn_thinking_tokens} tok > cap {replay_cap} — "
                            "transcript not carried into context."
                        )
                        protocol_for_memory = None
                        self._last_thinking_seed = None
                    self._add_message(
                        current_turn_text,
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
            plain_cap = getattr(self._llm, "thinking_replay_max_tokens", 0) or 0
            if plain_cap and plain_thinking_tokens > plain_cap:
                claude_assistant_message = None
            if complete_response:
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
        notes: List[str] = []
        if self._alarm_store is not None:
            notes.append(self._ALARM_CAPABILITY_NOTE)
        if self._uber_tools_active():
            notes.append(self._UBER_CAPABILITY_NOTE)
        if self._model_health_enabled:
            notes.append(self._MODEL_HEALTH_CAPABILITY_NOTE)
        if self._steam_enabled:
            notes.append(self._STEAM_CAPABILITY_NOTE)
        if self._memory_tools_active:
            notes.append(self._MEMORY_CAPABILITY_NOTE)
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
        """Short inline tag flagging that an MCP tool was used. Uber keeps its
        own glyph; any other MCP tool gets a generic 🔧 tag so EVERY tool call
        is visible in chat (あさひ audits usage from there when away)."""
        if tool_name.startswith("uber"):
            return "\n🍔 *Uber Eats*\n"
        if tool_name:
            return f"\n🔧 *ツール: {tool_name[:40]}*\n"
        return ""

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
        result = await web_fetch(url, max_chars=max_chars)
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
                        "incidents, verdict). Read-only; no LLM call."
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
                        "memory_update / memory_delete / memory_inject. Diary "
                        "hits carry a diary_uid, which memory_read_diary expands "
                        "to the full entry. Note: these results disappear at the "
                        "end of this turn — pin anything you want to keep "
                        "referring to with memory_inject. Meaning of importance: "
                        "user = curated by the user himself (resident every "
                        "session) / high = important (resident every session) / "
                        "low = enters context only when searched or recalled. To "
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
                                    "automatically."
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
                                "enum": ["high", "low"],
                                "description": (
                                    "high = important (resident in the system "
                                    "prompt from the next startup) / low = "
                                    "normal (recalled via search and RAG). "
                                    "Default: low."
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
                        "Rewrite an existing fact (correction or addition). "
                        "Confirm the target id with memory_search before using "
                        "this. importance is unchanged (content of user-tier "
                        "facts may still be corrected)."
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
                                "description": "The new text (replaces the whole fact).",
                            },
                        },
                        "required": ["fact_id", "new_fact"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_delete",
                    "description": (
                        "Delete one fact. Two-phase: calling it without "
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
                        "Read one diary entry in full. memory_search returns "
                        "diary hits at sentence granularity, so use this when "
                        "you need the surrounding context. The result lasts only "
                        "this turn — pin it with memory_inject if you want to "
                        "keep referring to it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diary_uid": {
                                "type": "string",
                                "description": "The diary_uid attached to a diary hit from memory_search.",
                            }
                        },
                        "required": ["diary_uid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_inject",
                    "description": (
                        "Pin the given facts / diary entries into the "
                        "conversation context. They stay visible from the next "
                        "user message onward, through the same mechanism as "
                        "automatic recall. Results from memory_search / "
                        "memory_read_diary vanish at the end of the turn, so "
                        "inject anything you want to keep using. Injected "
                        "content is not surfaced again by automatic recall."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "ids of facts to pin (max 10).",
                            },
                            "diary_uids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "diary_uids to pin (max 3; the full entry is included).",
                            },
                        },
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
                result = await web_search(
                    query,
                    provider=cfg.get("provider", "brave"),
                    api_key=cfg.get("api_key", ""),
                    max_results=5,
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
                result = await web_fetch(
                    url, max_chars=int(cfg.get("max_fetch_chars", 20000) or 20000)
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
                "content": json.dumps(result, ensure_ascii=False),
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
        "memory_inject",
    )
    # Per-operation marker text is built by _memory_marker (📝 *記憶◯◯: …*);
    # display/history only — stripped from the AI replay like all markers.
    # Approval phrases あさひ might actually type (JA/ZH/EN). Matched against
    # the TAIL of his latest real message — the second factor behind the
    # model's confirmed=true claim. Staged deletions expire after 15 min.
    _MEMORY_APPROVE_RE = re.compile(
        r"(同意|承認|許可|いいよ|いいです|ええよ|構わない|かまわない|どうぞ|"
        r"消していい|消しちゃっていい|削除していい|削除して構わない|オーケー|"
        r"删吧|删了吧|删掉|可以删|同意删除|はい|OK|ok|Ok)"
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
            force = bool(args.get("force", False))
            dup = None if force else await self._alarm_store.find_near(fire_at_utc)
            if dup is not None:
                # Near-duplicate: don't create. Hand the existing alarm back so the
                # model can reconsider this same turn and, if it still judges
                # another is needed, re-call with force=true.
                return "\n⏰ *Alarm set(重複スキップ)*\n", {
                    "status": "duplicate_nearby",
                    "message": (
                        f"近い時刻（{format_local(dup['fire_at_utc'])}）に"
                        f"既にアラームがある:「{dup.get('note', '')}」"
                        f"(id: {dup['id']})。同じ用件ならこれ以上設定しなくてよい。"
                        "別の用件で本当に必要だと自分で判断する場合のみ、"
                        "force=true を付けて set_alarm を呼び直すこと。"
                    ),
                    "existing": {
                        "id": dup["id"],
                        "at_local": format_local(dup["fire_at_utc"]),
                        "note": dup.get("note", ""),
                    },
                }
            record = await self._alarm_store.add(fire_at_utc=fire_at_utc, note=note)
            local = format_local(record["fire_at_utc"])
            return f"\n⏰ *Alarm set: {local}*\n", {
                "status": "ok",
                "message": f"アラームを {local} に設定しました。",
                "id": record["id"],
                "at_local": local,
                "note": note,
            }
        if name == "list_alarms":
            pending = await self._alarm_store.list_pending()
            return f"\n⏰ *Alarm list: {len(pending)}件*\n", {
                "status": "ok",
                "count": len(pending),
                "alarms": [
                    {
                        "id": a["id"],
                        "at_local": format_local(a["fire_at_utc"]),
                        "note": a.get("note", ""),
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
                )
            elif name == "memory_update":
                result = await mgr.update_fact_manual(
                    str(args.get("fact_id", "")).strip(),
                    str(args.get("new_fact", "")),
                )
            elif name == "memory_delete":
                result = await self._memory_delete_flow(args)
            elif name == "memory_read_diary":
                result = self._memory_read_diary_query(args)
            elif name == "memory_inject":
                result = self._memory_inject_query(args)
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
            kws = "、".join(
                str(k).strip() for k in (args.get("keywords") or []) if str(k).strip()
            )
            label = f"記憶検索(履歴): {_clip(kws)}"
        elif name == "memory_add":
            label = f"記憶追加: {_clip(args.get('fact'))}"
        elif name == "memory_update":
            label = f"記憶更新: {_clip(args.get('new_fact'))}"
        elif name == "memory_read_diary":
            label = f"日記閲覧: {_clip(result.get('date') or args.get('diary_uid'))}"
        elif name == "memory_inject":
            n = int(result.get("injected_facts", 0)) + int(
                result.get("injected_diaries", 0)
            )
            label = f"記憶注入: {n}件"
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
        keywords = [
            str(k).strip() for k in (args.get("keywords") or []) if str(k).strip()
        ]
        return await asyncio.to_thread(
            search_history,
            conf_uid,
            keywords,
            date_from=str(args.get("date_from", "") or "").strip(),
            date_to=str(args.get("date_to", "") or "").strip(),
        )

    def _memory_read_diary_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Full diary view — read-only, this-turn-only (pin via memory_inject)."""
        uid = str(args.get("diary_uid", "")).strip()
        if not uid:
            return {
                "status": "error",
                "message": "diary_uid が必要（memory_searchの日記ヒットに付いている）。",
            }
        entry = self._memory_manager.read_diary_full(uid)
        if not entry:
            return {"status": "error", "message": f"日記 {uid} が見つからない。"}
        return {
            "status": "ok",
            "diary_uid": uid,
            **entry,
            "note": "この結果はこのターン限り。以後も参照するなら memory_inject で固定。",
        }

    def _memory_inject_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Pin facts/diaries into the conversation context, RAG-style.

        Builds a block folded into the NEXT outgoing user message (persist-
        not-ephemeral, same seam as the RAG blocks) and marks the ids as
        session-injected so auto-RAG never re-surfaces them."""
        mgr = self._memory_manager
        fact_ids = [
            str(x).strip() for x in (args.get("fact_ids") or []) if str(x).strip()
        ][:10]
        diary_uids = [
            str(x).strip() for x in (args.get("diary_uids") or []) if str(x).strip()
        ][:3]
        if not fact_ids and not diary_uids:
            return {
                "status": "error",
                "message": "fact_ids か diary_uids のどちらかが必要。",
            }
        lines: List[str] = []
        injected_facts: List[str] = []
        injected_diaries: List[str] = []
        missing: List[str] = []
        for fid in fact_ids:
            f = mgr.find_fact(fid)
            if not f:
                missing.append(fid)
                continue
            date = str(f.get("updated", ""))[:10]
            lines.append(f"- [{date}] {f.get('fact', '')}")
            injected_facts.append(fid)
        for uid in diary_uids:
            e = mgr.read_diary_full(uid)
            if not e:
                missing.append(uid)
                continue
            lines.append(f"[日記 {e.get('date') or uid}]\n{e.get('content', '')}")
            injected_diaries.append(uid)
        if not lines:
            return {
                "status": "error",
                "message": "指定のidが1件も見つからない。",
                "missing": missing,
            }
        self._memory_pending_blocks.append(
            "【記憶（自分で呼び出して文脈に固定した分）】\n" + "\n".join(lines)
        )
        # Auto-RAG must not re-surface what is now pinned in context.
        self._session_injected_fact_ids.update(injected_facts)
        self._session_injected_uids.update(injected_diaries)
        result: Dict[str, Any] = {
            "status": "ok",
            "injected_facts": len(injected_facts),
            "injected_diaries": len(injected_diaries),
            "note": "次のユーザーメッセージから文脈に残り続ける。自動想起はこれらを重複表示しない。",
        }
        if missing:
            result["missing"] = missing
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
