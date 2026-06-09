from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
from datetime import datetime
from loguru import logger
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ...chat_history_manager import get_history, get_recent_histories
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
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
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

        # Tracks whether the current session's banner has already been
        # prepended in _memory. Set by set_memory_from_recent_histories
        # when the current session had pre-existing messages, OR by
        # _add_message when injecting it onto the first user message of
        # a freshly-started (empty-on-load) session.
        self._current_session_banner_added = False

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

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
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
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return

        self._memory.append(message_data)

    def set_memory_manager(self, manager) -> None:
        """Attach a PersistentMemoryManager for fact extraction and diary injection."""
        self._memory_manager = manager

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        """Format an ISO timestamp as '[YYYY-MM-DD HH:MM:SS Weekday]'."""
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        try:
            dt = datetime.fromisoformat(ts)
            return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[dt.weekday()]}]"
        except (ValueError, TypeError):
            return f"[{ts}]" if ts else ""

    @classmethod
    def _now_tag(cls) -> str:
        """Timestamp tag for messages happening right now."""
        return cls._format_timestamp(datetime.now().isoformat(timespec="seconds"))

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
        "現在のターンが直前のターンの「直後」だと自動的に仮定してはいけない。"
        "二つのターンの間に数時間・数日・数週間の空白があり得る。\n\n"
        "【時間に関する厳格なルール】\n\n"
        "時間・日付・経過・順序・「いつの話か」に少しでも関わる"
        "**あらゆる発言**を行う前に、必ず関連するタイムスタンプタグを参照すること。"
        "ユーザーの質問に答える時だけでなく、以下のすべての場合に適用される：\n"
        "- 自分から時刻・日付・経過時間・最近性に言及する時"
        "（「さっき」「先日」「今日は」「久しぶり」など）\n"
        "- 時刻に応じた挨拶をする時（おはよう・こんばんは等）\n"
        "- ユーザーに対して時間関連の質問・確認をする時"
        "（「今は何時頃？」「あれから〇日経った？」など）\n"
        "- 過去の出来事の時期や、二つの出来事の時間差を述べる時\n"
        "- 「現在」「最近」「以前」を基準とした推論をする時\n\n"
        "**タイムスタンプを見ずに時間関連の発言・質問を行うことは禁止する。** "
        "想像・推測・「直前の続き」と仮定して時間に言及することは許可されない。\n\n"
        "現在時刻が必要な場合は、"
        "**最新のユーザーメッセージのタイムスタンプを「現在」の基準とする**こと。\n\n"
        "【Web検索・Web取得について】\n\n"
        "あなたには2つのWebツールが備わっている可能性がある（環境設定による）：\n"
        "- **Web検索**（web_search）：キーワードで検索し、複数の結果を概要で得る\n"
        "- **Web取得**（web_fetch）：会話に既に出ているURLの全文を読む\n\n"
        "これらは情報源の拡張手段として、雑談の中でも積極的に使ってよい。"
        "次のような場面で自発的に使うことを推奨する：\n"
        "- ユーザーがURLを貼った時、または会話中に出てきたURLの内容が答えに必要な時"
        "→ web_fetch でその全文を読んでから答える\n"
        "- 最新の出来事・ニュース、変化する事実（価格・バージョン・天気・予定など）"
        "→ web_search で調べる\n"
        "- 雑談の中で新しい話題が出てきた時、関連する豆知識・最新情報・別角度を"
        "提供できそうなら web_search で調べて話題を広げてよい\n"
        "- あなたから新しい話題を持ち出す時、根拠や具体例を添えたいなら検索して構わない\n"
        "- あなたの知識が古い、または不確かで、推測で答えると間違える恐れがある時\n"
        "- ユーザーが明示的に調べるよう求めた時\n\n"
        "ただし、ごく短い相槌や挨拶、明らかに常識的な内容、純粋な意見・感想を"
        "述べる場面ではツール呼び出しは不要。"
        "不確かな事実を確認せず推測で断定するのは避けること——"
        "その場合は適切なツールで確認するか、「分からない」と正直に言うこと。\n\n"
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
        "Web検索で裏を取るか、率直に「分からない」と言えば、両立する。"
    )

    def _build_runtime_system(self) -> str:
        """Return the full system prompt as a plain string (used for non-Claude LLMs).

        Order matters: HISTORY_NOTE is appended last so it sits closest to
        the message history, giving the LLM the strictest instructions
        right before it encounters the data they apply to.
        """
        parts = [self._system, self._TIMESTAMP_NOTE]
        if self._memory_manager:
            mem_block = self._memory_manager.get_memory_prompt()
            if mem_block:
                parts.append(mem_block)
        parts.append(self._HISTORY_NOTE)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Prompt caching helpers (Claude only)
    # ------------------------------------------------------------------

    _CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}

    def _is_claude_llm(self) -> bool:
        return isinstance(self._llm, ClaudeAsyncLLM)

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
                "text": "\n\n".join([self._system, self._TIMESTAMP_NOTE]),
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
        """Mark the last message's last text block with cache_control.

        Returns a new list with the last message replaced; the original
        message objects (which live in self._memory) are not mutated.
        Only applies for Claude LLM — otherwise returns messages unchanged.
        """
        if not self._is_claude_llm() or not messages:
            return messages

        new_messages = list(messages)
        last = new_messages[-1]
        content = last.get("content")

        if isinstance(content, str):
            new_last = {
                **last,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": self._CACHE_CONTROL_1H,
                    }
                ],
            }
        elif isinstance(content, list) and content:
            new_content = [dict(c) for c in content]
            new_content[-1] = {
                **new_content[-1],
                "cache_control": self._CACHE_CONTROL_1H,
            }
            new_last = {**last, "content": new_content}
        else:
            return new_messages

        new_messages[-1] = new_last
        return new_messages

    @staticmethod
    def _session_header_text(uid: str, is_current: bool = False) -> str:
        """Format a session-boundary banner from a history UID.

        UID format: ``YYYY-MM-DD_HH-MM-SS_<hex>``. The banner is prepended
        to the first message of each session so the LLM can distinguish
        independent sessions in the otherwise-flat message stream.
        """
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        label = "現在進行中のセッション" if is_current else "セッション"
        parts = uid.split("_")
        if len(parts) >= 2 and len(parts[0]) == 10 and len(parts[1]) == 8:
            try:
                dt = datetime.strptime(
                    f"{parts[0]}_{parts[1]}", "%Y-%m-%d_%H-%M-%S"
                )
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
        if role == "user":
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
        self._memory = []
        # Reset banner state; will be set True below if the current
        # session already has messages here, or later by _add_message
        # when the first user message of a fresh session comes in.
        self._current_session_banner_added = False
        loaded_uids = []
        for uid, messages in sessions:
            loaded_uids.append(uid)
            first_in_session = True
            for msg in messages:
                entry = self._msg_from_history_record(msg)
                if not entry:
                    continue
                if first_in_session:
                    # Prepend a session-boundary banner so the LLM can tell
                    # where one past session ends and the next begins.
                    banner = self._session_header_text(uid, is_current=False)
                    entry["content"] = f"{banner}\n{entry['content']}"
                    first_in_session = False
                self._memory.append(entry)

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
                first_in_session = True
                for msg in current_messages:
                    entry = self._msg_from_history_record(msg)
                    if not entry:
                        continue
                    if first_in_session:
                        banner = self._session_header_text(
                            current_uid, is_current=True
                        )
                        entry["content"] = f"{banner}\n{entry['content']}"
                        first_in_session = False
                        self._current_session_banner_added = True
                    self._memory.append(entry)
            loaded_uids.append(current_uid)

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
        # Cache breakpoint goes on the last historical message — everything
        # up to and including it gets cached by Anthropic, while the fresh
        # user input appended below stays uncached. No-op for non-Claude.
        messages = self._attach_cache_breakpoint(messages)
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

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
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

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

        while True:
            stream = self._llm.chat_completion(messages, self._build_system_for_llm(), tools=tools)
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

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
                        f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
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
                # elif event["type"] == "message_delta":
                #     if event["data"]["delta"].get("stop_reason"):
                #         stop_reason = event["data"]["delta"].get("stop_reason")
                elif event["type"] == "message_stop":
                    break
                elif event["type"] == "error":
                    logger.error(f"LLM API Error: {event['message']}")
                    yield f"[Error from LLM: {event['message']}]"
                    return

            if pending_tool_calls:
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
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
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []

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

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
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

            messages = self._to_messages(input_data)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._formatted_tools_claude
                    llm_supports_native_tools = True
                elif isinstance(self._llm, OpenAICompatibleAsyncLLM):
                    tool_mode = "OpenAI"
                    tools = self._formatted_tools_openai
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else []
                ):
                    yield output
                return
            else:
                logger.info("Starting simple chat completion.")
                token_stream = self._llm.chat_completion(
                    messages, self._build_system_for_llm()
                )
                complete_response = ""
                async for event in token_stream:
                    text_chunk = ""
                    if isinstance(event, dict) and event.get("type") == "text_delta":
                        text_chunk = event.get("text", "")
                    elif (
                        isinstance(event, dict)
                        and event.get("type") == "web_search_marker"
                    ):
                        # Inline "searched the web" indicator. Streamed for
                        # display (Discord + web stay in sync) but deliberately
                        # NOT added to complete_response, so it isn't saved to
                        # history and the model can't learn to fake the marker.
                        marker = event.get("text", "")
                        if marker:
                            yield marker
                        continue
                    elif isinstance(event, str):
                        text_chunk = event
                    else:
                        continue
                    if text_chunk:
                        yield text_chunk
                        complete_response += text_chunk
                if complete_response:
                    self._add_message(complete_response, "assistant")

        return chat_with_memory

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
