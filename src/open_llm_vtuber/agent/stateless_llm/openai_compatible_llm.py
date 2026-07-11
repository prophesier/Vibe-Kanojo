"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

from typing import AsyncIterator, List, Dict, Any
import logging
import httpx
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    DefaultAsyncHttpxClient,
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface
from ...mcpp.types import ToolCallObject


# Seam the agent inserts into the runtime system prompt between the static
# persona block and the facts/diaries block, ONLY when explicit caching is
# active. The responses transport splits on it and puts an explicit cache
# breakpoint at the static block's end; it never reaches the model.
CACHE_SEAM_MARKER = "<<__CACHE_SEAM__>>"


class _OpenAIConnBridge(logging.Handler):
    """Surface new-TCP-connection events to the OpenAI API as [conn] INFO
    lines. Cache routing is per-connection (see the pinning note in
    AsyncLLM.__init__), so a [conn] line right before a [cache] read=0 on
    an image turn means the pinned connection was dropped (server-side
    idle/age/request-count limit) and the request got rerouted — the
    diagnostic that separates "pin broke" from "pin doesn't work"."""

    def emit(self, record):
        try:
            msg = record.getMessage()
            if "connect_tcp.started" not in msg or "openai" not in msg:
                return
            marker = "host='"
            i = msg.find(marker)
            host = msg[i + len(marker) :].split("'", 1)[0] if i >= 0 else "?"
            logger.info(f"[conn] new TCP connection → {host}")
        except Exception:
            pass


# Child logger level beats the WARNING clamp the Steam client puts on the
# "httpcore" parent; connect events carry no URLs/keys (host only).
_conn_logger = logging.getLogger("httpcore.connection")
_conn_logger.setLevel(logging.DEBUG)
if not any(isinstance(h, _OpenAIConnBridge) for h in _conn_logger.handlers):
    _conn_logger.addHandler(_OpenAIConnBridge())


class AsyncLLM(StatelessLLMInterface):
    _MAX_COMPLETION_TOKEN_MULTIPLIER = 4

    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        reasoning_effort: str = "",
        api_mode: str = "chat",
        cache_mode: str = "implicit",
    ):
        """
        Initializes an instance of the `AsyncLLM` class.

        Parameters:
        - model (str): The model to be used for language generation.
        - base_url (str): The base URL for the OpenAI API.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to "z".
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to "z".
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to "z".
        - temperature (float, optional): What sampling temperature to use, between 0 and 2. Defaults to 1.0.
        - reasoning_effort (str, optional): Default reasoning effort
          (none/low/medium/high; responses mode also xhigh/max). Empty =
          don't send (provider default: gpt-5.5/5.6 = medium, gpt-5.1 =
          none). Per-call callers (memory tasks) still override by passing
          the argument explicitly.
        - api_mode (str, optional): "chat" (default) = /v1/chat/completions;
          "responses" = /v1/responses (stateless, store=false). Responses
          mode is what allows function tools + reasoning to coexist on
          gpt-5.6 (mutually exclusive on the chat endpoint). Flip back to
          "chat" for instant rollback; a 404 in responses mode means the
          endpoint doesn't implement /v1/responses (no auto-fallback —
          fix the config).
        - cache_mode (str, optional): "implicit" (default) or "explicit"
          prompt caching, responses mode only. Explicit places up to 4
          breakpoints (static persona | history/current-session seam |
          previous user msg | current user msg) and makes image turns
          cache-immune (implicit deterministically misses on them —
          verified probes 5–7). Requests whose system prompt carries no
          CACHE_SEAM_MARKER (e.g. memory tasks) stay implicit.
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self._reasoning_effort = (reasoning_effort or "").strip()
        mode = (api_mode or "chat").strip().lower()
        if mode not in ("chat", "responses"):
            logger.warning(f"Unknown api_mode {api_mode!r}; falling back to 'chat'.")
            mode = "chat"
        self._api_mode = mode
        cmode = (cache_mode or "implicit").strip().lower()
        if cmode not in ("implicit", "explicit"):
            logger.warning(
                f"Unknown cache_mode {cache_mode!r}; falling back to 'implicit'."
            )
            cmode = "implicit"
        self._cache_mode = cmode
        if cmode == "explicit" and mode == "responses":
            logger.info("Explicit prompt caching active (ttl 30m, 4 breakpoints).")
        # Verbatim output items (reasoning/message/function_call) of recent
        # responses-mode tool-call rounds, keyed by call_id. Replayed on the
        # next hop of the SAME turn so encrypted reasoning survives the tool
        # loop (docs: "keep every output item"). Tool transcripts never enter
        # persistent memory, so entries die naturally with the turn; the dict
        # is size-capped as a leak guard.
        self._responses_replay: Dict[str, List[Dict[str, Any]]] = {}
        # Text content of user messages that were SENT with an image
        # attached (explicit cache mode). Their breakpoint entries may live
        # on a vision-pool machine and be unreadable from later text turns,
        # so the rotating "previous" breakpoint skips them. Ordered,
        # size-capped; texts embed a timestamp tag so they're unique enough.
        self._responses_image_texts: Dict[str, None] = {}
        # Stable per-character routing hint for OpenAI's prompt cache. Empty
        # until set_prompt_cache_key is called (e.g. with the conf_uid).
        self._prompt_cache_key: str = ""
        self._include_usage_supported = True
        self._completion_token_param = (
            "max_completion_tokens"
            if self._uses_official_openai_endpoint(base_url)
            else "max_tokens"
        )
        # Keep ONE TCP connection alive indefinitely instead of httpx's
        # 5-second keepalive default. OpenAI's LB pins a connection to one
        # backend, and the prompt cache is per-machine — with a fresh
        # connection per turn, image-carrying requests get routed to a
        # separate vision pool and miss the whole cache (probe15: 3/3
        # deterministic). A surviving connection keeps text AND image turns
        # on the same machine (probe16: image turn read the full text-written
        # cache after 150s and 400s idle, no heartbeat needed). Exactly one
        # keepalive connection: two idle connections could round-robin turns
        # across two backends and split the cache. If the server closes an
        # idle connection, the SDK reconnects transparently and the next
        # turn re-pins — self-healing, worst case is one image-turn miss.
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=1,
                    keepalive_expiry=None,
                ),
            ),
        )
        self.support_tools = True

        logger.info(
            f"Initialized AsyncLLM with the parameters: {self.base_url}, {self.model}"
        )

    def set_prompt_cache_key(self, key: str) -> None:
        """Set a stable OpenAI prompt_cache_key (e.g. the conf_uid) so this
        character's requests route to the same cache machine across turns."""
        self._prompt_cache_key = key or ""

    @staticmethod
    def _uses_official_openai_endpoint(base_url: str) -> bool:
        return bool(base_url and "api.openai.com" in base_url.lower())

    @staticmethod
    def _get_usage_value(obj: Any, key: str, default: Any = 0) -> Any:
        """Read a usage field from either OpenAI SDK models or plain dicts."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _log_openai_cache_usage(cls, usage: Any) -> None:
        """Log OpenAI prompt-cache usage from a stream usage chunk.

        ``cache_write_tokens`` exists from gpt-5.6 (explicit caching era);
        absent on older models → reported as 0, same as before."""
        prompt_tokens = cls._get_usage_value(usage, "prompt_tokens", 0) or 0
        completion_tokens = cls._get_usage_value(usage, "completion_tokens", 0) or 0
        prompt_details = cls._get_usage_value(usage, "prompt_tokens_details", None)
        cached = cls._get_usage_value(prompt_details, "cached_tokens", 0) or 0
        cache_write = cls._get_usage_value(prompt_details, "cache_write_tokens", 0) or 0
        completion_details = cls._get_usage_value(
            usage, "completion_tokens_details", None
        )
        reasoning = cls._get_usage_value(completion_details, "reasoning_tokens", 0) or 0

        fresh = max(prompt_tokens - cached, 0)
        hit_pct = (cached / prompt_tokens * 100) if prompt_tokens else 0
        logger.info(
            f"[cache] read={cached} write={cache_write} fresh={fresh} "
            f"(hit {hit_pct:.0f}%, input={prompt_tokens}, "
            f"output={completion_tokens}, reasoning={reasoning})"
        )

    @staticmethod
    def _is_usage_stream_option_unsupported(error: APIError) -> bool:
        message = str(error).lower()
        return (
            "stream_options" in message
            or "stream option" in message
            or "include_usage" in message
        )

    @staticmethod
    def _is_max_tokens_unsupported(error: APIError) -> bool:
        message = str(error).lower()
        return "max_tokens" in message and "max_completion_tokens" in message

    @staticmethod
    def _is_reasoning_effort_unsupported(error: APIError) -> bool:
        message = str(error).lower()
        return "reasoning_effort" in message or "reasoning effort" in message

    @staticmethod
    def _summarize_messages(messages: List[Dict[str, Any]]) -> str:
        """Summarize request messages for error logs without dumping chat text."""
        total_chars = 0
        roles = {}
        for message in messages:
            role = message.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or ""
                        if isinstance(text, str):
                            total_chars += len(text)
                    elif isinstance(part, str):
                        total_chars += len(part)
            elif content is not None:
                total_chars += len(str(content))

        role_summary = ", ".join(f"{role}={count}" for role, count in roles.items())
        return (
            f"{len(messages)} message(s), roles: {role_summary or 'none'}, "
            f"approx_content_chars={total_chars}"
        )

    def _completion_token_limit(self, max_tokens: int) -> int:
        """Map visible-output budget to OpenAI's total completion budget.

        Newer OpenAI reasoning models count hidden reasoning tokens against
        max_completion_tokens. Memory tasks pass max_tokens as the desired
        visible JSON headroom, so give those models extra room for reasoning
        while preserving exact max_tokens behavior for compatible endpoints.
        """
        if self._completion_token_param != "max_completion_tokens":
            return max_tokens
        return max_tokens * self._MAX_COMPLETION_TOKEN_MULTIPLIER

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        max_tokens: int = None,
        disable_server_tools: bool = False,
        reasoning_effort: str = None,
    ) -> AsyncIterator[str | List[ChoiceDeltaToolCall]]:
        """
        Generates a chat completion using the OpenAI API asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.
        - tools (List[Dict[str, str]], optional): List of tools to use for this completion.
        - max_tokens (int, optional): Cap on generated tokens. Memory tasks
          (diary/fact extraction) pass a large value so long JSON/diary
          output isn't truncated; chat leaves it None (provider default).
        - disable_server_tools (bool): Accepted for signature parity with the
          Claude LLM. OpenAI-compatible endpoints don't auto-inject server
          tools, so this is a no-op here — present only so callers like
          PersistentMemoryManager._call_llm can pass it uniformly.

        Yields:
        - str: The content of each chunk from the API response.
        - List[ChoiceDeltaToolCall]: The tool calls detected in the response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        if self._api_mode == "responses":
            async for event in self._responses_completion(
                messages,
                system=system,
                tools=tools,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ):
                yield event
            return

        stream = None
        # Tool call related state variables
        accumulated_tool_calls = {}
        in_tool_call = False
        emitted_text_chars = 0
        last_finish_reason = None

        try:
            # If system prompt is provided, add it to the messages
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]
            logger.debug(f"Messages: {messages_with_system}")

            available_tools = tools if self.support_tools else NOT_GIVEN

            request_kwargs = {
                "messages": messages_with_system,
                "model": self.model,
                "stream": True,
                "temperature": self.temperature,
                "tools": available_tools,
            }
            if max_tokens:
                request_kwargs[self._completion_token_param] = (
                    self._completion_token_limit(max_tokens)
                )
            if self._include_usage_supported:
                request_kwargs["stream_options"] = {"include_usage": True}
            # Reasoning effort. Per-call value wins (memory tasks pass their
            # own knob; "" there means "don't send"); None falls back to the
            # instance default from config (chat path never passes it). Sent
            # only when non-empty; dropped on the retry below if the
            # model/endpoint rejects it.
            if reasoning_effort is None:
                reasoning_effort = self._reasoning_effort
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = reasoning_effort
            # Pin cache routing. OpenAI keys its prompt cache per-machine; without
            # a stable prompt_cache_key, consecutive turns can scatter across
            # machines and miss an otherwise-valid 24h cache. The key is set
            # explicitly per character (see set_prompt_cache_key) rather than
            # derived from the system prompt: the system embeds facts/diaries that
            # the startup backfill rewrites mid-session, so hashing it would flip
            # the key and route the turn to a cold machine, losing even the
            # unchanged persona prefix. Sent via extra_body so it's
            # SDK-version-agnostic / ignored by endpoints that don't support it.
            if self._prompt_cache_key:
                request_kwargs["extra_body"] = {
                    "prompt_cache_key": self._prompt_cache_key
                }

            while True:
                try:
                    stream: AsyncStream[
                        ChatCompletionChunk
                    ] = await self.client.chat.completions.create(**request_kwargs)
                    break
                except APIError as e:
                    if (
                        self._include_usage_supported
                        and self._is_usage_stream_option_unsupported(e)
                    ):
                        self._include_usage_supported = False
                        logger.warning(
                            "OpenAI-compatible endpoint does not support "
                            "stream_options.include_usage; retrying without cache "
                            "usage logging."
                        )
                        request_kwargs.pop("stream_options", None)
                        continue
                    if self._is_reasoning_effort_unsupported(e):
                        # gpt-5.6 on /v1/chat/completions: function tools +
                        # reasoning are mutually exclusive ("use /v1/responses
                        # or set reasoning_effort to 'none'"). The default is
                        # medium, so DROPPING the param doesn't help — the
                        # error tells us to send an explicit 'none'.
                        if (
                            "'none'" in str(e)
                            and request_kwargs.get("reasoning_effort") != "none"
                        ):
                            request_kwargs["reasoning_effort"] = "none"
                            logger.warning(
                                "Endpoint requires reasoning_effort='none' with "
                                "function tools (chat/completions limitation); "
                                "retrying with none. For reasoning + tools this "
                                "model needs the /v1/responses API."
                            )
                            continue
                        if "reasoning_effort" in request_kwargs:
                            request_kwargs.pop("reasoning_effort", None)
                            logger.warning(
                                "Endpoint rejected reasoning_effort; retrying "
                                "without it."
                            )
                            continue
                    if (
                        max_tokens
                        and self._completion_token_param == "max_tokens"
                        and self._is_max_tokens_unsupported(e)
                    ):
                        self._completion_token_param = "max_completion_tokens"
                        logger.warning(
                            "OpenAI endpoint rejected max_tokens; retrying with "
                            "max_completion_tokens for this and future requests."
                        )
                        request_kwargs.pop("max_tokens", None)
                        request_kwargs["max_completion_tokens"] = (
                            self._completion_token_limit(max_tokens)
                        )
                        continue
                    raise
            logger.debug(
                f"Tool Support: {self.support_tools}, Available tools: {available_tools}"
            )

            served_model_logged = False
            async for chunk in stream:
                if not served_model_logged and getattr(chunk, "model", None):
                    logger.info(
                        f"[llm] requested={self.model!r} served={chunk.model!r}"
                    )
                    served_model_logged = True
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._log_openai_cache_usage(usage)

                # Guard against chunks with missing choices field (e.g., from OpenWebUI)
                if not chunk.choices:
                    continue

                finish_reason = getattr(chunk.choices[0], "finish_reason", None)
                if finish_reason:
                    last_finish_reason = finish_reason

                if self.support_tools:
                    has_tool_calls = (
                        hasattr(chunk.choices[0].delta, "tool_calls")
                        and chunk.choices[0].delta.tool_calls
                    )

                    if has_tool_calls:
                        logger.debug(
                            f"Tool calls detected in chunk: {chunk.choices[0].delta.tool_calls}"
                        )
                        in_tool_call = True
                        # Process tool calls in the current chunk
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            index = (
                                tool_call.index if hasattr(tool_call, "index") else 0
                            )

                            # Initialize tool call for this index if needed
                            if index not in accumulated_tool_calls:
                                accumulated_tool_calls[index] = {
                                    "index": index,
                                    "id": getattr(tool_call, "id", None),
                                    "type": getattr(tool_call, "type", None),
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Update tool call information
                            if hasattr(tool_call, "id") and tool_call.id:
                                accumulated_tool_calls[index]["id"] = tool_call.id
                            if hasattr(tool_call, "type") and tool_call.type:
                                accumulated_tool_calls[index]["type"] = tool_call.type

                            # Update function information
                            if hasattr(tool_call, "function"):
                                if (
                                    hasattr(tool_call.function, "name")
                                    and tool_call.function.name
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "name"
                                    ] = tool_call.function.name
                                if (
                                    hasattr(tool_call.function, "arguments")
                                    and tool_call.function.arguments
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "arguments"
                                    ] += tool_call.function.arguments

                        continue

                    # If we were in a tool call but now we're not, yield the tool call result
                    elif in_tool_call and not has_tool_calls:
                        in_tool_call = False
                        # Convert accumulated tool calls to the required format and output
                        logger.info(f"Complete tool calls: {accumulated_tool_calls}")

                        # Use the from_dict method to create a ToolCallObject instance from a dictionary
                        complete_tool_calls = [
                            ToolCallObject.from_dict(tool_data)
                            for tool_data in accumulated_tool_calls.values()
                        ]

                        yield complete_tool_calls
                        accumulated_tool_calls = {}  # Reset for potential future tool calls

                # Process regular content chunks
                if len(chunk.choices) == 0:
                    logger.info("Empty chunk received")
                    continue
                elif chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                content = chunk.choices[0].delta.content
                emitted_text_chars += len(content)
                yield content

            # If stream ends while still in a tool call, make sure to yield the tool call
            if in_tool_call and accumulated_tool_calls:
                logger.info(f"Final tool call at stream end: {accumulated_tool_calls}")

                # Create a ToolCallObject instance from a dictionary using the from_dict method.
                complete_tool_calls = [
                    ToolCallObject.from_dict(tool_data)
                    for tool_data in accumulated_tool_calls.values()
                ]

                yield complete_tool_calls

            if last_finish_reason == "length":
                logger.warning(
                    "OpenAI stream stopped because it reached the completion token "
                    f"limit; emitted_text_chars={emitted_text_chars}. "
                    "If this happens during memory tasks, increase their "
                    "max_tokens budget."
                )

        except APIConnectionError as e:
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            # OpenAI returns 429 for BOTH real rate-limiting (rate_limit_exceeded:
            # too many requests/tokens per minute) AND exhausted billing/quota
            # (insufficient_quota: out of credit / monthly budget cap). str(e)
            # carries the message + code, so log it to tell the two apart.
            detail = f"{getattr(e, 'code', '') or ''} {e}"
            logger.error(
                f"Error calling the chat endpoint: 429 Too Many Requests — {e}"
            )
            if "insufficient_quota" in detail:
                yield (
                    "Error calling the chat endpoint: OpenAI quota/credit exhausted "
                    "(insufficient_quota). Check your billing. See the logs for details."
                )
            else:
                yield (
                    "Error calling the chat endpoint: Rate limit exceeded. "
                    "Please try again later. See the logs for details."
                )

        except APIError as e:
            if "does not support tools" in str(e):
                self.support_tools = False
                logger.warning(
                    f"{self.model} does not support tools. Disabling tool support."
                )
                yield "__API_NOT_SUPPORT_TOOLS__"
                return
            logger.error(f"LLM API: Error occurred: {e}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Messages: {self._summarize_messages(messages)}")
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            # make sure the stream is properly closed
            # so when interrupted, no more tokens will being generated.
            if stream:
                logger.debug("Chat completion finished.")
                await stream.close()
                logger.debug("Stream closed.")

    # ------------------------------------------------------------------
    # /v1/responses transport (api_mode: "responses")
    #
    # Same contract as the chat path: yields str text deltas, then one
    # List[ToolCallObject] when the model requests tools. Everything above
    # this layer (agent tool loops, memory tasks) is unchanged.
    # ------------------------------------------------------------------

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Flatten a chat-format content field (str or parts list) to text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    text = p.get("text") or p.get("content") or ""
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(p, str):
                    parts.append(p)
            return "".join(parts)
        return "" if content is None else str(content)

    @staticmethod
    def _to_responses_tools(tools: Any) -> List[Dict[str, Any]] | None:
        """Chat-format nested function tools → responses flat format."""
        if not tools or isinstance(tools, NotGiven):
            return None
        flat = []
        for t in tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if fn:
                flat.append(
                    {
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )
            elif isinstance(t, dict):
                flat.append(t)  # already flat / non-function tool: pass through
        return flat or None

    _EXPLICIT_BP = {"mode": "explicit"}

    def _to_responses_input(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        explicit: bool = False,
    ) -> List[Dict[str, Any]]:
        """Chat-format message history → responses input items.

        The system prompt becomes input[0] as a message item (NOT the
        top-level `instructions` param: instructions is a plain string, so
        explicit cache breakpoints — which attach to content blocks — could
        never be placed on it).

        Assistant tool-call messages are replaced by the verbatim output
        items stashed from the previous hop when available (preserves
        encrypted reasoning); otherwise reconstructed as function_call items.

        With ``explicit``, up to 4 breakpoints are placed (entries are keyed
        to EXACT breakpoint positions — no longest-prefix matching, so the
        previous turn's position must be kept alive each turn):
          1. static persona block (system split on CACHE_SEAM_MARKER)
          2. the "_cache_seam"-tagged history message (history | current
             session boundary — survives a pure restart)
          3. previous user message  4. current user message (rotating pair
             → read the full previous-turn entry, write only ~the new turn)
        Breakpoints go on the last input_text block, never on images, so an
        image turn's entry stays byte-identical to what the next turn (which
        never re-sends the image) replays.
        """
        items: List[Dict[str, Any]] = []
        if system:
            if explicit and CACHE_SEAM_MARKER in system:
                static_part, _, memory_part = system.partition(CACHE_SEAM_MARKER)
                content = [
                    {
                        "type": "input_text",
                        "text": static_part,
                        "prompt_cache_breakpoint": self._EXPLICIT_BP,
                    }
                ]
                if memory_part.strip():
                    content.append({"type": "input_text", "text": memory_part})
                items.append({"role": "system", "content": content})
            else:
                # Marker never reaches the model in any mode.
                items.append(
                    {
                        "role": "system",
                        "content": system.replace(CACHE_SEAM_MARKER, ""),
                    }
                )
        # Rotating breakpoints: previous + current user message. The current
        # one is stable across the hops of a tool loop (tool items are
        # appended after it), so hop N reads hop 1's entry.
        #
        # The PREVIOUS slot skips user messages that rode with an image:
        # observed in production (2026-07-11 16:55) that an image-carrying
        # request can miss ALL breakpoint entries despite a byte-identical
        # prefix ([sys_fp] unchanged) and re-write everything — most
        # consistent with vision requests being served/cached on a different
        # machine pool (OpenAI-side; in-vitro probes with production-shaped
        # requests do NOT reproduce it). Entries written by such a turn may
        # be unreadable later, so anchoring the previous slot there would
        # drop the whole current-session prefix; anchor on the last
        # image-free user message instead.
        bp_indices: set = set()
        if explicit:
            user_indices = [
                i for i, m in enumerate(messages) if m.get("role") == "user"
            ]
            if user_indices:
                bp_indices.add(user_indices[-1])
                for i in reversed(user_indices[:-1]):
                    text = self._content_to_text(messages[i].get("content"))
                    if text not in self._responses_image_texts:
                        bp_indices.add(i)
                        break
            bp_indices.update(i for i, m in enumerate(messages) if m.get("_cache_seam"))
        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                first_id = (m["tool_calls"][0] or {}).get("id")
                stashed = self._responses_replay.get(first_id or "")
                if stashed:
                    items.extend(stashed)
                else:
                    text = self._content_to_text(m.get("content"))
                    if text:
                        items.append({"role": "assistant", "content": text})
                    for tc in m["tool_calls"]:
                        fn = tc.get("function") or {}
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.get("id"),
                                "name": fn.get("name"),
                                "arguments": fn.get("arguments") or "{}",
                            }
                        )
                continue
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.get("tool_call_id"),
                        "output": self._content_to_text(m.get("content")),
                    }
                )
                continue
            content = m.get("content")
            if content is None:
                continue
            # Breakpoints attach only to input_* blocks — assistant content
            # is output_text, so an assistant message can't carry one (the
            # agent only tags non-assistant messages).
            want_bp = i in bp_indices and role != "assistant"
            if isinstance(content, str):
                if want_bp:
                    items.append(
                        {
                            "role": role,
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": content,
                                    "prompt_cache_breakpoint": self._EXPLICIT_BP,
                                }
                            ],
                        }
                    )
                else:
                    items.append({"role": role, "content": content})
                continue
            # Multimodal parts list (user images): convert to responses types.
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    parts.append({"type": "input_text", "text": p.get("text", "")})
                elif p.get("type") == "image_url":
                    img = p.get("image_url") or {}
                    part = {"type": "input_image", "image_url": img.get("url", "")}
                    if img.get("detail"):
                        part["detail"] = img["detail"]
                    parts.append(part)
            if parts:
                if explicit and any(p.get("type") == "input_image" for p in parts):
                    # Remember this turn carried an image so later turns
                    # don't anchor their "previous" breakpoint on it.
                    text = "".join(
                        p.get("text", "")
                        for p in parts
                        if p.get("type") == "input_text"
                    )
                    self._responses_image_texts[text] = None
                    while len(self._responses_image_texts) > 32:
                        self._responses_image_texts.pop(
                            next(iter(self._responses_image_texts))
                        )
                if want_bp:
                    # Last TEXT block, never an image: keeps the entry equal
                    # to the image-free version the next turn replays.
                    for p in reversed(parts):
                        if p.get("type") == "input_text":
                            p["prompt_cache_breakpoint"] = self._EXPLICIT_BP
                            break
                items.append({"role": role, "content": parts})
        return items

    def _stash_replay(
        self, call_ids: List[str], raw_items: List[Dict[str, Any]]
    ) -> None:
        """Remember this hop's verbatim output items under each call_id so
        the next hop of the tool loop can replay them (encrypted reasoning
        included). Size-capped FIFO; stale entries are simply never looked
        up again once the turn's transcript is discarded."""
        for cid in call_ids:
            if cid:
                self._responses_replay[cid] = raw_items
        while len(self._responses_replay) > 128:
            self._responses_replay.pop(next(iter(self._responses_replay)))

    def _log_responses_usage(self, usage: Any) -> None:
        """Normalize responses-API usage field names onto the shared
        [cache] log line (input_tokens ↔ prompt_tokens etc.)."""
        in_details = self._get_usage_value(usage, "input_tokens_details", None)
        out_details = self._get_usage_value(usage, "output_tokens_details", None)
        self._log_openai_cache_usage(
            {
                "prompt_tokens": self._get_usage_value(usage, "input_tokens", 0),
                "completion_tokens": self._get_usage_value(usage, "output_tokens", 0),
                "prompt_tokens_details": {
                    "cached_tokens": self._get_usage_value(
                        in_details, "cached_tokens", 0
                    ),
                    "cache_write_tokens": self._get_usage_value(
                        in_details, "cache_write_tokens", 0
                    ),
                },
                "completion_tokens_details": {
                    "reasoning_tokens": self._get_usage_value(
                        out_details, "reasoning_tokens", 0
                    ),
                },
            }
        )

    def _log_pool_state(self) -> None:
        """One [conn] line describing THIS client's connection pool right
        before a request. The module-level bridge can't tell whose
        connection a connect event belongs to (embeddings / RAG judge also
        hit api.openai.com with default 5s-keepalive pools and reconnect
        every turn by design) — this line disambiguates: the same
        connection accumulating "Request Count" across turns = the pin
        holds; an empty pool each turn = the chat connection is being
        dropped and the image-routing fix is not effective."""
        try:
            pool = self.client._client._transport._pool
            infos = [c.info() for c in pool.connections]
            logger.info(f"[conn] pool({self.model}): {infos or 'empty'}")
        except Exception:
            pass

    async def _responses_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = None,
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        max_tokens: int = None,
        reasoning_effort: str = None,
    ) -> AsyncIterator[str | List[ToolCallObject]]:
        """Streaming completion over /v1/responses (stateless, store=false).

        Yield contract is identical to the chat path. Verified firsthand on
        gpt-5.6-sol (experiments/_responses_probe*.py): function tools +
        reasoning coexist here, encrypted reasoning items replay across tool
        hops, and both implicit cache (prompt_cache_key) and explicit
        breakpoints work with store=false.
        """
        stream = None
        try:
            if reasoning_effort is None:
                reasoning_effort = self._reasoning_effort

            # Explicit caching only for requests whose system prompt carries
            # the seam marker (= the chat path). Memory tasks and other
            # one-shot callers stay implicit — explicit writes bill 1.25×
            # and their prompts are never re-read.
            explicit_active = self._cache_mode == "explicit" and bool(
                system and CACHE_SEAM_MARKER in system
            )
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": self._to_responses_input(
                    messages, system, explicit=explicit_active
                ),
                "stream": True,
                "temperature": self.temperature,
                "store": False,
            }
            # extra_body carries fields the installed SDK has no typed param
            # for yet — version-agnostic, exactly like the chat path.
            extra_body: Dict[str, Any] = {}
            if explicit_active:
                extra_body["prompt_cache_options"] = {
                    "mode": "explicit",
                    "ttl": "30m",
                }
            flat_tools = self._to_responses_tools(tools if self.support_tools else None)
            if flat_tools:
                request_kwargs["tools"] = flat_tools
            if reasoning_effort:
                # summary:"auto" = the model's own reasoning digest, logged as
                # [thinking] below (raw CoT is never exposed on this API).
                request_kwargs["reasoning"] = {
                    "effort": reasoning_effort,
                    "summary": "auto",
                }
                # Without this, reasoning items come back without content and
                # the chain can't be replayed across tool hops (store=false).
                request_kwargs["include"] = ["reasoning.encrypted_content"]
            if max_tokens:
                # Reasoning tokens count against max_output_tokens, same as
                # max_completion_tokens on the chat path — give headroom.
                request_kwargs["max_output_tokens"] = (
                    max_tokens * self._MAX_COMPLETION_TOKEN_MULTIPLIER
                )
            if self._prompt_cache_key:
                extra_body["prompt_cache_key"] = self._prompt_cache_key
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            self._log_pool_state()
            while True:
                try:
                    stream = await self.client.responses.create(**request_kwargs)
                    break
                except APIError as e:
                    msg = str(e).lower()
                    # Degrade gracefully on partially-compatible providers
                    # (Ollama/Groq/vLLM implement the stateless subset with
                    # varying extras). Each drop is logged, never silent.
                    # Summary first: its error message also contains
                    # "reasoning", which would wrongly drop the whole knob.
                    if "summary" in msg and "summary" in (
                        request_kwargs.get("reasoning") or {}
                    ):
                        request_kwargs["reasoning"].pop("summary", None)
                        logger.warning(
                            "/v1/responses rejected reasoning summary (often an "
                            "unverified-org restriction); retrying without it — "
                            "[thinking] log lines unavailable."
                        )
                        continue
                    if (
                        "prompt_cache_options" in msg or "breakpoint" in msg
                    ) and "prompt_cache_options" in extra_body:
                        # Provider without explicit caching: strip options AND
                        # the per-block breakpoints, fall back to implicit.
                        extra_body.pop("prompt_cache_options", None)
                        if not extra_body:
                            request_kwargs.pop("extra_body", None)
                        request_kwargs["input"] = self._to_responses_input(
                            messages, system, explicit=False
                        )
                        logger.warning(
                            "/v1/responses rejected explicit prompt caching; "
                            "retrying implicit."
                        )
                        continue
                    droppable = [
                        ("include", "encrypted reasoning replay unavailable"),
                        ("temperature", "temperature not supported here"),
                        ("reasoning", "reasoning knob not supported here"),
                        ("prompt_cache_key", "cache routing key not supported"),
                    ]
                    dropped = False
                    for key, why in droppable:
                        if key not in msg:
                            continue
                        if key in request_kwargs:
                            request_kwargs.pop(key, None)
                        elif key in extra_body:
                            extra_body.pop(key, None)
                            if not extra_body:
                                request_kwargs.pop("extra_body", None)
                        else:
                            continue
                        logger.warning(
                            f"/v1/responses rejected '{key}' ({why}); "
                            "retrying without it."
                        )
                        dropped = True
                        break
                    if dropped:
                        continue
                    raise

            served_model_logged = False
            raw_output_items: List[Dict[str, Any]] = []
            func_call_items: List[Dict[str, Any]] = []
            emitted_text_chars = 0

            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "response.created":
                    served = getattr(getattr(event, "response", None), "model", None)
                    if not served_model_logged and served:
                        logger.info(f"[llm] requested={self.model!r} served={served!r}")
                        served_model_logged = True
                elif etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    emitted_text_chars += len(delta)
                    if delta:
                        yield delta
                elif etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item is None:
                        continue
                    item_dict = item.model_dump(exclude_none=True)
                    raw_output_items.append(item_dict)
                    if item_dict.get("type") == "function_call":
                        func_call_items.append(item_dict)
                    elif item_dict.get("type") == "reasoning":
                        # Same visibility as the Claude path's [thinking] line:
                        # log the model's reasoning summary, never shown to the
                        # user and never stored to chat history.
                        summary = " ".join(
                            s.get("text", "")
                            for s in item_dict.get("summary") or []
                            if isinstance(s, dict)
                        ).strip()
                        if summary:
                            logger.info(f"[thinking] {summary}")
                elif etype in ("response.completed", "response.incomplete"):
                    usage = getattr(getattr(event, "response", None), "usage", None)
                    if usage:
                        self._log_responses_usage(usage)
                    if etype == "response.incomplete":
                        logger.warning(
                            "/v1/responses stream ended incomplete (likely "
                            f"max_output_tokens); emitted_text_chars="
                            f"{emitted_text_chars}."
                        )
                elif etype in ("response.failed", "error"):
                    detail = getattr(
                        getattr(getattr(event, "response", None), "error", None),
                        "message",
                        None,
                    ) or getattr(event, "message", str(event))
                    logger.error(f"/v1/responses stream error: {detail}")
                    yield (
                        "Error calling the chat endpoint: Error occurred while "
                        "generating response. See the logs for details."
                    )
                    return

            if func_call_items:
                call_ids = [fc.get("call_id", "") for fc in func_call_items]
                self._stash_replay(call_ids, raw_output_items)
                yield [
                    ToolCallObject.from_dict(
                        {
                            "id": fc.get("call_id"),
                            "type": "function",
                            "index": i,
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": fc.get("arguments") or "{}",
                            },
                        }
                    )
                    for i, fc in enumerate(func_call_items)
                ]

        except APIConnectionError as e:
            logger.error(
                f"Error calling the responses endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details."

        except RateLimitError as e:
            detail = f"{getattr(e, 'code', '') or ''} {e}"
            logger.error(
                f"Error calling the responses endpoint: 429 Too Many Requests — {e}"
            )
            if "insufficient_quota" in detail:
                yield (
                    "Error calling the chat endpoint: OpenAI quota/credit exhausted "
                    "(insufficient_quota). Check your billing. See the logs for details."
                )
            else:
                yield (
                    "Error calling the chat endpoint: Rate limit exceeded. "
                    "Please try again later. See the logs for details."
                )

        except APIError as e:
            if getattr(e, "status_code", None) == 404:
                logger.error(
                    f"/v1/responses returned 404 — {self.base_url} doesn't "
                    "implement the responses API. Set api_mode: 'chat' for "
                    "this provider (no auto-fallback by design)."
                )
            elif "does not support tools" in str(e):
                self.support_tools = False
                logger.warning(
                    f"{self.model} does not support tools. Disabling tool support."
                )
                yield "__API_NOT_SUPPORT_TOOLS__"
                return
            else:
                logger.error(f"LLM API (responses): Error occurred: {e}")
                logger.info(f"Base URL: {self.base_url}")
                logger.info(f"Model: {self.model}")
                logger.info(f"Messages: {self._summarize_messages(messages)}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            if stream:
                logger.debug("Responses completion finished.")
                await stream.close()
                logger.debug("Responses stream closed.")
