"""Description: This file contains the implementation of the `AsyncLLM` class for Claude API.
This class is responsible for handling asynchronous interaction with Claude API endpoints
for language generation.
"""

import asyncio
import json
from typing import AsyncIterator, List, Dict, Any, Optional, Union

import anthropic
import httpx
from loguru import logger
from anthropic import AsyncAnthropic, NOT_GIVEN

from .stateless_llm_interface import StatelessLLMInterface


def _sniff_image_media_type(base64_data: str, declared: str) -> str:
    """Return the actual media type by inspecting base64 magic bytes.

    Anthropic strictly validates the declared media_type against image bytes
    and rejects mismatches with HTTP 400. Discord (and browser paste of
    screenshots) sometimes mislabel PNG as JPEG, so we override the declared
    type when the magic bytes contradict it.
    """
    sample = base64_data[:24]
    if sample.startswith("iVBORw0KGgo"):
        return "image/png"
    if sample.startswith("/9j/"):
        return "image/jpeg"
    if sample.startswith("R0lGOD"):
        return "image/gif"
    if sample.startswith("UklGR") and "V0VCUF" in base64_data[:64]:
        return "image/webp"
    return declared


def _budget_tokens_removed(model: str) -> bool:
    """Whether manual extended thinking (``budget_tokens``) is removed for this
    model. Still functional on Opus 4.6 and older, but 400s on Opus 4.7/4.8,
    Sonnet 5, Fable 5, and Mythos 5 — so forced thinking must fall back to
    adaptive there.

    ``opus-5`` is listed pre-emptively: no such model exists yet, so unlike the
    others it is a guess, not a verified 400. It costs nothing to carry — the
    fallback is adaptive, which every recent model supports.

    Matching is by substring, so version suffixes are covered
    (``claude-fable-5``, ``claude-sonnet-5-20260101``, ...). The 4.x ids are
    safe from false positives: ``claude-opus-4-5`` contains ``opus-4-5``, never
    ``opus-5``; ``claude-sonnet-4-5`` never contains ``sonnet-5``.
    """
    m = (model or "").lower()
    return any(
        x in m
        for x in ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable", "mythos")
    )


class AsyncLLM(StatelessLLMInterface):
    # Output ceiling for a chat turn when the caller doesn't specify one
    # (overridable per config via max_tokens). The old signature default of
    # 1024 was too small for a real reply and only ever went unnoticed because
    # the thinking floor below silently raised it.
    _DEFAULT_MAX_TOKENS = 8000
    # When thinking is on, the reply shares the max_tokens budget with the
    # (billed) thinking tokens. Raise the ceiling so reasoning doesn't truncate
    # a normal-length reply. Kept as a floor on top of the configured value:
    # a small configured max_tokens still gets lifted to this when thinking is
    # active, so reasoning can never eat the whole reply.
    _THINKING_MAX_TOKENS_FLOOR = 8000
    # Forced (always-on) extended thinking: the default budget (overridable per
    # config via thinking_budget), plus the reply headroom kept above it
    # (budget_tokens must stay under max_tokens).
    _THINKING_FORCED_BUDGET = 4096
    _THINKING_FORCED_REPLY_ROOM = 4000
    # API floor: budget_tokens must be >= 1024 or the request 400s. Anything
    # lower in the config is raised to this, loudly.
    _THINKING_BUDGET_MIN = 1024
    # Not a limit, just where Anthropic's docs stop recommending interactive
    # use ("for thinking budgets above 32k, use batch processing to avoid
    # networking issues"). Above this we warn but still send it.
    _THINKING_BUDGET_WARN = 32000

    def __init__(
        self,
        model: str = "claude-3-haiku-latest",
        base_url: str = None,
        llm_api_key: str = None,
        system: str = None,
        enable_web_search: bool = False,
        max_web_searches: int = 3,
        enable_web_fetch: bool = False,
        max_web_fetches: int = 5,
        max_fetch_tokens: int = 30000,
        thinking: bool = False,
        thinking_effort: str = "medium",
        thinking_force: bool = False,
        thinking_budget: int = _THINKING_FORCED_BUDGET,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize Claude LLM.

        Args:
            model (str): Model name
            base_url (str): Base URL for Claude API
            llm_api_key (str): Claude API key
            system (str): System prompt
            enable_web_search (bool): Declare Anthropic's native web_search
                server tool so Claude can search the web on its own decision.
            max_web_searches (int): Cap on searches per reply when enabled.
            enable_web_fetch (bool): Declare Anthropic's native web_fetch
                server tool so Claude can read full content from URLs
                already present in the conversation.
            max_web_fetches (int): Cap on URL fetches per reply when enabled.
            max_fetch_tokens (int): Truncate any single fetched page above
                this many tokens to bound per-turn input cost.
            thinking (bool): Enable extended thinking.
            thinking_effort (str): Effort level for the adaptive path.
            thinking_force (bool): Reason on every turn via manual
                budget_tokens, instead of letting adaptive skip turns.
            thinking_budget (int): Thinking tokens allowed per request on the
                FORCED path only — it becomes budget_tokens. Ignored on the
                adaptive path (which has no budget) and on models where manual
                thinking is unavailable. Note this is per REQUEST, not per user
                turn: a turn that goes N rounds through the tool loop pays it
                N+1 times. Clamped up to the API's 1024 minimum.
            max_tokens (int): Output ceiling for a chat turn when the caller
                does not pass one. Covers thinking + reply together, so when
                thinking is on the reply gets whatever reasoning leaves. One-
                shot utility calls (memory, diary, fact extraction) pass their
                own value and are unaffected.
        """
        self.model = model
        self.system = system
        self._enable_web_search = enable_web_search
        self._max_web_searches = max_web_searches
        self._enable_web_fetch = enable_web_fetch
        self._max_web_fetches = max_web_fetches
        self._max_fetch_tokens = max_fetch_tokens
        self._thinking = thinking
        self._thinking_effort = thinking_effort
        self._thinking_force = thinking_force
        self._thinking_budget = self._resolve_thinking_budget(thinking_budget)
        self._max_tokens = self._resolve_max_tokens(max_tokens)
        logger.info(f"Claude reply ceiling: max_tokens={self._max_tokens}.")
        if enable_web_search:
            logger.info(
                f"Claude native web search enabled (max {max_web_searches}/reply)."
            )
        if enable_web_fetch:
            logger.info(
                f"Claude native web fetch enabled "
                f"(max {max_web_fetches}/reply, max {max_fetch_tokens} tokens/page)."
            )
        if thinking:
            # Always name the resolved budget on the forced path: a typo in the
            # conf key is silently ignored (pydantic drops unknown keys), so
            # this log line is the only way to catch a setting that never
            # took effect.
            mode = (
                f"forced/every-turn, budget={self._thinking_budget}"
                if thinking_force and not _budget_tokens_removed(model)
                else f"adaptive, effort={thinking_effort}"
            )
            logger.info(f"Claude extended thinking enabled ({mode}).")

        # Initialize Claude client. The extended-cache-ttl beta header lets us
        # request 1-hour prompt cache TTL on cache_control blocks; without it
        # only the default 5-minute TTL is accepted.
        #
        # timeout/max_retries harden the connection against flaky networks:
        # the SDK retries connection failures and retryable statuses (429/5xx,
        # incl. 529 overload) BEFORE the stream starts consuming. A generous
        # read timeout accommodates long time-to-first-token on large cached
        # contexts. (Mid-stream disconnects can't be retried — they'd
        # duplicate already-streamed text — so the partial reply is kept.)
        self.client = AsyncAnthropic(
            api_key=llm_api_key,
            base_url=base_url if base_url else None,
            default_headers={"anthropic-beta": "extended-cache-ttl-2025-04-11"},
            timeout=httpx.Timeout(120.0, connect=10.0),
            max_retries=4,
        )
        # Fire-and-forget [usage]-log tasks (count_tokens is off the reply's
        # critical path). Strong refs so the event loop can't GC them mid-run.
        self._bg_tasks: set = set()
        # Message-framing overhead of count_tokens (the wrapper around the
        # counted text), calibrated once per process via a 1-token probe:
        # count("x") - 1. None = not yet calibrated (retried on next use).
        self._count_overhead: Optional[int] = None

        logger.info(f"Initialized Claude AsyncLLM with model: {self.model}")
        logger.debug(f"Base URL: {base_url}")

    def _spawn_usage_log(
        self, out_toks: int, think_toks: Any, visible_text: str
    ) -> None:
        """Emit the per-request [usage] line, counting the visible (summary)
        thinking text in TOKENS via the free count_tokens endpoint so it
        compares directly against the billed thinking_tokens. Runs as a
        fire-and-forget task — one extra round-trip that must not delay the
        reply's message_stop. Never raises; a failed count logs "?".

        count_tokens wraps the text in a user message, so the raw count
        includes a constant framing overhead. That constant is calibrated
        once per process (count("x") - 1 — the API rejects empty content, so
        a 1-token probe stands in for an empty one) and subtracted; BPE
        merges at the role boundary can wobble it by ±1 token, which is
        noise at the scale we compare. An uncalibrated line (probe failed)
        logs the raw count."""
        billed = think_toks if think_toks is not None else "n/a"
        if not visible_text:
            logger.info(
                f"[usage] output_tokens={out_toks} thinking_tokens={billed} "
                "visible_thinking_tokens=0 chars=0"
            )
            return

        async def _count(text: str) -> int:
            counted = await self.client.with_options(
                timeout=15.0, max_retries=1
            ).messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": text}],
            )
            return counted.input_tokens

        async def _count_and_log() -> None:
            if self._count_overhead is None:
                try:
                    self._count_overhead = max(0, await _count("x") - 1)
                except Exception as e:
                    logger.debug(f"count_tokens overhead probe failed: {e}")
            try:
                raw = await _count(visible_text)
                if self._count_overhead is not None:
                    visible = str(max(0, raw - self._count_overhead))
                else:
                    visible = str(raw)
            except Exception as e:
                logger.debug(f"count_tokens for [usage] failed: {e}")
                visible = "?"
            logger.info(
                f"[usage] output_tokens={out_toks} thinking_tokens={billed} "
                f"visible_thinking_tokens={visible} chars={len(visible_text)}"
            )

        task = asyncio.create_task(_count_and_log())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    @classmethod
    def _resolve_max_tokens(cls, value: Any) -> int:
        """Validate the configured reply ceiling. Never raises — a bad value
        falls back to the default rather than killing startup."""
        try:
            n = int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"max_tokens {value!r} is not an integer; "
                f"using {cls._DEFAULT_MAX_TOKENS}."
            )
            return cls._DEFAULT_MAX_TOKENS
        if n <= 0:
            logger.warning(
                f"max_tokens {n} is not positive; using {cls._DEFAULT_MAX_TOKENS}."
            )
            return cls._DEFAULT_MAX_TOKENS
        return n

    @classmethod
    def _resolve_thinking_budget(cls, value: Any) -> int:
        """Validate the configured forced-thinking budget, warning on anything
        the API or the docs would object to.

        Never raises: a bad value falls back to the default rather than killing
        startup, because this knob is not worth losing the whole server over.
        """
        try:
            budget = int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"thinking_budget {value!r} is not an integer; "
                f"using {cls._THINKING_FORCED_BUDGET}."
            )
            return cls._THINKING_FORCED_BUDGET
        if budget < cls._THINKING_BUDGET_MIN:
            # The API rejects budget_tokens < 1024 outright, so silently
            # honouring a smaller number would break every forced turn.
            logger.warning(
                f"thinking_budget {budget} is below the API minimum "
                f"({cls._THINKING_BUDGET_MIN}); raising it to the minimum."
            )
            return cls._THINKING_BUDGET_MIN
        if budget > cls._THINKING_BUDGET_WARN:
            logger.warning(
                f"thinking_budget {budget} is above {cls._THINKING_BUDGET_WARN}; "
                "Anthropic recommends batch processing beyond that (long "
                "requests, connection limits). Sending it anyway — expect slow "
                "turns and higher output-token cost."
            )
        return budget

    def _convert_message_format(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert message format to Claude's expected format."""
        # Assistant entries may carry preserved thinking blocks from the turn
        # that produced them (side field set by basic_memory_agent._add_message).
        # Rebuild them into ordered content blocks — thinking first, then the
        # visible text — and drop the side field so only valid API fields go on
        # the wire. Opus 4.5+ keeps replayed thinking from ALL prior turns in
        # context (reconstructed server-side from the signature). A
        # cache_control tag attached by the agent rides on the text block,
        # which stays last, so the breakpoint still covers the thinking blocks
        # ahead of it.
        thinking_blocks = message.get("thinking_blocks")
        if message.get("role") == "assistant" and thinking_blocks:
            new_content: List[Dict[str, Any]] = [dict(b) for b in thinking_blocks]
            content = message.get("content")
            if isinstance(content, str):
                if content:
                    new_content.append({"type": "text", "text": content})
            elif isinstance(content, list):
                new_content.extend(dict(c) for c in content)
            return {"role": "assistant", "content": new_content}

        # Handle potential tool_result content blocks
        if isinstance(message.get("content"), list):
            new_content = []
            is_tool_result = False
            for content_item in message["content"]:
                if content_item.get("type") == "image_url":
                    # Extract media type and base64 data from data URL
                    data_url = content_item["image_url"]["url"]
                    # Split 'data:image/jpeg;base64,/9j/4AAQ...' into parts
                    header, base64_data = data_url.split(",", 1)
                    # Extract media type from 'data:image/jpeg;base64'
                    declared_media_type = header.split(":")[1].split(";")[0]
                    # Discord (and pasted screenshots) sometimes mislabel PNG
                    # as JPEG; Anthropic rejects mismatches, so we sniff the
                    # actual bytes and correct the declaration here.
                    media_type = _sniff_image_media_type(
                        base64_data, declared_media_type
                    )
                    if media_type != declared_media_type:
                        logger.debug(
                            f"Image media type corrected: {declared_media_type} → {media_type}"
                        )

                    new_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        }
                    )
                elif content_item.get("type") == "tool_result":
                    is_tool_result = True
                    # Keep tool_result block as is, Anthropic SDK handles it
                    new_content.append(content_item)
                else:
                    # Assume text or other standard types
                    new_content.append(content_item)

            # For tool_result messages, the role should be 'user'
            # Ensure the role is correctly set before returning
            role = "user" if is_tool_result else message["role"]
            return {"role": role, "content": new_content}

        # Handle plain text content or non-list content
        return message

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: Union[str, List[Dict[str, Any]]] = None,
        tools: List[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        disable_server_tools: bool = False,
        reasoning_effort: str = None,  # noqa: ARG002 — accepted for signature parity (OpenAI-only); ignored here
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generates a chat completion using the Claude API asynchronously,
        handling text generation and tool use.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (Union[str, List[Dict[str, Any]]], optional): System prompt.
          Pass a list of content blocks to enable prompt caching via
          ``cache_control`` markers; pass a plain string for a normal request.
        - tools (List[Dict[str, Any]], optional): List of tools available.
        - max_tokens (int, optional): Output ceiling for this call. ``None``
          (the chat path) uses the configured value; one-shot utility callers
          pass their own. Thinking, when active, floors this further.

        Yields:
        - Dict[str, Any]: Events representing text deltas, tool use, or errors.
          Possible event types:
            - {"type": "message_start", "data": ...}
            - {"type": "text_delta", "text": "..."}
            - {"type": "tool_use_start", "data": {"id": ..., "name": ..., "input": None}}
            - {"type": "tool_input_delta", "tool_id": ..., "partial_json": "..."} # Optional
            - {"type": "tool_use_complete", "data": {"id": ..., "name": ..., "input": {...}}}
            - {"type": "thinking_complete", "data": {"type": "thinking", "thinking": ..., "signature": ...}}
            - {"type": "message_delta", "data": ...} # e.g., stop_reason
            - {"type": "message_stop"}
            - {"type": "error", "message": "..."}
        """
        # Resolve before anything reads it: the thinking branches below treat
        # max_tokens as a floor to raise, so it has to hold the configured
        # value by then, not None.
        if max_tokens is None:
            max_tokens = self._max_tokens
        text_emitted = False
        try:
            # Filter out system messages and convert message format
            converted_messages = [
                self._convert_message_format(msg)
                for msg in messages
                if msg["role"] != "system"
            ]

            # Build the tool list: caller-supplied tools plus Anthropic's
            # native server tools when enabled. Both web_search and
            # web_fetch run server-side within this same stream — no client
            # round-trip — so we just declare them and parse the extra
            # result blocks below.
            #
            # disable_server_tools=True skips auto-injection: callers that
            # do one-shot non-chat work (fact extraction, diary creation,
            # fact pruning, consolidation) don't want or need web tools,
            # and we'd otherwise pay a couple hundred wasted tokens per
            # call shipping the unused tool defns.
            final_tools = list(tools) if tools else []
            if not disable_server_tools and self._enable_web_search:
                final_tools.append(
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": self._max_web_searches,
                    }
                )
            if not disable_server_tools and self._enable_web_fetch:
                # Stay on the basic web_fetch version: the newer
                # web_fetch_20260209 adds dynamic filtering but requires
                # the code_execution tool to be enabled, which we don't
                # use here. max_content_tokens caps how big each fetched
                # page is allowed to grow in our context window.
                final_tools.append(
                    {
                        "type": "web_fetch_20250910",
                        "name": "web_fetch",
                        "max_uses": self._max_web_fetches,
                        "max_content_tokens": self._max_fetch_tokens,
                    }
                )

            logger.debug(f"Sending messages to Claude API: {converted_messages}")
            logger.debug(f"Tools provided: {final_tools}")

            # Extended thinking. Skipped for one-shot utility calls
            # (disable_server_tools) — fact extraction / diary / pruning don't
            # benefit from a reasoning pass and shouldn't pay for it. Passed via
            # extra_body so it goes straight to the API regardless of the SDK
            # version's typed-param support for thinking/output_config.
            think_kwargs: Dict[str, Any] = {}
            if self._thinking and not disable_server_tools:
                extra_body: Dict[str, Any] = {}
                if self._thinking_force and not _budget_tokens_removed(self.model):
                    # Forced extended thinking: reason on EVERY turn. Adaptive
                    # skips turns it judges simple — which is exactly where
                    # coherence errors slip through (e.g. delivering an alarm's
                    # content inline instead of at fire time). budget_tokens must
                    # stay under max_tokens, so lift the reply ceiling — this
                    # max() is what keeps the API's budget < max_tokens rule
                    # satisfied for any configured budget.
                    budget = self._thinking_budget
                    max_tokens = max(
                        max_tokens, budget + self._THINKING_FORCED_REPLY_ROOM
                    )
                    extra_body["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }
                else:
                    extra_body["thinking"] = {
                        "type": "adaptive",
                        "display": "summarized",
                    }
                    if self._thinking_effort:
                        extra_body["output_config"] = {"effort": self._thinking_effort}
                    if max_tokens < self._THINKING_MAX_TOKENS_FLOOR:
                        max_tokens = self._THINKING_MAX_TOKENS_FLOOR
                think_kwargs["extra_body"] = extra_body

            async with self.client.messages.stream(
                messages=converted_messages,
                system=system if system else (self.system if self.system else ""),
                model=self.model,
                max_tokens=max_tokens,
                tools=final_tools if final_tools else NOT_GIVEN,
                **think_kwargs,
            ) as stream:
                current_tool_call_info = None
                partial_json_accumulator = ""
                # Track an in-flight server-side tool call (web_search or
                # web_fetch) so its input-JSON deltas don't trip the
                # client-tool path. The accumulator collects the raw JSON;
                # the parse-and-log happens at content_block_stop.
                server_tool_index = None
                server_tool_name = ""
                server_tool_input_json = ""
                # Track an in-flight thinking block (adaptive thinking): collect
                # the summary text + signature so we can log it and hand the
                # complete block back for tool-loop replay (Anthropic requires
                # thinking blocks echoed verbatim, signature included, ahead of
                # any tool_use in the same turn).
                thinking_index = None
                thinking_text = ""
                thinking_signature = ""
                thinking_redacted_data = None
                # Per-REQUEST visible (summary) thinking text, across all
                # thinking blocks. Counted in tokens at the end (via
                # count_tokens, off-path) and logged next to the billed
                # thinking_tokens so "is the log text a summary or verbatim?"
                # is answerable per turn from two like-for-like numbers.
                visible_thinking_parts: List[str] = []

                async for event in stream:
                    if event.type == "message_start":
                        logger.debug("Stream: message_start")
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            fresh = getattr(usage, "input_tokens", 0) or 0
                            cache_read = (
                                getattr(usage, "cache_read_input_tokens", 0) or 0
                            )
                            cache_write = (
                                getattr(usage, "cache_creation_input_tokens", 0) or 0
                            )
                            total_input = fresh + cache_read + cache_write
                            hit_pct = (
                                (cache_read / total_input * 100) if total_input else 0
                            )
                            logger.info(
                                f"[cache] read={cache_read} write={cache_write} "
                                f"fresh={fresh} (hit {hit_pct:.0f}%)"
                            )
                        yield {
                            "type": "message_start",
                            "data": event.message.model_dump(exclude_none=True),
                        }
                    elif event.type == "content_block_start":
                        logger.debug(
                            f"Stream: content_block_start - Index: {event.index}, Type: {event.content_block.type}"
                        )
                        if event.content_block.type == "text":
                            pass  # Handled by text_delta
                        elif event.content_block.type == "thinking":
                            thinking_index = event.index
                            thinking_text = ""
                            thinking_signature = ""
                            thinking_redacted_data = None
                        elif event.content_block.type == "redacted_thinking":
                            thinking_index = event.index
                            thinking_text = ""
                            thinking_signature = ""
                            thinking_redacted_data = getattr(
                                event.content_block, "data", ""
                            )
                        elif event.content_block.type == "tool_use":
                            current_tool_call_info = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": None,
                                "index": event.index,  # Store index
                            }
                            partial_json_accumulator = ""
                            logger.debug(
                                f"Stream: tool_use started - ID: {current_tool_call_info['id']}, Name: {current_tool_call_info['name']}"
                            )
                            yield {
                                "type": "tool_use_start",
                                "data": current_tool_call_info.copy(),
                            }
                        elif event.content_block.type == "server_tool_use":
                            # Native server-side tool (e.g. web_search).
                            # Anthropic executes it inline; we only log it.
                            server_tool_index = event.index
                            server_tool_name = getattr(event.content_block, "name", "?")
                            server_tool_input_json = ""
                            tag = (
                                "web_fetch"
                                if server_tool_name == "web_fetch"
                                else "web_search"
                            )
                            logger.info(
                                f"[{tag}] model invoked server tool "
                                f"'{server_tool_name}'"
                            )
                        elif event.content_block.type == "web_search_tool_result":
                            # Search results from Anthropic. Not shown to the
                            # user; the model folds them into its next text.
                            result = getattr(event.content_block, "content", None)
                            count = len(result) if isinstance(result, list) else "?"
                            logger.info(
                                f"[web_search] received results ({count} item(s))"
                            )
                        elif event.content_block.type == "web_fetch_tool_result":
                            # Fetched page content from Anthropic. Same as
                            # search result: not surfaced as text, the model
                            # weaves it into its next text output.
                            inner = getattr(event.content_block, "content", None)
                            url = getattr(inner, "url", None) if inner else None
                            err = getattr(inner, "error_code", None) if inner else None
                            if err:
                                logger.info(
                                    f"[web_fetch] fetch failed: {err} "
                                    f"(url={url or '?'})"
                                )
                            else:
                                logger.info(
                                    f"[web_fetch] received content from {url or '?'}"
                                )
                    elif event.type == "content_block_delta":
                        logger.debug(
                            f"Stream: content_block_delta - Index: {event.index}, Delta Type: {event.delta.type}"
                        )
                        if event.delta.type == "text_delta":
                            text_emitted = True
                            yield {"type": "text_delta", "text": event.delta.text}
                        elif event.delta.type == "thinking_delta":
                            thinking_text += getattr(event.delta, "thinking", "")
                        elif event.delta.type == "signature_delta":
                            thinking_signature += getattr(event.delta, "signature", "")
                        elif event.delta.type == "input_json_delta":
                            if (
                                current_tool_call_info
                                and event.index == current_tool_call_info["index"]
                            ):
                                partial_json_accumulator += event.delta.partial_json
                                logger.trace(
                                    f"Stream: input_json_delta - Tool ID: {current_tool_call_info['id']}, Partial: {event.delta.partial_json}"
                                )
                            elif (
                                server_tool_index is not None
                                and event.index == server_tool_index
                            ):
                                # Accumulate the server tool's input JSON
                                # ({"query":"..."} for web_search,
                                # {"url":"..."} for web_fetch).
                                server_tool_input_json += event.delta.partial_json
                            else:
                                logger.warning(
                                    f"Received input_json_delta but no active tool call matching index {event.index}"
                                )
                    elif event.type == "content_block_stop":
                        logger.debug(
                            f"Stream: content_block_stop - Index: {event.index}"
                        )
                        # Thinking block finished — log the summary and hand the
                        # complete block (text + signature, or redacted data)
                        # back so the tool loop can replay it. Not shown to the
                        # user and not stored to chat history.
                        if thinking_index is not None and event.index == thinking_index:
                            if thinking_redacted_data is not None:
                                block = {
                                    "type": "redacted_thinking",
                                    "data": thinking_redacted_data,
                                }
                                logger.info("[thinking] (redacted)")
                            else:
                                block = {
                                    "type": "thinking",
                                    "thinking": thinking_text,
                                    "signature": thinking_signature,
                                }
                                summary = thinking_text.strip()
                                if summary:
                                    logger.info(f"[thinking] {summary}")
                                if thinking_text:
                                    visible_thinking_parts.append(thinking_text)
                            yield {"type": "thinking_complete", "data": block}
                            thinking_index = None
                            thinking_text = ""
                            thinking_signature = ""
                            thinking_redacted_data = None
                        # Server-side tool input finished — parse the JSON,
                        # log it, and emit the appropriate inline marker.
                        if (
                            server_tool_index is not None
                            and event.index == server_tool_index
                        ):
                            parsed: Dict[str, Any] = {}
                            try:
                                if server_tool_input_json.strip():
                                    parsed = json.loads(server_tool_input_json)
                            except json.JSONDecodeError:
                                parsed = {}

                            if server_tool_name == "web_search":
                                query_str = str(parsed.get("query", "")).strip()
                                logger.info(
                                    f"[web_search] query: {query_str or '(empty)'}"
                                )
                                # Inline marker shown to the user at the exact
                                # position the search was triggered (display
                                # only — see basic_memory_agent handling).
                                shown = query_str[:80] if query_str else "..."
                                yield {
                                    "type": "web_search_marker",
                                    "text": f"\n🔍 *Web検索: {shown}*\n",
                                }
                            elif server_tool_name == "web_fetch":
                                url_str = str(parsed.get("url", "")).strip()
                                logger.info(f"[web_fetch] url: {url_str or '(empty)'}")
                                shown = url_str[:200] if url_str else "..."
                                yield {
                                    "type": "web_search_marker",
                                    "text": f"\n🔗 *Web取得: {shown}*\n",
                                }
                            server_tool_index = None
                            server_tool_name = ""
                            server_tool_input_json = ""
                        # Check if this stop corresponds to the active tool call
                        if (
                            current_tool_call_info
                            and event.index == current_tool_call_info["index"]
                        ):
                            try:
                                if not partial_json_accumulator.strip():
                                    logger.warning(
                                        f"Empty JSON input received for tool ID: {current_tool_call_info['id']}. Using empty object."
                                    )
                                    tool_input = {}
                                else:
                                    tool_input = json.loads(partial_json_accumulator)
                                current_tool_call_info["input"] = tool_input
                                logger.debug(
                                    f"Stream: tool_use completed - ID: {current_tool_call_info['id']}, Input: {tool_input}"
                                )
                                # Yield the complete tool call info
                                yield {
                                    "type": "tool_use_complete",
                                    "data": current_tool_call_info.copy(),
                                }
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"Failed to decode tool input JSON: {partial_json_accumulator}. Error: {e}"
                                )
                                yield {
                                    "type": "error",
                                    "message": f"Failed to parse tool input JSON for tool ID {current_tool_call_info['id']}",
                                }
                            finally:
                                # Reset regardless of success or failure for this index
                                current_tool_call_info = None
                                partial_json_accumulator = ""
                    elif event.type == "message_delta":
                        logger.debug(
                            f"Stream: message_delta - Delta: {event.delta.model_dump(exclude_none=True)}, Usage: {event.usage}"
                        )
                        # Report how many server-tool calls this reply made.
                        # Search counts toward the per-request billing tier;
                        # fetch is free but content tokens hit input.
                        server_use = getattr(event.usage, "server_tool_use", None)
                        if server_use:
                            n_searches = getattr(server_use, "web_search_requests", 0)
                            n_fetches = getattr(server_use, "web_fetch_requests", 0)
                            if n_searches:
                                logger.info(
                                    f"[web_search] this reply used {n_searches} "
                                    "web search request(s)"
                                )
                            if n_fetches:
                                logger.info(
                                    f"[web_fetch] this reply used {n_fetches} "
                                    "web fetch request(s)"
                                )
                        # Per-request usage line, emitted on the delta that
                        # carries stop_reason (the final one). thinking_tokens
                        # is the BILLED raw reasoning (0 = the model skipped
                        # thinking this request — grep-friendly signal for
                        # trigger-rate stats). The visible summary text is
                        # token-counted off-path so both sides use the same
                        # unit; see _spawn_usage_log.
                        if getattr(event.delta, "stop_reason", None):
                            out_toks = getattr(event.usage, "output_tokens", 0) or 0
                            details = getattr(
                                event.usage, "output_tokens_details", None
                            )
                            if isinstance(details, dict):
                                think_toks = details.get("thinking_tokens")
                            else:
                                think_toks = getattr(details, "thinking_tokens", None)
                            self._spawn_usage_log(
                                out_toks,
                                think_toks,
                                "\n".join(visible_thinking_parts),
                            )
                        yield {
                            "type": "message_delta",
                            "data": {
                                "delta": event.delta.model_dump(exclude_none=True),
                                "usage": event.usage.model_dump(),
                            },
                        }
                    elif event.type == "message_stop":
                        logger.debug("Stream: message_stop")
                        yield {"type": "message_stop"}
                        # No need to break here, the context manager handles the end
                    elif event.type == "ping":
                        logger.trace("Stream: ping")
                        pass  # Ignore pings
                    # Anthropic SDK might raise errors directly, or via event.type == 'error'
                    # The outer try/except handles SDK-level errors.

        except Exception as e:
            is_transient = isinstance(
                e, (httpx.TransportError, anthropic.APIConnectionError)
            ) or type(e).__name__ in {"BrokenResourceError", "ReadError"}
            if is_transient and text_emitted:
                # Connection dropped mid-stream after we'd already streamed
                # part of the reply. Retrying would duplicate text, so finish
                # gracefully with whatever was delivered instead of raising
                # (which would surface a bare error and discard the partial).
                logger.warning(
                    f"Claude stream dropped mid-reply ({type(e).__name__}: {e}); "
                    "delivering partial response and ending turn cleanly."
                )
                yield {"type": "message_stop"}
                logger.debug("Chat completion stream processing finished (partial).")
                return

            logger.error(f"Claude API error occurred: {type(e).__name__}: {e}")
            logger.info(f"Model: {self.model}")
            # Yield an error event before raising
            yield {"type": "error", "message": f"Claude API error: {str(e)}"}
            raise

        # No finally block needed for stream.close() due to async with
        logger.debug("Chat completion stream processing finished.")
