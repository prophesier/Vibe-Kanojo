"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

from typing import AsyncIterator, List, Dict, Any
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface
from ...mcpp.types import ToolCallObject


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
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self._include_usage_supported = True
        self._completion_token_param = (
            "max_completion_tokens"
            if self._uses_official_openai_endpoint(base_url)
            else "max_tokens"
        )
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
        )
        self.support_tools = True

        logger.info(
            f"Initialized AsyncLLM with the parameters: {self.base_url}, {self.model}"
        )

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
        """Log OpenAI automatic prompt-cache usage from a stream usage chunk."""
        prompt_tokens = cls._get_usage_value(usage, "prompt_tokens", 0) or 0
        completion_tokens = cls._get_usage_value(usage, "completion_tokens", 0) or 0
        prompt_details = cls._get_usage_value(usage, "prompt_tokens_details", None)
        cached = cls._get_usage_value(prompt_details, "cached_tokens", 0) or 0
        completion_details = cls._get_usage_value(
            usage, "completion_tokens_details", None
        )
        reasoning = (
            cls._get_usage_value(completion_details, "reasoning_tokens", 0) or 0
        )

        fresh = max(prompt_tokens - cached, 0)
        hit_pct = (cached / prompt_tokens * 100) if prompt_tokens else 0
        logger.info(
            f"[cache] read={cached} write=0 fresh={fresh} "
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
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

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
