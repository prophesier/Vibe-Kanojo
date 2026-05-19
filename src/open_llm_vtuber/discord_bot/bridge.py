"""WebSocket client that bridges to the Open-LLM-VTuber backend.

The bridge owns a single long-lived WebSocket connection to the OLV server's
``/proxy-ws`` endpoint. Text input is forwarded with ``text-input`` messages;
incoming ``audio`` messages carry the AI's textual reply in ``display_text``,
which is dispatched to a registered callback so the Discord side can post it
back to the originating channel.

The bridge serialises in-flight conversations: only one Discord request is
sent to OLV at a time, and replies for that request are collected until an
end-of-turn signal arrives (the server sends a ``control`` message with
``conversation-chain-end`` after a turn completes). Other Discord messages
queue behind it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

import websockets
from loguru import logger


ReplyCallback = Callable[["TurnResult"], Awaitable[None]]


@dataclass
class TurnResult:
    """Aggregated reply for one user turn."""

    request_id: str
    text_parts: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def text(self) -> str:
        return " ".join(part.strip() for part in self.text_parts if part).strip()


@dataclass
class _PendingTurn:
    request_id: str
    text: str
    callback: ReplyCallback
    result: TurnResult = field(init=False)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.result = TurnResult(request_id=self.request_id)


class OLVBridge:
    """Async client for the OLV ``/proxy-ws`` WebSocket.

    Usage::

        bridge = OLVBridge("ws://localhost:12393/proxy-ws")
        await bridge.start()
        await bridge.send_text("hi", request_id="...", on_reply=cb)
        await bridge.stop()
    """

    def __init__(
        self,
        server_url: str,
        *,
        reconnect_initial_delay: float = 2.0,
        reconnect_max_delay: float = 30.0,
        turn_timeout: float = 120.0,
    ) -> None:
        self._server_url = server_url
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._turn_timeout = turn_timeout

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._send_lock = asyncio.Lock()
        self._receive_task: Optional[asyncio.Task[None]] = None
        self._ready_event = asyncio.Event()
        self._current_turn: Optional[_PendingTurn] = None
        self._turn_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._receive_task = asyncio.create_task(self._run_loop())
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.warning(
                "OLV bridge did not become ready within 10s — will keep retrying"
            )

    async def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._receive_task = None

    async def send_text(
        self,
        text: str,
        *,
        request_id: str,
        on_reply: ReplyCallback,
        images: Optional[Sequence[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send a single user turn and await its full reply.

        Calls ``on_reply`` exactly once with a :class:`TurnResult` containing
        the aggregated text (and an ``error`` message if something went wrong).
        Concurrent calls are serialised — later turns wait for earlier ones to
        finish so that replies don't interleave.
        """
        async with self._turn_lock:
            if not self.is_connected or self._ws is None:
                result = TurnResult(request_id=request_id, error="bridge not connected")
                await on_reply(result)
                return

            turn = _PendingTurn(request_id=request_id, text=text, callback=on_reply)
            self._current_turn = turn

            try:
                payload: dict[str, Any] = {"type": "text-input", "text": text}
                if images:
                    payload["images"] = list(images)
                if metadata:
                    payload["metadata"] = dict(metadata)
                async with self._send_lock:
                    await self._ws.send(json.dumps(payload))
                logger.debug(
                    f"Sent text-input to OLV (request_id={request_id}"
                    + (f", images={len(images)}" if images else "")
                    + ")"
                )

                try:
                    await asyncio.wait_for(turn.done.wait(), timeout=self._turn_timeout)
                except asyncio.TimeoutError:
                    turn.result.error = (
                        f"timed out waiting for reply after {self._turn_timeout:.0f}s"
                    )
            except Exception as e:
                turn.result.error = f"send failed: {e}"
            finally:
                self._current_turn = None

            try:
                await on_reply(turn.result)
            except Exception as e:
                logger.exception(f"on_reply callback raised: {e}")

    async def interrupt(self) -> None:
        if not self.is_connected or self._ws is None:
            return
        try:
            async with self._send_lock:
                await self._ws.send(
                    json.dumps({"type": "interrupt-signal", "text": ""})
                )
        except Exception as e:
            logger.warning(f"Failed to send interrupt: {e}")

    async def _run_loop(self) -> None:
        delay = self._reconnect_initial_delay
        while self._running:
            try:
                logger.info(f"Connecting to OLV at {self._server_url}")
                async with websockets.connect(
                    self._server_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._ready_event.set()
                    delay = self._reconnect_initial_delay
                    logger.info("OLV bridge connected")

                    # Ask OLV to create a fresh history file so messages get
                    # persisted. Without this, context.history_uid stays empty
                    # and store_message() drops everything on the floor.
                    try:
                        async with self._send_lock:
                            await ws.send(json.dumps({"type": "create-new-history"}))
                    except Exception as e:
                        logger.warning(f"Failed to request new history: {e}")

                    await self._receive_forever(ws)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"OLV bridge connection error: {e}")
            finally:
                self._ws = None
                self._ready_event.clear()
                self._fail_current_turn("connection lost")

            if not self._running:
                break

            logger.info(f"Reconnecting to OLV in {delay:.1f}s")
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            delay = min(delay * 2, self._reconnect_max_delay)

    async def _receive_forever(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Received non-JSON frame from OLV; ignoring")
                continue
            await self._handle_incoming(data)

    async def _handle_incoming(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type")

        if msg_type == "audio":
            display = data.get("display_text") or {}
            text = (display.get("text") or "").strip()
            if text and self._current_turn is not None:
                self._current_turn.result.text_parts.append(text)
            return

        if msg_type == "full-text":
            # Status banners like "Thinking..." — ignore for chat output.
            logger.debug(f"OLV status: {data.get('text')}")
            return

        if msg_type == "backend-synth-complete":
            # OLV waits for this ack before sending conversation-chain-end.
            # The browser sends it after playing audio; we ack immediately.
            if self._ws is not None:
                try:
                    async with self._send_lock:
                        await self._ws.send(
                            json.dumps({"type": "frontend-playback-complete"})
                        )
                except Exception as e:
                    logger.warning(f"Failed to send frontend-playback-complete: {e}")
            return

        if msg_type == "control":
            ctrl = data.get("text")
            if ctrl == "conversation-chain-end" and self._current_turn is not None:
                self._current_turn.done.set()
            return

        if msg_type == "error":
            message = data.get("message", "unknown error")
            logger.warning(f"OLV error: {message}")
            if self._current_turn is not None:
                self._current_turn.result.error = message
                self._current_turn.done.set()
            # The proxy only resets conversation_active on chain-end or
            # interrupt-signal. OLV doesn't send chain-end on errors, so send
            # interrupt-signal to unblock the proxy's message queue.
            if self._ws is not None:
                try:
                    async with self._send_lock:
                        await self._ws.send(
                            json.dumps({"type": "interrupt-signal", "text": ""})
                        )
                except Exception:
                    pass
            return

        # Other message types (set-model-and-conf, group-update, history-list, ...)
        # are not needed for the minimum text bridge.
        logger.trace(f"OLV ignored message type: {msg_type}")

    def _fail_current_turn(self, reason: str) -> None:
        turn = self._current_turn
        if turn is not None and not turn.done.is_set():
            turn.result.error = reason
            turn.done.set()
