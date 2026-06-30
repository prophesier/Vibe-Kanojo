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
    # Expression index for the face to attach on Discord: the LLM-marked primary
    # if any, otherwise the first expression seen this turn.
    face_index: Optional[int] = None

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
        # Resolves when an expression-capture-complete arrives (for /refresh-faces).
        self._capture_future: Optional[asyncio.Future] = None
        # Server-initiated (proactive) turns — e.g. a fired alarm or a cache
        # keepalive — arrive with no pending request. Buffer their text + face
        # and flush on chain-end.
        self._proactive_callback: Optional[
            Callable[[str, Optional[int]], Awaitable[None]]
        ] = None
        self._proactive_parts: list[str] = []
        # Expression face for the in-flight proactive turn (primary wins, else
        # first expression seen) — mirrors _current_turn.result.face_index.
        self._proactive_face: Optional[int] = None

    def set_proactive_callback(
        self, cb: Callable[[str, Optional[int]], Awaitable[None]]
    ) -> None:
        """Register a callback invoked with (text, face_index) of an unsolicited
        turn the server pushes (no pending request) — a fired alarm or a cache
        keepalive. ``face_index`` is the Live2D expression for that turn, or None."""
        self._proactive_callback = cb

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

    async def request_expression_capture(
        self, *, timeout: float = 120.0
    ) -> dict[str, Any]:
        """Trigger a frontend Live2D expression-face capture and wait for it.

        Returns ``{"count": int, "error": str | None}``. Raises on timeout or if
        the bridge is not connected.
        """
        if self._ws is None:
            raise RuntimeError("bridge not connected")
        loop = asyncio.get_running_loop()
        self._capture_future = loop.create_future()
        try:
            async with self._send_lock:
                await self._ws.send(json.dumps({"type": "request-expression-capture"}))
            return await asyncio.wait_for(self._capture_future, timeout=timeout)
        finally:
            self._capture_future = None

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
                    logger.debug(
                        f"Turn {request_id} complete: "
                        f"{len(turn.result.text)} chars accumulated"
                    )
                except asyncio.TimeoutError:
                    turn.result.error = (
                        f"timed out waiting for reply after {self._turn_timeout:.0f}s"
                    )
                    logger.warning(
                        f"Turn {request_id} timed out after {self._turn_timeout:.0f}s "
                        f"with {len(turn.result.text)} chars accumulated so far "
                        "(no conversation-chain-end received)"
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

        if msg_type == "expression-capture-complete":
            fut = self._capture_future
            if fut is not None and not fut.done():
                fut.set_result(
                    {"count": data.get("count", 0), "error": data.get("error")}
                )
            return

        if msg_type == "audio":
            display = data.get("display_text") or {}
            text = (display.get("text") or "").strip()
            if text and self._current_turn is not None:
                self._current_turn.result.text_parts.append(text)
            elif text:
                # No pending request → this is a proactive (server-initiated)
                # turn, e.g. a fired alarm. Buffer until chain-end.
                self._proactive_parts.append(text)
            # Pick the face for this turn: the LLM-marked primary wins; otherwise
            # fall back to the first expression seen.
            actions = data.get("actions") or {}
            primary = actions.get("primary_expression")
            if self._current_turn is not None:
                if primary is not None:
                    self._current_turn.result.face_index = primary
                elif self._current_turn.result.face_index is None:
                    exprs = actions.get("expressions") or []
                    if exprs:
                        self._current_turn.result.face_index = exprs[0]
            else:
                # Proactive turn (no pending request): capture its face too, so
                # alarms / keepalive posts carry an expression like normal replies.
                if primary is not None:
                    self._proactive_face = primary
                elif self._proactive_face is None:
                    exprs = actions.get("expressions") or []
                    if exprs:
                        self._proactive_face = exprs[0]
            return

        if msg_type == "full-text":
            # Status banners like "Thinking..." — ignore for chat output.
            logger.debug(f"OLV status: {data.get('text')}")
            return

        if msg_type == "backend-synth-complete":
            # Dormant by default: Discord runs through OLV's skip_tts (empty-TTS)
            # path, so finalize_conversation_turn does NOT send this to the bridge
            # — the turn finalizes on conversation-chain-end instead. Kept as a
            # safety net for if Discord ever re-enables real TTS (skip_tts=False):
            # OLV would then send this and wait for the ack below before chain-end.
            # When it does arrive we ack immediately (no audio to actually play).
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
            if ctrl == "conversation-chain-end":
                if self._current_turn is not None:
                    self._current_turn.done.set()
                else:
                    await self._flush_proactive()
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

    async def _flush_proactive(self) -> None:
        """Hand a completed proactive turn's text + face to the callback (if any)."""
        if not self._proactive_parts:
            self._proactive_face = None
            return
        text = " ".join(p.strip() for p in self._proactive_parts if p).strip()
        face = self._proactive_face
        self._proactive_parts = []
        self._proactive_face = None
        if text and self._proactive_callback is not None:
            try:
                await self._proactive_callback(text, face)
            except Exception as e:
                logger.warning(f"proactive callback raised: {e}")
