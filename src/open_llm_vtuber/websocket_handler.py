from typing import Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import base64
import json
import os
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[str]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, WebSocket] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}

        # Global session registry: one active session per conf_uid shared across
        # all connected clients (web + proxy).  Prevents a second client from
        # auto-creating a duplicate session when another is already running.
        self._active_history_uid: Dict[str, str] = {}   # conf_uid → history_uid
        self._session_ref_count: Dict[str, int] = {}    # conf_uid → # clients using it

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-end": self._handle_conversation_trigger,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
            "request-expression-capture": self._handle_request_expression_capture,
            "expression-capture-begin": self._handle_expression_capture_begin,
            "expression-capture-chunk": self._handle_expression_capture_chunk,
            "expression-capture-done": self._handle_expression_capture_done,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            session_service_context = await self._init_service_context(
                websocket.send_text, client_uid
            )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            logger.info(f"Connection established for client {client_uid}")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        self.client_connections[client_uid] = websocket
        self.client_contexts[client_uid] = session_service_context
        self.received_data_buffers[client_uid] = np.array([])

        self.chat_group_manager.client_group_map[client_uid] = ""
        await self.send_group_update(websocket, client_uid)

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": session_service_context.live2d_model.model_info,
                    "conf_name": session_service_context.character_config.conf_name,
                    "conf_uid": session_service_context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        # Start microphone
        await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            send_text=send_text,
            client_uid=client_uid,
            memory_manager=self.default_context_cache.memory_manager,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """
        msg_type = data.get("type")
        if not msg_type:
            logger.warning("Message received without type")
            return

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            if msg_type != "frontend-playback-complete":
                logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(self, client_uid: str) -> None:
        """Handle client disconnection"""
        # Snapshot context before any cleanup removes it from the dict.
        context = self.client_contexts.get(client_uid)

        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Decrement global session ref count for this client.
        if context:
            conf_uid = context.character_config.conf_uid
            history_uid = context.history_uid
            if history_uid and self._active_history_uid.get(conf_uid) == history_uid:
                remaining = self._session_ref_count.get(conf_uid, 1) - 1
                if remaining <= 0:
                    self._active_history_uid.pop(conf_uid, None)
                    self._session_ref_count.pop(conf_uid, None)
                    if context.memory_manager:
                        context.memory_manager.set_current_session("")
                else:
                    self._session_ref_count[conf_uid] = remaining

        # Clean up other client data
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        # Close context resources (e.g., MCPClient).
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(self, client_uid: str) -> None:
        """Clean up failed connection data"""
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: WebSocket, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
            )
        else:
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation or reuse of chat history.

        Uses a global session registry (per conf_uid) so that all connected
        clients — e.g. a Discord proxy and a web UI — share a single session.
        The second client to connect adopts the first's session instead of
        creating a new one.  Pass {"force": true} to explicitly start a new
        session (which ends the current shared session).
        """
        context = self.client_contexts[client_uid]
        conf_uid = context.character_config.conf_uid
        force = bool(data.get("force", False)) if isinstance(data, dict) else False

        # --- Adopt the globally active session if one already exists ---
        global_uid = self._active_history_uid.get(conf_uid)
        if global_uid and not force:
            if context.history_uid != global_uid:
                # This client is joining an existing session for the first time.
                context.history_uid = global_uid
                self._session_ref_count[conf_uid] = (
                    self._session_ref_count.get(conf_uid, 0) + 1
                )
                if context.memory_manager:
                    context.memory_manager.set_current_session(global_uid)
                mem_cfg = context.system_config.persistent_memory
                if mem_cfg.enabled and hasattr(
                    context.agent_engine, "set_memory_from_recent_histories"
                ):
                    context.agent_engine.set_memory_from_recent_histories(
                        conf_uid=conf_uid,
                        n=mem_cfg.recent_sessions,
                        current_uid=global_uid,
                    )
                else:
                    context.agent_engine.set_memory_from_history(
                        conf_uid=conf_uid,
                        history_uid=global_uid,
                    )
            logger.info(
                f"Client {client_uid} adopted globally active session {global_uid}"
            )
            await websocket.send_text(
                json.dumps({"type": "new-history-created", "history_uid": global_uid})
            )
            return

        # --- End the previous session (diary + facts) if this context held one ---
        if context.memory_manager and context.history_uid:
            old_uid = context.history_uid
            # Only finalize when no other client is still on that session.
            old_ref_count = self._session_ref_count.get(conf_uid, 1) - 1
            if old_ref_count <= 0:
                from .chat_history_manager import get_history as _get_history
                old_messages = _get_history(conf_uid, old_uid)
                llm = getattr(context.agent_engine, "_llm", None)
                persona = getattr(context.agent_engine, "_system", "") or ""
                if old_messages and llm:
                    asyncio.create_task(
                        context.memory_manager.end_of_session_async(
                            old_messages, old_uid, llm, persona=persona
                        )
                    )
                self._active_history_uid.pop(conf_uid, None)
                self._session_ref_count.pop(conf_uid, None)
                if context.memory_manager:
                    context.memory_manager.set_current_session("")
            else:
                self._session_ref_count[conf_uid] = old_ref_count

        # --- Create new session and register it globally ---
        history_uid = create_new_history(conf_uid)
        if history_uid:
            context.history_uid = history_uid
            self._active_history_uid[conf_uid] = history_uid
            self._session_ref_count[conf_uid] = 1
            if context.memory_manager:
                context.memory_manager.set_current_session(history_uid)

            mem_cfg = context.system_config.persistent_memory
            if mem_cfg.enabled and hasattr(
                context.agent_engine, "set_memory_from_recent_histories"
            ):
                context.agent_engine.set_memory_from_recent_histories(
                    conf_uid=conf_uid,
                    n=mem_cfg.recent_sessions,
                    current_uid=history_uid,
                )
            else:
                context.agent_engine.set_memory_from_history(
                    conf_uid=conf_uid,
                    history_uid=history_uid,
                )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        audio_data = data.get("audio", [])
        if audio_data:
            self.received_data_buffers[client_uid] = np.append(
                self.received_data_buffers[client_uid],
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "interrupt"})
                    )
                elif audio_bytes == b"<|RESUME|>":
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=self.client_contexts[client_uid],
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            await context.handle_config_switch(websocket, config_file_name)

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                {
                    "type": "set-model-and-conf",
                    "model_info": context.live2d_model.model_info,
                    "conf_name": context.character_config.conf_name,
                    "conf_uid": context.character_config.conf_uid,
                    "client_uid": client_uid,
                }
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")

    async def _handle_request_expression_capture(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Ask connected frontend client(s) to capture the Live2D expression faces.

        Triggered from Discord ``/refresh-faces`` via the proxy bridge. Broadcast
        to every other client; only a real frontend handles ``capture-expressions``
        (the bridge ignores it). The frontend replies with
        ``expression-capture-result``.
        """
        msg = json.dumps({"type": "capture-expressions"})
        sent = 0
        for uid, ws in list(self.client_connections.items()):
            if uid == client_uid:
                continue  # don't echo back to the requester (the bridge)
            try:
                await ws.send_text(msg)
                sent += 1
            except Exception as e:
                logger.warning(f"capture-expressions send to {uid} failed: {e}")
        logger.info(f"[expression-capture] requested; broadcast to {sent} client(s)")
        if sent == 0:
            try:
                await websocket.send_json(
                    {
                        "type": "expression-capture-complete",
                        "count": 0,
                        "error": "no frontend client connected",
                    }
                )
            except Exception:
                pass

    def _discord_faces_dir(self, client_uid: str) -> str:
        """``cache/discord_faces/<conf_uid>`` for the given client's character."""
        context = self.client_contexts.get(client_uid)
        conf_uid = (
            context.character_config.conf_uid
            if context and context.character_config
            else "default"
        )
        return os.path.join("cache", "discord_faces", conf_uid)

    async def _handle_expression_capture_begin(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Start of a face capture: (re)create a clean output directory.

        Images are streamed one-per-message (``expression-capture-chunk``) to stay
        well under the WebSocket message-size limit; a single combined message can
        run to several MB and gets the connection dropped.
        """
        out_dir = self._discord_faces_dir(client_uid)
        try:
            if os.path.isdir(out_dir):
                for name in os.listdir(out_dir):
                    if name.endswith(".png"):
                        os.remove(os.path.join(out_dir, name))
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"[expression-capture] could not clear {out_dir}: {e}")

    async def _handle_expression_capture_chunk(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Save one captured expression PNG as ``<index>.png`` (overwrite)."""
        index = data.get("index")
        data_url = data.get("image") or ""
        out_dir = self._discord_faces_dir(client_uid)
        os.makedirs(out_dir, exist_ok=True)
        try:
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(b64)
            with open(os.path.join(out_dir, f"{index}.png"), "wb") as f:
                f.write(raw)
        except Exception as e:
            logger.warning(f"failed to save expression face {index}: {e}")

    async def _handle_expression_capture_done(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """End of a face capture: announce completion back toward Discord."""
        count = data.get("count", 0)
        error = data.get("error")
        logger.info(
            f"[expression-capture] done: {count} faces in "
            f"{self._discord_faces_dir(client_uid)}"
        )
        # Broadcast so the proxy bridge (which forwards it to Discord) receives it;
        # real frontends ignore this type.
        complete = json.dumps(
            {"type": "expression-capture-complete", "count": count, "error": error}
        )
        for uid, ws in list(self.client_connections.items()):
            try:
                await ws.send_text(complete)
            except Exception:
                pass
