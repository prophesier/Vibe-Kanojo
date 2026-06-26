"""Event-driven scheduler that fires due alarms.

Not a poller: it sleeps exactly until the earliest pending alarm's time, and
re-arms immediately when one is added or cancelled (the store's ``changed``
event). When an alarm comes due it asks a delivery callback to speak it.

If nothing can receive the alarm right now (no client connected — e.g. just
after startup, before Discord/the frontend reconnect), the callback reports
"not delivered" and the alarm stays pending; the websocket layer delivers it
the moment a client connects. So the scheduler never spins on an undeliverable
alarm, and nothing is ever lost.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Awaitable, Callable, Dict, List

from loguru import logger

from .store import AlarmStore

# callback(due_alarms) -> True if delivered (and marked), False if it couldn't
# be delivered right now (no client) and should wait for a connect.
DeliverCallback = Callable[[List[Dict]], Awaitable[bool]]


class AlarmScheduler:
    def __init__(self, store: AlarmStore, deliver: DeliverCallback) -> None:
        self._store = store
        self._deliver = deliver
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._run())
            logger.info("[alarm] scheduler started.")

    async def stop(self) -> None:
        self._running = False
        self._store.changed.set()  # nudge the loop out of its wait
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Never let a bad tick kill the loop or the server.
                logger.error(f"[alarm] scheduler tick failed: {e}")
                await asyncio.sleep(5)

    async def _tick(self) -> None:
        self._store.changed.clear()
        next_at = await self._store.next_fire_at()

        if next_at is None:
            # Nothing pending — sleep until something is added/cancelled.
            await self._store.changed.wait()
            return

        now = datetime.datetime.now(datetime.timezone.utc)
        delay = (next_at - now).total_seconds()
        if delay > 0:
            # Sleep until the alarm is due, but wake early if the set changes.
            try:
                await asyncio.wait_for(self._store.changed.wait(), timeout=delay)
                return  # changed → recompute
            except asyncio.TimeoutError:
                pass  # reached the fire time

        due = await self._store.get_due()
        if not due:
            return
        logger.info(f"[alarm] {len(due)} alarm(s) due; delivering.")
        delivered = await self._deliver(due)
        if not delivered:
            # No client (ready) to receive it. Leave the alarms pending and
            # retry: wake on the next add/cancel, or re-check every 30s so a
            # client that connects a moment later still gets it (covers the
            # startup window where clients reconnect after the server).
            logger.info("[alarm] no client ready; will retry on connect / in 30s.")
            try:
                await asyncio.wait_for(self._store.changed.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass
