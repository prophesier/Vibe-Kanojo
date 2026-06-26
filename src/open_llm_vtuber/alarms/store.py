"""Persistent store for self-set alarms, one JSON file per character.

Design notes:
- One file `chat_history/<conf_uid>/alarms.json`, holding a list of records.
  No database, no extra dependency — single user, low volume.
- Times are stored as absolute UTC ISO strings. Absolute + UTC means a
  restart, a timezone change, or DST never shifts a reminder; an offline
  server just finds passed alarms "overdue" when it next looks.
- A per-character singleton (`get_alarm_store`) is shared by the tool handler
  that creates alarms and the scheduler that fires them, so they share one
  lock and one `changed` event (the scheduler waits on it to re-arm its sleep).

Status machine: ``pending`` → ``delivered`` (fired and spoken) | ``cancelled``.
There is no "missed": an overdue alarm is just a pending one whose time has
passed; it is still delivered, only annotated as late by the caller.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

STATUS_PENDING = "pending"
STATUS_DELIVERED = "delivered"
STATUS_CANCELLED = "cancelled"


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_iso(value: str) -> Optional[datetime.datetime]:
    """Parse a stored ISO timestamp, tolerating a trailing 'Z'."""
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None


class AlarmStore:
    """CRUD over the per-character alarms file, plus scheduler helpers."""

    def __init__(self, conf_uid: str) -> None:
        self._conf_uid = conf_uid
        self._dir = os.path.join("chat_history", conf_uid)
        self._path = os.path.join(self._dir, "alarms.json")
        self._lock = asyncio.Lock()
        # Set whenever the pending set changes (add/cancel) so a sleeping
        # scheduler wakes and recomputes its next fire time.
        self.changed = asyncio.Event()

    # ------------------------------------------------------------------ io
    def _read(self) -> List[Dict[str, Any]]:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning(f"[alarm] {self._path} is not a list; treating as empty.")
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[alarm] failed to read {self._path}: {e}")
        return []

    def _write(self, alarms: List[Dict[str, Any]]) -> None:
        os.makedirs(self._dir, exist_ok=True)
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(alarms, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)  # atomic on the same filesystem

    def _wake_scheduler(self) -> None:
        try:
            self.changed.set()
        except RuntimeError:
            # No running loop (e.g. called from a sync context in tests).
            pass

    # -------------------------------------------------------------- public
    async def add(self, *, fire_at_utc: datetime.datetime, note: str) -> Dict[str, Any]:
        """Create a pending alarm and persist it."""
        record = {
            "id": uuid.uuid4().hex[:12],
            "conf_uid": self._conf_uid,
            "fire_at_utc": fire_at_utc.astimezone(datetime.timezone.utc).isoformat(),
            "note": (note or "").strip(),
            "created_at_utc": _now_utc().isoformat(),
            "status": STATUS_PENDING,
        }
        async with self._lock:
            alarms = self._read()
            alarms.append(record)
            self._write(alarms)
        logger.info(
            f"[alarm] set {record['id']} for {record['fire_at_utc']} "
            f"note={record['note']!r}"
        )
        self._wake_scheduler()
        return record

    async def cancel(self, alarm_id: str) -> Optional[Dict[str, Any]]:
        """Cancel a pending alarm by id. Returns the record, or None if not
        found / not pending."""
        async with self._lock:
            alarms = self._read()
            for a in alarms:
                if a.get("id") == alarm_id and a.get("status") == STATUS_PENDING:
                    a["status"] = STATUS_CANCELLED
                    a["cancelled_at_utc"] = _now_utc().isoformat()
                    self._write(alarms)
                    self._wake_scheduler()
                    logger.info(f"[alarm] cancelled {alarm_id}")
                    return a
        return None

    async def list_pending(self) -> List[Dict[str, Any]]:
        """All still-pending alarms, earliest first."""
        async with self._lock:
            pending = [a for a in self._read() if a.get("status") == STATUS_PENDING]
        pending.sort(key=lambda a: a.get("fire_at_utc", ""))
        return pending

    async def get_due(
        self, now_utc: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        """Pending alarms whose fire time has passed (read-only), earliest
        first. The caller delivers them and then calls ``mark_delivered``."""
        now = now_utc or _now_utc()
        due: List[Dict[str, Any]] = []
        async with self._lock:
            for a in self._read():
                if a.get("status") != STATUS_PENDING:
                    continue
                fire_at = _parse_iso(a.get("fire_at_utc", ""))
                if fire_at is not None and fire_at <= now:
                    due.append(a)
        due.sort(key=lambda a: a.get("fire_at_utc", ""))
        return due

    async def mark_delivered(self, alarm_ids: List[str]) -> None:
        """Mark the given alarms delivered. No-op for ids already gone."""
        ids = set(alarm_ids)
        if not ids:
            return
        async with self._lock:
            alarms = self._read()
            changed = False
            for a in alarms:
                if a.get("id") in ids and a.get("status") == STATUS_PENDING:
                    a["status"] = STATUS_DELIVERED
                    a["delivered_at_utc"] = _now_utc().isoformat()
                    changed = True
            if changed:
                self._write(alarms)

    async def next_fire_at(self) -> Optional[datetime.datetime]:
        """Earliest pending fire time (past or future), or None. The scheduler
        sleeps until this; a value in the past means 'fire now'."""
        async with self._lock:
            times = [
                t
                for a in self._read()
                if a.get("status") == STATUS_PENDING
                and (t := _parse_iso(a.get("fire_at_utc", ""))) is not None
            ]
        return min(times) if times else None


# --------------------------------------------------------------- singletons
_STORES: Dict[str, AlarmStore] = {}


def get_alarm_store(conf_uid: str) -> AlarmStore:
    """Return the shared AlarmStore for a character (created on first use)."""
    store = _STORES.get(conf_uid)
    if store is None:
        store = AlarmStore(conf_uid)
        _STORES[conf_uid] = store
    return store


# ------------------------------------------------------------ time helpers
_LOCAL_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M")


def _parse_local(
    text: str, now_local: datetime.datetime
) -> Optional[datetime.datetime]:
    """Parse a user/character-supplied local time into an aware local datetime.

    Accepts 'YYYY-MM-DD HH:MM[:SS]' or a bare 'HH:MM[:SS]'. A bare clock time
    resolves to its next occurrence (today if still ahead, else tomorrow).
    """
    text = (text or "").strip().replace("/", "-").replace("T", " ")
    tz = now_local.tzinfo
    for fmt in _LOCAL_FORMATS:
        try:
            naive = datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
        if "%Y" in fmt:
            return naive.replace(tzinfo=tz)
        # Time-only: anchor to today, roll to tomorrow if already past.
        candidate = now_local.replace(
            hour=naive.hour,
            minute=naive.minute,
            second=naive.second,
            microsecond=0,
        )
        if candidate <= now_local:
            candidate += datetime.timedelta(days=1)
        return candidate
    return None


def resolve_fire_at(
    in_minutes: Optional[float] = None, at: Optional[str] = None
) -> Tuple[Optional[datetime.datetime], Optional[str]]:
    """Turn a relative (``in_minutes``) or absolute (``at``) request into an
    absolute UTC datetime. Returns (fire_at_utc, None) or (None, error)."""
    now_local = datetime.datetime.now().astimezone()
    if in_minutes is not None:
        try:
            mins = float(in_minutes)
        except (TypeError, ValueError):
            return None, "in_minutes must be a number"
        if mins < 0:
            return None, "in_minutes must be zero or positive"
        fire_local = now_local + datetime.timedelta(minutes=mins)
        return fire_local.astimezone(datetime.timezone.utc), None
    if at:
        fire_local = _parse_local(at, now_local)
        if fire_local is None:
            return None, (
                f"could not understand the time {at!r}; "
                "use 'HH:MM' or 'YYYY-MM-DD HH:MM'"
            )
        return fire_local.astimezone(datetime.timezone.utc), None
    return None, "provide either in_minutes or at"


def format_local(utc_iso: str, *, with_date: bool = True) -> str:
    """Render a stored UTC ISO timestamp in the server's local time for display."""
    dt = _parse_iso(utc_iso)
    if dt is None:
        return utc_iso
    local = dt.astimezone()
    return local.strftime("%Y-%m-%d %H:%M" if with_date else "%H:%M")
