"""Self-set alarms: the character can schedule a reminder for itself.

A single JSON file per character (no database, no extra deps). Each alarm
carries an absolute UTC fire time, so it survives process/PC restarts and a
server that was offline simply finds it overdue on the next run.
"""

from .store import AlarmStore, get_alarm_store, resolve_fire_at, format_local

__all__ = ["AlarmStore", "get_alarm_store", "resolve_fire_at", "format_local"]
