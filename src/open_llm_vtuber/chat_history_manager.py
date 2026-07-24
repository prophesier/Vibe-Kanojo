import os
import re
import json
import uuid
import unicodedata
from datetime import datetime
from typing import Literal, List, TypedDict, Optional
from loguru import logger


class HistoryMessage(TypedDict):
    role: Literal["human", "ai"]
    timestamp: str
    content: str
    # Optional display information for the message
    name: Optional[str]
    avatar: Optional[str]


def _is_safe_filename(filename: str) -> bool:
    """Validate filename for safety and allowed characters"""
    if not filename or len(filename) > 255:
        return False

    # Allow alphanumeric, hyphen, underscore, and common unicode characters
    # Block any filesystem special characters, control characters, and path separators
    pattern = re.compile(r"^[\w\-_\u0020-\u007E\u00A0-\uFFFF]+$")
    return bool(pattern.match(filename))


def _sanitize_path_component(component: str) -> str:
    """Sanitize and validate a path component"""
    # Remove any path components, get just the basename
    sanitized = os.path.basename(component.strip())

    if not _is_safe_filename(sanitized):
        raise ValueError(f"Invalid characters in path component: {component}")

    return sanitized


def _ensure_conf_dir(conf_uid: str) -> str:
    """Ensure the directory for a specific conf exists and return its path"""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")

    safe_conf_uid = _sanitize_path_component(conf_uid)
    base_dir = os.path.join("chat_history", safe_conf_uid)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_safe_history_path(conf_uid: str, history_uid: str) -> str:
    """Get sanitized path for history file"""
    safe_conf_uid = _sanitize_path_component(conf_uid)
    safe_history_uid = _sanitize_path_component(history_uid)
    base_dir = os.path.join("chat_history", safe_conf_uid)
    full_path = os.path.normpath(os.path.join(base_dir, f"{safe_history_uid}.json"))
    if not full_path.startswith(base_dir):
        raise ValueError("Invalid path: Path traversal detected")
    return full_path


def create_new_history(conf_uid: str) -> str:
    """Create a new history file with a unique ID and return the history_uid"""
    if not conf_uid:
        logger.warning("No conf_uid provided")
        return ""

    # Use uuid.uuid4().hex to generate a UUID without hyphens
    # New format: UUID_YYYY-MM-DD_HH-MM-SS
    history_uid = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"
    conf_dir = _ensure_conf_dir(conf_uid)  # conf_uid is sanitized here

    # Create history file with empty metadata
    try:
        filepath = os.path.join(conf_dir, f"{history_uid}.json")
        initial_data = [
            {
                "role": "metadata",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to create new history file: {e}")
        return ""

    logger.debug(f"Created new history file with empty metadata: {filepath}")
    return history_uid


def store_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai"],
    content: str,
    name: str | None = None,
    avatar: str | None = None,
    thinking_seed: dict | None = None,
):
    """Store a message in a specific history file

    Args:
        conf_uid: Configuration unique identifier
        history_uid: History unique identifier
        role: Message role ("human" or "ai")
        content: Message content
        name: Optional display name (default None)
        avatar: Optional avatar URL (default None)
        thinking_seed: Optional Claude final-message payload
            ({"model": ..., "content": [blocks]}) persisted alongside the
            visible text so a restart can re-seed thinking precedent
            (see basic_memory_agent._apply_thinking_seeds)
    """
    if not conf_uid or not history_uid:
        if not conf_uid:
            logger.warning("Missing conf_uid")
        if not history_uid:
            logger.warning("Missing history_uid")
        return

    filepath = _get_safe_history_path(conf_uid, history_uid)
    logger.debug(f"Storing {role} message to {filepath}")

    history_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception:
            logger.error(f"Failed to load history file: {filepath}")
            pass

    now_str = datetime.now().isoformat(timespec="seconds")
    new_item = {
        "role": role,
        "timestamp": now_str,
        "content": content,
    }

    # Add optional display information if provided
    if name is not None:
        new_item["name"] = name
    if avatar is not None:
        new_item["avatar"] = avatar
    if thinking_seed is not None:
        new_item["thinking_seed"] = thinking_seed

    history_data.append(new_item)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    logger.debug(f"Successfully stored {role} message")


def pop_last_message(
    conf_uid: str, history_uid: str, expected_role: Literal["human", "ai"]
) -> bool:
    """Remove the NEWEST message from a history file if it matches
    ``expected_role``.

    Used by the safety-refusal path: the refused user input must leave the
    on-disk record too, or a restart reloads it into context and the
    classifier keeps firing forever. The role guard makes a mistimed call a
    no-op instead of deleting an innocent record. Returns True when a record
    was removed."""
    if not conf_uid or not history_uid:
        return False
    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)
    except Exception:
        logger.error(f"Failed to load history file: {filepath}")
        return False
    if not history_data or history_data[-1].get("role") != expected_role:
        return False
    removed = history_data.pop()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    logger.info(
        f"Removed last '{expected_role}' message from history {history_uid} "
        f"({len(str(removed.get('content', '')))} chars)."
    )
    return True


def get_metadata(conf_uid: str, history_uid: str) -> dict:
    """Get metadata from history file"""
    if not conf_uid or not history_uid:
        return {}

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if history_data and history_data[0]["role"] == "metadata":
            return history_data[0]
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
    return {}


def update_metadate(conf_uid: str, history_uid: str, metadata: dict) -> bool:
    """Set metadata in history file

    Updates existing metadata with new fields, preserving existing ones.
    If no metadata exists, creates new metadata entry.
    """
    if not conf_uid or not history_uid:
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if history_data and history_data[0]["role"] == "metadata":
            # Update existing metadata while preserving other fields
            history_data[0].update(metadata)
        else:
            # Create new metadata with timestamp if none exists
            new_metadata = {
                "role": "metadata",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            new_metadata.update(metadata)  # Add new fields
            history_data.insert(0, new_metadata)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Updated metadata for history {history_uid}")
        return True
    except Exception as e:
        logger.error(f"Failed to set metadata: {e}")
    return False


def get_history(
    conf_uid: str, history_uid: str, *, quiet: bool = False
) -> List[HistoryMessage]:
    """Read chat history for the given conf_uid and history_uid.

    ``quiet=True`` suppresses the "file not found" warning for callers
    that legitimately expect the file might not exist (e.g. probing a
    freshly-created session whose empty metadata file was just cleaned
    up by get_history_list).
    """
    if not conf_uid or not history_uid:
        if not conf_uid:
            logger.warning("Missing conf_uid")
        if not history_uid:
            logger.warning("Missing history_uid")
        return []

    filepath = _get_safe_history_path(conf_uid, history_uid)

    if not os.path.exists(filepath):
        if not quiet:
            logger.warning(f"History file not found: {filepath}")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)
            # Filter out metadata
            return [msg for msg in history_data if msg["role"] != "metadata"]
    except Exception:
        return []


def delete_history(conf_uid: str, history_uid: str) -> bool:
    """Delete a specific history file"""
    if not conf_uid or not history_uid:
        logger.warning("Missing conf_uid or history_uid")
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Successfully deleted history file: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete history file: {e}")
    return False


def get_history_list(conf_uid: str) -> List[dict]:
    """Get list of histories with their latest messages"""
    if not conf_uid:
        return []

    histories = []
    conf_dir = _ensure_conf_dir(conf_uid)
    empty_history_uids = []

    try:
        for filename in os.listdir(conf_dir):
            if not filename.endswith(".json"):
                continue
            # Skip sidecar files that share the conf dir but aren't chat history:
            # the facts store, the self-set alarms store, and the diary/fact RAG
            # embedding indexes. (These would otherwise be parsed as sessions and
            # error out per scan.)
            if filename in ("facts.json", "alarms.json") or filename.endswith(
                ".embeddings.json"
            ):
                continue

            history_uid = filename[:-5]
            filepath = os.path.join(conf_dir, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    messages = json.load(f)

                    # Filter out metadata for checking if history is empty
                    actual_messages = [
                        msg for msg in messages if msg["role"] != "metadata"
                    ]
                    if not actual_messages:
                        empty_history_uids.append(history_uid)
                        continue

                    latest_message = actual_messages[-1]
                    history_info = {
                        "uid": history_uid,
                        "latest_message": latest_message,
                        "timestamp": (
                            latest_message["timestamp"] if latest_message else None
                        ),
                    }
                    histories.append(history_info)
            except Exception as e:
                logger.error(f"Error reading history file {filename}: {e}")
                continue

        # Clean up empty histories if there are other non-empty ones
        if len(empty_history_uids) > 0 and len(os.listdir(conf_dir)) > 1:
            for uid in empty_history_uids:
                try:
                    os.remove(os.path.join(conf_dir, f"{uid}.json"))
                    logger.info(f"Removed empty history file: {uid}")
                except Exception as e:
                    logger.error(f"Failed to remove empty history file {uid}: {e}")

        histories.sort(
            key=lambda x: x["timestamp"] if x["timestamp"] else "", reverse=True
        )
        return histories

    except Exception as e:
        logger.error(f"Error listing histories: {e}")
        return []


def modify_latest_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    new_content: str,
) -> bool:
    """Modify the latest message in a specific history file if it matches the given role"""
    if not conf_uid or not history_uid:
        logger.warning("Missing conf_uid or history_uid")
        return False

    filepath = _get_safe_history_path(conf_uid, history_uid)
    if not os.path.exists(filepath):
        logger.warning(f"History file not found: {filepath}")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if not history_data:
            logger.warning("History is empty")
            return False

        latest_message = history_data[-1]
        if latest_message["role"] != role:
            logger.warning(
                f"Latest message role ({latest_message['role']}) doesn't match requested role ({role})"
            )
            return False

        latest_message["content"] = new_content
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Successfully modified latest {role} message")
        return True

    except Exception as e:
        logger.error(f"Failed to modify latest message: {e}")
        return False


def get_recent_histories(
    conf_uid: str, n: int, exclude_uid: str = ""
) -> List[tuple[str, List[HistoryMessage]]]:
    """Return the N most recent non-empty histories, ordered oldest→newest.

    Each element is (history_uid, messages).
    exclude_uid is skipped so callers can keep the active session separate
    and append it themselves, ensuring the sliding window membership is
    stable across reconnects (preventing spurious cache invalidations).
    """
    history_list = get_history_list(conf_uid)  # already newest-first
    result = []
    for entry in history_list:
        if len(result) >= n:
            break
        uid = entry["uid"]
        if uid == exclude_uid:
            continue
        messages = get_history(conf_uid, uid)
        if messages:
            result.append((uid, messages))
    result.reverse()  # oldest first so memory builds chronologically
    return result


def get_latest_history_uid(conf_uid: str) -> str:
    """Return the most recent NON-EMPTY history uid (or "" if none).

    Used by resume mode to continue the previous conversation instead of
    starting a fresh session.
    """
    recent = get_recent_histories(conf_uid, 1)
    return recent[-1][0] if recent else ""


# ---------------------------------------------------------------------------
# history_search — keyword full-scan over the chat log (memory-tool family).
#
# Deliberately NOT RAG: the corpus is too large to embed and the use case is
# exact recall ("did we talk about X"), so a plain scan wins. Matching is
# OR + automatic bigram fragmentation: each keyword is split into contiguous
# 2-char fragments and a message containing ANY fragment is a candidate,
# ranked by fragment coverage. This is what makes a single call with the full
# shop name hit conversations that only ever mentioned part of it — the
# calling model is lazy with tools and won't retry with sub-strings itself.
# ---------------------------------------------------------------------------

_SEARCH_BUDGET_CHARS = 2800
_SEARCH_MAX_BLOCKS = 8
_SEARCH_MAX_KEYWORDS = 8
_SEARCH_BLOCK_MAX_MSGS = 7  # merged-block span cap (common-word searches)
_HIT_WINDOW = 160  # chars shown for the matched message, centered on the hit
_NEIGHBOR_WINDOW = 120  # chars shown for ±1 neighbor messages
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _search_norm(text: str) -> str:
    """Normalize for matching: NFKC (full/half-width) + lowercase."""
    return unicodedata.normalize("NFKC", text or "").lower()


def _search_fragments(keyword_norm: str) -> set:
    """Contiguous 2-char fragments of a normalized keyword (whitespace removed).
    Single-char keywords return an empty set — handled as whole-substring."""
    compact = "".join(keyword_norm.split())
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _search_coverage(
    msg_norm: str, msg_fragments: set, kw_norm: str, kw_fragments: set
) -> float:
    """Fraction of the keyword's fragments present in the message (0..1)."""
    if not kw_fragments:  # single-char keyword
        compact = "".join(kw_norm.split())
        return 1.0 if compact and compact in msg_norm else 0.0
    return len(kw_fragments & msg_fragments) / len(kw_fragments)


def _search_snippet(
    content: str, keywords_norm: List[str], width: int, is_hit: bool
) -> str:
    """One-line excerpt of a message. Hit messages get a window centered on the
    first keyword (or keyword-fragment) occurrence; neighbors get the head."""
    text = content or ""
    pos = 0
    if is_hit:
        # Position search on .lower() (length-preserving in practice, unlike
        # NFKC) so the window indexes the original text. Best-effort: a
        # full-width-only match falls back to the message head.
        low = text.lower()
        found = -1
        for kw in keywords_norm:
            compact = "".join(kw.split())
            if compact:
                found = low.find(compact)
                if found >= 0:
                    break
        if found < 0:
            for kw in keywords_norm:
                compact = "".join(kw.split())
                for i in range(len(compact) - 1):
                    found = low.find(compact[i : i + 2])
                    if found >= 0:
                        break
                if found >= 0:
                    break
        if found >= 0:
            pos = max(0, found - width // 2)
    start, end = pos, pos + width
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + " ".join(text[start:end].split()) + suffix


def search_history(
    conf_uid: str,
    keywords: List[str],
    date_from: str = "",
    date_to: str = "",
) -> dict:
    """Keyword search over ALL stored sessions of a conf (full scan, no index).

    Returns ``{"status", "total_hits", "shown_hits", "text"}`` where ``text``
    is the ready-to-read report: header (全N件中 K件表示 + narrowing hint),
    then blocks newest-first. A block is a matched message ±1 neighbor
    (overlapping/adjacent blocks in the same session are merged), each message
    clipped to a short excerpt, hit lines prefixed with ►. Budgets: max
    ``_SEARCH_MAX_BLOCKS`` blocks / ~``_SEARCH_BUDGET_CHARS`` chars — blocks
    are SELECTED by coverage score (then recency) but DISPLAYED newest-first,
    with whole oldest blocks dropped when over budget.
    """
    keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not keywords:
        return {"status": "error", "message": "keywords が空。"}
    keywords = keywords[:_SEARCH_MAX_KEYWORDS]
    for d, label in ((date_from, "date_from"), (date_to, "date_to")):
        if d and not _DATE_RE.match(d):
            return {
                "status": "error",
                "message": f"{label} は YYYY-MM-DD 形式で指定すること: {d!r}",
            }

    kw_norm = [_search_norm(k)[:40] for k in keywords]
    kw_frags = [_search_fragments(k) for k in kw_norm]

    # ---- scan: flatten all sessions, score every message -------------------
    # sessions: uid -> list of raw messages; hits: (score, ts, uid, index)
    sessions: dict = {}
    hits: List[tuple] = []
    for entry in get_history_list(conf_uid):  # newest-first
        uid = entry["uid"]
        messages = get_history(conf_uid, uid, quiet=True)
        if not messages:
            continue
        sessions[uid] = messages
        for idx, msg in enumerate(messages):
            content = str(msg.get("content", ""))
            if not content:
                continue
            ts = str(msg.get("timestamp", ""))
            day = ts[:10]
            if date_from and (not day or day < date_from):
                continue
            if date_to and (not day or day > date_to):
                continue
            msg_norm = _search_norm(content)
            msg_frags = _search_fragments(msg_norm)
            score = 0.0
            for kn, kf in zip(kw_norm, kw_frags):
                score += _search_coverage(msg_norm, msg_frags, kn, kf)
            if score > 0.0:
                hits.append((score, ts, uid, idx))

    total_hits = len(hits)
    if total_hits == 0:
        hint = "ヒットなし。別の語・より特徴的な語で試すこと。"
        if date_from or date_to:
            hint += "期間指定（date_from/date_to）を外すか広げるのも有効。"
        return {"status": "ok", "total_hits": 0, "shown_hits": 0, "text": hint}

    # ---- select blocks by (coverage desc, recency desc) --------------------
    hits.sort(key=lambda h: (h[0], h[1]), reverse=True)  # score desc, ts desc
    # blocks: per session, list of [start, end, set(hit_indexes)]
    blocks: dict = {}
    n_blocks = 0
    for _score, _ts, uid, idx in hits:
        last = len(sessions[uid]) - 1
        start, end = max(0, idx - 1), min(last, idx + 1)
        merged = False
        for blk in blocks.get(uid, []):
            if start <= blk[1] + 1 and end >= blk[0] - 1:
                new_start, new_end = min(blk[0], start), max(blk[1], end)
                if new_end - new_start + 1 <= _SEARCH_BLOCK_MAX_MSGS:
                    blk[0], blk[1] = new_start, new_end
                    blk[2].add(idx)
                elif blk[0] <= idx <= blk[1]:
                    # already inside a full block — mark as hit, don't grow
                    blk[2].add(idx)
                # else: adjacent to a full block — drop from display (a
                # separate block would duplicate lines); still in total_hits
                merged = True
                break
        if not merged:
            if n_blocks >= _SEARCH_MAX_BLOCKS:
                continue
            blocks.setdefault(uid, []).append([start, end, {idx}])
            n_blocks += 1

    # ---- render newest-first under the char budget -------------------------
    rendered: List[tuple] = []  # (block_ts, text, n_hit_msgs)
    for uid, blks in blocks.items():
        for start, end, hit_idxs in blks:
            msgs = sessions[uid]
            ts = str(msgs[start].get("timestamp", ""))
            header = f"〔{ts[:16].replace('T', ' ')}〕"
            lines = [header]
            for i in range(start, end + 1):
                m = msgs[i]
                name = m.get("name") or (
                    "ユーザー" if m.get("role") == "human" else "AI"
                )
                is_hit = i in hit_idxs
                width = _HIT_WINDOW if is_hit else _NEIGHBOR_WINDOW
                snippet = _search_snippet(
                    str(m.get("content", "")), kw_norm, width, is_hit
                )
                lines.append(f"{'► ' if is_hit else ''}{name}: {snippet}")
            rendered.append((ts, "\n".join(lines), len(hit_idxs)))
    rendered.sort(key=lambda b: b[0], reverse=True)

    shown_blocks: List[str] = []
    shown_hits = 0
    used = 0
    for _ts, text, n_hit in rendered:
        if shown_blocks and used + len(text) > _SEARCH_BUDGET_CHARS:
            break
        shown_blocks.append(text)
        shown_hits += n_hit
        used += len(text)

    header = f"全{total_hits}件ヒット中 {shown_hits}件を表示（新しい順）。"
    if shown_hits < total_hits:
        header += (
            "表示しきれない分がある——date_from/date_to で期間を絞るか、"
            "より特徴的な語で検索し直すと精度が上がる。"
        )
    return {
        "status": "ok",
        "total_hits": total_hits,
        "shown_hits": shown_hits,
        "text": header + "\n\n" + "\n\n".join(shown_blocks),
    }


def rename_history_file(
    conf_uid: str, old_history_uid: str, new_history_uid: str
) -> bool:
    """Rename a history file with a new history_uid"""
    if not conf_uid or not old_history_uid or not new_history_uid:
        logger.warning("Missing required parameters for rename")
        return False

    old_filepath = _get_safe_history_path(conf_uid, old_history_uid)
    new_filepath = _get_safe_history_path(conf_uid, new_history_uid)

    try:
        if os.path.exists(old_filepath):
            os.rename(old_filepath, new_filepath)
            logger.info(
                f"Renamed history file from {old_history_uid} to {new_history_uid}"
            )
            return True
    except Exception as e:
        logger.error(f"Failed to rename history file: {e}")
    return False
