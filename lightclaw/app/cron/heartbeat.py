"""
Heartbeat: sensor-gated proactive care loop.

Phase 1: deterministic sensor checks (no LLM, < 100ms).
  If no triggers and not a medium/heavy window → HEARTBEAT_OK (zero LLM).
Phase 2: conditional agent turn when triggers exist or medium/heavy window.
  Low-budget, small context.  No Dreaming / prune / distill.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightclaw.config import get_heartbeat_config, get_heartbeat_query_path, load_config
from lightclaw.constant import (
    ERROR_FINGERPRINT_COOLDOWN_SECONDS,
    ERROR_FINGERPRINT_RESOLVED_AFTER_CYCLES,
    ERROR_FINGERPRINT_STATE_MAX_ENTRIES,
    ERROR_FINGERPRINT_STATE_TTL_DAYS,
    HEARTBEAT_MAX_AGENT_ITERS,
    HEARTBEAT_MAX_INPUT_TOKENS,
    HEARTBEAT_TOTAL_TIMEOUT_SECONDS,
    WORKING_DIR,
)

logger = logging.getLogger(__name__)

# ── Interval parsing ────────────────────────────────────────────────────────

_EVERY_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$",
    re.IGNORECASE,
)

# ── Thresholds ──────────────────────────────────────────────────────────────

MEMORY_MD_SIZE_THRESHOLD = 20_000       # 20 KB
DAILY_TALK_SIZE_THRESHOLD = 30_000      # 30 KB
NEW_ERROR_THRESHOLD = 3                 # new ERROR/CRITICAL in lightclaw.log
LEARNINGS_PENDING_THRESHOLD = 3         # .learnings/ pending items
MEDIUM_HEARTBEAT_INTERVAL_H = 4         # medium heartbeat every ~4 hours
CONSECUTIVE_ABNORMAL_THRESHOLD = 3       # consecutive abnormal heartbeats

# ── Heartbeat state directory ───────────────────────────────────────────────

_HEARTBEAT_DIR = WORKING_DIR / "memory" / ".heartbeat"
_LOG_FILE_TEMPLATE = "{date}.log"
_STATE_FILE = "_state.json"
_LAST_CHECK_MARKER = "_last_check_marker"
_SUMMARY_FILE = "_summary.md"
_ERROR_FINGERPRINTS_FILE = "_error_fingerprints.json"


def parse_heartbeat_every(every: str) -> int:
    """Parse interval string (e.g. '30m', '1h') to total seconds."""
    every = (every or "").strip()
    if not every:
        return 30 * 60
    m = _EVERY_PATTERN.match(every)
    if not m:
        logger.warning("heartbeat every=%r invalid, using 30m", every)
        return 30 * 60
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        return 30 * 60
    return total


def _ensure_heartbeat_dir() -> Path:
    _HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    return _HEARTBEAT_DIR


# ── State management ────────────────────────────────────────────────────────

def _load_state() -> dict[str, Any]:
    _ensure_heartbeat_dir()
    sp = _HEARTBEAT_DIR / _STATE_FILE
    if sp.is_file():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "last_check_at": None,
        "last_status": "OK",
        "consecutive_abnormal": 0,
        "heartbeat_count_today": 0,
        "last_medium_heartbeat_at": None,
        "last_heavy_heartbeat_at": None,
        "triggers": [],
        "last_log_position": 0,
    }


def _save_state(state: dict[str, Any]) -> None:
    _ensure_heartbeat_dir()
    tmp = _HEARTBEAT_DIR / (_STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _HEARTBEAT_DIR / _STATE_FILE)


def _write_heartbeat_log(status: str, triggers: list[str] | None = None,
                         details: str | None = None) -> None:
    _ensure_heartbeat_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = _HEARTBEAT_DIR / _LOG_FILE_TEMPLATE.format(date=today)
    now = datetime.now().strftime("%H:%M:%S")
    parts = [now, "|", status]
    if triggers:
        parts.append("| triggers=" + ",".join(triggers))
    if details:
        parts.append("| " + details)
    line = " ".join(parts) + "\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def _read_log_position_marker() -> int:
    _ensure_heartbeat_dir()
    mp = _HEARTBEAT_DIR / _LAST_CHECK_MARKER
    if mp.is_file():
        try:
            return int(mp.read_text(encoding="utf-8").strip())
        except (ValueError, TypeError):
            return 0
    return 0


def _write_log_position_marker(pos: int) -> None:
    _ensure_heartbeat_dir()
    tmp = _HEARTBEAT_DIR / (_LAST_CHECK_MARKER + ".tmp")
    tmp.write_text(str(pos), encoding="utf-8")
    os.replace(tmp, _HEARTBEAT_DIR / _LAST_CHECK_MARKER)


# ── Error fingerprint: deduplication + cooldown + resolved detection ────────

# Ordered list of (keyword, regex_pattern) — first match wins
_FINGERPRINT_PATTERNS: list[tuple[str, str]] = [
    ("websocket_send_failed", r"send\s*failed.*connection\s*not\s*available"),
    ("websocket_disconnect", r"(?:disconnect|reconnect)[a-z]*"),
    ("stream_turns_error", r"stream_turns\s+(?:error|fail)"),
    ("stream_chunk_timeout", r"stream(?:ing)?\s*chunk\s*timeout"),
    ("timeout", r"(?:timeout|timed?\s*out)"),
    ("connection_error", r"(?:connection|connect)\s*(?:error|fail|refused|reset)"),
    ("api_error", r"(?:API|status\s*code)\s*(?:error|fail|429|500|502|503)"),
    ("import_error", r"(?:ImportError|ModuleNotFoundError)"),
    ("syntax_error", r"SyntaxError"),
    ("runtime_error", r"(?:RuntimeError|runtime_error)"),
    ("heartbeat_failed", r"heartbeat\s+run\s+failed"),
    ("unknown_error", r"ERROR|CRITICAL"),
]

_MODULE_PATH_RE = re.compile(r"(?:lightclaw/)?((?:[a-zA-Z_][\w]*/)*[a-zA-Z_][\w]*\.py)")


def _extract_module_path(line: str) -> str:
    """Extract relative module path from a log line, e.g. app/channels/yuanbao/ws_client.py."""
    m = _MODULE_PATH_RE.search(line)
    if m:
        return m.group(1)
    return "unknown"


def _strip_noise(text: str) -> str:
    """Remove timestamps, UUIDs, numeric IDs, turn/session IDs to normalize error messages."""
    # ISO timestamps
    text = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?", " ", text)
    # UUIDs
    text = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", " ", text)
    # Long hex strings (>= 16 chars)
    text = re.sub(r"\b[0-9a-fA-F]{16,}\b", " ", text)
    # Pure digit sequences >= 4 digits
    text = re.sub(r"\b\d{4,}\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_fingerprint(line: str) -> tuple[str, str, str]:
    """Extract (error_keyword, module_path, normalized_head) for dedup.

    Returns a 3-tuple that serves as the composite fingerprint key.
    """
    module_path = _extract_module_path(line)
    normalized = _strip_noise(line)

    # Determine error keyword by pattern matching
    lower = normalized.lower()
    for keyword, pattern in _FINGERPRINT_PATTERNS:
        if re.search(pattern, lower):
            error_keyword = keyword
            break
    else:
        error_keyword = "unknown_error"

    # Truncate normalized message for the fingerprint head
    msg_head = normalized[:200] if len(normalized) > 200 else normalized

    return (error_keyword, module_path, msg_head)


def _make_fp_key(error_keyword: str, module_path: str, msg_head: str) -> str:
    """Composite fingerprint key for storage."""
    return f"{error_keyword}|{module_path}|{msg_head}"


def _load_fingerprints() -> dict[str, Any]:
    """Load error fingerprint state from disk. Returns default on any failure."""
    _ensure_heartbeat_dir()
    fp_path = _HEARTBEAT_DIR / _ERROR_FINGERPRINTS_FILE
    default: dict[str, Any] = {"version": 1, "fingerprints": {}}
    if not fp_path.is_file():
        return default
    try:
        data = json.loads(fp_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("fingerprints"), dict):
            return data
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Failed to load error fingerprints, rebuilding from scratch")
    return default


def _save_fingerprints(data: dict[str, Any]) -> None:
    """Atomically save fingerprint state to disk. Prunes stale entries."""
    fps = data.get("fingerprints", {})
    if not fps:
        return

    now = datetime.now(timezone.utc)
    ttl_seconds = ERROR_FINGERPRINT_STATE_TTL_DAYS * 86400
    max_entries = ERROR_FINGERPRINT_STATE_MAX_ENTRIES

    # Prune: remove resolved entries older than TTL, then keep most recent N
    to_remove = []
    for key, entry in fps.items():
        if entry.get("resolved"):
            try:
                last_dt = datetime.fromisoformat(entry["last_seen"])
                if (now - last_dt).total_seconds() > ttl_seconds:
                    to_remove.append(key)
            except (ValueError, TypeError):
                pass
    for key in to_remove:
        del fps[key]

    # If still too many, keep the max_entries with most recent last_seen
    if len(fps) > max_entries:
        sorted_keys = sorted(
            fps.keys(),
            key=lambda k: fps[k].get("last_seen", ""),
            reverse=True,
        )
        for key in sorted_keys[max_entries:]:
            del fps[key]

    data["fingerprints"] = fps
    _ensure_heartbeat_dir()
    tmp = _HEARTBEAT_DIR / (_ERROR_FINGERPRINTS_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _HEARTBEAT_DIR / _ERROR_FINGERPRINTS_FILE)


def _update_fingerprints_and_count(
    lines: list[str],
) -> tuple[int, dict[str, Any]]:
    """Classify error lines into fingerprints, apply cooldown + resolved detection.

    Returns (actionable_count, stats_dict) where:
      - actionable_count: number of new/actionable fingerprint hits
      - stats_dict: {"raw_error_lines": N, "actionable_fingerprints": N,
                      "suppressed_by_cooldown": N, "resolved": N,
                      "total_active_fingerprints": N}
    """
    data = _load_fingerprints()
    fps = data.get("fingerprints", {})
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cooldown = ERROR_FINGERPRINT_COOLDOWN_SECONDS
    resolved_cycles = ERROR_FINGERPRINT_RESOLVED_AFTER_CYCLES

    actionable_count = 0
    suppressed = 0
    raw_error_lines = 0

    # Track which fingerprints appear in this batch
    seen_this_cycle: set[str] = set()

    for line in lines:
        # Only match lines whose log LEVEL (prefix) is ERROR or CRITICAL,
        # not lines that merely contain those words in the message body.
        if not (line.startswith("ERROR ") or line.startswith("CRITICAL ")):
            continue
        raw_error_lines += 1

        error_keyword, module_path, msg_head = _extract_fingerprint(line)
        fp_key = _make_fp_key(error_keyword, module_path, msg_head)
        seen_this_cycle.add(fp_key)

        existing = fps.get(fp_key)

        if existing is None:
            # New fingerprint — always actionable
            fps[fp_key] = {
                "first_seen": now_iso,
                "last_seen": now_iso,
                "last_reported": now_iso,
                "seen_count": 1,
                "missed_cycles": 0,
                "resolved": False,
                "sample": line.strip()[:300],
            }
            actionable_count += 1
            continue

        # Update last_seen and seen_count
        existing["last_seen"] = now_iso
        existing["seen_count"] = existing.get("seen_count", 0) + 1
        existing["missed_cycles"] = 0

        if existing.get("resolved"):
            # Previously resolved, now re-appeared — actionable
            existing["resolved"] = False
            existing["first_seen"] = now_iso
            existing["last_reported"] = now_iso
            existing["seen_count"] = 1
            actionable_count += 1
            continue

        # Check cooldown
        last_reported = existing.get("last_reported", "")
        if last_reported:
            try:
                last_dt = datetime.fromisoformat(last_reported)
                if (now - last_dt).total_seconds() < cooldown:
                    suppressed += 1
                    continue
            except (ValueError, TypeError):
                pass

        # Cooldown expired — actionable again
        existing["last_reported"] = now_iso
        actionable_count += 1

    # Update missed_cycles for fingerprints NOT seen this cycle
    resolved_count = 0
    for fp_key, entry in fps.items():
        if fp_key not in seen_this_cycle:
            entry["missed_cycles"] = entry.get("missed_cycles", 0) + 1
            if not entry.get("resolved") and entry["missed_cycles"] >= resolved_cycles:
                entry["resolved"] = True
                resolved_count += 1

    data["fingerprints"] = fps
    _save_fingerprints(data)

    stats = {
        "raw_error_lines": raw_error_lines,
        "actionable_fingerprints": actionable_count,
        "suppressed_by_cooldown": suppressed,
        "resolved": resolved_count,
        "total_active_fingerprints": sum(1 for e in fps.values() if not e.get("resolved")),
    }
    return actionable_count, stats


# ── Sensor checks ───────────────────────────────────────────────────────────

def _count_new_errors() -> tuple[int, dict[str, Any]]:
    """Count new ERROR/CRITICAL lines with fingerprint dedup + cooldown.

    Returns (actionable_count, fingerprint_stats).
    actionable_count = number of distinct actionable fingerprints (not raw line count).
    """
    log_path = Path("/root/.lightclaw/logs/lightclaw.log")
    if not log_path.is_file():
        return 0, {}
    last_pos = _read_log_position_marker()
    try:
        file_size = log_path.stat().st_size
    except OSError:
        return 0, {}
    if file_size <= last_pos:
        _write_log_position_marker(file_size)
        return 0, {}
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            if last_pos > 0:
                f.seek(last_pos)
            new_lines = f.readlines()
    except Exception:
        return 0, {}
    _write_log_position_marker(file_size)
    return _update_fingerprints_and_count(new_lines)


def _count_learnings_pending() -> int:
    """Count pending items in workspace/.learnings/."""
    learnings_dir = WORKING_DIR / ".learnings"
    if not learnings_dir.is_dir():
        return 0
    count = 0
    for f in learnings_dir.iterdir():
        if f.is_file() and f.suffix == ".md":
            try:
                text = f.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if "**Status**:" in line and "pending" in line.lower():
                        count += 1
                    elif "Status:" in line and "pending" in line.lower():
                        count += 1
            except Exception:
                pass
    return count


def _is_medium_window(state: dict[str, Any]) -> bool:
    """Check if we've reached a medium heartbeat window (every ~4 hours)."""
    last_medium = state.get("last_medium_heartbeat_at")
    if not last_medium:
        return True
    try:
        last_dt = datetime.fromisoformat(last_medium)
        hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        return hours_since >= MEDIUM_HEARTBEAT_INTERVAL_H
    except (ValueError, TypeError):
        return True


def _is_heavy_window(state: dict[str, Any]) -> bool:
    """Heavy heartbeat: first of the day or end of day."""
    today = datetime.now().strftime("%Y-%m-%d")
    last_heavy = state.get("last_heavy_heartbeat_at", "")
    if not last_heavy or not last_heavy.startswith(today):
        return True
    return False


def _run_sensor_checks(existing_state: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Run all deterministic sensor checks. Returns (triggers, check_details)."""
    triggers: list[str] = []
    details: dict[str, Any] = {}
    heartbeat_dir = _ensure_heartbeat_dir()

    # 1. DAILY_TALK.md check
    daily_talk = WORKING_DIR / "DAILY_TALK.md"
    if daily_talk.is_file():
        dt_size = daily_talk.stat().st_size
        details["daily_talk_size"] = dt_size
        if dt_size > DAILY_TALK_SIZE_THRESHOLD:
            triggers.append(f"daily_talk_size_exceeded:{dt_size}")
    else:
        details["daily_talk_exists"] = False

    # 2. MEMORY.md size check
    memory_md = WORKING_DIR / "MEMORY.md"
    if memory_md.is_file():
        mem_size = memory_md.stat().st_size
        details["memory_size"] = mem_size
        if mem_size > MEMORY_MD_SIZE_THRESHOLD:
            triggers.append(f"memory_size_exceeded:{mem_size}")
    else:
        details["memory_exists"] = False

    # 3. .dreams/status.json last_error check
    dreams_status = WORKING_DIR / "memory" / ".dreams" / "status.json"
    if dreams_status.is_file():
        try:
            status = json.loads(dreams_status.read_text(encoding="utf-8"))
            last_error = status.get("last_error")
            details["dreaming_last_error"] = last_error
            if last_error:
                triggers.append(f"dreaming_error:{last_error[:80]}")
        except (json.JSONDecodeError, TypeError):
            details["dreaming_status_parse_error"] = True

    # 4. New errors in lightclaw.log (fingerprint-deduped + cooldown)
    new_errors, fp_stats = _count_new_errors()
    details["log_errors_new"] = new_errors
    if fp_stats:
        details["error_fingerprint_stats"] = fp_stats
    if new_errors >= NEW_ERROR_THRESHOLD:
        triggers.append(f"log_errors_new:{new_errors}")

    # 5. .learnings/ pending
    pending = _count_learnings_pending()
    details["learnings_pending"] = pending
    if pending >= LEARNINGS_PENDING_THRESHOLD:
        triggers.append(f"learnings_pending:{pending}")

    # 6. Consecutive abnormal heartbeats
    consecutive = existing_state.get("consecutive_abnormal", 0)
    if consecutive >= CONSECUTIVE_ABNORMAL_THRESHOLD:
        triggers.append(f"consecutive_abnormal:{consecutive}")

    # 7. Medium heartbeat window
    if _is_medium_window(existing_state):
        triggers.append("medium_window")
        details["is_medium_window"] = True

    # 8. Heavy heartbeat window
    if _is_heavy_window(existing_state):
        triggers.append("heavy_window")
        details["is_heavy_window"] = True

    # 9. Placeholder for unread_messages / pending_tasks
    details["unread_messages"] = "no_source"
    details["pending_tasks"] = "no_source"

    return triggers, details


# ── Agent turn boundary ─────────────────────────────────────────────────────

# Heartbeat agent tool whitelist: only safe read-only tools.
# The heartbeat agent relies primarily on the sensor summary injected into its
# prompt; these tools are available for lightweight state reading only.
_HEARTBEAT_AGENT_ALLOWED: tuple[str, ...] = (
    "read_status_file",
    "read_recent_heartbeat_errors",
    "get_current_time",
)


# ── Main entry point ────────────────────────────────────────────────────────

async def run_heartbeat_once(
    *,
    runner: Any,
    channel_manager: Any,
) -> str | None:
    """Run one heartbeat cycle: sensor-gated proactive care loop.

    Returns 'HEARTBEAT_OK' when no action needed (no LLM called).
    Returns None when an agent turn was triggered.
    """
    # ── Active hours check ──
    config = load_config()
    hb = get_heartbeat_config()
    if hb is None:
        logger.debug("heartbeat skipped: disabled")
        return "HEARTBEAT_OK"
    if not _in_active_hours(hb.active_hours):
        logger.debug("heartbeat skipped: outside active hours")
        return "HEARTBEAT_OK"

    # ── Ensure memory_manager exists ──
    mm = getattr(runner, "memory_manager", None)
    if mm is None:
        logger.debug("heartbeat skipped: no memory_manager")
        return "HEARTBEAT_OK"

    # ── Phase 1: Load state + sensor checks ──
    state = _load_state()
    now = datetime.now(timezone.utc)
    today = datetime.now().strftime("%Y-%m-%d")

    # Reset heartbeat count on new day
    state_date = (state.get("last_check_at") or "")[:10]
    if state_date != today:
        state["heartbeat_count_today"] = 0
    state["heartbeat_count_today"] += 1
    state["last_check_at"] = now.isoformat()

    triggers, details = _run_sensor_checks(state)
    state["triggers"] = triggers

    logger.info(
        "Heartbeat Phase 1: count_today=%d triggers=%s details=%s",
        state["heartbeat_count_today"], triggers,
        {k: v for k, v in details.items() if v},
    )
    # Log fingerprint stats separately for visibility
    fp_stats = details.get("error_fingerprint_stats")
    if fp_stats:
        logger.info(
            "heartbeat error sensor: raw_error_lines=%d actionable=%d "
            "suppressed_by_cooldown=%d resolved=%d active_total=%d",
            fp_stats.get("raw_error_lines", 0),
            fp_stats.get("actionable_fingerprints", 0),
            fp_stats.get("suppressed_by_cooldown", 0),
            fp_stats.get("resolved", 0),
            fp_stats.get("total_active_fingerprints", 0),
        )

    # ── Phase 2a: No trigger, no window → HEARTBEAT_OK (zero LLM) ──
    actionable_triggers = [
        t for t in triggers
        if t not in ("medium_window", "heavy_window")
    ]
    has_window_triggers = "medium_window" in triggers or "heavy_window" in triggers

    if not actionable_triggers and not has_window_triggers:
        state["last_status"] = "OK"
        state["consecutive_abnormal"] = 0
        _save_state(state)
        _write_heartbeat_log("OK")
        return "HEARTBEAT_OK"

    # ── Phase 2b: Trigger or window → conditional agent turn ──
    is_abnormal = bool(actionable_triggers)
    if is_abnormal:
        state["consecutive_abnormal"] += 1
    else:
        state["consecutive_abnormal"] = 0

    # Mark medium/heavy window as served
    if "medium_window" in triggers:
        state["last_medium_heartbeat_at"] = now.isoformat()
    if "heavy_window" in triggers:
        state["last_heavy_heartbeat_at"] = now.isoformat()

    status_str = "ERROR" if actionable_triggers else "ACTION"
    state["last_status"] = status_str
    _save_state(state)
    _write_heartbeat_log(status_str, triggers=triggers,
                        details=f"windows={'M' if 'medium_window' in triggers else ''}{'H' if 'heavy_window' in triggers else ''}")

    # ── Build small-context agent prompt ──
    path = get_heartbeat_query_path()
    heartbeat_md = ""
    if path.is_file():
        heartbeat_md = path.read_text(encoding="utf-8")

    # Read recent DAILY_TALK fragment
    daily_talk_fragment = ""
    dt_path = WORKING_DIR / "DAILY_TALK.md"
    if dt_path.is_file():
        dt_text = dt_path.read_text(encoding="utf-8")
        daily_talk_fragment = dt_text[-3000:] if len(dt_text) > 3000 else dt_text

    # Build compact triggers summary
    triggers_summary = "\n".join(f"  - {t}" for t in triggers) if triggers else "  (无异常)"
    state_summary = (
        f"last_status={state.get('last_status')} "
        f"consecutive_abnormal={state.get('consecutive_abnormal')} "
        f"count_today={state.get('heartbeat_count_today')}"
    )

    agent_query = (
        f"## 心跳状态\n{state_summary}\n\n"
        f"## 触发事项\n{triggers_summary}\n\n"
        f"## 心跳任务\n{heartbeat_md}\n\n"
        f"## 最近对话片段\n{daily_talk_fragment[-2000:]}\n"
        f"\n## 硬性约束（必须遵守）\n"
        f"- 禁止读取完整 lightclaw.log\n"
        f"- 禁止读取超过 200 行日志文件\n"
        f"- 禁止使用 shell 做大范围 grep 搜索\n"
        f"- 禁止写文件\n"
        f"- 禁止修改配置\n"
        f"- 只基于心跳 sensor summary 做判断\n"
        f"- 如果信息不足，输出 HEARTBEAT_OK 或简短告警即可，不要展开调查\n"
        f"- 最多使用 3 次工具调用，之后必须输出结论\n"
    )

    target = (hb.target or "").strip().lower()
    from lightclaw.app.cron.agent_task import run_cron_agent

    try:
        await run_cron_agent(
            query_text=agent_query,
            runner=runner,
            channel_manager=channel_manager,
            config=config,
            target=target,
            session_id="heartbeat",
            user_id="main",
            source="heartbeat",
            timeout=HEARTBEAT_TOTAL_TIMEOUT_SECONDS,
            allowed_tools=list(_HEARTBEAT_AGENT_ALLOWED),
            tool_policy_source="heartbeat",
            runtime_profile="heartbeat",
            max_input_length_override=HEARTBEAT_MAX_INPUT_TOKENS,
            max_iters_override=HEARTBEAT_MAX_AGENT_ITERS,
        )
        _write_heartbeat_log(
            "AGENT_TURN_DONE",
            triggers=triggers,
            details="heartbeat agent turn completed",
        )
    except Exception:
        logger.exception("heartbeat agent turn failed")
        _write_heartbeat_log(
            "AGENT_TURN_ERROR",
            triggers=triggers,
            details="heartbeat agent turn failed",
        )

    return None


def _in_active_hours(active_hours: Any) -> bool:
    """Return True if current local time is within [start, end]."""
    if not active_hours or not hasattr(active_hours, "start") or not hasattr(active_hours, "end"):
        return True
    try:
        from datetime import time

        start_parts = active_hours.start.strip().split(":")
        end_parts = active_hours.end.strip().split(":")
        start_t = time(
            int(start_parts[0]),
            int(start_parts[1]) if len(start_parts) > 1 else 0,
        )
        end_t = time(
            int(end_parts[0]),
            int(end_parts[1]) if len(end_parts) > 1 else 0,
        )
    except (ValueError, IndexError, AttributeError):
        return True
    now = datetime.now().time()
    if start_t <= end_t:
        return start_t <= now <= end_t
    return now >= start_t or now <= end_t
