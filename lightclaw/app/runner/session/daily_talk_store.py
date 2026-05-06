"""Global daily talk memory file for prompt personalization.

Stores all users' natural-language user messages in:
  - WORKING_DIR/DAILY_TALK.md

Rollover strategy:
  - Lazy trigger on first normal conversation of the next day.
  - DAILY_TALK.md is reset for the current day (previous content is discarded).

Condensation strategy:
  - When DAILY_TALK.md exceeds DAILY_TALK_MAX_CHARS, LLM condenses it in-place
    to DAILY_TALK_CONDENSE_TARGET characters, preserving core information.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from lightclaw.constant import (
    DAILY_TALK_CONDENSE_TARGET,
    DAILY_TALK_MAX_CHARS,
    MEMORY_MD_COMPACT_TARGET,
    MEMORY_MD_MAX_CHARS,
)

logger = logging.getLogger(__name__)

_DATE_MARKER_RE = re.compile(r"<!--\s*daily_talk_date:\s*(\d{4}-\d{2}-\d{2})\s*-->")

# Minimum character count (after stripping) to consider a message worth storing.
_MIN_MEANINGFUL_CHARS = 5

# Characters that are considered "content" for the noise check.
# A message consisting only of punctuation / emoji / whitespace is discarded.
_CONTENT_CHAR_CATEGORIES = frozenset({"L", "N"})  # Unicode Letter or Number


def _is_low_value_message(text: str) -> bool:
    """Return True when a message carries too little information to store.

    Three cheap heuristic checks (no LLM required):
    1. Too short  — stripped length < _MIN_MEANINGFUL_CHARS.
    2. No letters or digits — the message is pure punctuation / emoji.
    3. Exact duplicate of the last stored message (checked by caller).
    """
    stripped = text.strip()
    if len(stripped) < _MIN_MEANINGFUL_CHARS:
        return True
    # Check whether the message contains at least one letter or digit.
    has_content = any(unicodedata.category(ch)[0] in _CONTENT_CHAR_CATEGORIES for ch in stripped)
    return not has_content


def _to_local_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if now.tzinfo is None:
        return now.astimezone()
    return now.astimezone()


def _current_day(now: datetime | None = None) -> str:
    return _to_local_now(now).strftime("%Y-%m-%d")


def _current_time(now: datetime | None = None) -> str:
    return _to_local_now(now).strftime("%H:%M:%S")


class DailyTalkStore:
    """Global user-talk memory store under WORKING_DIR."""

    def __init__(self, working_dir: str | Path) -> None:
        root = Path(working_dir)
        self._daily_path = root / "DAILY_TALK.md"
        self._lock = asyncio.Lock()

    @property
    def daily_path(self) -> Path:
        return self._daily_path

    async def needs_rollover(self, *, now: datetime | None = None) -> bool:
        """Return whether DAILY_TALK.md belongs to a previous day."""
        today = _current_day(now)
        async with self._lock:
            text = await asyncio.to_thread(self._read_text, self._daily_path)
        stored = self._extract_daily_date(text)
        return bool(stored and stored != today)

    async def append_user_message(
        self,
        text: str,
        *,
        now: datetime | None = None,
        actor_meta: dict[str, str] | None = None,
    ) -> None:
        """Append one USER_MESSAGE into today's DAILY_TALK.md.

        The message text is truncated at ``DAILY_TALK_MAX_CHARS`` to prevent
        a single oversized message from blowing up the file.

        Low-value messages are silently dropped before writing:
        - Too short (< _MIN_MEANINGFUL_CHARS chars after stripping).
        - Contains no letters or digits (pure punctuation / emoji).
        - Exact duplicate of the most recent stored message.

        Optional *actor_meta* dict adds channel/session attribution, e.g.:
        ``{"channel": "yuanbao", "chat_type": "private", "session": "yuanbao-xxx"}``.
        """
        normalized = (text or "").strip()
        if not normalized:
            return

        # Drop low-value messages early — no LLM needed.
        if _is_low_value_message(normalized):
            logger.debug("Daily talk: skipping low-value message %r", normalized[:40])
            return

        # Truncate single message to prevent oversized writes.
        if len(normalized) > DAILY_TALK_MAX_CHARS:
            normalized = normalized[:DAILY_TALK_MAX_CHARS]

        today = _current_day(now)
        stamp = _current_time(now)

        async with self._lock:
            # Duplicate check: read last stored message and skip if identical.
            existing = await asyncio.to_thread(self._read_text, self._daily_path)
            if self._is_duplicate_of_last(existing, normalized):
                logger.debug("Daily talk: skipping duplicate message %r", normalized[:40])
                return
            await asyncio.to_thread(
                self._append_message_sync,
                normalized,
                today,
                stamp,
                existing,
                actor_meta,
            )

    async def maybe_rollover(
        self,
        *,
        model: Any,
        language: str = "zh",
        now: datetime | None = None,
    ) -> bool:
        """Reset DAILY_TALK.md for today when date has changed.

        Previous day's content is simply discarded — the "fresh tail" is
        only meant to cover the current day.
        """
        today = _current_day(now)
        async with self._lock:
            daily_text = await asyncio.to_thread(self._read_text, self._daily_path)
        stored_date = self._extract_daily_date(daily_text)
        if not stored_date or stored_date == today:
            return False

        logger.info("Daily talk rollover: %s -> %s", stored_date, today)
        async with self._lock:
            await asyncio.to_thread(
                self._write_text_atomic,
                self._daily_path,
                self._daily_header(today),
            )
        return True

    async def build_prompt_context(self) -> str:
        """Build context snippets injected into system prompt."""
        async with self._lock:
            daily_text = await asyncio.to_thread(self._read_text, self._daily_path)

        parts: list[str] = []
        daily_body = self._daily_body(daily_text).strip()
        daily_date = self._extract_daily_date(daily_text) or ""
        if daily_body:
            attr = f' date="{daily_date}"' if daily_date else ""
            parts.append(f"<daily-talk-context{attr}>\n{daily_body}\n</daily-talk-context>")

        return "\n\n".join(parts)

    def _append_message_sync(self, text: str, today: str, stamp: str, existing: str = "", actor_meta: dict[str, str] | None = None) -> None:
        stored = self._extract_daily_date(existing)
        if stored != today:
            existing = self._daily_header(today)

        payload = existing.rstrip()
        if payload:
            payload += "\n\n"
        # Format metadata line: [channel=X type=Y session=Z]
        if actor_meta:
            meta_parts = []
            for key in ("channel", "chat_type", "session"):
                val = actor_meta.get(key, "")
                if val:
                    meta_parts.append(f"{key}={val}")
            if meta_parts:
                payload += f"## {stamp} | {' '.join(meta_parts)}\n{text}\n"
            else:
                payload += f"## {stamp}\n{text}\n"
        else:
            payload += f"## {stamp}\n{text}\n"
        self._write_text_atomic(self._daily_path, payload)

    async def condense_if_needed(
        self,
        *,
        invoke_llm: Any,
        language: str = "zh",
    ) -> bool:
        """Condense DAILY_TALK.md in-place when it exceeds the size threshold.

        Calls LLM to shorten the content to ``DAILY_TALK_CONDENSE_TARGET``
        characters while preserving the core information.  The condensed
        text overwrites DAILY_TALK.md (keeping today's date header).

        Returns True if condensation was performed.
        """
        async with self._lock:
            daily_text = await asyncio.to_thread(self._read_text, self._daily_path)

        if len(daily_text) < DAILY_TALK_MAX_CHARS:
            return False

        daily_body = self._daily_body(daily_text).strip()
        if not daily_body:
            return False

        logger.info(
            "Daily talk condensation triggered: %d chars (threshold %d, target %d)",
            len(daily_text),
            DAILY_TALK_MAX_CHARS,
            DAILY_TALK_CONDENSE_TARGET,
        )

        from lightclaw.agent.prompt_catalog import get_daily_talk_condensation_prompt

        system = get_daily_talk_condensation_prompt(language, target_chars=DAILY_TALK_CONDENSE_TARGET)
        try:
            result = await invoke_llm(system, daily_body)
        except Exception:
            logger.exception("Daily talk condensation LLM call failed")
            return False

        condensed = (result or "").strip()
        if not condensed:
            logger.warning("Daily talk condensation returned empty result, keeping original")
            return False

        today = _current_day()
        new_content = self._daily_header(today) + "\n" + condensed + "\n"

        async with self._lock:
            await asyncio.to_thread(
                self._write_text_atomic,
                self._daily_path,
                new_content,
            )
        logger.info(
            "Daily talk condensed from %d to %d chars",
            len(daily_text),
            len(new_content),
        )
        return True

    async def compact_memory_md_if_needed(
        self,
        *,
        invoke_llm: Any,
        language: str = "zh",
    ) -> bool:
        """Compact MEMORY.md when it exceeds the size threshold.

        Oversized content is offloaded to files under ``memory/``; the main
        MEMORY.md is rewritten with a compacted version targeting
        ``MEMORY_MD_COMPACT_TARGET`` characters.

        Returns True if compaction was performed.
        """
        root = self._daily_path.parent
        memory_md_path = root / "MEMORY.md"

        memory_text = self._read_text(memory_md_path)
        if len(memory_text) < MEMORY_MD_MAX_CHARS:
            return False

        logger.info(
            "MEMORY.md compaction triggered: %d chars (threshold %d, target %d)",
            len(memory_text),
            MEMORY_MD_MAX_CHARS,
            MEMORY_MD_COMPACT_TARGET,
        )

        from lightclaw.agent.prompt_catalog import get_memory_md_compaction_prompt

        system = get_memory_md_compaction_prompt(language, target_chars=MEMORY_MD_COMPACT_TARGET)
        try:
            result = await invoke_llm(system, memory_text)
        except Exception:
            logger.exception("MEMORY.md compaction LLM call failed")
            return False

        compacted, offloaded = self._parse_compaction_result(result)

        # Write offloaded content to separate files
        memory_dir = root / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        for item in offloaded:
            filename = item.get("filename", "").strip()
            content = item.get("content", "").strip()
            if filename and content:
                offload_path = memory_dir / filename
                await asyncio.to_thread(
                    self._write_text_atomic,
                    offload_path,
                    content,
                )
                logger.info("Offloaded %d chars to memory/%s", len(content), filename)

        # Overwrite MEMORY.md with compacted content
        if compacted.strip():
            await asyncio.to_thread(
                self._write_text_atomic,
                memory_md_path,
                compacted.strip() + "\n",
            )
            logger.info(
                "MEMORY.md compacted from %d to %d chars",
                len(memory_text),
                len(compacted),
            )
        return True

    @staticmethod
    def _parse_compaction_result(raw: str) -> tuple[str, list[dict]]:
        """Parse LLM compaction JSON output into (compacted, offloaded_list)."""
        import json as _json

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            data = _json.loads(cleaned)
            compacted = data.get("compacted", "")
            offloaded = data.get("offloaded", [])
            if not isinstance(offloaded, list):
                offloaded = []
            return compacted, offloaded
        except (_json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse compaction JSON, keeping original")
            return raw, []

    @staticmethod
    def _append_to_file_sync(path: Path, text: str) -> None:
        """Append text to a file, creating it and parent dirs if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text)

    @staticmethod
    def _daily_header(day: str) -> str:
        return f"<!-- daily_talk_date: {day} -->\n# DAILY_TALK\n"

    @staticmethod
    def _extract_daily_date(text: str) -> str:
        if not text:
            return ""
        match = _DATE_MARKER_RE.search(text)
        return match.group(1) if match else ""

    def _daily_body(self, text: str) -> str:
        if not text:
            return ""
        body = _DATE_MARKER_RE.sub("", text, count=1)
        lines = body.splitlines()
        if lines and lines[0].strip() == "# DAILY_TALK":
            lines = lines[1:]
        return "\n".join(lines).strip()

    @staticmethod
    def _is_duplicate_of_last(existing: str, new_text: str) -> bool:
        """Return True if ``new_text`` is identical to the last stored message.

        Parses the last ``## HH:MM:SS`` block from the file and compares its
        body to ``new_text`` (stripped).  This prevents storing the same
        message multiple times when the user repeats themselves.
        """
        if not existing:
            return False
        # Find all message blocks; the last one is the most recent.
        blocks = re.split(r"\n## \d{2}:\d{2}:\d{2}\n", existing)
        if len(blocks) < 2:
            return False
        last_block = blocks[-1].strip()
        return last_block == new_text.strip()

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to read %s", path)
            return ""

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
